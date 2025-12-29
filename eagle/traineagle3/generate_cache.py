#!/usr/bin/env python3
"""
Generate cache.pt file before training to avoid multiprocessing issues with loaded model.
This should be run before starting the main training.
"""
import argparse
import torch
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer
import multiprocessing

def build_input_mask_qwen(tokenizer, messages):
    """Build input_ids and loss_mask for Qwen2/Qwen3 models."""
    all_input_ids = []
    all_loss_mask = []
    im_start = "<|im_start|>"
    im_end = "<|im_end|>"

    for i, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]
        formatted = f"{im_start}{role}\n{content}{im_end}\n"
        tokens = tokenizer(formatted, add_special_tokens=(i==0)).input_ids

        if role == "assistant":
            # Compute loss on assistant content and <|im_end|> token
            header = f"{im_start}{role}\n"
            header_tokens = tokenizer(header, add_special_tokens=False).input_ids
            header_len = len(header_tokens)
            footer_no_loss_len = 1  # only the final \n
            content_and_im_end_len = len(tokens) - header_len - footer_no_loss_len
            if content_and_im_end_len > 0:
                mask = [0] * header_len + [1] * content_and_im_end_len + [0] * footer_no_loss_len
            else:
                mask = [0] * len(tokens)
            if len(mask) != len(tokens):
                mask = mask[:len(tokens)] if len(mask) > len(tokens) else mask + [0] * (len(tokens) - len(mask))
        else:
            mask = [0] * len(tokens)

        all_input_ids.extend(tokens)
        all_loss_mask.extend(mask)

    return all_input_ids, all_loss_mask


def process_data(data_chunk):
    token_dict = Counter()
    input_ids = data_chunk["input_ids"]
    loss_mask = data_chunk["loss_mask"]
    for i in range(len(input_ids)):
        ids = input_ids[i][0]
        mask = loss_mask[i][0]
        for j in range(len(ids)):
            if mask[j] == 1:
                token_dict[ids[j]] += 1
    return token_dict


def merge_dicts(dicts):
    """Merge multiple Counter dicts"""
    result = Counter()
    for d in dicts:
        result.update(d)
    return result


def generate_cache(datapath, tokenizerpath, vocab_size, draft_vocab_size, model_type, max_len=2048):
    """Generate cache.pt file with draft vocabulary mapping."""

    print(f"Loading tokenizer from {tokenizerpath}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizerpath)

    print(f"Loading dataset from {datapath}...")
    dataset = load_dataset('json', data_files=datapath)
    dataset = dataset['train']
    original_columns = dataset.column_names

    print(f"Dataset size: {len(dataset)}")

    def preprocess_function(examples):
        new_examples = {
            "input_ids": [],
            "loss_mask": []
        }

        for i in range(len(examples['id'])):
            # Parse messages (support both ShareGPT and OpenAI format)
            if 'messages' in examples and examples['messages'][i]:
                messages = examples['messages'][i]
            elif 'conversations' in examples and examples['conversations'][i]:
                # Convert ShareGPT to OpenAI format
                roles = {"human": "user", "gpt": "assistant", "system": "system"}
                source = examples['conversations'][i]
                if not source:
                    continue
                messages = []
                for conv in source:
                    role = roles.get(conv.get("from", ""), None)
                    if role:
                        messages.append({"role": role, "content": conv.get("value", "")})
            else:
                continue

            if not messages:
                continue

            # Add system prompt if not present
            if messages[0]["role"] != "system":
                if model_type in ["qwen2", "qwen3"]:
                    system_prompt = "You are a helpful assistant."
                else:
                    system_prompt = "You are a helpful, respectful and honest assistant."
                messages = [{"role": "system", "content": system_prompt}] + messages

            # Build input_ids and loss_mask based on model type
            if model_type in ["qwen2", "qwen3"]:
                input_ids_list, loss_mask_list = build_input_mask_qwen(tokenizer, messages)
                if len(input_ids_list) > max_len:
                    continue
                if len(input_ids_list) == 0:
                    continue
                input_ids = torch.tensor(input_ids_list)
                loss_mask = torch.tensor(loss_mask_list)
            else:
                # LLaMA3 format
                conversation = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                if not tokenizer.pad_token_id:
                    tokenizer.pad_token_id = tokenizer.unk_token_id
                input_ids = tokenizer(conversation, return_tensors="pt", add_special_tokens=False).input_ids[0]
                if len(input_ids) > max_len:
                    continue
                loss_mask = torch.ones_like(input_ids)

            new_examples["input_ids"].append(input_ids[None, :])
            new_examples["loss_mask"].append(loss_mask[None, :])

        return new_examples

    print("Processing dataset...")
    num_proc = min(8, len(dataset))  # Use appropriate number of processes
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
        load_from_cache_file=False
    )

    print("Computing token statistics...")
    num_processes = min(num_proc, len(dataset))
    chunk_size = len(dataset) // num_processes + (len(dataset) % num_processes > 0)
    chunks = [dataset[i:i + chunk_size] for i in range(0, len(dataset), chunk_size)]

    with multiprocessing.Pool(num_processes) as pool:
        results = pool.map(process_data, chunks)

    token_dict = merge_dicts(results)

    total_frequency = sum(token_dict.values())
    top_N = token_dict.most_common(draft_vocab_size)
    top_N_frequency_sum = sum(freq for key, freq in top_N)
    top_N_ratio = top_N_frequency_sum / total_frequency

    print(f"Total tokens: {total_frequency}")
    print(f"Top {draft_vocab_size} token frequency ratio: {top_N_ratio:.2%}")

    used_tokens = [key for key, freq in top_N]
    used_tokens.sort()
    d2t = [used_tokens[i] - i for i in range(len(used_tokens))]
    t2d = [i in used_tokens for i in range(vocab_size)]
    d2t = torch.tensor(d2t)
    t2d = torch.tensor(t2d)

    cache = {
        "d2t": d2t,
        "t2d": t2d
    }

    print("Saving cache.pt...")
    torch.save(cache, "cache.pt")
    print("Done! cache.pt has been generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate cache.pt for EAGLE training')
    parser.add_argument('--trainpath', type=str, required=True, help='Path to training data JSONL file')
    parser.add_argument('--tokenizerpath', type=str, required=True, help='Path to tokenizer')
    parser.add_argument('--model_type', type=str, default='qwen3', choices=['llama3', 'qwen2', 'qwen3'])
    parser.add_argument('--vocab_size', type=int, default=152064, help='Full vocabulary size')
    parser.add_argument('--draft_vocab_size', type=int, default=80000, help='Draft vocabulary size')
    parser.add_argument('--max_len', type=int, default=2048, help='Maximum sequence length')

    args = parser.parse_args()

    generate_cache(
        datapath=args.trainpath,
        tokenizerpath=args.tokenizerpath,
        vocab_size=args.vocab_size,
        draft_vocab_size=args.draft_vocab_size,
        model_type=args.model_type,
        max_len=args.max_len
    )
