import argparse
import deepspeed

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--basepath', type=str, default='/home/lyh/weights/hf/llama31chat/8B/')
parser.add_argument('--trainpath', type=str,
                    default="/home/lyh/code/nlp/developing/vllmbase/vllm/gedata/l318b.jsonl")
parser.add_argument('--testpath', type=str,
                    default="/home/lyh/code/nlp/developing/vllmbase/vllm/gedata/0318.json")
parser.add_argument('--savedir', type=str, default='0')
parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
parser.add_argument('--model_type', type=str, default='qwen3',
                    choices=['llama3', 'qwen2', 'qwen3'], help='Target model type for chat template')
parser.add_argument('--data_format', type=str, default='openai',
                    choices=['sharegpt', 'openai'], help='Input data format')
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()
import json
import re

deepspeed_config = args.deepspeed_config
with open(deepspeed_config) as f:
    ds_config = json.load(f)
train_config = {
    "bs": ds_config["train_micro_batch_size_per_gpu"],
    "num_epochs": 2,
    "num_workers": 2,
    "max_len": 2048,
    "config_path": "config.json",
    "gradient_checkpoint": True
}

from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
from cnets import padding

torch.backends.cuda.matmul.allow_tf32 = True
from accelerate.utils import set_seed

set_seed(0)
from cnets import Model
from configs import EConfig
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
# import accelerate
import numpy as np
from transformers import PreTrainedTokenizerBase, get_linear_schedule_with_warmup


# ============= Helper Functions for Multi-turn Conversation =============

def convert_sharegpt_to_openai(conversations):
    """Convert ShareGPT format to OpenAI format."""
    if not conversations:
        return []

    roles = {"human": "user", "gpt": "assistant", "system": "system"}
    messages = []

    for conv in conversations:
        role = roles.get(conv.get("from", ""), None)
        if role is None:
            continue
        messages.append({
            "role": role,
            "content": conv.get("value", "")
        })

    # Ensure first message is from user (skip if from assistant)
    if messages and messages[0]["role"] == "assistant":
        messages = messages[1:]

    return messages


def get_default_system_prompt(model_type):
    """Get default system prompt for model type."""
    if model_type in ["qwen2", "qwen3"]:
        return "You are a helpful assistant."
    else:  # llama3
        return ("You are a helpful, respectful and honest assistant. "
                "Always answer as helpfully as possible, while being safe. "
                "Your answers should not include any harmful, unethical, racist, sexist, "
                "toxic, dangerous, or illegal content. Please ensure that your responses "
                "are socially unbiased and positive in nature.\n\n"
                "If a question does not make any sense, or is not factually coherent, "
                "explain why instead of answering something not correct. If you don't know "
                "the answer to a question, please don't share false information.")


def validate_messages(messages):
    """Validate message structure."""
    if not messages:
        return False

    # Check alternating user/assistant pattern (after system)
    start_idx = 1 if messages[0]["role"] == "system" else 0
    expected_roles = ["user", "assistant"]

    for i, msg in enumerate(messages[start_idx:]):
        expected = expected_roles[i % 2]
        if msg["role"] != expected:
            return False

    return True


def build_input_mask_qwen2(tokenizer, messages):
    """
    Build input_ids and loss_mask for Qwen2/2.5 models.

    Qwen2 chat format:
    <|im_start|>system
    {content}<|im_end|>
    <|im_start|>user
    {content}<|im_end|>
    <|im_start|>assistant
    {content}<|im_end|>

    Strategy: Tokenize segment by segment to avoid offset calculation errors.
    Only compute loss on assistant content (excluding header and footer).
    """
    all_input_ids = []
    all_loss_mask = []

    im_start = "<|im_start|>"
    im_end = "<|im_end|>"

    for i, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]

        # Build the formatted message
        # Format: <|im_start|>{role}\n{content}<|im_end|>\n
        formatted = f"{im_start}{role}\n{content}{im_end}\n"
        tokens = tokenizer(formatted, add_special_tokens=(i==0)).input_ids

        if role == "assistant":
            # Compute loss on assistant content and <|im_end|> token
            # Header: <|im_start|>assistant\n (no loss)
            header = f"{im_start}{role}\n"
            header_tokens = tokenizer(header, add_special_tokens=False).input_ids
            header_len = len(header_tokens)

            # Footer: only final \n has no loss, <|im_end|> should have loss
            # <|im_end|>\n is 2 tokens: [151645, 198]
            footer_no_loss_len = 1  # only the final \n
            content_and_im_end_len = len(tokens) - header_len - footer_no_loss_len

            if content_and_im_end_len > 0:
                mask = [0] * header_len + [1] * content_and_im_end_len + [0] * footer_no_loss_len
            else:
                mask = [0] * len(tokens)

            # Ensure mask length matches tokens length
            if len(mask) != len(tokens):
                mask = mask[:len(tokens)] if len(mask) > len(tokens) else mask + [0] * (len(tokens) - len(mask))
        else:
            # No loss on system/user messages
            mask = [0] * len(tokens)

        all_input_ids.extend(tokens)
        all_loss_mask.extend(mask)

    return all_input_ids, all_loss_mask


def build_input_mask_llama3(tokenizer, messages):
    """
    Build input_ids and loss_mask for LLaMA3 models.

    LLaMA3 chat format:
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    {assistant_message}<|eot_id|>
    """
    all_input_ids = []
    all_loss_mask = []

    for i, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]

        # Build header tokens
        if i == 0:
            header = f"<|begin_of_text|><|start_header_id|>{role}<|end_header_id|>\n\n"
        else:
            header = f"<|start_header_id|>{role}<|end_header_id|>\n\n"

        footer = "<|eot_id|>"

        # Tokenize separately
        header_tokens = tokenizer(header, add_special_tokens=False).input_ids
        content_tokens = tokenizer(content, add_special_tokens=False).input_ids
        footer_tokens = tokenizer(footer, add_special_tokens=False).input_ids

        all_tokens = header_tokens + content_tokens + footer_tokens

        # Build mask
        if role == "assistant":
            mask = [0] * len(header_tokens) + [1] * len(content_tokens) + [0] * len(footer_tokens)
        else:
            mask = [0] * len(all_tokens)

        all_input_ids.extend(all_tokens)
        all_loss_mask.extend(mask)

    return all_input_ids, all_loss_mask


# ============= End of Helper Functions =============


def build_dataset_rank(
        tokenizer, datapath
):

    ds = load_dataset('json', data_files=datapath)
    ds = ds['train']
    ds = ds.shuffle(seed=42)
    ds1 = ds
    original_columns1 = ds1.column_names
    num_proc = 8

    def preprocess_function(examples):
        """
        Preprocess conversation data with proper loss masking.
        Supports both OpenAI and ShareGPT formats, and Qwen2/Qwen3/LLaMA3 models.

        Loss mask: 1 for assistant content (compute loss), 0 for system/user (ignore loss)
        """
        new_examples = {
            "attention_mask": [],
            "input_ids": [],
            "loss_mask": []
        }

        model_type = args.model_type
        data_format = args.data_format

        for i in range(len(examples['id'])):
            # Step 1: Parse messages based on data format
            if data_format == 'openai':
                # OpenAI format: messages array with role/content
                messages = examples.get('messages', [[]])[i]
                if not messages:
                    continue
            else:
                # ShareGPT format: conversations with from/value
                conversations = examples.get('conversations', [[]])[i]
                if not conversations:
                    continue
                messages = convert_sharegpt_to_openai(conversations)

            if not messages:
                continue

            # Step 2: Add default system prompt if not present
            if messages[0]["role"] != "system":
                system_prompt = get_default_system_prompt(model_type)
                messages = [{"role": "system", "content": system_prompt}] + messages

            # Step 3: Validate message structure (optional, can be lenient)
            # Skip validation for flexibility with real-world data
            # if not validate_messages(messages):
            #     continue

            # Step 4: Build input_ids and loss_mask using segment-by-segment approach
            if model_type in ["qwen2", "qwen3"]:
                input_ids_list, loss_mask_list = build_input_mask_qwen2(tokenizer, messages)
            else:  # llama3
                input_ids_list, loss_mask_list = build_input_mask_llama3(tokenizer, messages)

            # Step 5: Filter by max_len
            if len(input_ids_list) > train_config["max_len"]:
                continue

            if len(input_ids_list) == 0:
                continue

            input_ids = torch.tensor(input_ids_list)
            loss_mask = torch.tensor(loss_mask_list)
            attention_mask = torch.ones_like(loss_mask)

            new_examples["input_ids"].append(input_ids[None, :])
            new_examples["loss_mask"].append(loss_mask[None, :])
            new_examples["attention_mask"].append(attention_mask[None, :])

        return new_examples

    ds1 = ds1.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns1,
        load_from_cache_file=False
    )


    ds1.set_format(type="torch")
    return ds1


class DataCollatorWithPadding:

    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        # padding_tensor = torch.zeros(B, N - n, S,dtype=intensors.dtype)
        padding_tensor = torch.zeros(B, N - n, S, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item['input_ids'].shape[1] for item in features)
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        batch_attention_mask = torch.cat(
            [self.paddingtensor2D(item['attention_mask'], max_length) for item in features])
        batch_loss_mask = torch.cat(
            [self.paddingtensor2D(item['loss_mask'], max_length) for item in features])

        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        return batch


tokenizer = AutoTokenizer.from_pretrained(args.basepath)
traindataset = build_dataset_rank(tokenizer, args.trainpath)
testdataset = build_dataset_rank(tokenizer, args.testpath)

config = EConfig.from_pretrained(train_config["config_path"])
model = Model(config, ds_config, train_config, path=args.basepath, load_emb=True, load_head=True, model_type=args.model_type)
model.scandata(args.trainpath, args.basepath)


criterion = nn.SmoothL1Loss(reduction="none")

num_epochs = train_config["num_epochs"]

model_engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                                     model=model,
                                                     model_parameters=model.parameters(),
                                                     )

global_rank = deepspeed.comm.get_rank()
rank = deepspeed.comm.get_local_rank()
world_size = deepspeed.comm.get_world_size()
if global_rank == 0:
    import wandb

    wandb.login(key="")
    wandb.init(project="l382", entity="yuhui-li", config=ds_config)

os.makedirs(args.savedir, exist_ok=True)

sampler = DistributedSampler(testdataset, num_replicas=world_size, rank=global_rank, shuffle=False)
test_loader = DataLoader(testdataset, batch_size=train_config["bs"], sampler=sampler, num_workers=4, pin_memory=True,
                         collate_fn=DataCollatorWithPadding())

train_sampler = DistributedSampler(traindataset, num_replicas=world_size, rank=global_rank, shuffle=True)
train_loader = DataLoader(traindataset, batch_size=train_config["bs"], sampler=train_sampler, num_workers=4,
                          pin_memory=True,
                          collate_fn=DataCollatorWithPadding())


def find_max_state_with_file(directory, filename="zero_to_fp32.py"):
    max_a = -1
    for subdir in os.listdir(directory):
        match = re.match(r"state_(\d+)", subdir)
        if match:
            a_value = int(match.group(1))
            subdir_path = os.path.join(directory, subdir)
            file_path = os.path.join(subdir_path, filename)
            if os.path.isdir(subdir_path) and os.path.exists(file_path):
                max_a = max(max_a, a_value)
    if max_a == -1:
        return None, 0
    return f"{directory}/state_{max_a}", max_a + 1


checkpoint_path, start_epoch = find_max_state_with_file(args.savedir)
if checkpoint_path:
    print(f"load from {checkpoint_path}")
    model_engine.load_checkpoint(checkpoint_path)



for epoch in range(start_epoch, num_epochs):
    train_sampler.set_epoch(epoch+1)
    print(f"Now training epoch {epoch}")

    model.train()
    epoch_acces = [[] for _ in range(model.length)]
    epoch_plosses = [[] for _ in range(model.length)]


    for batch_idx, data in enumerate(tqdm(train_loader)):

        model.zero_grad()

        plosses, vlosses, acces = model_engine(input_ids=data["input_ids"].to(rank),
                                               attention_mask=data["attention_mask"].to(rank),
                                               loss_mask=data["loss_mask"],
                                               )

        ploss_weight = [0.8 ** i for i in range(len(plosses))]
        ploss = sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
        loss = ploss
        model_engine.backward(loss)


        model_engine.step()

        if global_rank == 0:
            logdict = {"train/lr": optimizer.optimizer.param_groups[0]["lr"]}
            for i in range(len(plosses)):
                logdict[f"train/ploss_{i}"] = plosses[i].item()
            for i in range(len(acces)):
                logdict[f"train/acc_{i}"] = acces[i]
            wandb.log(logdict)
        epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
        epoch_plosses = [epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))]


    for i in range(len(epoch_acces)):
        acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
        deepspeed.comm.all_reduce(acc_i, op=deepspeed.comm.ReduceOp.AVG)
        acc_i = acc_i.item()
        if global_rank == 0:
            wandb.log({f"train/epochacc_{i}": acc_i})
            print(f"Train Epoch [{epoch + 1}/{num_epochs}], position {i},  Acc: {acc_i:.2f}")

    for i in range(len(epoch_plosses)):
        loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
        deepspeed.comm.all_reduce(loss_i, op=deepspeed.comm.ReduceOp.AVG)
        loss_i = loss_i.item()
        if global_rank == 0:
            wandb.log({f"train/epochploss_{i}": loss_i})
            print(f"Train Epoch [{epoch + 1}/{num_epochs}], position {i}, pLoss: {loss_i:.2f}")

    epoch_acces = [[] for _ in range(model.length)]
    epoch_plosses = [[] for _ in range(model.length)]

    for batch_idx, data in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            plosses, vlosses, acces = model_engine(input_ids=data["input_ids"].to(rank),
                                                   attention_mask=data["attention_mask"].to(rank),
                                                   loss_mask=data["loss_mask"],
                                                   )
            epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
            epoch_plosses = [epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))]

    for i in range(len(epoch_acces)):
        acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
        deepspeed.comm.all_reduce(acc_i, op=deepspeed.comm.ReduceOp.AVG)
        acc_i = acc_i.item()
        if global_rank == 0:
            wandb.log({f"test/epochacc_{i}": acc_i})
            print(f"Test Epoch [{epoch + 1}/{num_epochs}], position {i},  Acc: {acc_i:.2f}")

    for i in range(len(epoch_plosses)):
        loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
        deepspeed.comm.all_reduce(loss_i, op=deepspeed.comm.ReduceOp.AVG)
        loss_i = loss_i.item()
        if global_rank == 0:
            wandb.log({f"test/epochploss_{i}": loss_i})
            print(f"Test Epoch [{epoch + 1}/{num_epochs}], position {i}, pLoss: {loss_i:.2f}")
    # clear out the redundance cahce after each step
    torch.cuda.empty_cache()

    model_engine.save_16bit_model(f"{args.savedir}/state_{epoch}", exclude_frozen_parameters=True)
    if epoch % 10 == 0:
        deepspeed.DeepSpeedEngine.save_checkpoint(model_engine, save_dir=f"{args.savedir}/state_{epoch}")
