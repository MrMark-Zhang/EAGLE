import argparse
import deepspeed

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--basepath', type=str, default='/workspace/Qwen3-8B')
parser.add_argument('--trainpath', type=str,
                    default="/workspace/EAGLE/eagle/traineagle3/data/train_data_filtered.jsonl")
parser.add_argument('--testpath', type=str,
                    default="/workspace/EAGLE/eagle/traineagle3/data/test_data_filtered.jsonl")
parser.add_argument('--savedir', type=str, default='0')
parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()
import json
import re

deepspeed_config = args.deepspeed_config
with open(deepspeed_config) as f:
    ds_config = json.load(f)
train_config = {
    "bs": ds_config["train_micro_batch_size_per_gpu"],
    "num_epochs": 40,
    "num_workers": 2,
    "max_len": 2048,
    "config_path": "config.json",
    "gradient_checkpointing": True
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
        new_examples = {
            "attention_mask": [],
            "input_ids": [],
            "loss_mask": []
        }
        # Only show debug output for first example to avoid overwhelming logs
        debug_example_idx = 0

        for i in range(len(examples['id'])):
            messages = [
                {"role": "system",
                 "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
            ]
            convroles = ["user", "assistant"]
            roles = {"human": "user", "gpt": "assistant"}
            source = examples['conversations'][i]
            if not source:
                continue
            if roles[source[0]["from"]] != "user":
                # Skip the first one if it is not from human
                source = source[1:]
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == convroles[j % 2], f"{i}"
                # if sentence["from"]=="gpt":
                #     sentence["value"]=" "+sentence["value"]
                messages.append(
                    {"role": role, "content": sentence["value"]}
                )
            conversation = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id = tokenizer.unk_token_id

            input_ids = tokenizer(
                conversation,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids[0]
            # filtering out the samples which is longer than max_len
            if len(input_ids) > train_config["max_len"]:
                continue
            loss_mask = torch.ones_like(input_ids)
            # print(i)

            sep = "<|im_end|>\n<|im_start|>assistant\n"        #qwen3-8b针对assistant的分隔符
            sep2 = "<|im_end|>\n<|im_start|>user\n"

            total_len = len(input_ids)

            # Debug: Print original conversation (only for first example)
            if i == debug_example_idx:
                print("=" * 80)
                print(f"ORIGINAL CONVERSATION (Example {i}):")
                print("-" * 80)
                print(conversation)
                print("-" * 80)

            turns = conversation.split(sep2)

            # Debug: Print result after splitting by sep2
            if i == debug_example_idx:
                print(f"\nAFTER SPLITTING BY sep2 ('{sep2.encode('unicode_escape').decode()}'):")
                print(f"Number of turns: {len(turns)}")
                for idx, turn in enumerate(turns):
                    print(f"\n--- Turn {idx} ---")
                    print(turn)
                print("-" * 80)

            turns[1] = turns[0] + sep2 + turns[1]
            turns = turns[1:]

            # Debug: Print reconstructed turns
            if i == debug_example_idx:
                print(f"\nRECONSTRUCTED TURNS (after turns[1:]):")
                print(f"Number of turns: {len(turns)}")
                for idx, turn in enumerate(turns):
                    print(f"\n--- Reconstructed Turn {idx} ---")
                    print(turn)
                print("-" * 80)

            cur_len = 1
            loss_mask[:cur_len] = 0

            if i == debug_example_idx:
                print(f"\nPROCESSING EACH TURN (splitting by sep):")
                print(f"sep = '{sep.encode('unicode_escape').decode()}'")
                print("-" * 80)

            for turn_idx, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)

                parts = turn.split(sep)

                # Debug: Print parts after splitting by sep
                if i == debug_example_idx:
                    print(f"\n>>> Turn {turn_idx} - Split into {len(parts)} parts:")
                    for part_idx, part in enumerate(parts):
                        print(f"\n  [Part {part_idx}]:")
                        print(f"  {part}")

                if len(parts) != 2:
                    if i == debug_example_idx:
                        print(f"  WARNING: Expected 2 parts, got {len(parts)}. Breaking.")
                    break
                parts[0] += sep

                if i == debug_example_idx:
                    print(f"\n  After adding sep to Part 0:")
                    print(f"  Part 0 (instruction): {parts[0]}")
                    print(f"  Part 1 (response): {parts[1]}")
                    print(f"  Part 0 token length: {len(tokenizer(parts[0]).input_ids) - 1}")
                    print(f"  Turn token length: {turn_len}")
                # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

                # Ignore the user instructions
                if turn_idx == 0:
                    loss_mask[cur_len: cur_len + instruction_len - 2] = 0
                else:
                    loss_mask[cur_len - 3: cur_len + instruction_len + 1] = 0
                cur_len += turn_len
                if turn_idx != 0:
                    cur_len += 3
                # cur_len+=2

                # if i != 0 and not tokenizer.legacy:
                #     # The legacy and non-legacy modes handle special tokens differently
                #     cur_len -= 1

            loss_mask[cur_len:] = 0
            attention_mask = torch.ones_like(loss_mask)

            # Debug: Print final summary
            if i == debug_example_idx:
                print("\n" + "=" * 80)
                print("FINAL SUMMARY:")
                print("-" * 80)
                print(f"Total input_ids length: {len(input_ids)}")
                print(f"Total conversation length: {total_len}")
                print(f"Final cur_len: {cur_len}")
                print(f"Loss mask shape: {loss_mask.shape}")
                print(f"Number of tokens to train on (loss_mask=1): {loss_mask.sum().item()}")
                print(f"Number of tokens to ignore (loss_mask=0): {(loss_mask == 0).sum().item()}")
                print("=" * 80 + "\n")

            # new_examples["conversation"].append(conversation)
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
model = Model(config, ds_config, train_config, path=args.basepath, load_emb=True, load_head=True)
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
# Disabled wandb logging
# if global_rank == 0:
#     import wandb
#
#     wandb.login(key="")
#     wandb.init(project="l382", entity="yuhui-li", config=ds_config)

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
            # wandb.log(logdict)  # Disabled wandb logging
        epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
        epoch_plosses = [epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))]


    for i in range(len(epoch_acces)):
        acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
        deepspeed.comm.all_reduce(acc_i, op=deepspeed.comm.ReduceOp.AVG)
        acc_i = acc_i.item()
        if global_rank == 0:
            # wandb.log({f"train/epochacc_{i}": acc_i})  # Disabled wandb logging
            print(f"Train Epoch [{epoch + 1}/{num_epochs}], position {i},  Acc: {acc_i:.2f}")

    for i in range(len(epoch_plosses)):
        loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
        deepspeed.comm.all_reduce(loss_i, op=deepspeed.comm.ReduceOp.AVG)
        loss_i = loss_i.item()
        if global_rank == 0:
            # wandb.log({f"train/epochploss_{i}": loss_i})  # Disabled wandb logging
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
            # wandb.log({f"test/epochacc_{i}": acc_i})  # Disabled wandb logging
            print(f"Test Epoch [{epoch + 1}/{num_epochs}], position {i},  Acc: {acc_i:.2f}")

    for i in range(len(epoch_plosses)):
        loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
        deepspeed.comm.all_reduce(loss_i, op=deepspeed.comm.ReduceOp.AVG)
        loss_i = loss_i.item()
        if global_rank == 0:
            # wandb.log({f"test/epochploss_{i}": loss_i})  # Disabled wandb logging
            print(f"Test Epoch [{epoch + 1}/{num_epochs}], position {i}, pLoss: {loss_i:.2f}")
    # clear out the redundance cahce after each step
    torch.cuda.empty_cache()

    model_engine.save_16bit_model(f"{args.savedir}/state_{epoch}", exclude_frozen_parameters=True)
    if epoch % 10 == 0:
        deepspeed.DeepSpeedEngine.save_checkpoint(model_engine, save_dir=f"{args.savedir}/state_{epoch}")
