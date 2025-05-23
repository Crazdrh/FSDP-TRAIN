import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, BackwardPrefetch, FullStateDictConfig, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data.distributed import DistributedSampler

# ---- Dataset ----
class PromptTextDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=2048):
        self.files = sorted(glob.glob(os.path.join(data_dir, '*.txt')))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], 'r', encoding='utf-8') as f:
            prompt = f.read()
        enc = self.tokenizer(prompt, truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = enc['input_ids'].squeeze(0)
        attention_mask = enc['attention_mask'].squeeze(0)
        return input_ids, attention_mask

def collate_fn(batch):
    input_ids, attention_masks = zip(*batch)
    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels = input_ids.clone()  # Causal LM: predict next token
    return input_ids, attention_masks, labels

# ---- Distributed init ----
def setup_distributed():
    dist.init_process_group("nccl")

def cleanup_distributed():
    dist.destroy_process_group()

# ---- Training loop ----
def train(rank, world_size, args):
    torch.cuda.set_device(rank)
    setup_distributed()
    # Tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", use_fast=True)
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    model = model.to(torch.float32)  # Explicitly set to fp32
    model = model.cuda(rank)

    # FSDP wrapping
    fsdp_policy = transformer_auto_wrap_policy
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=fsdp_policy,
        mixed_precision=MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        ),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=rank,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    )

    # Dataset and loader
    dataset = PromptTextDataset(args['data_dir'], tokenizer, max_length=args['max_length'])
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args['batch_size'],
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )

    optimizer = optim.AdamW(fsdp_model.parameters(), lr=args['lr'])
    fsdp_model.train()

    for epoch in range(args['epochs']):
        sampler.set_epoch(epoch)
        for step, (input_ids, attention_mask, labels) in enumerate(dataloader):
            input_ids = input_ids.cuda(rank, non_blocking=True)
            attention_mask = attention_mask.cuda(rank, non_blocking=True)
            labels = labels.cuda(rank, non_blocking=True)
            outputs = fsdp_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 10 == 0 and rank == 0:
                print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")

        # Save only on rank 0
        if rank == 0:
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, save_policy):
                cpu_state = fsdp_model.state_dict()
            torch.save(cpu_state, f"{args['save_dir']}/llama3.1-8b-epoch{epoch}.pt")

    cleanup_distributed()

# ---- Main ----
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=1e-5)
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    train(rank, world_size, vars(args))
