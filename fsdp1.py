import argparse
from contextlib import contextmanager
import functools
from itertools import chain
import json
import multiprocessing
import os
import time
from pathlib import Path
import logging

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel,
    CPUOffload,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint import load, save

import tqdm
from datasets import load_dataset, load_from_disk, DatasetDict

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
)
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding

# fixes for reset_parameters not existing
LlamaRMSNorm.reset_parameters = lambda self: torch.nn.init.ones_(self.weight)
LlamaRotaryEmbedding.reset_parameters = lambda _: None

LOGGER = logging.getLogger(__name__)


@record
def main():
    parser = _get_parser()
    args = parser.parse_args()

    dist.init_process_group()

    rank = dist.get_rank()
    local_rank = rank % torch.cuda.device_count()
    world_size = dist.get_world_size()

    logging.basicConfig(
        format=f"[rank={rank}] [%(asctime)s] %(levelname)s:%(message)s",
        level=logging.INFO,
    )

    LOGGER.info(os.environ)
    LOGGER.info(args)
    LOGGER.info(f"local_rank={local_rank} rank={rank} world size={world_size}")

    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16
    torch.cuda.set_device(device)

    torch.manual_seed(args.seed)

    LOGGER.info(f"Loading model from HF_HOME={os.environ.get('HF_HOME', 'Not Set')}")


    config = AutoConfig.from_pretrained(args.model_name, use_cache=False)
    if rank == 0:
        with torch.device("cpu"):
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=dtype,
                attn_implementation="eager",
                use_cache=False,
            )
    else:
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(
                config,
                torch_dtype=dtype,
                attn_implementation="eager",
            )
    LOGGER.info(f"{sum(p.numel() for p in model.parameters())} model parameters")

    LOGGER.info(f"Before FSDP: {get_mem_stats(device)}")

    from torch.nn import Embedding
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer, Embedding},
    )
    model = FullyShardedDataParallel(
        model,
        device_id=local_rank,
        param_init_fn=lambda m: m.to_empty(device=device, recurse=False),
        sync_module_states=True,
        # NOTE: FULL_SHARD is equivalent to deepspeed ZeRO stage 3
        auto_wrap_policy=wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=CPUOffload(offload_params=args.cpu_offload == "on"),
    )

    LOGGER.info(f"After FSDP: {get_mem_stats(device)}")
    LOGGER.info(f"FSDP architecture: {model}")

    # Applying gradient checkpointing - note that only the LlamaDecoderLayer supports this,
    # so we can just reuse our existing wrap_policy.
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=checkpoint_wrapper, auto_wrap_policy=wrap_policy
    )

    # NOTE: since this can download data, make sure to do the main process first on each node
    # since we manually specified HF_HOME to be a node local drive.
    with rank_ordered(should_go_first=local_rank == 0):
        train_data = _load_and_preprocess_data(args, config)
    LOGGER.info(f"{len(train_data)} training samples")

    dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        collate_fn=default_data_collator,
        num_workers=1,
        prefetch_factor=2,
        # NOTE: this sampler will split dataset evenly across workers
        sampler=DistributedSampler(train_data, shuffle=True, drop_last=True),
    )
    LOGGER.info(f"{len(dataloader)} batches per epoch")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=1000, eta_min=args.lr * 1e-2
    )

    exp_dir: Path = Path(args.save_dir) / args.experiment_name

    # NOTE: full_state_dict=False means we will be saving sharded checkpoints.
    ckpt_opts = StateDictOptions(full_state_dict=False, cpu_offload=True)

    # attempt resume
    state = {
        "epoch": 0,
        "global_step": 0,
        "epoch_step": 0,
        "running_loss": 0,
    }
    resumed = False
    if (exp_dir / "state.json").exists():
        sharded_model_state, sharded_optimizer_state = get_state_dict(
            model, optimizer, options=ckpt_opts
        )
        load(
            dict(model=sharded_model_state, optimizer=sharded_optimizer_state),
            checkpoint_id=exp_dir / "checkpoint",
        )
        set_state_dict(
            model,
            optimizer,
            model_state_dict=sharded_model_state,
            optim_state_dict=sharded_optimizer_state,
            options=ckpt_opts,
        )
        lr_scheduler.load_state_dict(
            torch.load(
                exp_dir / "lr_scheduler.pt", map_location=device, weights_only=True
            )
        )
        with open(exp_dir / "state.json") as fp:
            state = json.load(fp)
        resumed = True
    LOGGER.info(f"Resumed={resumed} | {state}")
    dist.barrier()

    if (exp_dir.is_mount() and rank == 0) or (
        not exp_dir.is_mount() and local_rank == 0
    ):
        LOGGER.info(f"Creating experiment root directory")
        exp_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()


    timers = {k: LocalTimer(device) for k in ["data", "forward", "backward", "update"]}

    for state["epoch"] in range(state["epoch"], args.num_epochs):
        LOGGER.info(f"Begin epoch {state['epoch']} at step {state['epoch_step']}")

        progress_bar = tqdm.tqdm(range(len(dataloader)), disable=True)
        if state["epoch_step"] > 0:
            progress_bar.update(state["epoch_step"])

        dataloader.sampler.set_epoch(state["epoch"])
        batches = iter(dataloader)

        for i_step in range(len(dataloader)):
            with timers["data"], torch.no_grad():
                batch = next(batches)
                batch = {k: v.to(device=device) for k, v in batch.items()}

            if i_step < state["epoch_step"]:
                # NOTE: for resuming
                continue

            with timers["forward"]:
                outputs = model(**batch)

            with timers["backward"]:
                outputs.loss.backward()

            with timers["update"]:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.cpu_offload == "off")

            state["global_step"] += 1
            state["epoch_step"] += 1
            state["running_loss"] += outputs.loss.item()
            progress_bar.update(1)

            if state["global_step"] % args.log_freq == 0:
                tok_per_step = world_size * args.batch_size * args.seq_length
                ms_per_step = sum(t.avg_elapsed_ms() for t in timers.values())
                info = {
                    "global_step": state["global_step"],
                    "lr": lr_scheduler.get_last_lr()[0],
                    "running_loss": state["running_loss"] / args.log_freq,
                    "epoch": state["epoch"],
                    "epoch_progress": state["epoch_step"] / len(dataloader),
                    "num_batches_remaining": len(dataloader) - i_step,
                    **get_mem_stats(device),
                    "tok/s": 1000 * tok_per_step / ms_per_step,
                    "time/total": ms_per_step,
                    "time/total": sum(t.avg_elapsed_ms() for t in timers.values()),
                    **{
                        f"time/{k}": timer.avg_elapsed_ms()
                        for k, timer in timers.items()
                    },
                }

                LOGGER.info(info)

                torch.cuda.reset_peak_memory_stats(device)
                state["running_loss"] = 0
                for t in timers.values():
                    t.reset()

            if state["global_step"] % args.ckpt_freq == 0:
                LOGGER.info("Saving checkpoint.")
                dist.barrier()
                # NOTE: we have to call this on ALL ranks
                sharded_model_state, sharded_optimizer_state = get_state_dict(
                    model, optimizer, options=ckpt_opts
                )
                save(
                    dict(model=sharded_model_state, optimizer=sharded_optimizer_state),
                    checkpoint_id=exp_dir / "checkpoint",
                )
                if rank == 0:
                    torch.save(lr_scheduler.state_dict(), exp_dir / "lr_scheduler.pt")
                    with open(exp_dir / "state.json", "w") as fp:
                        json.dump(state, fp)
                dist.barrier()

        state["epoch_step"] = 0


def _load_and_preprocess_data(args, config):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Detect if args.dataset_name is a local path or a Hugging Face dataset name
    if os.path.exists(args.dataset_name):
        data = load_from_disk(args.dataset_name)
    else:
        data = load_dataset(args.dataset_name, trust_remote_code=True)

    # Robustly handle both DatasetDict and Dataset
    if isinstance(data, DatasetDict):
        dataset = data.get("train", next(iter(data.values())))
    else:
        dataset = data

    column_names = dataset.column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    seq_length = args.seq_length or tokenizer.model_max_length
    if seq_length > config.max_position_embeddings:
        seq_length = min(1024, config.max_position_embeddings)

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length > seq_length:
            total_length = (total_length // seq_length) * seq_length
        result = {
            k: [t[i : i + seq_length] for i in range(0, total_length, seq_length)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {seq_length}",
    )

    # If your dataset is a DatasetDict (with splits), return ["train"], else just return it.
    return lm_datasets["train"] if isinstance(lm_datasets, DatasetDict) else lm_datasets


def get_mem_stats(device=None):
    mem = torch.cuda.memory_stats(device)
    props = torch.cuda.get_device_properties(device)
    return {
        "total_mem_in_gb": 1e-9 * props.total_memory,
        "curr_alloc_in_gb": 1e-9 * mem["allocated_bytes.all.current"],
        "peak_alloc_in_gb": 1e-9 * mem["allocated_bytes.all.peak"],
        "curr_resv_in_gb": 1e-9 * mem["reserved_bytes.all.current"],
        "peak_resv_in_gb": 1e-9 * mem["reserved_bytes.all.peak"],
    }


@contextmanager
def rank_ordered(*, should_go_first: bool):
    if should_go_first:
        yield
    dist.barrier()
    if not should_go_first:
        yield
    dist.barrier()


class LocalTimer:
    def __init__(self, device: torch.device):
        if device.type == "cpu":
            self.synchronize = lambda: torch.cpu.synchronize(device=device)
        elif device.type == "cuda":
            self.synchronize = lambda: torch.cuda.synchronize(device=device)
        self.measurements = []
        self.start_time = None

    def __enter__(self):
        self.synchronize()
        self.start_time = time.time()
        return self

    def __exit__(self, type, value, traceback):
        if traceback is None:
            self.synchronize()
            end_time = time.time()
            self.measurements.append(end_time - self.start_time)
        self.start_time = None

    def avg_elapsed_ms(self):
        return 1000 * (sum(self.measurements) / len(self.measurements))

    def reset(self):
        self.measurements = []
        self.start_time = None


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment-name", default=None, required=True)
    parser.add_argument("-d", "--dataset-name", default=None, required=True)
    parser.add_argument("-m", "--model-name", default=None, required=True)
    parser.add_argument("--save-dir", default="../outputs")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num-epochs", default=6, type=int)
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument("-b", "--batch-size", default=1, type=int)
    parser.add_argument("--log-freq", default=100, type=int)
    parser.add_argument("--ckpt-freq", default=15, type=int)
    parser.add_argument("-s", "--seq-length", default=1024, type=int)
    parser.add_argument("--cpu-offload", default="on", choices=["on", "off"])
    return parser


if __name__ == "__main__":
    main()
