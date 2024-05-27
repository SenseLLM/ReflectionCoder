import os
import shutil

import torch.distributed as dist

from transformers import set_seed
from utils.model_utils import get_model
from utils.trainer import CodeLLMTrainer

from dataclasses import field, dataclass
from datasets import load_dataset, concatenate_datasets
from transformers import HfArgumentParser, TrainingArguments

@dataclass
class ReflexionTrainingArguments(TrainingArguments):

    # data
    max_len: int = field(default=4096)
    num_workers: int = field(default=64)
    train_file: list[str] = field(default=None)

    # model
    model_cfg: str = field(default=None)

    # mask
    all_mask: bool = field(default=False)
    block_mask: bool = field(default=False)
    
    random_block_mask: bool = field(default=False)
    sequential_block_mask: bool = field(default=False)

    random_token_mask: bool = field(default=False)
    sequential_token_mask: bool = field(default=False)
    
    block_order: str = field(default='etc')

    logit: bool = field(default=False)

    def __post_init__(self):
        super().__post_init__()

        block_trans = dict(t="<|text|>", c="<|code|>", e="<|execution|>")
        self.block_order = [block_trans[s] for s in self.block_order]

        self.gradient_checkpointing_kwargs = dict(use_reentrant=False)

def set_env(args):
    if os.path.exists(args.output_dir):
        if args.overwrite_output_dir:
            if args.process_index == 0:
                shutil.rmtree(args.output_dir)
        else:
            raise ValueError("Output directory already exists.")
    
    set_seed(args.seed)

    if args.world_size > 1:
        dist.barrier()

def tokenize_dataset(args, processor, files):

    with args.main_process_first(desc="dataset map tokenization"):

        train_sets, train_lens = [], []

        for file in files:
            dataset = load_dataset('json', data_files=file, split='train')

            dataset = dataset.map(
                processor.process_tokenize,
                batched=True,
                num_proc=args.num_workers,
                remove_columns=list(dataset.features),
                desc="Running tokenizer on dataset",
            )
        
            train_sets.append(dataset)
            train_lens.append(len(dataset))
    
        train_sets = concatenate_datasets(train_sets)
        train_sets.lengths = train_lens
    
    return train_sets

def train():
    parser = HfArgumentParser(ReflexionTrainingArguments)
    
    args = parser.parse_args_into_dataclasses()[0]
    
    set_env(args)

    model, tokenizer, processor = get_model(args)

    train_set = tokenize_dataset(args, processor, args.train_file)

    trainer = CodeLLMTrainer(
        args=args,
        model=model, 
        tokenizer=tokenizer,
        train_dataset=train_set,
    )

    trainer.train()

    trainer.save_model(os.path.join(args.output_dir, "checkpoint-final"))

if __name__ == "__main__":
    train()
