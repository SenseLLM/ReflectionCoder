import math
import torch
import numpy as np

import torch.nn.functional as F

from transformers import Trainer, TrainerState, TrainingArguments
from transformers.trainer_pt_utils import LengthGroupedSampler, get_length_grouped_indices

from torch.nn.utils.rnn import pad_sequence
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class MaskCollator:

    # message type
    user_token: str = "<|user|>"
    assistant_token: str = "<|assistant|>"

    # block type
    text_token: str = "<|text|>"
    code_token: str = "<|code|>"
    execution_token: str = "<|execution|>"

    # end block
    end_of_block_token: str = "<|endofblock|>"
    end_of_message_token: str = "<|endofmessage|>"
    
    def __init__(self, args: TrainingArguments, state: TrainerState, tokenizer: PreTrainedTokenizerBase):
        self.args = args
        self.state = state
        self.tokenizer = tokenizer

        self.np_rng = np.random.RandomState(args.seed)

        self.token2id = {}
        self.id2token = {}

        for token in [
            self.user_token, self.assistant_token, 
            self.text_token, self.code_token, self.execution_token, 
            self.end_of_block_token, self.end_of_message_token
        ]:  
            _id = tokenizer.convert_tokens_to_ids(token)
            self.token2id[token] = _id
            self.id2token[_id] = token
        
        self.block_begin = [self.token2id[token] for token in [self.text_token, self.code_token, self.execution_token]]
        self.block_end = self.token2id[self.end_of_block_token]
        
        self.mask_func = []

        for name in ['all_mask', 'random_blockmask', 'sequential_block_mask', 'block_mask', 'random_token_mask', 'sequential_token_mask']:
            if getattr(args, name, False):
                self.mask_func.append(getattr(self, name))

    def split_block(self, input_id):
        blocks, block, block_type = [], [], ''
        for token_id in input_id:
            if token_id in self.block_begin:
                block_type = self.id2token[token_id]
            
            block.append(token_id)
            
            if token_id == self.block_end:
                blocks.append((block_type, block))
                block, block_type = [], ''

        return blocks

    def all_mask(self, input_id, mask_rate):
        
        return [0] * len(input_id)
    
    def random_token_mask(self, input_id, mask_rate):
            
        num = len(input_id)
        mask_num = math.ceil(num * mask_rate)

        attention_mask = [0 for _ in range(mask_num)] + [1 for _ in range(num - mask_num)]
        self.np_rng.shuffle(attention_mask)

        return attention_mask
    
    def sequential_token_mask(self, input_id, mask_rate):

        num = len(input_id)
        mask_num = math.ceil(num * mask_rate)
        
        attention_mask = [0 for _ in range(mask_num)] + [1 for _ in range(num - mask_num)]

        return attention_mask
    
    def random_block_mask(self, input_id, mask_rate):
            
        split_blocks = self.split_block(input_id)

        mask_num = math.ceil(len(split_blocks) * mask_rate)
        mask_indices = self.np_rng.choice(len(split_blocks), mask_num, replace=False)

        attention_mask = []
        for i, (block_type, block) in enumerate(split_blocks):
            if i in mask_indices:
                attention_mask += [0] * len(block)
            else:
                attention_mask += [1] * len(block)

        return attention_mask

    def sequential_block_mask(self, input_id, mask_rate):

        split_blocks = self.split_block(input_id)

        mask_num = math.ceil(len(split_blocks) * mask_rate)

        attention_mask = []
        for i, (block_type, block) in enumerate(split_blocks):
            if i < mask_num:
                attention_mask += [0] * len(block)
            else:
                attention_mask += [1] * len(block)

        return attention_mask

    def block_mask(self, input_id, mask_rate):

        split_blocks = self.split_block(input_id)

        mask_block = [self.args.block_order[0]]
        
        if mask_rate >= 0.33:
            mask_block.append(self.args.block_order[1])
        
        if mask_rate >= 0.67:
            mask_block.append(self.args.block_order[2])

        attention_mask = []
        for block_type, block in split_blocks:
            if block_type in mask_block:
                attention_mask += [0] * len(block)
            else:
                attention_mask += [1] * len(block)

        return attention_mask
    
    def mask(self, input_id, mask_rate):
        
        if len(self.mask_func) == 0:
            return [1] * len(input_id)
        
        start = input_id.index(self.token2id[self.assistant_token]) + 1
        end = start + input_id[start:].index(self.token2id[self.end_of_message_token])
        
        func = self.np_rng.choice(self.mask_func)
        masked_attention = func(input_id[start:end], mask_rate)

        return [0] * start + masked_attention + [0] + [1] * (len(input_id) - end - 1)

    def __call__(self, inputs):
        input_ids = [torch.tensor(i['input_ids']) for i in inputs]
        labels = [torch.tensor(i['labels']) for i in inputs]

        attention_mask = [torch.ones_like(torch.tensor(i['input_ids'])) for i in inputs]

        results = {
            "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id).long(),
            "attention_mask": pad_sequence(attention_mask, batch_first=True, padding_value=0).long(),
            "labels": pad_sequence(labels, batch_first=True, padding_value=-100).long(),
        }

        if inputs[0]['type'] != 'generation' and self.args.logit:
            if self.state.max_steps == 0:
                mask_rate = 0
            else:
                cur_step = self.state.global_step + 1
                mask_rate = cur_step / self.state.max_steps
                mask_rate = min(mask_rate, 1)
                
            distillation_attention_mask = [torch.tensor(self.mask(i['input_ids'], mask_rate))  for i in inputs]
            distillation_mask = [torch.tensor(i['distillation_mask']) for i in inputs]
            distillation_label = [torch.tensor(i['distillation_labels']) for i in inputs]

            results['distillation_attention_mask'] = pad_sequence(distillation_attention_mask, batch_first=True, padding_value=0).long()
            results['distillation_mask'] = pad_sequence(distillation_mask, batch_first=True, padding_value=0).long()
            results['distillation_labels'] = pad_sequence(distillation_label, batch_first=True, padding_value=-100).long()
        
        return results

class MultiDatasetsLengthGroupedSampler(LengthGroupedSampler):
    
    def __init__(self, batch_size, datasets, model_input_name=None, seed=3407):
        self.datasets = datasets
        self.batch_size = batch_size
        self.model_input_name = model_input_name

        self.target_lengths = [l // batch_size * batch_size for l in datasets.lengths]

        self.lengths = [len(feature) for feature in datasets[model_input_name]]

        self.np_rng = np.random.RandomState(seed)

    def __iter__(self):

        target_indices = []
        
        offset = 0
        for target_len, origin_len in zip(self.target_lengths, self.datasets.lengths):
            
            indices = get_length_grouped_indices(self.lengths[offset:offset + origin_len], self.batch_size)
            indices = np.array(indices[:target_len]) + offset

            target_indices.append(indices)

            offset += origin_len
        
        target_indices = np.concatenate(target_indices).reshape(-1, self.batch_size)
        self.np_rng.shuffle(target_indices)
        target_indices = target_indices.reshape(-1).tolist()

        return iter(target_indices)

    def __len__(self):
        return sum(self.target_lengths)

class CodeLLMTrainer(Trainer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data_collator = MaskCollator(self.args, self.state, self.tokenizer)
    
    def _get_train_sampler(self):

        sampler = MultiDatasetsLengthGroupedSampler(
            self.args.train_batch_size * self.args.world_size,
            self.train_dataset,
            self.tokenizer.model_input_names[0],
            self.args.seed
        )
        
        return sampler

    def compute_loss(self, model, inputs, return_outputs=False):

        self.data_collator.state = self.state

        if "distillation_labels" not in inputs:
            inputs['use_cache'] = False
            return super().compute_loss(model, inputs, return_outputs)

        outputs = model(
            torch.cat([inputs['input_ids'], inputs['input_ids']], dim=0),
            attention_mask=torch.cat([inputs['attention_mask'], inputs['distillation_attention_mask']], dim=0),
            labels=torch.cat([inputs['labels'], inputs['distillation_labels']], dim=0),
            use_cache=False
        )

        vocab_size = outputs.logits.size(-1)

        mask = torch.ne(inputs['distillation_labels'].unsqueeze(-1), -100)

        teacher_logits, student_logits = torch.chunk(outputs.logits, 2, dim=0)

        teacher_logits = teacher_logits.masked_select(mask).view(-1, vocab_size)
        student_logits = student_logits.masked_select(mask).view(-1, vocab_size)
        
        logit_align_loss = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1).detach(), 
            reduction='batchmean'
        )

        outputs.loss += logit_align_loss

        return (outputs.loss, outputs) if return_outputs else outputs.loss
    