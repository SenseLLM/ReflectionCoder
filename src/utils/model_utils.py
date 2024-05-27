import torch
import deepspeed

from utils.processor import CodeProcessor
from transformers.tokenization_utils import AddedToken
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.integrations import is_deepspeed_zero3_enabled

def update_new_token(_input_emb, _output_emb, _tokenizer, _src_tokenizer):
    for _id in range(len(_src_tokenizer), len(_tokenizer)):
        _str = _tokenizer.convert_ids_to_tokens(_id)
        _ids = _src_tokenizer.encode(_str, add_special_tokens=False)

        _input_emb.weight.data[_id] = torch.mean(_input_emb.weight.data[_ids], dim=0)
        _output_emb.weight.data[_id] = torch.mean(_output_emb.weight.data[_ids], dim=0)
    
    return _input_emb, _output_emb

def get_model(args):
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_cfg,
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_cfg, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    processor = CodeProcessor(args.max_len, tokenizer)
    special_tokens = [AddedToken(t) for t in processor.get_special_token()]
    
    tokenizer.add_special_tokens(
        dict(additional_special_tokens=special_tokens), 
        replace_additional_special_tokens=False
    )

    model.resize_token_embeddings(len(tokenizer))
    
    src_tokenizer = AutoTokenizer.from_pretrained(args.model_cfg, trust_remote_code=True)

    input_emb = model.get_input_embeddings()
    output_emb = model.get_output_embeddings()

    if is_deepspeed_zero3_enabled():
        with deepspeed.zero.GatheredParameters([input_emb.weight, output_emb.weight], modifier_rank=0):
            if torch.distributed.get_rank() == 0:
                input_emb, output_emb = update_new_token(input_emb, output_emb, tokenizer, src_tokenizer)  
    else:
        input_emb, output_emb = update_new_token(input_emb, output_emb, tokenizer, src_tokenizer)

    model.set_input_embeddings(input_emb)
    model.set_output_embeddings(output_emb)

    return model, tokenizer, processor