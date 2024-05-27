import os
import json
import torch
import argparse

from vllm import LLM, SamplingParams
from evalplus.data import get_human_eval_plus, get_mbpp_plus

os.environ["TOKENIZERS_PARALLELISM"] = "false"

TASK = dict()

def registry(name):

    def _registry(_class):
        TASK[name] = _class
        return _class
    
    return _registry

def load_jsonl(path):
    data = []
    with open(path, 'r') as fr:
        for line in fr.readlines():
            data.append(json.loads(line))
    return data

def save_jsonl(data, path, mode='w'):
    with open(path, mode, encoding='utf-8') as fw:
        for d in data:
            fw.write(json.dumps(d, ensure_ascii=False) + '\n')

def post_process(code, stop=None):
    for string in ['<|text|>', '<|code|>', '<|endofblock|>', '<|endofmessage|>']:
        code = code.replace(string, '')
    
    try:
        if code.count('```') % 2 == 1:
            code = code[:code.rfind('```')]
        else:
            code = code[code.find('```') + 3:]
            code = code[code.find('\n') + 1:]
    except:
        pass

    stops = ['\n# Test', '\nif', '\nassert', '\nprint', "\n```"]

    if stop is not None:
        stops = stop + ["\n```"]

    for string in stops:
        if string in code:
            code = code[:code.find(string)]
    
    return code.strip()

@registry('humaneval')
class Humaneval:

    dataset = 'humaneval'
    get_dataset_func = get_human_eval_plus

    @classmethod
    def get_prompt(cls, sample):

        prompt = f"<|user|><|text|>Write a complete Python function for the problem.\n\n{sample['prompt'].strip()}<|endofblock|><|endofmessage|><|assistant|><|text|>```python\n"

        return prompt

    @classmethod
    def test(cls, model, generate_params, result_path):

        task_ids, prompts = [], []
        for task_id, sample in cls.get_dataset_func().items():
            task_ids.append(task_id)
            prompts.append(cls.get_prompt(sample))
        
        completions = model.generate(prompts, generate_params)

        results = []
        for task_id, completion in zip(task_ids, completions):
            for output in completion.outputs:
                results.append(dict(task_id=task_id, completion=post_process(output.text)))
        
        target_file = os.path.join(result_path, f'{cls.dataset}.jsonl')

        save_jsonl(results, target_file)

@registry('mbpp')
class MBPP(Humaneval):

    dataset = 'mbpp'
    get_dataset_func = get_mbpp_plus
    
    @classmethod
    def get_prompt(cls, sample):

        prompt = sample['prompt'].replace('"""', '').strip().split('\n')
        prompt = f"{prompt[0]}\nYour code should satisfy the following assertion:\n```python\n{prompt[1]}\n```"

        prompt = f"<|user|><|text|>{prompt}<|endofblock|><|endofmessage|><|assistant|><|text|>```python\n"

        return prompt

@registry('multiple')
class MultiPLE:

    @classmethod
    def get_prompt(cls, sample, lang):

        prompt = f"<|user|><|text|>Write a complete {lang} function for the problem.\n\n{sample['prompt'].strip()}<|endofblock|><|endofmessage|><|assistant|><|text|>```{lang.lower()}\n{sample['prompt']}"

        return prompt

    @classmethod
    def test(cls, model, generate_params, result_path):

        generate_params.stop += ['```']

        trans = dict(cpp="C++", cs="CSharp", java="Java", js='Javascript', php="PHP", ts='Typescript', sh='Bash', swift='Swift', rs='Rust')
        stops = dict(cpp=['\nint main'], js=['\n}'], php=['\n}'], ts=['\n}'], rs=['\nfn main()'])

        for lang in args.langs:

            test_samples = load_jsonl(f'data/multiple/{lang}.jsonl')

            prompts = []
            for test_sample in test_samples:
                prompts.append(cls.get_prompt(test_sample, trans[lang]))
            print(prompts[0])
            
            completions = model.generate(prompts, generate_params)

            results = []
            for sample, completion in zip(test_samples, completions):

                if lang in ['php', 'js']:
                    sample['stop_tokens'] = []

                outs = []
                for output in completion.outputs:
                    out = output.text
                    for string in stops.get(lang, []) + sample['stop_tokens']:
                        if string in out:
                            out = out[:out.rfind(string)]
                    out = sample['prompt'] + out

                    if lang in ['js', 'php', 'ts']:
                        out += '\n}'

                    outs.append(out)

                results.append(outs)
            
            with open(os.path.join(result_path, f'multiple_{lang}.json'), 'w') as fw:
                json.dump(results, fw, indent=4)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', required=True, type=str)
    parser.add_argument('-t', '--task', default=None, type=str, nargs='+')

    parser.add_argument('-tp', '--tp', default=None, type=int)

    args = parser.parse_args()
  
    model = LLM(
        model=args.path, 
        trust_remote_code=True, 
        tensor_parallel_size=args.tp or torch.cuda.device_count()
    )

    sample_params = SamplingParams(
        temperature=0,
        max_tokens=1024,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
        stop=['<|endofblock|>', '<|endofmessage|>']
    )
    
    if args.task is None:
        args.task = ["humaneval", "mbpp"]

    if isinstance(args.task, str):
        args.task = [args.task]
    
    os.makedirs(os.path.join(args.path, 'results'), exist_ok=True)

    for task in args.task:
        TASK[task].test(model, sample_params, os.path.join(args.path, 'results'))
