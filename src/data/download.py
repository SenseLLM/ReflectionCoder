import os
import html

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from datasets import load_dataset

def process(sample):
    result = [{"role": "user", "content": [{"type": "text", "content": html.unescape(sample['query'])}]}]

    output = html.unescape(sample['answer']).replace("\r\n", "\n").replace("\t", "    ").strip()

    result.append({"role": "assistant", "content": [{"type": "text", "content": output}]})

    return {"messages": result, "type": "generation"}

def code_instruction_data():
    os.makedirs('train', exist_ok=True)

    data = load_dataset('m-a-p/CodeFeedback-Filtered-Instruction', split='train')
    data.map(process, remove_columns=list(data.features)).to_json('train/code_instruct.jsonl')

def code_reflection_data():
    os.makedirs('train', exist_ok=True)

    data = load_dataset('SenseLLM/RelfectionSeq-GPT', split='train')
    data.to_json('train/reflection_gpt.jsonl')

    data = load_dataset('SenseLLM/RelfectionSeq-DS', split='train')
    data.to_json('train/reflection_ds.jsonl')

if __name__ == '__main__':
    code_instruction_data()
    code_reflection_data()
    