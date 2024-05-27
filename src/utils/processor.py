
class CodeProcessor:

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

    trans_dict = dict(
        user=user_token, 
        assistant=assistant_token,
        text=text_token, 
        code=code_token,
        execution=execution_token
    )

    ignore_tokens = {
        user_token: end_of_message_token,
        execution_token: end_of_block_token
    }

    special_token = [
        user_token, assistant_token, 
        text_token, code_token, execution_token, 
        end_of_block_token, end_of_message_token
    ]

    def __init__(self, max_lan, tokenizer):

        self.max_len = max_lan
        self.tokenizer = tokenizer

    def get_special_token(self):
        return self.special_token
    
    def process_ignore(self, input_ids, labels):
        ignore_token_ids = dict()
        for k, v in self.ignore_tokens.items():
            ignore_token_ids[self.tokenizer.convert_tokens_to_ids(k)] = self.tokenizer.convert_tokens_to_ids(v)
        
        final_labels = []
        for input_id, label in zip(input_ids, labels):
            index = 0
            while index < len(input_id):
                if input_id[index] in ignore_token_ids:
                    end = ignore_token_ids[input_id[index]]
                    while index < len(input_id) and input_id[index] != end:
                        label[index], index = -100, index + 1
                    if index < len(input_id): 
                        label[index], index = -100, index + 1
                else:
                    index = index + 1
            final_labels.append(label)
        
        return final_labels
    
    def process_distillation(self, labels):
        user_id = self.tokenizer.convert_tokens_to_ids(self.user_token)
        assistant_id = self.tokenizer.convert_tokens_to_ids(self.assistant_token)

        distillation_mask, distillation_labels = [], []
        for label in labels:
            index = max(loc for loc, val in enumerate(label) if val == user_id)
            distillation_mask.append([0] * index + [1] * (len(label[index:])))

            index = max(loc for loc, val in enumerate(label) if val == assistant_id)
            distillation_labels.append([-100] * index + label[index:-1] + [-100])
        
        return distillation_mask, distillation_labels

    def process_tokenize(self, examples):

        texts = []
        for example in examples['messages']:
            text = ''
            for e in example:
                block = ''
                for content in e['content']:
                    block += f"{self.trans_dict[content['type']]}{content['content']}{self.end_of_block_token}"  
                text += f"{self.trans_dict[e['role']]}{block}{self.end_of_message_token}"
            
            texts.append(text)
        
        inputs = self.tokenizer(texts, add_special_tokens=False)

        input_ids = [
            [self.tokenizer.bos_token_id] + input_id + [self.tokenizer.eos_token_id] 
            for input_id in inputs['input_ids'] if len(input_id) <= self.max_len - 2
        ]
        labels = [
            [-100] + label + [self.tokenizer.eos_token_id] 
            for label in inputs['input_ids'] if len(label) <= self.max_len - 2
        ]

        types = examples['type'][0:1] * len(input_ids)
        
        distillation_mask, distillation_labels = self.process_distillation(labels)

        ignore_labels = self.process_ignore(input_ids, labels)
        
        return dict(
            input_ids=input_ids, 
            labels=ignore_labels, 
            distillation_mask=distillation_mask, 
            distillation_labels=distillation_labels,
            type=types
        )
    