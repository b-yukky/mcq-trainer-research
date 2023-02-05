import pandas as pd
import numpy as np
from torch import index_put_
from transformers import (
    T5TokenizerFast as T5Tokenizer,
    )
from torch.utils.data import Dataset

class MultiDataset(Dataset):

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: T5Tokenizer,
        source_max_token_len: int,
        target_max_token_len: int,
        ):

        self.tokenizer = tokenizer
        self.data = data
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len
        self.SEP_TOKEN = '<sep>'
        self.MASKING_CHANCE = 0.7
    def __len__(self):
        return len(self.data)
    
    def encode_qa(self, data_row):
        if np.random.rand() > self.MASKING_CHANCE:
            answer = data_row['answer']
        else:
            answer = '[MASK]'
            
        context = '[MASK]' if data_row['context'] in ['', None] else data_row['context']
        question = '[MASK]' if data_row['question'] in ['', None] else data_row['question']
        
        source_encoding = self.tokenizer(
            'qa: {} {} {}'.format(
                answer, self.SEP_TOKEN, context
            ),
            max_length= self.source_max_token_len,
            padding='max_length',
            truncation= True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
            ) 
        
        target_encoding = self.tokenizer(
            '{} {} {}'.format(data_row['answer'], self.SEP_TOKEN, question),
            max_length=self.target_max_token_len,
            padding='max_length',
            truncation = True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
            )

        labels = target_encoding['input_ids']  
        labels[labels == 0] = -100
        
        return dict(
            answer = data_row['answer'],
            context = data_row['context'],
            question = data_row['question'],
            input_ids = source_encoding['input_ids'].flatten(),
            attention_mask = source_encoding['attention_mask'].flatten(),
            labels=labels.flatten()
            )
    
    def encode_distractors(self, data_row):
        
        question = '[MASK]' if data_row['question'] in ['', None] else data_row['question']
        context = '[MASK]' if data_row['context'] in ['', None] else data_row['context']   
        incorrect1 = 'N/A' if type(data_row['incorrect1']) == float else data_row['incorrect2']
        incorrect2 = 'N/A' if type(data_row['incorrect3']) == float else data_row['incorrect2']
        incorrect3 = 'N/A' if type(data_row['incorrect3']) == float else data_row['incorrect3']
             
        source_encoding = self.tokenizer(
            'choices: {} {} {} {} {}'.format(
                data_row['answer'], self.SEP_TOKEN, 
                question, self.SEP_TOKEN, 
                context
            ),
            max_length= self.source_max_token_len,
            padding='max_length',
            truncation= True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
            )

    
        target_encoding = self.tokenizer(
            '{} {} {} {} {}'.format(
                incorrect1, self.SEP_TOKEN,
                incorrect2, self.SEP_TOKEN, 
                incorrect3
            ),
            max_length=self.target_max_token_len,
            padding='max_length',
            truncation = True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
            )

        labels = target_encoding['input_ids']  
        labels[labels == 0] = -100
        
        return dict(
            answer = data_row['answer'],
            context = data_row['context'],
            question = data_row['question'],
            input_ids = source_encoding['input_ids'].flatten(),
            attention_mask = source_encoding['attention_mask'].flatten(),
            labels=labels.flatten()
            )

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        
        if type(data_row['incorrect1']) == float or  data_row['incorrect1'] in ['', None]:
            return self.encode_qa(data_row)
        else:
            return self.encode_distractors(data_row)
