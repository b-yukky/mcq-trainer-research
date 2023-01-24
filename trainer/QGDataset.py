import pandas as pd
import numpy as np
from transformers import (
    T5TokenizerFast as T5Tokenizer,
    )
from torch.utils.data import Dataset

class QGDataset(Dataset):

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
        self.MASKING_CHANCE = 0.3
        self.SEP_TOKEN = '<sep>'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        
        if np.random.rand() > self.MASKING_CHANCE:
            answer = data_row['answer_text']
        else:
            answer = '[MASK]'

        source_encoding = self.tokenizer(
            '{} {} {}'.format(answer, self.SEP_TOKEN, data_row['context']),
            max_length= self.source_max_token_len,
            padding='max_length',
            truncation= True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
            )
    
        target_encoding = self.tokenizer(
            '{} {} {}'.format(data_row['answer_text'], self.SEP_TOKEN, data_row['question']),
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
            answer_text = data_row['answer_text'],
            context = data_row['context'],
            question = data_row['question'],
            input_ids = source_encoding['input_ids'].flatten(),
            attention_mask = source_encoding['attention_mask'].flatten(),
            labels=labels.flatten()
            )
