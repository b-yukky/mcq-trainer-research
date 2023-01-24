from typing import List, Dict
import json
import pandas as pd
import numpy as np

import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from transformers import (
    T5TokenizerFast as T5Tokenizer,
    )
from pytorch_lightning import Trainer

import time
from QGModel import QGModel
from QGDataset import QGDataset
from DTDataset import DTDataset

pl.seed_everything(42)


class QGDataModule(pl.LightningDataModule):

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: T5Tokenizer,
        batch_size,
        source_max_token_len: int,
        target_max_token_len: int,
        custom: bool = False
        ): 
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len
        self.custom = custom

    def setup(self, stage='fit'):
        if self.custom:
            self.train_dataset = DTDataset(self.train_df, self.tokenizer, self.source_max_token_len, self.target_max_token_len)
            self.val_dataset = DTDataset(self.val_df, self.tokenizer, self.source_max_token_len, self.target_max_token_len)
            self.test_dataset = DTDataset(self.test_df, self.tokenizer, self.source_max_token_len, self.target_max_token_len)
        else:
            self.train_dataset = QGDataset(self.train_df, self.tokenizer, self.source_max_token_len, self.target_max_token_len)
            self.val_dataset = QGDataset(self.val_df, self.tokenizer, self.source_max_token_len, self.target_max_token_len)
            self.test_dataset = QGDataset(self.test_df, self.tokenizer, self.source_max_token_len, self.target_max_token_len)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle=True, num_workers = 0)

    def val_dataloader(self): 
        return DataLoader(self.val_dataset, batch_size=1, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=0)


class QADTrainer():
    
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, dev_df: pd.DataFrame, \
            model_name: str, epochs: int, learning_rate: float, df_take_percentage: int = 1, datasets_name: str = 'squad') -> None:
        self.train_df = train_df
        self.test_df = test_df
        self.dev_df = dev_df
        self.datasets_name = datasets_name
        self.SEP_TOKEN = '<sep>'
        self.MODEL_NAME = model_name
        self.SOURCE_MAX_TOKEN_LEN = 512
        self.TARGET_MAX_TOKEN_LEN = 128
        self.MAX_EPOCHS = epochs
        self.BATCH_SIZE = 16
        self.LEARNING_RATE = learning_rate
        self.dataset_take_percentage(df_take_percentage)
        self.loading_model(model_name)
        self.init_data_module(custom=True if self.datasets_name.__contains__('multi-qad') else False)
        self.timenow = time.strftime("%m-%d_%Hh%M")
        self.init_checkpoint_callback()
        self.init_trainer()
        self.init_model()
        
    def dataset_take_percentage(self, percent):
        self.TAKE_TRAIN = int(len(self.train_df) * percent)
        self.TAKE_DEV = int(len(self.dev_df) * percent)
        self.TAKE_TEST = int(len(self.test_df) * percent)
    
    def loading_model(self, model_name):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.tokenizer.add_tokens(self.SEP_TOKEN)
        self.TOKENIZER_LEN = len(self.tokenizer)
    
    def init_data_module(self, custom):        
        self.data_module = QGDataModule(self.train_df[:self.TAKE_TRAIN], self.dev_df[:self.TAKE_DEV], self.test_df[:self.TAKE_TEST], \
            self.tokenizer, self.BATCH_SIZE, self.SOURCE_MAX_TOKEN_LEN, self.TARGET_MAX_TOKEN_LEN, custom=custom)
        self.data_module.setup()

    def init_checkpoint_callback(self):
        self.checkpoint_callback = ModelCheckpoint(
            dirpath=f'checkpoints/{self.MODEL_NAME}',
            filename=f'best-checkpoint',
            save_top_k=-1,
            verbose=True,
            monitor='val_loss',   
            mode='min'
        )
    
    def init_trainer(self):
        self.trainer = pl.Trainer(
            callbacks= [self.checkpoint_callback],
            max_epochs=self.MAX_EPOCHS,
            enable_progress_bar=True,
            accelerator='cuda' if torch.cuda.is_available() else 'cpu',
            devices=1,
            val_check_interval=0.2
        )
        
    def init_model(self):
        self.model = QGModel(self.MODEL_NAME, self.TOKENIZER_LEN, self.LEARNING_RATE)
        self.model.to('cuda:0')

    def start_training(self):
        self.trainer.fit(self.model, self.data_module)


