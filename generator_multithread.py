import multiprocessing
from tqdm import tqdm

import torch
from time import perf_counter
import random
import pandas as pd
import numpy as np
from mlflow import log_metric, log_param, log_artifacts
from QGDataset import QGDataset
import csv
from datasets import load_dataset
from QAGenerator import QAGenerator
from DGenerator import DGenerator

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process, Queue
from mlflow import log_metric

random.seed(42)

import os
os.environ["PYTHONUNBUFFERED"] = "1"

CHECKPOINT_PATH ='checkpoints\\t5-base\\multi-qad-161k\\DGen\\best-checkpoint-v2.ckpt'
BASE_MODEL = 't5-base'


def write_results(q, filename):
    header = ['question', 'context', 'answer', 'incorrect1', 'incorect2', 'incorrect3']
    output = pd.DataFrame(columns=header)
    one_missing = 0
    two_missing = 0
    three_missing = 0
    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(header)
        count = 0

        while True:
            result = q.get()
            if result is None:
                break
            
            print(result[2:], count)
            writer.writerow(result)
            output = pd.concat([pd.DataFrame([result], columns=header), output], ignore_index=True)
            count += 1
            if result[3] == '':
                three_missing += 1
            elif result[4] == '':
                two_missing += 1
            elif result[5] == '':
                one_missing += 1
                
            log_param('count', count)
            log_param('one_missing', one_missing)
            log_param('two_missing', two_missing)
            log_param('three_missing', three_missing)
            if count % 100 == 0 :
                output.to_csv(filename+'.backup')
    
def process_df(q, df):
    # Do something with the row
    d_model = DGenerator(BASE_MODEL, CHECKPOINT_PATH)
    dataset = QGDataset(df, d_model.tokenizer, d_model.SOURCE_MAX_TOKEN_LEN, d_model.TARGET_MAX_TOKEN_LEN)
    
    for i in tqdm(range(dataset.__len__())):
        question = dataset.data.iloc[i]['question']
        context = dataset.data.iloc[i]['context']
        answer = dataset.data.iloc[i]['answer']
        
        t1_start = perf_counter()
        distractors = d_model.generate(answer, question, context)
        log_metric('generation_time', perf_counter() - t1_start)
        
        while len(distractors) < 3:
            distractors.append('')

        q.put([question, context, answer, distractors[0], distractors[1], distractors[2]])

    
def gen_d():
    
    
    dataset_path = 'datasets/processed/train_squad.csv'
    dataset_df = pd.read_csv(dataset_path)
    
    dataset_df.rename(columns = {'answer_text':'answer'}, inplace = True)
        
    dataset_name = dataset_path.split('/')[-1]
    

    log_param('experiment_type', 'dataset_evaluation')
    log_param('base_model', BASE_MODEL)
    log_param('checkpoint', CHECKPOINT_PATH)
    log_param('dataset_path', dataset_path)
    
    output_file = f'datasets/eval/dgen-{BASE_MODEL}_{dataset_name}'
    
    workers = multiprocessing.cpu_count() - 2
    
    split_dfs = np.array_split(dataset_df, workers)
    
    q = Queue()
    processes = []
    # Start a process to write the results to the output file
    p = Process(target=write_results, args=(q, output_file,))
    processes.append(p)
    p.start()
    
    for df in split_dfs:
        p = Process(target=process_df, args=(q, df))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
        
if __name__ == '__main__':
    gen_d()
