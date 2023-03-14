import multiprocessing
from async_timeout import timeout
from tqdm import tqdm

import torch
from time import perf_counter
import random
import pandas as pd
import numpy as np
from mlflow import log_metric, log_param, log_artifacts
from  models.QGDataset import QGDataset
import csv
from datasets import load_dataset
from  models.QAGenerator import QAGenerator
from models.DGenerator import DGenerator
from  models.MDTGenerator import MDTGenerator
from  models.MCQGenerator import MCQGenerator
from  models.MultiGenerator import MultiGenerator

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process, Queue

random.seed(42)

import os
os.environ["PYTHONUNBUFFERED"] = "1"

CHECKPOINT_PATH ='checkpoints\\google\\davinci-50k\\base\\0002-checkpoint-v3.ckpt'
BASE_MODEL = 'google/flan-t5-base'


def write_results(write_queue, filename, length):
    header = ['context', 'question', 'answer', 'incorrect1', 'incorect2', 'incorrect3']
    output = pd.DataFrame(columns=header)

    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(header)
        count = 0

        while True:
            result = write_queue.get()
            if result is None:
                break
            
            print(f"{result[1:]} | {count} | {round(count/length, 4)*100}%")
            writer.writerow(result)
            output = pd.concat([pd.DataFrame([result], columns=header), output], ignore_index=True)
            count += 1

            if count % 500 == 0 :
                output.to_csv(filename+'.backup')
    
def process_df(data_queue, write_queue):
    # Do something with the row
    d_model = MCQGenerator(BASE_MODEL, CHECKPOINT_PATH)

    while True:
        row = data_queue.get(timeout=3)
        
        if row is None:
            break
        
        context = row['context']
        
        t1_start = perf_counter()
        
        output = d_model.generate(context)

        log_metric('generation_time', perf_counter() - t1_start)
        
        write_queue.put_nowait([context, output[1],  output[0],  output[2],  output[3],  output[4]])

    
def gen_d():
    
    
    dataset_path = 'datasets/processed/adversarial-qa.csv'
    dataset_df = pd.read_csv(dataset_path).drop_duplicates(subset=['context'])
    
    # dataset_df.rename(columns = {'answer_text':'answer'}, inplace = True)
        
    dataset_name = dataset_path.split('/')[-1]
    

    log_param('experiment_type', 'dataset_evaluation')
    log_param('base_model', BASE_MODEL)
    log_param('checkpoint', CHECKPOINT_PATH)
    log_param('dataset_path', dataset_path)
    
    output_file = f'datasets/eval/{BASE_MODEL}_{dataset_name}'
    
    workers = 2
    
    write_queue = Queue()
    data_queue = Queue()
    processes = []
    
    for _, row in dataset_df.iterrows():
        data_queue.put(row)
    data_queue.put(None)
    data_queue.close()

    # Start a process to write the results to the output file
    p = Process(target=write_results, args=(write_queue, output_file, data_queue.qsize()))
    processes.append(p)
    p.start()
    
    
    for i in range(workers):
        p = Process(target=process_df, args=(data_queue, write_queue,))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
        
if __name__ == '__main__':
    gen_d()
