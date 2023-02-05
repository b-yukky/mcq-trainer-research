import multiprocessing
from tqdm import tqdm

import torch
import random
import pandas as pd
import numpy as np
import csv

from  models.MCQGenerator import MCQGenerator
from  models.MultiGenerator import MultiGenerator
from models.LeafGenerator import LeafBaseMCQGenerator
from models.OpenAIGenerator import OpenAIGenerator

from multiprocessing import Process, Queue

random.seed(42)

import os
os.environ["PYTHONUNBUFFERED"] = "1"

CHECKPOINT_PATH ='checkpoints\\google\\multimcq-1M\\base\\00005-checkpoint-v3.ckpt'
BASE_MODEL = 'google/flan-t5-base'
BASE_MODEL = '/text-davinci-003'

DATASET_NAME = 'wikipedia-10T'

def write_results(write_queue, filename, length, model_name):
    global DATASET_NAME
    header = ['topic', 'context', 'question', 'answer', 'choices', 'model', 'dataset']
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
            writer.writerow(result + [model_name] + [DATASET_NAME])
            output = pd.concat([pd.DataFrame([result + [model_name] + [DATASET_NAME]], columns=header), output])
            count += 1
            
            if count == 100:
                output.to_csv(filename, index=False)
                print('Saved!')
        
        print('Done!')
    
    exit(0)

    
def process_df(data_queue, write_queue):
    # Do something with the row
    # model = MultiGenerator(BASE_MODEL, CHECKPOINT_PATH)
    model = OpenAIGenerator()

    while True:
        row = data_queue.get(timeout=3)
        
        if row is None:
            break
        
        topic = row['topic']
        context = row['context']
                
        output = model.generate(context)
        
        write_queue.put_nowait([topic, context, output[1],  output[0],  [output[2],  output[3],  output[4]]])

    
def gen_d():
    
    
    dataset_path = 'experiment_wikipedia-10T.csv'
    dataset_df = pd.read_csv(dataset_path).drop_duplicates(subset=['context'])
    
    # dataset_df.rename(columns = {'answer_text':'answer'}, inplace = True)
    
    base_model_name = BASE_MODEL.split('/')[1]
    train_dataset_name = 'gpt-3'
    
    model_name = f'{base_model_name}-{train_dataset_name}'

    output_file = f'datasets/experiment/{model_name}v2.csv'
    
    
    workers = 1
    
    write_queue = Queue()
    data_queue = Queue()
    processes = []
    
    for _, row in dataset_df.iterrows():
        data_queue.put(row)
    data_queue.put(None)
    data_queue.close()

    # Start a process to write the results to the output file
    p = Process(target=write_results, args=(write_queue, output_file, data_queue.qsize(), model_name))
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
