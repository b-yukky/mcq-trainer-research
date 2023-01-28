from tqdm import tqdm

import torch
from time import perf_counter
import random
import pandas as pd
from mlflow import log_metric, log_param, log_artifacts
from QGDataset import QGDataset
import csv
from datasets import load_dataset
from QAGenerator import QAGenerator
from DGenerator import DGenerator
import json
from datasets import Dataset

random.seed(42)


def gen_qa():
    
    
    checkpoint_path ='checkpoints\\t5-base\\squad\\best-checkpoint-v10.ckpt'
    
    BASE_MODEL = 't5-base'
    
    dataset_path = 'pattern_recognition_example_text.json'
    with open(dataset_path,'r') as fin:
        example_text= json.load(fin)
    

    dataset_ds = Dataset.from_dict({"id": ['']*len(example_text)})
    examples = dataset_ds.add_column(name="context", column=example_text)
    examples = examples.add_column(name="answer", column=['']*len(example_text))
    examples = examples.add_column(name="question", column=['']*len(example_text))
    
    dataset_name = "-".join(dataset_path.split('/')[-2:])
    
    qa_model = QAGenerator(BASE_MODEL, checkpoint_path)
        

    log_param('experiment_type', 'dataset_evaluation')
    log_param('base_model', BASE_MODEL)
    log_param('checkpoint', checkpoint_path)
    log_param('dataset_path', dataset_path)
    
    output_file = f'datasets/eval/test-{BASE_MODEL}_{dataset_name}'
    
    
    with open(output_file, "w", newline='', encoding='utf-8') as output:
        writer = csv.writer(output, delimiter=',')
        writer.writerow(['question', 'context', 'answer', 'generated_question', 'generated_answer'])
        for i in tqdm(range(examples.__len__())):
            question = examples['question'][i]
            context = examples['context'][i]
            answer = examples['answer'][i]
            
            t1_start = perf_counter()
            generated_answer, generated_question = qa_model.generate(context)
            print(' '+ generated_question, generated_answer)
            log_metric('generation_time', perf_counter() - t1_start)

            writer.writerow([question, context, answer, generated_question, generated_answer])
    
def gen_d():
    
    checkpoint_path ='checkpoints\\t5-base\\multi-qad-161k\\DGen\\best-checkpoint-v2.ckpt'
    
    BASE_MODEL = 't5-base'
    
    dataset_path = 'datasets/processed/train_squad.csv'
    dataset_df = pd.read_csv(dataset_path)
    
    dataset_df.rename(columns = {'answer_text':'answer'}, inplace = True)
        
    dataset_name = dataset_path.split('/')[-1]
    
    d_model = DGenerator(BASE_MODEL, checkpoint_path)
    
    
    dataset = QGDataset(dataset_df, d_model.tokenizer, d_model.SOURCE_MAX_TOKEN_LEN, d_model.TARGET_MAX_TOKEN_LEN)
    

    log_param('experiment_type', 'dataset_evaluation')
    log_param('base_model', BASE_MODEL)
    log_param('checkpoint', checkpoint_path)
    log_param('dataset_path', dataset_path)
    
    output_file = f'datasets/eval/dgen-{BASE_MODEL}_{dataset_name}'
    
    
    with open(output_file, "w", newline='', encoding='utf-8') as output:
        writer = csv.writer(output, delimiter=',')
        writer.writerow(['question', 'context', 'answer', 'incorrect1', 'incorect2', 'incorrect3'])
        for i in tqdm(range(dataset.__len__())):
            question = dataset.data.iloc[i]['question']
            context = dataset.data.iloc[i]['context']
            answer = dataset.data.iloc[i]['answer']
            
            t1_start = perf_counter()
            distractors = d_model.generate(answer, question, context)
            log_metric('generation_time', perf_counter() - t1_start)
            print(distractors)
            while len(distractors) < 3:
                distractors.append('')

            writer.writerow([question, context, answer, distractors[0], distractors[1], distractors[2]])
    
        dataset.data.to_csv(f'datasets/eval/{BASE_MODEL}_{dataset_name}')


if __name__ == '__main__':
    gen_qa()
