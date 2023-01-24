import pandas as pd
from QADTrainer import QADTrainer
from datasets import load_dataset

def preprocess_dataset():
    
    squad_train_df = pd.read_csv('datasets/squad/train_df.csv')
    squad_dev_df = pd.read_csv('datasets/squad/dev_df.csv')

    context_name = 'context_para'
    drop_context = 'context_sent' 

    df = squad_train_df.copy()
    df = df.dropna()
    df.rename(columns = {context_name: 'context'}, inplace=True)
    df.drop(columns=[drop_context, 'answer_start', 'answer_end'], inplace=True)

    test_df = df[:11877]
    train_df = df[11877:]

    dev_df = squad_dev_df.copy()
    dev_df.rename(columns = {context_name: 'context'}, inplace=True)
    dev_df.drop(columns=[drop_context, 'answer_start', 'answer_end'], inplace=True)

    return train_df, test_df, dev_df

def main():
    
    datasets_name = 'multi-qad-200k'
    
    # train_df, test_df, dev_df = preprocess_dataset()
    
    datasets = load_dataset('b-yukky/multi-qad-200k', use_auth_token='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')
    
    train_df = datasets['train'].to_pandas()
    test_df = datasets['test'].to_pandas()
    dev_df = datasets['validation'].to_pandas()
    
    trainer = QADTrainer(
        train_df,
        test_df,
        dev_df,
        model_name= 't5-small',
        epochs = 6,
        learning_rate= 0.0001,
        df_take_percentage = 1,
        datasets_name = datasets_name
    )
    
    trainer.start_training()

if __name__ == '__main__':
    main()
