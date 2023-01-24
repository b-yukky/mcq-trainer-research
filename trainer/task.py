import pandas as pd
from QADTrainer import QADTrainer
from datasets import load_dataset

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
        epochs = 4,
        learning_rate= 0.0001,
        df_take_percentage = 1,
        datasets_name = datasets_name
    )
    
    trainer.start_training()

if __name__ == '__main__':
    main()
