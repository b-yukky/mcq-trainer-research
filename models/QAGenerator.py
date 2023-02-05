from transformers import T5TokenizerFast as T5Tokenizer
from .QGModel import QGModel
from typing import List, Dict, Tuple

class QAGenerator():
    
    def __init__(self, base_model: str, checkpoint_path: str) -> None:
        self.SEP_TOKEN = '<sep>'
        self.SOURCE_MAX_TOKEN_LEN = 300
        self.TARGET_MAX_TOKEN_LEN = 80
        self.tokenizer = T5Tokenizer.from_pretrained(base_model)
        self.tokenizer.add_tokens(self.SEP_TOKEN)
        self.qg_model = QGModel.load_from_checkpoint(checkpoint_path=checkpoint_path, model_name=base_model,tokenizer_len=len(self.tokenizer))
        self.qg_model.freeze()
        self.qg_model.eval()
        
    def generate(self, context: str) -> Tuple[str, str]:
        answer_mask = '[MASK]'
        model_output = self._model_predict(answer_mask, context)

        qna_pair = model_output.split('<sep>')

        if len(qna_pair) < 2:
            generated_answer = ''
            generated_question = qna_pair[0]
        else:
            generated_answer = qna_pair[0]
            generated_question = qna_pair[1]

        return generated_answer, generated_question
    
    def _model_predict(self, answer: str, context: str):
        source_encoding = self.tokenizer(
            '{} {} {}'.format(answer, self.SEP_TOKEN, context),
            max_length=self.SOURCE_MAX_TOKEN_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        generated_ids = self.qg_model.model.generate(
            input_ids=source_encoding['input_ids'],
            attention_mask=source_encoding['attention_mask'],
            num_beams=1,
            max_length=self.TARGET_MAX_TOKEN_LEN,
            repetition_penalty=1.0,
            length_penalty=1.0,
            early_stopping=True,
            use_cache=True
        )

        preds = {
            self.tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for generated_id in generated_ids
        }

        return ''.join(preds)
