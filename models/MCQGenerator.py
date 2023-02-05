from dis import dis
from transformers import T5TokenizerFast as T5Tokenizer
from .QGModel import QGModel
from typing import List, Dict, Tuple
import torch
from sentence_transformers import SentenceTransformer, util

class MCQGenerator():
    
    def __init__(self, base_model: str, checkpoint_path: str) -> None:
        self.SEP_TOKEN = '<sep>'
        self.SOURCE_MAX_TOKEN_LEN = 300
        self.TARGET_MAX_TOKEN_LEN = 80
        self.tokenizer = T5Tokenizer.from_pretrained(base_model)
        self.tokenizer.add_tokens(self.SEP_TOKEN)
        self.qg_model = QGModel.load_from_checkpoint(checkpoint_path=checkpoint_path, model_name=base_model,tokenizer_len=len(self.tokenizer))
        self.sentence_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.qg_model.freeze()
        self.qg_model.eval()
        
    def generate(self, context: str) -> Tuple[str, str]:
        answer_mask = '[MASK]'
        model_output = self._model_predict(answer_mask, context)

        mcq = model_output.split('<sep>')
                
        generated_answer = mcq[0].strip() if len(mcq) > 0 else ''
        generated_question = mcq[1].strip() if len(mcq) > 1 else ''
        
        incorrect1 = mcq[2] if len(mcq) > 2 else ''
        incorrect2 = mcq[3] if len(mcq) > 3 else ''
        incorrect3 = mcq[4] if len(mcq) > 4 else ''
        
        distractors = [incorrect1, incorrect2, incorrect3]
        distractors = list(map(lambda x: x.strip(), distractors))
        distractors = list(map(lambda x: x[:-1] if x[-1:] in '.,;\\!][+@&?:;><-=}{#~' else x, distractors))
        distractors = self._remove_similar_distractors([generated_answer], distractors)
        
        while len(distractors) < 3:
            distractors.append('')
        
        return generated_answer, generated_question, distractors[0], distractors[1], distractors[2]
    
    
    def _remove_similar_distractors(self, answer_list: list[str], distractors: list[str], threshold: float = 0.9):
        
        embeddings = self.sentence_transformer.encode(answer_list + distractors)
        cos_sim = util.cos_sim(embeddings, embeddings)

        all_sentence_combinations = []
        for i in range(len(cos_sim)-1):
            for j in range(i+1, len(cos_sim)):
                all_sentence_combinations.append([cos_sim[i][j], i, j])

        all_sentence_combinations = sorted(all_sentence_combinations, key=lambda x: x[0], reverse=True)

        to_delete = []

        for _, i, j in all_sentence_combinations[0:int(len(all_sentence_combinations)/2)]:
            if cos_sim[i][j] > threshold:
                to_delete.append(distractors[j-1])

        for v in to_delete:
            if (answer_list + distractors).count(v) > 1:
                distractors.remove(v)
            
        return distractors
    
    def _model_predict(self, answer: str, context: str):
        source_encoding = self.tokenizer(
            'fullmcq: {} {} {}'.format(
                answer, self.SEP_TOKEN, context
            ),
            max_length=self.SOURCE_MAX_TOKEN_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        with torch.no_grad():
            generated_ids = self.qg_model.model.generate(
                input_ids=source_encoding['input_ids'],
                attention_mask=source_encoding['attention_mask'],
                num_beams=1,
                max_length=self.TARGET_MAX_TOKEN_LEN,
                repetition_penalty=1.6,
                length_penalty=1.3,
                early_stopping=True,
                use_cache=True,
            )

        preds = {
            self.tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for generated_id in generated_ids
        }

        return ''.join(preds)
    
    def _correct_index_of(self, text:str, substring: str, start_index: int = 0):
        try:
            index = text.index(substring, start_index)
        except ValueError:
            index = -1

        return index
    
    def _replace_all_extra_id(self, text: str):
        new_text = text
        start_index_of_extra_id = 0

        while (self._correct_index_of(new_text, '<extra_id_') >= 0):
            start_index_of_extra_id = self._correct_index_of(new_text, '<extra_id_', start_index_of_extra_id)
            end_index_of_extra_id = self._correct_index_of(new_text, '>', start_index_of_extra_id)

            new_text = new_text[:start_index_of_extra_id] + '<sep>' + new_text[end_index_of_extra_id + 1:]

        return new_text
