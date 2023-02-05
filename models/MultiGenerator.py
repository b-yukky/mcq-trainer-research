from dis import dis
from transformers import T5TokenizerFast as T5Tokenizer
from .QGModel import QGModel
from typing import List, Dict, Tuple
import torch
from sentence_transformers import SentenceTransformer, util

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class MultiGenerator():
    
    def __init__(self, base_model: str, checkpoint_path: str) -> None:
        self.SEP_TOKEN = '<sep>'
        self.SOURCE_MAX_TOKEN_LEN = 512
        self.TARGET_MAX_TOKEN_LEN = 128
        self.tokenizer = T5Tokenizer.from_pretrained(base_model)
        self.tokenizer.add_tokens(self.SEP_TOKEN)
        self.multi_model = QGModel.load_from_checkpoint(checkpoint_path=checkpoint_path, model_name=base_model,tokenizer_len=len(self.tokenizer))
        self.sentence_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.multi_model.model.to(device)
        self.multi_model.freeze()
        self.multi_model.eval()
        
    def generate(self, context: str) -> Tuple[str, str]:
        
        output_qa = self._generate_qa(context)

        answer, question = output_qa.split(self.SEP_TOKEN)
        
        answer = answer.strip()
        question = question.strip()
        
        output_distractors = self._generate_distractors(context, question, answer)
        
        cleaned_distractors = self._replace_all_extra_id(output_distractors.replace('<pad>', '').replace('</s>', self.SEP_TOKEN))
        
        distractors = cleaned_distractors.split(self.SEP_TOKEN)[:-1]
        distractors = list(map(lambda x: x.strip(), distractors))
        distractors = list(map(lambda x: x[:-1] if x[-1:] in '.,;\\!][+@&?:;><-=}{#~' else x, distractors))
        distractors = self._remove_similar_distractors([answer], distractors)
        
        while len(distractors) < 3:
            distractors.append('')
        
        return answer, question, distractors[0], distractors[1], distractors[2]
    
    
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
    
    def _generate_qa(self, context: str, answer: str = '[MASK]'):
        source_encoding = self.tokenizer(
            'qa: {} {} {}'.format(answer, self.SEP_TOKEN, context),
            max_length=self.SOURCE_MAX_TOKEN_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        
        
        source_encoding['input_ids'] = source_encoding['input_ids'].to(device)
        source_encoding['attention_mask'] = source_encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            generated_ids = self.multi_model.model.generate(
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
    
    def _generate_distractors(self, context: str, question: str, answer: str, generate_count: int = 3):
        
        source_encoding = self.tokenizer(
            'choices: {} {} {} {} {}'.format(
                answer, self.SEP_TOKEN, 
                question, self.SEP_TOKEN, 
                context
            ),
            max_length=self.SOURCE_MAX_TOKEN_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        
        source_encoding['input_ids'] = source_encoding['input_ids'].to(device)
        source_encoding['attention_mask'] = source_encoding['attention_mask'].to(device)

        with torch.no_grad():
            generated_ids = self.multi_model.model.generate(
                input_ids=source_encoding['input_ids'],
                attention_mask=source_encoding['attention_mask'],
                num_beams=generate_count,
                num_return_sequences=generate_count,
                max_length=self.TARGET_MAX_TOKEN_LEN,
                repetition_penalty=3.0,
                length_penalty=1.0,
                temperature=1.5,
                early_stopping=True,
                use_cache=True
            )

        preds = {
            self.tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for generated_id in generated_ids
        }
    
        return ' <sep> '.join(preds)
    
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

            new_text = new_text[:start_index_of_extra_id] + self.SEP_TOKEN + new_text[end_index_of_extra_id + 1:]

        return new_text
