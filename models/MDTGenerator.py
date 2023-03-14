from transformers import T5TokenizerFast as T5Tokenizer
from models.QGModel import QGModel
from typing import List, Dict, Tuple
import string
from sentence_transformers import SentenceTransformer, util
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class MDTGenerator():
    
    def __init__(self, base_model: str, checkpoint_path: str) -> None:
        self.SEP_TOKEN = '<sep>'
        self.SOURCE_MAX_TOKEN_LEN = 512
        self.TARGET_MAX_TOKEN_LEN = 128
        self.tokenizer = T5Tokenizer.from_pretrained(base_model)
        self.tokenizer.add_tokens(self.SEP_TOKEN)
        self.dg_model = QGModel.load_from_checkpoint(checkpoint_path=checkpoint_path, model_name=base_model,tokenizer_len=len(self.tokenizer))
        self.sentence_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.dg_model.freeze()
        self.dg_model.eval()
        
    def generate(self, answer: str, question:str, incorrect1: str, incorrect2: str, generate_count: int = 9) -> List[str]:
        
        loop = 0
        distractors = []
        
        while len(distractors) < 3 and loop < 3:
            generate_triples_count = int(generate_count / 3.01) + 1 #since this model generates 3 distractors per generation
            
            model_output = self._model_predict(answer, question, incorrect1, incorrect2, generate_triples_count)

            cleaned_result = model_output.replace('<pad>', '').replace('</s>', '<sep>')
            cleaned_result = self._replace_all_extra_id(cleaned_result)
            new_distractors = cleaned_result.split('<sep>')[:-1]
            # new_distractors = [x.translate(str.maketrans('', '', string.punctuation)) for x in new_distractors]
            new_distractors = list(map(lambda x: x.strip(), new_distractors))
            # new_distractors = list(map(lambda x: x.replace('<unk>', ''), new_distractors))
            new_distractors = list(map(lambda x: x[:-1] if x[-1:] in '.,;\\!][+@&?:;><-=}{#~' else x, new_distractors))
            
            distractors.extend(new_distractors)
            distractors = self._remove_similar_distractors([answer], distractors)

            loop += 1

        return distractors
    
    
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
            if v in distractors:
                distractors.remove(v)
            
        return distractors
    
    def _model_predict(self, answer: str, question: str, incorrect1: str, incorrect2: str, generate_count: int):
        
        source_encoding = self.tokenizer(
            '{} {} {} {} {} {} {} {}'.format(
                answer, self.SEP_TOKEN, 
                question, self.SEP_TOKEN,
                incorrect1, self.SEP_TOKEN,
                incorrect2, self.SEP_TOKEN
            ),
            max_length= self.SOURCE_MAX_TOKEN_LEN,
            padding='max_length',
            truncation= True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
            )
        
        source_encoding['input_ids'] = source_encoding['input_ids'].to(device)
        source_encoding['attention_mask'] = source_encoding['attention_mask'].to(device)
        self.dg_model.model.to(device)

        with torch.no_grad():
            generated_ids = self.dg_model.model.generate(
                input_ids=source_encoding['input_ids'],
                attention_mask=source_encoding['attention_mask'],
                num_beams=generate_count,
                num_return_sequences=generate_count,
                max_length=self.TARGET_MAX_TOKEN_LEN,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True,
                use_cache=True
            )

        preds = {
            self.tokenizer.decode(generated_id, skip_special_tokens=False, clean_up_tokenization_spaces=True)
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
