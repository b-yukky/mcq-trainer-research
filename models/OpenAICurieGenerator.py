import openai
from dotenv import load_dotenv
import os
import re
import random
import backoff
load_dotenv()
from sentence_transformers import SentenceTransformer, util

class OpenAICurieGenerator():
    
    def __init__(self) -> None:
        ''' init '''
        self.API_KEY = os.environ.get("OPENAI_API_KEY")
        openai.api_key = self.API_KEY
        self.models = openai.Model.list()
        self.engines = openai.Engine.list()
        self.sentence_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    
    def format_text_regex(self, text):
        
        lines = text.strip().split("\n")
        # question_pattern = re.compile(r"\s*(?P<question>.+\?)")
        question_pattern = re.compile(r"\s*(?P<question>.+\?)")
        answer_pattern = re.compile(r"answer:\s*(?P<answer>.+)")
        choice_pattern = re.compile(r"incorrect[0-9]:\s*(?P<choice>.+)")
        distractors = []

        for line in lines:
            question_match = question_pattern.search(line)
            answer_match = answer_pattern.search(line)
            choice_match = choice_pattern.search(line)
            if question_match:
                question = question_match.group('question').replace('\n', '')
            elif answer_match:
                answer = answer_match.group('answer')
            elif choice_match:
                choice = choice_match.group('choice')
                distractors.append(choice)
        
        return question, answer, distractors
    
    def generate(self, context: str) -> tuple:
        
        questions_list = []
        answers_list = []
        distractors_list = []
        
        @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
        def generate_question_with_backoff(**kwargs):
            return openai.Completion.create(
            model="curie:ft-personal:multiqad400k-2023-02-08-15-23-54",
            prompt=(f'''{context}'''),
            max_tokens=256,
            n = 1,
            best_of = 1,
            stop = " [END]",
            temperature = 0.1,
            top_p = 1,
            presence_penalty = 1,
            frequency_penalty = 0.5,
        )
        
        response = generate_question_with_backoff()
        
        output = response["choices"]

        for qad_pair in output:
            print(qad_pair)
            try:
                question, answer, distractors = self.format_text_regex(qad_pair['text'])
            except Exception:
                break
            questions_list.append(question)
            answers_list.append(answer)
            
            distractors = list(map(lambda x: x.strip(), distractors))
            distractors = list(map(lambda x: x[:-1] if x[-1:] in '.,;\\!][+@&?:;><-=}{#~' else x, distractors))
            distractors = self._remove_similar_distractors([answer], distractors)
            
            distractors_list.append(distractors)

        if len(answers_list) < 1:
            answers_list.append('')
            
        if len(questions_list) < 1:
            questions_list.append('')
            
        if len(distractors_list) < 1:
            distractors_list.append([])
                    
        while len(distractors_list[0]) < 3:
            distractors_list[0].append('')
        
        return answers_list[0], questions_list[0], distractors_list[0][0], distractors_list[0][1], distractors_list[0][2]
    
    
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
