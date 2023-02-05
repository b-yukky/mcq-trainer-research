import openai
from dotenv import load_dotenv
import os
import re
import random

load_dotenv()

class OpenAIGenerator():
    
    def __init__(self) -> None:
        ''' init '''
        self.API_KEY = os.environ.get("OPENAI_API_KEY")
        openai.api_key = self.API_KEY
        self.models = openai.Model.list()
        self.engines = openai.Engine.list()
    
    def generate_random_answer_option(self):
        potential_answers = ['A', 'B', 'C', 'D']
        return random.choice(potential_answers)
    
    def format_text_regex(self, text, ans):
        
        lines = text.strip().split("\n")
        # question_pattern = re.compile(r"\s*(?P<question>.+\?)")
        question_pattern = re.compile(r"\s*(?P<question>.+\?)|(Q:|Question:)\s*.+")
        answer_pattern = re.compile(r"({}:|{}\.|{}\))\s*(?P<answer>.+)?".format(ans, ans, ans))
        choice_pattern = re.compile(r"[a-zA-Z]+(:|\.|\))\s*(?P<choice>.+)")
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

    def select_language(self, lang):
        
        if lang == 'fr':
            return 'in french '
        else:
            return ''
    
    def generate(self, context: str, desired_count: int = 1, lang='en') -> tuple:
        
        questions_list = []
        answers_list = []
        distractors_list = []
        
        answer_option = self.generate_random_answer_option()
        language = self.select_language(lang)
        
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=(f'''generate {desired_count} question with 4 options {language}on the following text: "{context}". the true answer is option {answer_option}'''),
            max_tokens=256,
            n = 1,
            stop = None,
            temperature = 0.2,
            top_p = 1,
        )
        
        print(f'''generate {desired_count} question with 4 options on the following text: "{context}". the true answer is option {answer_option}''')
        
        output = response["choices"]

        for qad_pair in output:
            print(qad_pair)
            try:
                question, answer, distractors = self.format_text_regex(qad_pair['text'], answer_option)
            except Exception:
                break
            questions_list.append(question)
            answers_list.append(answer)
            distractors_list.append(distractors)

        if len(answers_list) < 1:
            answers_list.append('')
            
        if len(questions_list) < 1:
            questions_list.append('')
            
        if len(distractors_list) < 1:
            questions_list.append([])
            
        while len(distractors_list[0]) < 3:
            distractors_list[0].append('')
        
        return answers_list[0], questions_list[0], distractors_list[0][0], distractors_list[0][1], distractors_list[0][2]
    
