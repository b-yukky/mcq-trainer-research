import openai
from dotenv import load_dotenv
import os
import re
import random
import backoff
load_dotenv()

class OpenAIChatGPTGenerator():
    
    def __init__(self) -> None:
        ''' init '''
        self.API_KEY = os.environ.get("OPENAI_API_KEY")
        openai.api_key = self.API_KEY
        self.models = openai.Model.list()
        self.engines = openai.Engine.list()
    
    def generate_random_answer_option(self):
        potential_answers = ['A', 'B', 'C', 'D']
        return random.choice(potential_answers)
    
    def double_lines_extraction(self, lines, ans):
        question_pattern = re.compile(r"\s*(?P<question>.+\?)|(Q:|Question:)\s*.+")
        print('double lines', lines)
        for line in lines:
            question_match = question_pattern.search(line)
            if line.endswith('?') or line.endswith(':') or question_match:
                question = question_match.group('question').replace('\n', '')
            else:
                r = self.extract_single_line_choices(line, answer_option=ans)
                if r != []:
                    answer = r[0]
                r = self.extract_single_line_choices(line.replace(answer[0], ''))
                if r != []:
                    distractors = r
        
        return question, answer, distractors
    
    def multi_lines_extraction(self, lines, ans):
        question_pattern = re.compile(r"\s*(?P<question>.+\?)|(Q:|Question:)\s*.+")
        answer_pattern = re.compile(r"({}:|{}\.|{}\))\s*(?P<answer>.+)?".format(ans, ans, ans))
        choice_pattern = re.compile(r"[a-zA-Z]+(:|\.|\))\s*(?P<choice>.+)")
        distractors = []
        for line in lines:
            question_match = question_pattern.search(line)
            answer_match = answer_pattern.search(line)
            choice_match = choice_pattern.search(line)
            if line.endswith('?') or line.endswith(':') or question_match:
                question = question_match.group('question')
            elif answer_match:
                answer = answer_match.group('answer')
            elif choice_match:
                choice = choice_match.group('choice')
                distractors.append(choice)
        
        return question, answer, distractors
    
    def format_text_regex(self, text, ans):
        
        # split text into lines
        lines = text.strip().split("\n")
        # remove empty lines
        lines = [line.strip() for line in lines if len(line) > 3]
        
        if len(lines) > 2:
            question, answer, distractors = self.multi_lines_extraction(lines, ans)
        else:
            question, answer, distractors = self.double_lines_extraction(lines, ans)
        
        question = self.clean_question(question)
        answer = self.clean_answer(answer)
        distractors = self.clean_distractors(distractors)
                    
        return question, answer, distractors

    def select_language(self, lang):
        
        if lang == 'fr':
            return 'in french '
        else:
            return ''
    
    def extract_single_line_choices(self, string, answer_option=None):
        if answer_option:
            pattern = re.compile(r'{}[\.\)](\s[^ ]+)(( [\w\-\'\"]+)+)?(?![\.\)])'.format(answer_option))
        else:
            # pattern = re.compile(r'[A-Z][\.\)](\s[\w]+)(( \w+)+)?(?![\.\)])')
            pattern = re.compile(r'[A-Z][\.\)](\s[^ ]+)(( [\w\-\'\"]+)+)?(?![\.\)])')
        choices = pattern.findall(string)
        choices = [(choice[0] + choice[1]).strip() for choice in choices]
        return choices
    
    def clean_question(self, question):
        question = question.replace('\n', '').replace('Q:', '').replace('Question:', '').replace('Q.', '').replace('Q)', '').strip()
        return question
    
    def clean_distractors(self, distractors):
        # replace any option letter with empty string
        distractors = [re.sub(r'[A-F][\.\)]', '', distractor).strip() for distractor in distractors]
        return distractors
    
    def clean_answer(self, answer):
        # remove answer option letter
        answer = re.sub(r'[A-F][\.\)]', '', answer).strip()
        return answer
    
    def generate(self, context: str, desired_count: int = 1, lang='en') -> tuple:
        
        questions_list = []
        answers_list = []
        distractors_list = []
        
        answer_option = self.generate_random_answer_option()
        language = self.select_language(lang)
        
        @backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.APIError), max_tries=5, max_time=60)
        def generate_question_with_backoff(**kwargs):
            return openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": f'''generate {desired_count} question with 4 options {language}on the following text: "{context}". the true answer is option {answer_option}'''}
                ]
            )
         
        response = generate_question_with_backoff()
        
        print(f'''generate {desired_count} question with 4 options on the following text: "{context}". the true answer is option {answer_option}''')
        
        output = [ response["message"] for response in response["choices"]]

        for qad_pair in output:
            print(qad_pair)
            try:
                question, answer, distractors = self.format_text_regex(qad_pair['content'], answer_option)
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
            distractors_list.append([])
            
        while len(distractors_list[0]) < 3:
            distractors_list[0].append('')
        
        return answers_list[0], questions_list[0], distractors_list[0][0], distractors_list[0][1], distractors_list[0][2]
    