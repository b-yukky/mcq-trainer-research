from typing import List
from nltk.tokenize import sent_tokenize
import toolz

from utils.duplicate_removal import remove_duplicates, remove_distractors_duplicate_with_correct_answer
from utils.text_cleaning import clean_text
from models.DGenerator import DGenerator
from .QAGenerator import QAGenerator

import time
from typing import List

import re


class Question:
    def __init__(self, answerText:str, questionText: str = '', distractors: List[str] = []):
        self.answerText = answerText
        self.questionText = questionText
        self.distractors = distractors

class LeafBaseMCQGenerator():
    def __init__(self, is_verbose=True):
        start_time = time.perf_counter()
        print('Loading ML Models...')

        self.question_generator = QAGenerator('t5-small', 'E:\\mcq-trainer-research\\checkpoints\\t5-small\\multitask-qg-ag.ckpt')
        print('Loaded QAGenerator in', round(time.perf_counter() - start_time, 2), 'seconds.') if is_verbose else ''

        self.distractor_generator = DGenerator('t5-small', 'E:\\mcq-trainer-research\\checkpoints\\t5-small\\race-distractors.ckpt')
        print('Loaded DistractorGenerator in', round(time.perf_counter() - start_time, 2), 'seconds.') if is_verbose else ''


    # Main function
    def generate(self, context: str, desired_count: int = 1) -> tuple:
        cleaned_text =  clean_text(context)

        questions = self._generate_question_answer_pairs(cleaned_text, desired_count)
        questions = self._generate_distractors(cleaned_text, questions)
        
        questions_list = []
        answers_list = []
        distractors_list = []

        for question in questions:
            questions_list.append(question.questionText)
            answers_list.append(question.answerText)
            distractors_list.append(question.distractors)
        
        if len(answers_list) < 1:
            answers_list.append('')
            
        if len(questions_list) < 1:
            questions_list.append('')
            
        if len(distractors_list) < 1:
            questions_list.append([])
            
        while len(distractors_list[0]) < 3:
            distractors_list[0].append('')
            
        return answers_list[0], questions_list[0], distractors_list[0][0], distractors_list[0][1], distractors_list[0][2]


    def _generate_question_answer_pairs(self, context: str, desired_count: int) -> List[Question]:
        context_splits = self._split_context_according_to_desired_count(context, desired_count)

        questions = []

        for split in context_splits:
            answer, question = self.question_generator.generate(split)
            questions.append(Question(answer.capitalize(), question))

        questions = list(toolz.unique(questions, key=lambda x: x.answerText))

        return questions

    def _generate_distractors(self, context: str, questions: List[Question]) -> List[Question]:
        for question in questions:
            t5_distractors =  self.distractor_generator.generate(question.answerText, question.questionText, context, generate_count=5)

            distractors = t5_distractors

            distractors = remove_duplicates(distractors)
            distractors = remove_distractors_duplicate_with_correct_answer(question.answerText, distractors)
            #TODO - filter distractors having a similar bleu score with another distractor

            question.distractors = distractors

        return questions


    #TODO: refactor to create better splits closer to the desired amount
    def _split_context_according_to_desired_count(self, context: str, desired_count: int) -> List[str]:
        sents = sent_tokenize(context)
        sent_ratio = len(sents) / desired_count

        context_splits = []

        if sent_ratio < 1:
            return sents
        else:
            take_sents_count = int(sent_ratio + 1)

            start_sent_index = 0

            while start_sent_index < len(sents):
                context_split = ' '.join(sents[start_sent_index: start_sent_index + take_sents_count])
                context_splits.append(context_split)
                start_sent_index += take_sents_count - 1

        return context_splits
    