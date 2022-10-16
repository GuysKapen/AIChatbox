# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
import json

import requests
import spacy
from typing import List, Dict, Text, Any

from rasa_sdk import Action, Tracker
from .components import QueryProcessor, DocumentRetrieval, PassageRetrieval, AnswerExtractor

from dotenv import load_dotenv
import os

load_dotenv()

IMAGE_GENERATE_API_URL = os.environ['IMAGE_GENERATE_API_URL']


class ActionAnswerInfoQuestion(Action):
    def name(self) -> Text:
        return "action_answer_info_question"

    # noinspection PyPep8Naming
    async def run(self, dispatcher, tracker: Tracker, domain) -> List[Dict[Text, Any]]:
        # SPACY_MODEL = os.environ.get('SPACY_MODEL', 'en_core_web_sm')
        # QA_MODEL = os.environ.get('QA_MODEL', 'distilbert-base-cased-distilled-squad')
        SPACY_MODEL = 'en_core_web_sm'
        QA_MODEL = 'distilbert-base-cased-distilled-squad'
        nlp = spacy.load(SPACY_MODEL, disable=['ner', 'parser', 'textcat'])
        query_processor = QueryProcessor(nlp)
        document_retriever = DocumentRetrieval()
        passage_retriever = PassageRetrieval(nlp)
        answer_extractor = AnswerExtractor(QA_MODEL, QA_MODEL)

        question = tracker.latest_message.get('text')

        query = query_processor.generate_query(question)
        docs = document_retriever.search(query)
        passage_retriever.fit(docs)
        passages = passage_retriever.most_similar(question)
        answers = answer_extractor.extract(question, passages)
        if len(answers) == 0:
            dispatcher.utter_message("Not found")
        dispatcher.utter_message(
            text=answers[0].get('answer'))
        return []


class ActionImageGenerate(Action):

    def name(self) -> Text:
        return "action_image_generate"

    async def run(self, dispatcher, tracker: Tracker, domain):
        result = requests.post(IMAGE_GENERATE_API_URL,
                               json={"prompt": tracker.get_slot("prompt")})
        dispatcher.utter_message(response="utter_generate_image", json_message=json.loads(result.content))
        return []
