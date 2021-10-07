from fasttext import FastText
from os import path
import re

MODEL_FILE = path.join(path.dirname(__file__), 'model', 'lid.176.ftz')

class WhatTheLang(object):
    def __init__(self):
        self.model_file = MODEL_FILE
        self.model = self.load_model()
        self.unknown = "CANT_PREDICT"

    def load_model(self):
        return FastText.load_model(self.model_file)

    def _clean_up(self,txt):
        txt = re.sub(r"\b\d+\b", "", txt)
        return txt

    def _flatten(self,pred):
        return [item[0] if len(item)!=0 else self.unknown for item in pred]

    def _get_langs(self):
        return self.model.labels

    def _label_to_output(self, label):
        return label.replace("__label__", "")

    def predict_lang(self,inp):
        if type(inp) != list:
            cleaned_txt = self._clean_up(inp)
            if cleaned_txt == "":
                raise ValueError("Not enough text to predict language")
            pred, _ = self.model.predict([cleaned_txt])
            pred = pred[0]
            if len(pred) == 0:
                return self.unknown
            return self._label_to_output(pred[0])
        else:
            batch = [self._clean_up(i) for i in inp]
            pred, _ = self.model.predict(batch)
            return [
                self._label_to_output(label)
                for label in self._flatten(pred)
            ]

    def pred_prob(self,inp):
        if type(inp) != list:
            inp = self._clean_up(inp)
            pred, prob = self.model.predict([inp])
            return [
                (self._label_to_output(label), probability)
                for label, probability in zip(pred[0], prob[0])
            ]
        else:
            pred, prob = self.model.predict(inp)
            return [
                [
                    (self._label_to_output(label), probability)
                    for label, probability in zip(item_predictions, item_probabilities)
                ]
                for item_predictions, item_probabilities in zip(pred, prob)
            ]



