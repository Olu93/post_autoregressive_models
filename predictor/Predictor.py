import tempfile
from typing import Dict, Iterable, List, Tuple
import numpy as np

import torch

from allennlp.common.util import JsonDict
from allennlp.data import (
    DataLoader,
    DatasetReader,
    Instance,
    Vocabulary,
    TextFieldTensors,
)
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.nn import util
from allennlp.predictors import Predictor
from allennlp.training import Trainer, GradientDescentTrainer
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.optimizers import AdamOptimizer
from allennlp.training.util import evaluate
import random as r
from reader.LanguageModelReader import LanguageModelReader


class ExtensiveLanguageModelPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader, frozen: bool = True) -> None:
        super().__init__(model, dataset_reader, frozen=frozen)
        self._vocab = model.vocab

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["sentence"]
        return self._dataset_reader.text_to_instance(self._dataset_reader.tokenizer.tokenize(sentence)[:-1])

    def display_qualitative_model_test(self, test_inputs: List[str] = None, quick_mode: bool = False, k: int = 10, probablistic:bool=False):
        test_inputs = test_inputs or self.sample_instances(k)
        prepared_test_inputs = [self._dataset_reader.text_to_instance(txt[:-1], txt[1:])
                       for txt in test_inputs]
        for instance in prepared_test_inputs:
            if not quick_mode:
                print("==================================================")
            all_class_probs = self.predict_instance(instance)["class_probabilities"]
            probs = np.array(all_class_probs if not quick_mode else [all_class_probs[-1]])
            true_labels = instance["labels"] if not quick_mode else [instance["labels"][-1]]
            # https://numpy.org/doc/stable/reference/generated/numpy.take.html AND https://stackoverflow.com/a/40475357/4162265
            chosen_indxs = np.argmax(probs, axis=1) if not probablistic else (np.random.rand(len(probs.cumsum(axis=1)), 1) < probs.cumsum(axis=1)).argmax(axis=1)
            max_probs = np.take(probs, list(chosen_indxs))
            token_strings = [str(token) for token in instance['text']]
            for token, (next_token_idx, pred_token_idx,
                        pred_confidence) in enumerate(zip(true_labels, chosen_indxs, max_probs)):
                next_token = next_token_idx
                pred_token = self._vocab.get_token_from_index(pred_token_idx, "labels")
                end = token + 1 if not quick_mode else -1
                print(f"{' '.join(token_strings[:end])} -> [{pred_token} ({next_token})] : {pred_confidence} ")

    def sample_instances(self, k: int = 10):
        splits = (sent['text'][:r.randint(2, max([2, len(sent['text'])]))] for sent in self._dataset_reader.read(LanguageModelReader.VAL) if len(sent['text']) > 1)
        return [inp for i, inp in enumerate(splits) if (i < k)] 

    def predict_sentence(self, sentence: str, max_len:int=50) -> JsonDict:
        final_token = self._vocab.get_token_index(".", "labels")
        input_sentence = sentence
        for _ in range(max_len):
            all_class_probs = self.predict_json({"sentence": input_sentence})["class_probabilities"]
            probs = np.array([all_class_probs[-1]])
            chosen_index = np.argmax(probs, axis=1)
            chosen_word = self._vocab.get_token_from_index(chosen_index, "labels")
            if chosen_word == self._vocab.get_token_from_index(final_token, "labels"):
                break
            input_sentence = f"{input_sentence} {chosen_word}"

        return f"{input_sentence} {chosen_word}"