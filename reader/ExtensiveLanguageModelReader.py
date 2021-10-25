from dataclasses import Field
from re import DEBUG
from typing import Dict, Iterable, Final, List
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField, Field
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WhitespaceTokenizer, SpacyTokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.tokenizers.token_class import Token
from datasets import load_dataset, Dataset
import random
from tqdm import tqdm
from overrides import overrides

from reader.LanguageModelReader import LanguageModelReader


DEBUG_LEVEL = -1  # Normal
# DEBUG_LEVEL = 0 # No Debug
DEBUG_LEVEL = 1 # Simple Mode
# DEBUG_LEVEL = 2  # Quick Mode
# DEBUG_LEVEL = 3 # Super quick mode
# MAX_SIZE = 50000

# Maxsize is...
# 942210it [44:12, 355.23it/s]
# 8916it [00:21, 406.60it/s]
MAX_SIZE = 50000

if DEBUG_LEVEL == 0:
    MAX_SIZE = None
if DEBUG_LEVEL == 1:
    MAX_SIZE = 10000
if DEBUG_LEVEL == 2:
    MAX_SIZE = 1000
if DEBUG_LEVEL == 3:
    MAX_SIZE = 250


class ExtensiveLanguageModelReader(LanguageModelReader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_instances = MAX_SIZE

    def load_dataset(self, set_type: str):
        is_allowed = any([set_type.startswith(prefix) for prefix in ExtensiveLanguageModelReader.allowed])
        assert is_allowed, f"You can use {set_type} for dataset {ExtensiveLanguageModelReader.DS_NAME}. Use one of those: {ExtensiveLanguageModelReader.allowed}"

        if not self.all_datasets:
            self.init_dataset()

        ds = self.all_datasets.get(set_type)

        sentence_sets = (self.sentence_splitter.split_sentences(txt) for txt in ds)
        lines = (self.tokenizer.tokenize(line) for sent in sentence_sets for line in sent)
        if set_type in [ExtensiveLanguageModelReader.TRAIN, ExtensiveLanguageModelReader.VAL]:
            ds = (line for line in lines if len(line) > 2)
        if set_type in [ExtensiveLanguageModelReader.TEST]:
            ds = (line[:i] for line in lines for i in range(2, len(line)) if len(line) > 2)

        return ds

    def _read(self, file_path: str) -> Iterable[Instance]:
        lines = self.load_dataset(file_path)
        for tokens in tqdm(lines):
            prev_words, next_words = tokens[:-1], tokens[1:]
            yield self.text_to_instance(prev_words, next_words)

    @overrides
    def text_to_instance(self, prev_words: List[Token], next_words: List[Token]) -> Instance:
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        fields: Dict[str, Field] = {}
        sequence = TextField(prev_words, self.token_indexers)
        fields["tokens"] = sequence
        fields["metadata"] = MetadataField({"words": [x.text for x in prev_words]})
        if next_words is not None:
            fields["tags"] = SequenceLabelField([lbl.text for lbl in next_words], sequence)
        return Instance(fields)


if __name__ == "__main__":
    reader = ExtensiveLanguageModelReader()
    for i, instance in enumerate(reader.read(ExtensiveLanguageModelReader.TRAIN)):
        print(f"{i}: {instance}")
        if i > 5:
            break
    for i, instance in enumerate(reader.read(ExtensiveLanguageModelReader.VAL)):
        print(f"{i}: {instance}")
        if i > 5:
            break
    for i, instance in enumerate(reader.read(ExtensiveLanguageModelReader.TEST)):
        print(f"{i}: {instance}")
        if i > 5:
            break
