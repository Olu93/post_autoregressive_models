from re import DEBUG
from typing import Dict, Iterable, Final
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WhitespaceTokenizer, SpacyTokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers import DatasetReader
from datasets import load_dataset, Dataset
import random
from tqdm import tqdm

# DEBUG_LEVEL = 0 # No Debug
# DEBUG_LEVEL = 1 # Simple Mode
DEBUG_LEVEL = 2  # Quick Mode
# DEBUG_LEVEL = 3


class LanguageModelReader(DatasetReader):
    TRAIN: Final = "train"
    VAL: Final = "val"
    TEST: Final = "test"
    ALL: Final = "unsupervised"
    DS_NAME: Final = "imdb"
    allowed: Final = [TRAIN, TEST, VAL]

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_tokens: int = None,
                 **kwargs):
        super().__init__(**kwargs, max_instances=1000 if DEBUG_LEVEL > 1 else None)
        self.tokenizer = tokenizer or SpacyTokenizer()
        self.sentence_splitter = tokenizer or SpacySentenceSplitter()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_tokens = max_tokens
        self.all_datasets: Dict[str, Dataset] = None

    @classmethod
    def cast(cls, other: DatasetReader):
        # Copy attributes only if other is of good type
        if isinstance(other, cls):
            new_object = LanguageModelReader()
            new_object.__dict__ = other.__dict__.copy()
        return new_object

    def init_dataset(self, frac=0.2):
        suffix = (f"[:{10000}]" if DEBUG_LEVEL > 0 else "")
        dataset = load_dataset(LanguageModelReader.DS_NAME,
                               split=LanguageModelReader.TRAIN + suffix)
        ds = dataset.shuffle().train_test_split(test_size=frac)
        self.all_datasets = {
            LanguageModelReader.TRAIN: ds['train']['text'],
            LanguageModelReader.VAL: ds['test']['text'],
            LanguageModelReader.TEST: load_dataset(LanguageModelReader.DS_NAME, split=LanguageModelReader.TEST)['text'],
        }

    def load_dataset(self, set_type: str):
        is_allowed = any([set_type.startswith(prefix) for prefix in LanguageModelReader.allowed])
        assert is_allowed, f"You can use {set_type} for dataset {LanguageModelReader.DS_NAME}. Use one of those: {LanguageModelReader.allowed}"

        if not self.all_datasets:
            self.init_dataset()

        ds = self.all_datasets.get(set_type)

        sentence_sets = (self.sentence_splitter.split_sentences(txt) for txt in ds)
        lines = (self.tokenizer.tokenize(line) for sent in sentence_sets for line in sent)
        if set_type in [LanguageModelReader.TRAIN, LanguageModelReader.VAL]:
            ds = (line[:random.randint(2, len(line))] for line in lines if len(line) > 2 for _ in range(3))
        if set_type in [LanguageModelReader.TEST]:
            ds = (line[:i] for line in lines for i in range(2, len(line)) if len(line) > 2)

        return ds

    def _read(self, file_path: str) -> Iterable[Instance]:
        lines = self.load_dataset(file_path)

        for tokens in tqdm(lines):
            text, next_word = tokens[:-1], tokens[-1]
            text_field = TextField(text, self.token_indexers)
            label_field = LabelField(str(next_word), skip_indexing=False)
            yield Instance({"text": text_field, "label": label_field})


if __name__ == "__main__":
    reader = LanguageModelReader()
    for i, instance in enumerate(reader.read(LanguageModelReader.TRAIN)):
        print(f"{i}: {instance}")
        if i > 5:
            break
    for i, instance in enumerate(reader.read(LanguageModelReader.VAL)):
        print(f"{i}: {instance}")
        if i > 5:
            break
    for i, instance in enumerate(reader.read(LanguageModelReader.TEST)):
        print(f"{i}: {instance}")
        if i > 5:
            break
