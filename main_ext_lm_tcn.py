import tempfile
from typing import Dict, Iterable, List, Tuple, Type
from allennlp.common.util import END_SYMBOL, START_SYMBOL
from allennlp.data import (
    DataLoader,
    DatasetReader,
    Instance,
    Vocabulary,
)
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers.sentence_splitter import SentenceSplitter, SpacySentenceSplitter
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from allennlp.data.tokenizers.tokenizer import Tokenizer
import numpy as np
from allennlp.models import Model
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.seq2vec_encoders.pytorch_seq2vec_wrapper import LstmSeq2VecEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.training import Trainer, GradientDescentTrainer
from allennlp.training.optimizers import AdamOptimizer
from encoder.ExtensiveTCNEncoder import ExtensiveTCNEncoder
from encoder.TCNEncoder import TCNEncoder
from model.ExtensiveLanguageTCNModel import ExtensiveLanguageTCNModel
from model.TextTCNModel import TextTCNModel
from predictor.Predictor import ExtensiveLanguageModelPredictor

from reader.ExtensiveLanguageModelReader import ExtensiveLanguageModelReader
from reader.LanguageModelReader import LanguageModelReader
import torch

DEBUG_LEVEL = -1  # Normal
# DEBUG_LEVEL = 0 # No Debug
# DEBUG_LEVEL = 1 # Simple Mode
# DEBUG_LEVEL = 2  # Quick Mode
DEBUG_LEVEL = 3  # Super quick mode
# MAX_SIZE = 50000

# Maxsize is...
# 942210it [44:12, 355.23it/s]
# 8916it [00:21, 406.60it/s]
MAX_SIZE = 25000

if DEBUG_LEVEL == 0:
    MAX_SIZE = None
if DEBUG_LEVEL == 1:
    MAX_SIZE = 10000
if DEBUG_LEVEL == 2:
    MAX_SIZE = 1000
if DEBUG_LEVEL == 3:
    MAX_SIZE = 250


class ExtensiveRunner():
    def __init__(
        self,
        reader: LanguageModelReader,
        num_epochs: int = 10,
    ) -> None:
        self.reader = reader
        self.num_epochs = num_epochs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def read_data(self, reader: DatasetReader, re_init=True, frac=0.01) -> Tuple[List[Instance], List[Instance]]:
        print('Initializing data')
        if re_init and isinstance(reader, LanguageModelReader):
            reader.init_dataset(frac=frac)
        print("Reading data")
        training_data = list(reader.read(LanguageModelReader.TRAIN))
        validation_data = list(reader.read(LanguageModelReader.VAL))
        return training_data, validation_data

    def build_vocab(self, instances: Iterable[Instance]) -> Vocabulary:
        print("Building the vocabulary")
        return Vocabulary.from_instances(instances)

    def build_model(self, vocab: Vocabulary) -> Model:
        print("Building the model")
        vocab_size = vocab.get_vocab_size("tokens")
        embedding_vector_size = 30
        embedder = BasicTextFieldEmbedder(
            {"tokens": Embedding(embedding_dim=embedding_vector_size, num_embeddings=vocab_size)})
        encoder = ExtensiveTCNEncoder(embedding_dims=(embedding_vector_size, 10, 10, 10), kernel_size=2)
        return ExtensiveLanguageTCNModel(vocab, embedder, encoder)

    def build_data_loaders(
        self,
        train_data: List[Instance],
        dev_data: List[Instance],
    ) -> Tuple[DataLoader, DataLoader]:
        train_loader = SimpleDataLoader(train_data, 8, shuffle=True)
        dev_loader = SimpleDataLoader(dev_data, 8, shuffle=False)
        return train_loader, dev_loader

    def build_trainer(
        self,
        model: Model,
        serialization_dir: str,
        train_loader: DataLoader,
        dev_loader: DataLoader,
    ) -> Trainer:
        parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        optimizer = AdamOptimizer(parameters)  # type: ignore
        trainer = GradientDescentTrainer(
            model=model,
            serialization_dir=serialization_dir,
            data_loader=train_loader,
            validation_data_loader=dev_loader,
            num_epochs=self.num_epochs,
            optimizer=optimizer,
        )
        return trainer

    def run_training_loop(self, model: Model = None):
        dataset_reader = self.reader

        train_data, dev_data = self.read_data(dataset_reader)

        vocab = self.build_vocab(train_data + dev_data)
        model = self.build_model(vocab) if model is None else model
        model = model.to(self.device)

        train_loader, dev_loader = self.build_data_loaders(train_data, dev_data)
        train_loader.index_with(vocab)
        dev_loader.index_with(vocab)

        # You obviously won't want to create a temporary file for your training
        # results, but for execution in binder for this guide, we need to do this.
        with tempfile.TemporaryDirectory() as serialization_dir:
            trainer = self.build_trainer(model, serialization_dir, train_loader, dev_loader)
            trainer.train()

        predictor = ExtensiveLanguageModelPredictor(model, dataset_reader)
        predictor.display_qualitative_model_test(quick_mode=True, probablistic=False)

        return model, dataset_reader


if __name__ == "__main__":
    reader = ExtensiveLanguageModelReader(
        max_instances=MAX_SIZE,
        tokenizer=SpacyTokenizer(start_tokens=[START_SYMBOL]),
        sentence_splitter=SpacySentenceSplitter(),
        token_indexers=SingleIdTokenIndexer(lowercase_tokens=True),
    )
    runner = ExtensiveRunner(reader=reader, num_epochs=1)
    for i in range(10):
        model = None
        model, dataset_reader = runner.run_training_loop(model)
    predictor = ExtensiveLanguageModelPredictor(model, dataset_reader)
    predictor.display_qualitative_model_test(quick_mode=False, probablistic=False)
# %%
