import tempfile
from typing import Dict, Iterable, List, Tuple
from allennlp.data import (
    DataLoader,
    DatasetReader,
    Instance,
    Vocabulary,
)
from allennlp.data.data_loaders import SimpleDataLoader
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
DEBUG_LEVEL = 2  # Quick Mode
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

def build_dataset_reader() -> DatasetReader:
    return ExtensiveLanguageModelReader(max_instances=MAX_SIZE)


def read_data(reader: DatasetReader, re_init=True, frac=0.01) -> Tuple[List[Instance], List[Instance]]:
    print('Initializing data')
    if re_init and isinstance(reader, LanguageModelReader):
        reader.init_dataset(frac=frac)
    print("Reading data")
    training_data = list(reader.read(LanguageModelReader.TRAIN))
    validation_data = list(reader.read(LanguageModelReader.VAL))
    return training_data, validation_data


def build_vocab(instances: Iterable[Instance]) -> Vocabulary:
    print("Building the vocabulary")
    return Vocabulary.from_instances(instances)


def build_model(vocab: Vocabulary) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")
    embedding_vector_size = 30
    embedder = BasicTextFieldEmbedder(
        {"tokens": Embedding(embedding_dim=embedding_vector_size, num_embeddings=vocab_size)})
    encoder = ExtensiveTCNEncoder(embedding_dims=(embedding_vector_size, 10, 10, 10), kernel_size=2)
    return ExtensiveLanguageTCNModel(vocab, embedder, encoder)


def build_data_loaders(
    train_data: List[Instance],
    dev_data: List[Instance],
) -> Tuple[DataLoader, DataLoader]:
    train_loader = SimpleDataLoader(train_data, 8, shuffle=True)
    dev_loader = SimpleDataLoader(dev_data, 8, shuffle=False)
    return train_loader, dev_loader


def build_trainer(
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
        num_epochs=50,
        optimizer=optimizer,
    )
    return trainer


def run_training_loop():
    dataset_reader = build_dataset_reader()

    train_data, dev_data = read_data(dataset_reader)

    vocab = build_vocab(train_data + dev_data)
    model = build_model(vocab)
    model = model.to(device='cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, dev_loader = build_data_loaders(train_data, dev_data)
    train_loader.index_with(vocab)
    dev_loader.index_with(vocab)

    # You obviously won't want to create a temporary file for your training
    # results, but for execution in binder for this guide, we need to do this.
    with tempfile.TemporaryDirectory() as serialization_dir:
        trainer = build_trainer(model, serialization_dir, train_loader, dev_loader)
        trainer.train()

    predictor = ExtensiveLanguageModelPredictor(model, dataset_reader)
    test_input_var = "A good"
    qualitative_model_test(dataset_reader, vocab, predictor, test_input_var)
    test_input_var = "My home is"
    qualitative_model_test(dataset_reader, vocab, predictor, test_input_var)
    test_input_var = "That is a"
    qualitative_model_test(dataset_reader, vocab, predictor, test_input_var)
    test_input_var = "I love"
    qualitative_model_test(dataset_reader, vocab, predictor, test_input_var)
    test_input_var = "You can use a lot of"
    qualitative_model_test(dataset_reader, vocab, predictor, test_input_var)


    return model, dataset_reader


def qualitative_model_test(dataset_reader, vocab, predictor, test_input_var):
    output = predictor.predict(test_input_var)
    probs = output["class_probabilities"][-1]
    print(
        dataset_reader.tokenizer.tokenize(test_input_var)[:-1] + [vocab.get_token_from_index(np.argmax(probs), "labels")],
        np.max(probs),
    )


if __name__ == "__main__":
    run_training_loop()
# %%
