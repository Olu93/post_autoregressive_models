import tempfile
from typing import Dict, Iterable, List, Tuple

from allennlp.data import (
    DataLoader,
    DatasetReader,
    Instance,
    Vocabulary,
)
from allennlp.data.data_loaders import SimpleDataLoader

from allennlp.models import Model
from allennlp.modules.seq2vec_encoders.pytorch_seq2vec_wrapper import LstmSeq2VecEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.training import Trainer, GradientDescentTrainer
from allennlp.training.optimizers import AdamOptimizer
from encoder.TCNEncoder import TCNEncoder
from model.SimpleClassifier import SimpleClassifier
from model.TextTCNModel import TextTCNModel

from reader.TextReader import LanguageModelReader


def build_dataset_reader() -> DatasetReader:
    return LanguageModelReader()


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
    embedding_vector_size = 10
    embedder = BasicTextFieldEmbedder(
        {"tokens": Embedding(embedding_dim=embedding_vector_size, num_embeddings=vocab_size)})
    encoder = TCNEncoder(embedding_dims=(embedding_vector_size, 10), kernel_size=2)
    return TextTCNModel(vocab, embedder, encoder)


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

    train_loader, dev_loader = build_data_loaders(train_data, dev_data)
    train_loader.index_with(vocab)
    dev_loader.index_with(vocab)

    # You obviously won't want to create a temporary file for your training
    # results, but for execution in binder for this guide, we need to do this.
    with tempfile.TemporaryDirectory() as serialization_dir:
        trainer = build_trainer(model, serialization_dir, train_loader, dev_loader)
        trainer.train()

    return model, dataset_reader


if __name__ == "__main__":
    run_training_loop()