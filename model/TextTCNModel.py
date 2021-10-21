from typing import Dict
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.training.metrics.categorical_accuracy import CategoricalAccuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.TCN import ResidualTCN
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.nn import util


class TextTCNModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
    ):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.decoder = nn.Linear(self.encoder.get_output_dim(),  num_labels)
        self.accuracy = CategoricalAccuracy()

    def forward(self, text: TextFieldTensors, label: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)  # Check if necessary
        # Shape: (batch_size, new_num_tokens, embedding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # print("=====================================")
        # print(encoded_text.shape)
        logits = self.decoder(encoded_text)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits)
        # Shape: (1,)
        loss = torch.nn.functional.cross_entropy(logits, label)
        self.accuracy(logits, label)
        output = {"loss": loss, "probs": probs}
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}
