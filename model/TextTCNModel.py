from typing import Dict
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.data.vocabulary import Vocabulary
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
        input_size,
        embedding_sizes,
        vocab: Vocabulary,
        embedder: TextFieldEmbedder,
        kernel_size=2,
    ):
        super(TextTCNModel, self).__init__()
        assert embedding_sizes[
            -1] == input_size, f"Out emb size must be the same as input emb size. Given: {embedding_sizes[-1]}, {input_size}"
        super().__init__(vocab)
        self.embedder = embedder
        num_labels = vocab.get_vocab_size("labels")
        # self.encoder = nn.Embedding(num_labels, input_size)

        self.encoder = ResidualTCN(input_size, embedding_sizes, kernel_size)
        self.decoder = nn.Linear(embedding_sizes[-1], num_labels)

    def forward(self, text: TextFieldTensors, label: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""
        emb = self.embedder(text)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text) # Check if necessary
        enc = self.encoder(emb.transpose(1, 2)).transpose(1, 2)
        logits = self.decoder(enc)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits)
        # Shape: (1,)
        loss = torch.nn.functional.cross_entropy(logits, label)
        self.accuracy(logits, label)
        output = {"loss": loss, "probs": probs}
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}
