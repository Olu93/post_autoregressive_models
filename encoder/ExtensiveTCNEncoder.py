from typing import Dict
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.modules.feedforward import FeedForward
from allennlp.nn import util
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder, Seq2SeqEncoder
from allennlp.modules.seq2seq_encoders import PassThroughEncoder, FeedForwardEncoder
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.models import Model
from overrides.overrides import overrides
import torch
from torch import nn

from modules.TCN import SequentialTemporalConvolutionNetworkModel


class ExtensiveTCNEncoder(PassThroughEncoder):
    def __init__(self, embedding_dims: int, kernel_size: int) -> None:
        super().__init__(embedding_dims[0])
        self.emb_dims = embedding_dims
        self.tcn = SequentialTemporalConvolutionNetworkModel(embedding_dims, kernel_size)

    # def __init__(self, embedding_dims: int, kernel_size: int, averaged: bool = False) -> None:

    @overrides
    def forward(self, inputs: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        # mask it first
        masked_input = inputs if mask is None else (inputs * mask.unsqueeze(dim=-1))
        # (batch_size, seq_len, emb_size/n_channel) ->  (batch_size, emb_size/n_channel, seq_len)
        emb_for_TCN_encoder = masked_input.transpose(1, 2)
        # (batch_size, emb_size/n_channel, seq_len) -> (batch_size, seq_len, emb_size/n_channel)
        output_after_conv = self.tcn(emb_for_TCN_encoder).transpose(1, 2)

        # return self._feedforward(output_after_conv)
        return output_after_conv

    @overrides
    def get_output_dim(self):
        return self.emb_dims[-1]