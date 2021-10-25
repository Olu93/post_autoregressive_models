from typing import Any, Dict, List
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.training.metrics.categorical_accuracy import CategoricalAccuracy
from overrides.overrides import overrides
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.TCN import TemporalConvolutionNetworkModel
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, TimeDistributed
from allennlp.nn import util
from allennlp.nn import InitializerApplicator
from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.nn import util as nn_util


class ExtensiveLanguageTCNModel(Model):
    def __init__(
            self,
            vocab: Vocabulary,
            embedder: TextFieldEmbedder,
            encoder: Seq2SeqEncoder,
            verbose_metrics: bool = False,
            initializer: InitializerApplicator = InitializerApplicator(),
            **kwargs,
    ):
        super().__init__(vocab, **kwargs)
        self.embedder = embedder
        self.encoder = encoder
        self._verbose_metrics = verbose_metrics
        self.num_classes = vocab.get_vocab_size("labels")

        self.decoder = TimeDistributed(nn.Linear(self.encoder.get_output_dim(), self.num_classes))
        check_dimensions_match(
            self.embedder.get_output_dim(),
            encoder.get_input_dim(),
            "text field embedding dim",
            "encoder input dim",
        )
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "accuracy_top3": CategoricalAccuracy(top_k=3),
        }

        initializer(self)

    @overrides
    def forward(
        self,  # type: ignore
        tokens: TextFieldTensors,
        tags: torch.LongTensor = None,
        metadata: List[Dict[str, Any]] = None,
        ignore_loss_on_o_tags: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""
        # Shape: (batch_size, num_tokens, embedding_dim)
        # print("=====================================")
        embedded_text = self.embedder(tokens)
        batch_size, sequence_length, _ = embedded_text.size()

        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(tokens)  # Check if necessary
        # Shape: (batch_size, new_num_tokens, embedding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # print(encoded_text.shape) - (batch_size, new_num_tokens, embedding_dim)
        logits = self.decoder(encoded_text)

        # (batch_size x new_num_tokens, embedding_dim)
        reshaped_log_probs = logits.view(-1, self.num_classes)
        class_probabilities = F.softmax(
            reshaped_log_probs,
            dim=-1,
        ).view([batch_size, sequence_length, self.num_classes])
        output_dict = {"logits": logits, "class_probabilities": class_probabilities}
        if tags is not None:
            if ignore_loss_on_o_tags:
                o_tag_index = self.vocab.get_token_index("O", namespace=self.label_namespace)
                tag_mask = mask & (tags != o_tag_index)
            else:
                tag_mask = mask
            loss = util.sequence_cross_entropy_with_logits(logits, tags, tag_mask)
            for metric in self.metrics.values():
                metric(logits, tags, mask)
            output_dict["loss"] = loss
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

        return metrics_to_return
