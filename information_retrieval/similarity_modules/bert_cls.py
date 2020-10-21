from typing import Dict, Iterator, List

import torch
import torch.nn as nn
from registrable import Registrable

from allennlp.modules import Seq2VecEncoder
from allennlp.data import TextFieldTensors
from allennlp.modules.seq2vec_encoders.cls_pooler import ClsPooler

class RelevanceMatcher(Registrable, torch.nn.Module):
    def __init__(
        self,
        input_dim: int
    ):
        super().__init__()
        self.dense = torch.nn.Linear(input_dim, 1, bias=False)


    def forward(
        self, 
        query_embeddings: TextFieldTensors, 
        candidates_embeddings: TextFieldTensors,
        query_mask: torch.Tensor = None,
        candidates_mask: torch.Tensor = None
    ):
        raise NotImplementedError()

@RelevanceMatcher.register('bert_cls')
class BertCLS(RelevanceMatcher):
    def __init__(
        self,
        input_dim: int,
        **kwargs
    ):
        super().__init__(input_dim=input_dim*4, **kwargs)
      
        # Gets the [CLS] token from BERT
        self._seq2vec_encoder = ClsPooler(embedding_dim=input_dim)

    def forward(
        self,
        lyric_embeddings: torch.Tensor,
        option_embeddings: torch.Tensor,
        lyric_mask: torch.Tensor = None,
        option_mask: torch.Tensor = None
    )-> torch.Tensor:

        lyric_encoded = self._seq2vec_encoder(lyric_embeddings, lyric_mask)
        option_encoded = self._seq2vec_encoder(option_embeddings, option_mask)

        interaction_vector = torch.cat(
            [
                lyric_encoded,
                option_encoded,
                torch.abs(lyric_encoded-option_encoded),
                lyric_encoded*option_encoded
            ],
            dim=1,
        )
        dense_out = self.dense(interaction_vector)
        score = torch.squeeze(dense_out,1)

        return score