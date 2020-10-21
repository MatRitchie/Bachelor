from typing import Dict, Optional

from overrides import overrides
import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder, TimeDistributed
from allennlp.nn import InitializerApplicator, util
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, BooleanAccuracy, Auc, F1Measure, FBetaMeasure, PearsonCorrelation

from information_retrieval.similarity_modules.bert_cls import RelevanceMatcher
from information_retrieval.evaluation.mrr import MRR

@Model.register("ranker")
class DocumentRanker(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        relevance_matcher: RelevanceMatcher,
        dropout: float = None,
        num_labels: int = None,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:

        super().__init__(vocab, **kwargs)
        self._text_field_embedder = text_field_embedder
        self._relevance_matcher = TimeDistributed(relevance_matcher) 
        self._mrr = MRR(padding_value=-1)
        self._loss = torch.nn.MSELoss(reduction='none')
        initializer(self)
        
    def forward(  # type: ignore
        self, 
        lyric: TextFieldTensors,               # batch * words
        title_options: TextFieldTensors,       # batch * num_options * words
        labels: torch.IntTensor = None         # batch * num_options
    ) -> Dict[str, torch.Tensor]:
      
        embedded_text = self._text_field_embedder(lyric, num_wrapping_dims=0)
        mask = get_text_field_mask(lyric).long()

        embedded_options = self._text_field_embedder(title_options, num_wrapping_dims=1) 
        options_mask = get_text_field_mask(title_options, num_wrapping_dims=1).long() #har til føjet det med num_wrapping_dim

        #make pairs of (lyric, title) instead of (lyric, (title1, title2,...))
        embedded_text = embedded_text.unsqueeze(1).expand(-1, embedded_options.size(1), -1, -1)
        mask = mask.unsqueeze(1).expand(-1, embedded_options.size(1), -1)
        scores = self._relevance_matcher(embedded_text, embedded_options, mask, options_mask).squeeze(-1)
        probs = torch.sigmoid(scores)

        output_dict = {"logits": scores, "probs": probs}
        output_dict["token_ids"] = util.get_token_ids_from_text_field_tensors(lyric)
        
        if labels is not None:
            label_mask = (labels != -1)
            self._mrr(probs, labels, label_mask)
            probs = probs.view(-1)
            labels = labels.view(-1)
            label_mask = label_mask.view(-1)   
            loss = self._loss(probs, labels)
            output_dict["loss"] = loss.masked_fill(~label_mask, 0).sum() / label_mask.sum()

        return output_dict

    #@overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            "mrr": self._mrr.get_metric(reset)  
        }
        return metrics