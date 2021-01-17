from typing import Dict, Optional, List


from overrides import overrides
import torch


from registrable import Registrable
from allennlp.data import TextFieldTensors, Vocabulary, TokenIndexer
from allennlp.models.model import Model
from allennlp.data.fields import MetadataField
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder, TimeDistributed
from allennlp.nn import InitializerApplicator, util
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import BLEU
from allennlp.training.util import get_batch_size


from information_retrieval.similarity_modules.bert_cls import RelevanceMatcher
from information_retrieval.similarity_modules.boe_encoder import RelevanceMatcher
from information_retrieval.evaluation.mrr import MRR



@Model.register("ranker")
class DocumentRanker(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        relevance_matcher: RelevanceMatcher,
        num_labels: int = None,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:

        super().__init__(vocab, **kwargs)
        self._text_field_embedder = text_field_embedder
        self._relevance_matcher = TimeDistributed(relevance_matcher) 
        self._mrr = MRR(padding_value=-1)
        self._bleu = BLEU(ngram_weights=(0.5, 0.5), exclude_indices={0})
        self._loss = torch.nn.CrossEntropyLoss(reduction = 'mean') 
        
    def forward(  
        self, 
        lyric: TextFieldTensors,               # batch * words
        title_options: TextFieldTensors,       # batch * num_options * words
        title_list: TextFieldTensors,
        meta_lyric,
        meta_title,
        labels: torch.IntTensor = None         # batch * num_options
    ) -> Dict[str, torch.Tensor]:
      
        
        embedded_text = self._text_field_embedder(lyric, num_wrapping_dims=0)
        mask = get_text_field_mask(lyric).long()

        embedded_options = self._text_field_embedder(title_options, num_wrapping_dims=1) 
        options_mask = get_text_field_mask(title_options, num_wrapping_dims=1).long() 

        #make pairs of (lyric, title) instead of (lyric, (title1, title2,...))
        embedded_text = embedded_text.unsqueeze(1).expand(-1, embedded_options.size(1), -1, -1)
        mask = mask.unsqueeze(1).expand(-1, embedded_options.size(1), -1)
        scores = self._relevance_matcher(embedded_text, embedded_options, mask, options_mask).squeeze(-1)

        output_dict = {"logits": scores}
       

        if labels is not None:

            #get index for title with max score
            max_score_index = torch.argmax(scores,1, keepdim=True)          # (batch, 1)
            max_score_index = max_score_index.squeeze()                     # (batch)
 
            #get index for correct title
            label_index = torch.argmax(labels, 1, keepdim=True)             # (batch, 1)
            label_index = label_index.squeeze()                             # (batch)

            batch_size = get_batch_size(scores)  
            count = torch.arange(batch_size)
            title_options_id = util.get_token_ids_from_text_field_tensors(title_list) # (batch, title_options, words)


            candidate_title = title_options_id[count, max_score_index]                   # (batch, words)
            reference_title = title_options_id[count, label_index]                       # (batch, words)


            label_mask = (labels != -1)   #the padding value of labels are set to -1
            
            self._mrr(scores, labels, label_mask)      
            self._bleu(candidate_title, reference_title)

            loss = self._loss(scores, torch.max(labels, 1)[1]) #inputs the raw scores and index for correct class
            loss = loss.masked_fill(~label_mask, 0).sum()       # zero out masked parts of the loss
            non_zero_elements = label_mask.sum()                #gets number of non-zero elements 
            output_dict["loss"] = loss / non_zero_elements      #gets the mean without masked elements

        return output_dict

  
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            "mrr": self._mrr.get_metric(reset),
            "bleu": self._bleu.get_metric(reset).get('BLEU')  #Extracting the value from dict
        }
        return metrics

    