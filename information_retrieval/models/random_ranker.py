from typing import Dict, Optional, List
from overrides import overrides
import torch
from torch.autograd import Variable

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

import numpy as np



@Model.register("random_ranker")
class DocumentRankerTFIDF(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        num_labels: int = None,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)
        self._mrr = MRR(padding_value=-1)
        self._bleu = BLEU(ngram_weights=(0.5, 0.5), exclude_indices={0})
        self._loss = torch.nn.CrossEntropyLoss(reduction = 'mean') 
        
    def forward(  
        self, 
        lyric: TextFieldTensors,               # batch * words
        title_options: TextFieldTensors,       # batch * num_options * words
        title_list: TextFieldTensors,
        meta_lyric: str,
        meta_title: List[str],
        labels: torch.IntTensor = None         # batch * num_options
    ) -> Dict[str, torch.Tensor]:
      
    
        scores = np.random.rand(len(meta_title[0])) # retuns an array of random numbers between 0 and 1
       
        scores = torch.from_numpy(scores)           # convertS to tensor 
        scores = scores.unsqueeze(0)                # simulate dimension of batch= 1 (1, number of titles)

        output_dict = {"logits": scores}  

        if labels is not None:
            
            #get index for title with max score
            max_score_index = torch.argmax(scores)          
 
            #get index for correct title
            label_index = torch.argmax(labels, 1, keepdim=True)             # (batch, 1)
            label_index = label_index.squeeze()                             # (batch)

            batch_size = get_batch_size(labels)  
            count = torch.arange(batch_size)
            title_options_id = util.get_token_ids_from_text_field_tensors(title_list) # (batch, title_options, words)

            candidate_title = title_options_id[count, max_score_index]                   # (batch, words)
            reference_title = title_options_id[count, label_index]                       # (batch, words)


            label_mask = (labels != -1)   #the padding value of labels are set to -1
            
            self._mrr(scores, labels, label_mask)      
            self._bleu(candidate_title, reference_title)
            loss = self._loss(scores, torch.max(labels, 1)[1])

            output_dict["loss"] = loss 
            
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

    