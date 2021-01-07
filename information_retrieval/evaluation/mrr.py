from typing import Optional, List

import numpy as np
import torch

from allennlp.training.metrics.metric import Metric
from information_retrieval.evaluation.Ranking_metric import RankingMetric


@Metric.register("mrr")
class MRR(RankingMetric):
    def get_metric(self, reset: bool = False):
        predictions = torch.cat(self._all_predictions, dim=0)
        labels = torch.cat(self._all_gold_labels, dim=0)
        masks = torch.cat(self._all_masks, dim=0)

        score = mrr(predictions, labels, masks).item()  #item copies element from array an return as scalar

        if reset:
            self.reset()
        return score

def first_nonzero(t):
    t = t.masked_fill(t != 0, 1)
    idx = torch.arange(t.size(-1), 0, -1).type_as(t)
    indices = torch.argmax(t * idx, 1, keepdim=True)
    return indices

def mrr(y_pred, y_true, mask):
    y_pred = y_pred.masked_fill(~mask, -1)                              #Replaces all mask values with -1
    #y_true = y_true.ge(y_true.max(dim=-1, keepdim=True).values).float() # This binarizes the label, which isn't needed, since it is a one-hot label

    _, rank = y_pred.sort(descending=True, dim=-1)                      #Sort the predicted scores in descending order, gets predicted ranking
    ordered_truth = y_true.gather(1, rank)                              #gather indexes from predicted scores from true scores
    
    gold = torch.arange(y_true.size(-1)).unsqueeze(0).type_as(y_true)   #Ordered indices
    _mrr = (ordered_truth / (gold + 1)) * mask                          #Devides the truth by indices, and zeros out all the masks
    
    return _mrr.gather(1, first_nonzero(ordered_truth)).mean()         #gets the mean of the batch for the first non-zero element 