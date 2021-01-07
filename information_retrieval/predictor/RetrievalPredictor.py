from typing import Dict, List
import logging
import json
import numpy as np

from overrides import overrides
 
from allennlp.predictors.predictor  import Predictor
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.nn import InitializerApplicator, util


from allennlp.models.model import Model
from allennlp.data.vocabulary import Vocabulary

from information_retrieval.dataset_readers import DatasetReaderRanker, Instance
from information_retrieval.models import DocumentRanker

@Predictor.register('retrieval-predictor')
class InformationRetrievalPredictor(Predictor):

    @overrides
    def predict_json(self, json_dict: JsonDict) -> JsonDict:
        instance = self._json_to_instance(json_dict)           #Turning json into an instance
        prediction = self.predict_instance(instance)          #Getting prediction for instance
        return {"instance": prediction}

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        lyric = json_dict['lyric']
        title_options = json_dict['title_options'] #it might make more sense to just use the title_options from the training data
        return self._dataset_reader.text_to_instance(title_options=title_options, lyric=lyric)
        
    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
       
        #getting the five most likely songs
        highest_n = 5
        top_5_predected_titles = []
        top_5_predected_indices = outputs['logits'].argsort()[::-1][:highest_n]
        for i in range(highest_n):
            top_5_predected_titles.append( instance.fields['title_options'][top_5_predected_indices[i]])

        outputs['lyric'] = [str(token) for token in instance.fields['lyric'].tokens]
        predicted_title = instance.fields['title_options'][outputs['logits'].argmax()]  #gets index with max score and indexes into title_options
        outputs['predicted_title'] = [str(token) for token in predicted_title.tokens]
        outputs['top_5_predected_titles'] = [str(token.tokens) for token in top_5_predected_titles]
        
        return sanitize(outputs)
