from typing import Dict, List
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field, ListField, ArrayField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer, WhitespaceTokenizer, PretrainedTransformerTokenizer

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

@DatasetReader.register("ranker")
class DatasetReaderRanker(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_tokens: int = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.tokenizer = tokenizer 
        self.token_indexers = token_indexers 
        self.max_tokens = max_tokens
        

    @overrides
    def _read(self, file_path: str):
        feature_names = ['lyric', 'title']
        df = pd.read_csv(cached_path(file_path),header=None, skiprows=1, names = feature_names)            #using pandas to open file 
        df = df.dropna(thresh=df.shape[1])


        title_list = df['title'].tolist() #getting a list of all the titles ['title1', 'title2'...]
        df.drop(columns=['title'], axis=1, inplace=True)      #Dropping columsn, that wont be used any more

        #Creates the one hot vector, which indicates the correct title for the lyric 
        one_hot_vector = []
        for i in range(df.shape[0]): 
          zero_list = [0] * df.shape[0]
          zero_list[i] = 1
          one_hot_vector.append(zero_list)
       
        #Text_to_instance is called on every row in the dataset. The list of one hot vectors is iterated through, such that the correct vector is passed on
        count = 0
        for row in df.index:
            yield self.text_to_instance( title_list, df['lyric'][row], one_hot_vector[count])
            count += 1   
            
    def _make_textfield(self, text: str):
        tokens = self.tokenizer.tokenize(text)
        if self.max_tokens:
            tokens = tokens[:self.max_tokens]
        return TextField(tokens, token_indexers=self.token_indexers)


    #this method is for being able to compare between bert and word2vec, which otherwise use different tokenizers
    def _make_textfield_eval(self, text: str):
        tokens = self.tokenizer.tokenize(text)
        if self.max_tokens:
            tokens = tokens[:self.max_tokens]
        return TextField(tokens, token_indexers= {'tokens': SingleIdTokenIndexer()})

    @overrides
    def text_to_instance(
        self,
        title_options: List[str],
        lyric: str, 
        labels: List[int] = None #Because of how the predictor works, the label has to be able to be none
    ) -> Instance: 
        
        lyric_field = self._make_textfield(lyric)
        options_field = ListField([self._make_textfield(o) for o in title_options])
        title_list = ListField([self._make_textfield_eval(o) for o in title_options])  

        meta_lyric = MetadataField(lyric)
        meta_title = MetadataField(title_options)

        #change to meta field 
        fields = { 'lyric': lyric_field, 'title_options': options_field, 'title_list': title_list, 'meta_lyric': meta_lyric, 'meta_title': meta_title   }

        if labels:           
            fields['labels'] = ArrayField(np.array(labels), padding_value=-1)
        
        return Instance(fields)
