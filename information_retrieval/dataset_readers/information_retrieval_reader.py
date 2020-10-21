from typing import Dict, List
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field, ListField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer, WhitespaceTokenizer, PretrainedTransformerTokenizer
from allennlp.common.checks import ConfigurationError

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
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.max_tokens = max_tokens
        

    @overrides
    def _read(self, file_path: str):
        logger.info("Reading instances from lines in file at: %s", file_path)
        feature_names = ['artist', 'title','lyric', 'percentage']
        df = pd.read_csv(cached_path(file_path),header=None, skiprows=1, names = feature_names)            #using pandas to open file 
        df = df.dropna(thresh=df.shape[1])


        title_list = df['title'].tolist() #getting a list of all the titles ['title1', 'title2'...]
        df.drop(columns=['artist', 'title', 'percentage'], axis=1, inplace=True)      #columns, which won't be used 

        res = []
        for i in range(df.shape[0]): 
          zero_list = [0] * df.shape[0]
          zero_list[i] = 1
          res.append(zero_list)
       
        i = 0
        for row in df.to_dict(orient='records'):       #‘records’ : list like [{column -> value}, … , {column -> value}]
            yield self.text_to_instance( res[i], title_list,**row)
            i = i + 1
            
    def _make_textfield(self, text: str):
        tokens = self.tokenizer.tokenize(text)
        if self.max_tokens:
            tokens = tokens[:self.max_tokens]
        return TextField(tokens, token_indexers=self.token_indexers)

    @overrides
    def text_to_instance(
        self,
        labels: List[int],
        title_options: List[str],
        lyric: str, 
    ) -> Instance: 
        
        if labels:
            assert all(l >= 0 for l in labels)
            assert all((l == 0) for l in labels[len(title_options):])
            labels = labels[:len(title_options)]
        
        
        lyric_field = self._make_textfield(lyric)
        options_field = ListField([self._make_textfield(o) for o in title_options])
      
        fields = { 'lyric': lyric_field, 'title_options': options_field }

        if labels:
            labels = list(map(float, filter(lambda x: not pd.isnull(x), labels)))            
            fields['labels'] = ArrayField(np.array(labels), padding_value=-1)
        
        return Instance(fields)
