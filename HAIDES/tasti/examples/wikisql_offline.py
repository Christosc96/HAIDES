'''
This code allows you to reproduce the results in the paper corresponding to the "WikiSQL" dataset.
The term 'offline' refers to the fact that all the target dnn outputs have already been computed.
Look at the README.md file for information about how to get the data to run this code.
'''
import os
import tasti
import jsonlines
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import defaultdict
from torchnlp.word_to_vector import FastText

# Feel free to change this!
ROOT_DATA_DIR = '/work/823656/data/datasets/wikiSQL/WikiSQL/data'

class WikiSQLDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl):
        self.mode = 'input'
        data = []
        with jsonlines.open(jsonl) as reader:
            for obj in reader:
                sql = obj['sql']
                label = (sql['agg'], len(sql['conds']))
                text = obj['question'].strip().lower()
                text = text.replace('?', '')
                data.append((text, sql['agg'], len(sql['conds'])))
        self.df = pd.DataFrame(data, columns=['text', 'agg', 'conds'])
        #self.df.to_csv('./wiqisql.csv')
        self.vectors = FastText(cache='./cache/vectors.pth')

    def __len__(self):
        return len(self.df)

    def embed(self, text):
        words = text.split(' ') # FIXME
        emb = self.vectors[words[0]]
        for word in words[1:]:
            emb += self.vectors[word]
        emb /= len(words)
        return emb
    
    def __getitem__(self, idx):
        text = self.df.loc[idx, 'text']
        agg = self.df.loc[idx, 'agg']
        conds = self.df.loc[idx, 'conds']
        if self.mode == 'input':
            return self.embed(text)
        else:
            return agg, conds
        
class Embedder(nn.Module):
    def __init__(self, nb_out=128):
        super().__init__()
        self.mlp = nn.Sequential(
                nn.Linear(300, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, nb_out)
        )

    def forward(self, x):
        return self.mlp(x)

class WikiSQLOfflineIndex(tasti.Index):
    def get_target_dnn(self):
        model = torch.nn.Identity()
        return model
    
    def get_pretrained_embedding_dnn(self):
        model = torch.nn.Identity()
        return model
        
    def get_embedding_dnn(self):
        model = Embedder()
        return model
    
    def get_target_dnn_dataset(self, train_or_test):
        dataset_fp = os.path.join(ROOT_DATA_DIR, 'train.jsonl')
        sql_dataset = WikiSQLDataset(dataset_fp)
        sql_dataset.mode = 'input'
        return sql_dataset
    
    def get_embedding_dnn_dataset(self, train_or_test):
        dataset_fp = os.path.join(ROOT_DATA_DIR, 'train.jsonl')
        sql_dataset = WikiSQLDataset(dataset_fp)
        sql_dataset.mode = 'input'
        return sql_dataset
    
    def override_target_dnn_cache(self, target_dnn_cache, train_or_test):
        dataset_fp = os.path.join(ROOT_DATA_DIR, 'train.jsonl')
        sql_dataset = WikiSQLDataset(dataset_fp)
        sql_dataset.mode = 'output'
        return sql_dataset
    
    def is_close(self, label1, label2):
        return label1 == label2
    
class WikiSQLAggregateQuery(tasti.AggregateQuery):
    def score(self, target_dnn_output):
        return target_dnn_output[1]
    
class WikiSQLSUPGPrecisionQuery(tasti.SUPGPrecisionQuery):
    def score(self, target_dnn_output):
        return 1.0 if target_dnn_output[0] == 0 else 0.0
    
class WikiSQLSUPGRecallQuery(tasti.SUPGRecallQuery):
    def score(self, target_dnn_output):
        return 1.0 if target_dnn_output[0] == 0 else 0.0

class WikiSQLLimitQuery(tasti.LimitQuery):
    def score(self, target_dnn_output):
        return target_dnn_output[1] if target_dnn_output[0] == 0 else 0

class WikiSQLQueryPredicate1(tasti.BaseQuery):
    def score(self, target_dnn_output):
        return 1.0 if target_dnn_output[0] != 0 else 0.0
    
class WikiSQLOfflineConfig(tasti.IndexConfig):
    def __init__(self):
        super().__init__()
        self.do_mining = False
        self.do_training = False
        self.do_infer = False
        self.do_bucketting = False
        self.nb_train = 3000
        #self.nb_buckets = 1000
        self.batch_size = 1

        self.max_k = 1
        self.nb_buckets = 64
        self.index_type = 'HAIDES'
        self.max_oracle_calls = 20000
        
if __name__ == '__main__':
    config = WikiSQLOfflineConfig()
    index = WikiSQLOfflineIndex(config)
    index.init()
    '''
    query = WikiSQLAggregateQuery(index)
    result = query.execute_metrics(err_tol=0.01, confidence=0.05)
    print(result)
    
    query = WikiSQLLimitQuery(index)
    result = query.execute_metrics(want_to_find=4, nb_to_find=10)
    print(result)
    '''
    print(len(index.embeddings))
    for i in [5000, 10000, 15000, 20000]:
        config = WikiSQLOfflineConfig()
        index = WikiSQLOfflineIndex(config)
        config.do_bucketting = False
        index.init()
        config.max_oracle_calls = i
        query = WikiSQLQueryPredicate1(index)
        index.adapt_to_predicate(query.score, config.max_oracle_calls, verbose=0)
        query.execute()