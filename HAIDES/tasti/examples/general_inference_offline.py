'''
Given a set of embeddings in the form of a numpy array, and a correspnding set of labels, cosntruct indexes and use the to perform proxy inference

'''
import os
import cv2
cv2.setNumThreads(0)
import swag
import json
import tasti
import torch
import pandas as pd
import numpy as np
import torchvision
from scipy.spatial import distance
import torchvision.transforms as transforms
from collections import defaultdict
from tqdm.autonotebook import tqdm
from blazeit.aggregation.samplers import ControlCovariateSampler

# Feel free to change this!
ROOT_DATA_DIR = '/work/823656/data/datasets/amazon_reviews_100000_40/'

'''
VideoDataset allows you to access frames of a given video.
'''
class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, video_fp, list_of_idxs=[], transform_fn=lambda x: x):
        self.video_fp = video_fp
        self.list_of_idxs = []
        self.transform_fn = transform_fn
        self.cap = swag.VideoCapture(self.video_fp)
        self.video_metadata = json.load(open(self.video_fp + '.json', 'r'))
        self.cum_frames = np.array(self.video_metadata['cum_frames'])
        self.cum_frames = np.insert(self.cum_frames, 0, 0)
        self.length = self.cum_frames[-1]
        self.current_idx = 0
        self.init()
        
    def init(self):
        if len(self.list_of_idxs) == 0:
            self.frames = None
        else:
            self.frames = []
            for idx in tqdm(self.list_of_idxs, desc="Video"):
                self.seek(idx)
                frame = self.read()
                self.frames.append(frame)
            
    def transform(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = self.transform_fn(frame)
        return frame

    def seek(self, idx):
        if self.current_idx != idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx - 1)
            self.current_idx = idx
        
    def read(self):
        _, frame = self.cap.read()
        frame = self.transform(frame)
        self.current_idx += 1
        return frame
    
    def __len__(self):
        return self.length if len(self.list_of_idxs) == 0 else len(self.list_of_idxs)
    
    def __getitem__(self, idx):
        if len(self.list_of_idxs) == 0:
            self.seek(idx)
            frame = self.read()
        else:
            frame = self.frames[idx]
        return frame   

'''
LabelDataset loads the target dnn .csv files and allows you to access the target dnn outputs of given frames.
'''
class LabelDataset(torch.utils.data.Dataset):
    def __init__(self, labels_fp, length):
        df = pd.read_csv(labels_fp)
        df = df[df['object_name'].isin(['car'])]
        frame_to_rows = defaultdict(list)
        for row in df.itertuples():
            frame_to_rows[row.frame].append(row)
        labels = []
        for frame_idx in range(length):
            labels.append(frame_to_rows[frame_idx])
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.labels[idx]

class GeneralInferenceIndex(tasti.Index):
    def get_target_dnn(self):
        '''
        In this case, because we are running the target dnn offline, so we just return the identity.
        '''
        model = torch.nn.Identity()
        return model
        
    def get_embedding_dnn(self):
        '''
        Results are already precomputed, no need for an embeddings DNN
        '''
        model = torch.nn.Identity()
        return model
    
    def get_pretrained_embedding_dnn(self):
        '''
        Results are already precomputed, no need to train an embeddings DNN
        '''
        model = torch.nn.Identity()
        return model
    
    def get_target_dnn_dataset(self, train_or_test):
        '''
        no real use for this, just return the length of the dataset for a sanity check
        '''
        embeddings = np.load(ROOT_DATA_DIR + 'X_train.npy')
        return embeddings
    
    def get_embedding_dnn_dataset(self, train_or_test):
        '''
        no real use for this, just return the length of the dataset for a sanity check
        '''
        embeddings = np.load(ROOT_DATA_DIR + 'X_train.npy')
        print("embeddings length", len(embeddings))
        return embeddings
    
    def override_target_dnn_cache(self, target_dnn_cache, train_or_test):
        
        labels = np.load(ROOT_DATA_DIR + 'y_train.npy')
        print("target DNN cache length", len(labels))

        return labels

class PredicateQuery(tasti.BaseQuery):
    def score(self, target_dnn_output):
        return target_dnn_output
    
class GeneralInferenceConfig(tasti.IndexConfig):
    def __init__(self):
        super().__init__()
        self.do_mining = False
        self.do_training = False
        self.do_infer = False
        self.do_bucketting = True
        
        self.batch_size = 16
        self.nb_train = 3000
        self.train_margin = 1.0
        self.train_lr = 1e-4
        self.max_k = 1
        self.nb_training_its = 12000

        self.nb_buckets = 10000
        self.index_type = 'TASTI'
        self.max_oracle_calls = 7000
    
if __name__ == '__main__':

    for i in [16, 32, 48, 64, 128]:
        config = GeneralInferenceConfig()
        config.nb_buckets = 10000
        index = GeneralInferenceIndex(config)
        index.init()

        query = PredicateQuery(index)
        #index.adapt_to_predicate(query.score, config.max_oracle_calls, verbose=0)
        query.execute()
