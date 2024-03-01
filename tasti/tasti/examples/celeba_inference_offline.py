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
ROOT_DATA_DIR = '/work/823656/data/datasets/celebA/'

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
        embeddings = np.load(ROOT_DATA_DIR + 'celeba_embeddings.npy')
        return embeddings
    
    def get_embedding_dnn_dataset(self, train_or_test):
        '''
        no real use for this, just return the length of the dataset for a sanity check
        '''
        embeddings = np.load(ROOT_DATA_DIR + 'celeba_embeddings.npy')
        print("embeddings length", len(embeddings))
        return embeddings
    
    def override_target_dnn_cache(self, target_dnn_cache, train_or_test):
        
        labels = np.load(ROOT_DATA_DIR + 'celeba_attributes.npy')
        print("target DNN cache length", len(labels))
        attribute_names = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 
        'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 
        'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 
        'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 
        'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 
        'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 
        'Wearing_Necktie', 'Young']

        index = attribute_names.index("Male")

        '''
        rare sbset
        '''
        index2 = attribute_names.index('Blond_Hair')

        labels2 = labels[:,index2]
        
        labels = labels[:,index]


        return labels & labels2

class PredicateQuery(tasti.BaseQuery):
    def score(self, target_dnn_output):
        return target_dnn_output
    
class GeneralInferenceConfig(tasti.IndexConfig):
    def __init__(self):
        super().__init__()
        self.do_mining = False
        self.do_training = False
        self.do_infer = False
        self.do_bucketting = False
        
        self.batch_size = 16
        self.nb_train = 3000
        self.train_margin = 1.0
        self.train_lr = 1e-4
        self.max_k = 1
        self.nb_training_its = 12000

        self.nb_buckets = 64
        self.index_type = 'HAIDES'
        self.max_oracle_calls = 20000
    
if __name__ == '__main__':


    for i in [10000, 15000, 20000, 25000]:

        config = GeneralInferenceConfig()
        index = GeneralInferenceIndex(config)
        config.do_bucketting = False
        index.init()
        config.max_oracle_calls = i
        query = PredicateQuery(index)
        index.adapt_to_predicate(query.score, config.max_oracle_calls, verbose=0)
        query.execute()
