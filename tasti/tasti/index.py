import torch
import torchvision
import tasti
import numpy as np
from tqdm.autonotebook import tqdm
import pickle
import os; import psutil; 
import sklearn

class Index:
    def __init__(self, config):
        self.config = config
        self.target_dnn_cache = tasti.DNNOutputCache(
            self.get_target_dnn(),
            self.get_target_dnn_dataset(train_or_test='train'),
            self.target_dnn_callback
        )
        self.target_dnn_cache = self.override_target_dnn_cache(self.target_dnn_cache, train_or_test='train')
        self.seed = self.config.seed
        self.rand = np.random.RandomState(seed=self.seed)
        self.index_tree = None
        
    def override_target_dnn_cache(self, target_dnn_cache, train_or_test='train'):
        '''
        This allows you to override tasti.utils.DNNOutputCache if you already have the target dnn
        outputs available somewhere. Returning a list or another 1-D indexable element will work.
        '''
        return target_dnn_cache
        
    def is_close(self, a, b):
        '''
        Define your notion of "closeness" as described in the paper between records "a" and "b".
        Return a Boolean.
        '''
        raise NotImplementedError
        
    def get_target_dnn_dataset(self, train_or_test='train'):
        '''
        Define your target_dnn_dataset under the condition of "train_or_test".
        Return a torch.utils.data.Dataset object.
        '''
        raise NotImplementedError
    
    def get_embedding_dnn_dataset(self, train_or_test='train'):
        '''
        Define your embedding_dnn_dataset under the condition of "train_or_test".
        Return a torch.utils.data.Dataset object.
        '''
        raise NotImplementedError
        
    def get_target_dnn(self):
        '''
        Define your Target DNN.
        Return a torch.nn.Module.
        '''
        raise NotImplementedError
        
    def get_embedding_dnn(self):
        '''
        Define your Embeding DNN.
        Return a torch.nn.Module.
        '''
        raise NotImplementedError
        
    def get_pretrained_embedding_dnn(self):
        '''
        Define your Embeding DNN.
        Return a torch.nn.Module.
        '''
        return self.get_pretrained_embedding_dnn()
        
    def target_dnn_callback(self, target_dnn_output):
        '''
        Often times, you want to process the output of your target dnn into something nicer.
        This function is called everytime a target dnn output is computed and allows you to process it.
        If it is not defined, it will simply return the input.
        '''
        return target_dnn_output

    def do_mining(self):
        '''
        The mining step of constructing a TASTI. We will use an embedding dnn to compute embeddings
        of the entire dataset. Then, we will use FPFRandomBucketter to choose "distinct" datapoints
        that can be useful for triplet training.
        '''
        if self.config.do_mining:
            model = self.get_pretrained_embedding_dnn()
            try:
                model.cuda()
                model.eval()
            except:
                pass
            
            dataset = self.get_embedding_dnn_dataset(train_or_test='train')
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=56,
                pin_memory=True
            )
            
            embeddings = []
            for batch in tqdm(dataloader, desc='Embedding DNN'):
                if torch.cuda.is_available():
                    batch = batch.cuda()
                with torch.no_grad():
                    output = model(batch).cpu()
                embeddings.append(output)  
            embeddings = torch.cat(embeddings, dim=0)
            embeddings = embeddings.numpy()
            
            bucketter = tasti.bucketters.FPFRandomBucketter(self.config.nb_train, self.seed)
            reps, _, _ = bucketter.bucket(embeddings, self.config.max_k)
            self.training_idxs = reps
        else:
            self.training_idxs = self.rand.choice(
                    len(self.get_embedding_dnn_dataset(train_or_test='train')),
                    size=self.config.nb_train,
                    replace=False
            )
            
    def do_training(self):
        '''
        Fine-tuning the embedding dnn via triplet loss. 
        '''
        if self.config.do_training:
            model = self.get_target_dnn()
            model.eval()
            model.cuda()
            
            for idx in tqdm(self.training_idxs, desc='Target DNN'):
                self.target_dnn_cache[idx]
            
            dataset = self.get_embedding_dnn_dataset(train_or_test='train')
            triplet_dataset = tasti.data.TripletDataset(
                dataset=dataset,
                target_dnn_cache=self.target_dnn_cache,
                list_of_idxs=self.training_idxs,
                is_close_fn=self.is_close,
                length=self.config.nb_training_its
            )
            dataloader = torch.utils.data.DataLoader(
                triplet_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=56,
                pin_memory=True
            )
            
            model = self.get_embedding_dnn()
            try:
                model.cuda()
            except:
                pass
            model.train()
            loss_fn = tasti.TripletLoss(self.config.train_margin)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.train_lr)
            
            for anchor, positive, negative in tqdm(dataloader, desc='Training Step'):
                '''
                anchor = anchor.cuda(non_blocking=True)
                positive = positive.cuda(non_blocking=True)
                negative = negative.cuda(non_blocking=True)
                '''
                
                e_a = model(anchor)
                e_p = model(positive)
                e_n = model(negative)
                
                optimizer.zero_grad()
                loss = loss_fn(e_a, e_p, e_n)
                loss.backward()
                optimizer.step()
                
            torch.save(model.state_dict(), './cache/wikisqlmodel.pt')
            self.embedding_dnn = model
        else:
            self.embedding_dnn = self.get_pretrained_embedding_dnn()
            
        del self.target_dnn_cache
        self.target_dnn_cache = tasti.DNNOutputCache(
            self.get_target_dnn(),
            self.get_target_dnn_dataset(train_or_test='test'),
            self.target_dnn_callback
        )
        self.target_dnn_cache = self.override_target_dnn_cache(self.target_dnn_cache, train_or_test='test')
        
            
    def do_infer(self):
        '''
        With our fine-tuned embedding dnn, we now compute embeddings for the entire dataset.
        '''
        if self.config.do_infer:
            model = self.embedding_dnn
            model.eval()
            try:
                model.cuda()
            except:
                pass
            dataset = self.get_embedding_dnn_dataset(train_or_test='test')
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=56,
                pin_memory=True
            )

            embeddings = []
            for batch in tqdm(dataloader, desc='Inference'):
                try:
                    batch = batch.cuda()
                except:
                    pass
                with torch.no_grad():
                    output = model(batch).cpu()
                embeddings.append(output)  
            embeddings = torch.cat(embeddings, dim=0)
            embeddings = embeddings.numpy()

            np.save('./cache/_wikisqlembeddings.npy', embeddings)
            self.embeddings = embeddings
        else:
            #self.embeddings = np.load('./cache/trained_embeddings.npy') #choose different embeddings for different datasets
            #self.embeddings = np.load('/work/823656/data/datasets/celebA/celeba_embeddings.npy')
            self.embeddings = np.load('./cache/_wikisqlembeddings.npy')
    def do_bucketting(self):
        '''
        Given our embeddings, cluster them and store the reps, topk_reps, and topk_dists to finalize our TASTI.
        '''
        print(len(self.embeddings))
        if self.config.do_bucketting:
            if(self.config.index_type == 'TASTI'):
                #self.embeddings = sklearn.preprocessing.normalize(self.embeddings, copy=False)
                bucketter = tasti.bucketters.FPFRandomBucketter(self.config.nb_buckets, self.seed)
                self.reps, self.topk_reps, self.topk_dists = bucketter.bucket(self.embeddings, self.config.max_k)

                np.save('./cache/reps.npy', self.reps)
                np.save('./cache/topk_reps.npy', self.topk_reps)
                np.save('./cache/topk_dists.npy', self.topk_dists)
            else:
                #self.embeddings = sklearn.preprocessing.normalize(self.embeddings, copy=False)
                pbar = tqdm(desc='Hierarchical K-means index construction', total =  self.config.nb_buckets) 
                bucketter = tasti.hierarchical_bucketters.HaidesBucketter(self.embeddings, self.config.nb_buckets, self.seed)
                bucketter.bucket(bucketter.original_embeddings, bucketter.original_embedding_indices, depth = 0, node = bucketter.tree_root, bar = pbar) 
                self.index_tree = bucketter
                #self.reps = bucketter.saveIndex(path='./cache/random/', dataset='wikisql', pkl_tree = 1)
                self.reps = bucketter.saveIndex(path='./cache/', dataset='nighstreet', pkl_tree = 1)

        else:
            if(self.config.index_type == 'TASTI'):
                self.reps = np.load('./cache/reps.npy')
                self.topk_reps = np.load('./cache/topk_reps.npy')
                self.topk_dists = np.load('./cache/topk_dists.npy')
            else:
                #load haides index
                path='./cache/'
                dataset='wikiSQL' #change this to the dataset we want to load               sql for random SQL fro kmeans
                #dataset='celeba' #change this to the dataset we want to load
                #dataset = 'nightstreet'
        
                load_path = path + dataset + '_' + str(len(self.embeddings)) + '_' + str(self.config.nb_buckets)  #remove _ for wikisql
                #print(load_path)
                #load_path = '/work/823656/tasti/tasti/tasti/examples/cache/nighstreet_973136_256_haides_reps.npy'
                self.reps = np.load(load_path + 'haides_reps.npy')
                load_path = load_path + 'hindex_tree.pkl'
                self.index_tree = pickle.load( open(load_path, "rb"))
                print("Loaded haides with branch factor: ", self.config.nb_buckets)
                
    def adapt_to_predicate(self, score_fn, max_reps, verbose = 1):
        if(verbose):
            self.index_tree.count_Nodes(node = self.index_tree.tree_root)
            self.index_tree.count_Leaf_Nodes(node = self.index_tree.tree_root)

        self.index_tree.initialize_tree_of_coverings(self.index_tree.tree_root, score_fn, self.target_dnn_cache)

        self.topk_reps, self.topk_dists = self.index_tree.adaptive_index_descent(max_reps, self.index_tree.original_embeddings, score_fn, self.target_dnn_cache)

        return
        
    def slice_hierarchical_index(self, budget):
        self.topk_reps, self.topk_dists = self.index_tree.partition_at_depth(budget)


    def crack(self):
        cache = [self.target_dnn_cache[rep] for rep in self.reps]
        cached_idxs = []
        for idx in range(len(cache)):
            if cache[idx] != None:
                cached_idxs.append(idx)        
        cached_idxs = np.array(cached_idxs)
        print(cached_idxs.shape)
        bucketter = tasti.bucketters.CrackingBucketter(self.config.nb_buckets)
        self.reps, self.topk_reps, self.topk_dists = bucketter.bucket(self.embeddings, self.config.max_k, cached_idxs)
        np.save('./cache/reps.npy', self.reps)
        np.save('./cache/topk_reps.npy', self.topk_reps)
        np.save('./cache/topk_dists.npy', self.topk_dists)

    def extend_for_query(self):
        bucketter = tasti.bucketters.FPFRandomExtendingBucketter(self.config.nb_buckets, self.seed)
        self.reps, self.topk_reps, self.topk_dists = bucketter.bucket(self.embeddings, self.config.max_k)



    def init(self):
        self.do_mining()
        self.do_training()
        self.do_infer()
        self.do_bucketting()
        
        for rep in tqdm(self.reps, desc='Target DNN Invocations'):
            self.target_dnn_cache[rep]
