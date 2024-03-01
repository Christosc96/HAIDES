import numpy as np
import tqdm
from numba import njit, prange
from tasti.bucketters import Bucketter
import sklearn
import sklearn.cluster
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances_argmin_min
import pickle
import math
from collections import Counter
import itertools
from queue import PriorityQueue

from sklearn.metrics import pairwise_distances_argmin_min
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from random import shuffle


@njit(parallel=True)
def get_dists(x, embeddings):
    dists = np.zeros(len(embeddings), dtype=np.float32)
    for i in prange(len(embeddings)):
        dists[i] = np.sqrt(np.sum((x - embeddings[i]) ** 2))
    return dists

@njit(parallel=True)
def get_and_update_dists(x, embeddings, min_dists):
    dists = np.zeros(len(embeddings), dtype=np.float32)
    for i in prange(len(embeddings)):
        dists[i] = np.sqrt(np.sum((x - embeddings[i]) ** 2))
        if dists[i] < min_dists[i]:
            min_dists[i] = dists[i]
    return dists

def recursive_bar_update(bar, depth, depth_to_check = 2):
    if(depth==depth_to_check):
        bar.update(1)
    return

def entropy(p):
    reward = None
    if(p == 0.0 or p == 1.0):
        reward = 0.0
    else:
        reward = -(p*math.log2(p) + (1-p)*math.log2((1-p)))
    return reward

class Node(object):
    def __init__(self):
        """"
        This is a herarchical index node
        """

        # children and their assignments
        self.children = {}
        self.asssignments = {}

        # values required by HOO and the covering generator algorithm
        self.activated = False
        self.counter = 0
        self.mean = 0
        self.U_value = None
        self.representatives = np.array([])
        self.depth = -1
        self.embeddings = None
        self.len_embeddings = -1
        self.sampled_reps = set()

        self.rep_reward = -1
        self.rep = -1
        self.data_point_rewards = 0

        self.B_value = np.inf
        self.mean = -1
        self.counter = 0
        self.radius = 0

        self.embedding_indexes = np.array([])

class HaidesBucketter(Bucketter):
    def __init__(
            self,
            embeddings,
            nb_buckets: int,
            seed: int=123456,
            max_depth: int=10,
            algorithm = 'KMEANS',
            subset = False
    ):
        super().__init__(nb_buckets)

        self.init = algorithm

        self.nb_buckets = nb_buckets
        self.max_depth = max_depth
        self.tree_root = Node() #root node


        self.original_embeddings = embeddings                                                                  #array of embedding vectors
        self.original_embedding_indices = np.array(range(len(self.original_embeddings)))                       #hold embedding indices

        if(subset):
            sample_rows = np.random.choice(embeddings.shape[0], size=50000, replace=False)
            self.original_embeddings = embeddings[sample_rows, :]                                                                #array of embedding vectors
            self.original_embedding_indices = np.array(range(len(self.original_embeddings)))                       #hold embedding indices

        self.global_reps = np.array([]) 
        self.topdepthreps = np.array([])                                    #for each embedding vector, hold its representative at each depth
        self.topdepthdists = np.array([])

        self.depth_indexes = np.array([])

    def topk(self, k, dists, nb_buckets):
        topks = []
        for i in range(len(dists)):
            if(nb_buckets == 1):
                topks.append(0)
            else:
                topks.append(np.argpartition(dists[i], k)[0:k])
        return np.array(topks)

    def bucket(
            self,
            embeddings: np.ndarray,
            embedding_indexes:np.ndarray,
            depth: int,
            node: Node ,
            max_k = 1,
            percent_fpf=0.75,
            bar = None
    ):

        node.depth = depth
        node.embeddings = embeddings
        node.embedding_indexes = embedding_indexes
        node.len_embeddings = len(embeddings)
        nb_buckets = self.nb_buckets    #/(2**depth)
        
        depth+=1                        #starting depth is 1 not 0
        '''
        if(nb_buckets <= 32):
            nb_buckets = 32
        '''

        if(depth==1):
            self.topdepthreps = np.full((len(embeddings), (self.max_depth+1)), -1)
            self.topdepthdists = np.full((len(embeddings), (self.max_depth+1)), -1.00)

        if((len(node.embeddings) <= nb_buckets) or depth >=4):
             #nodes with less than nb_buckets data will not get clustered (or at depth larger than 3 )
            return
            
        if((len(np.unique(node.embeddings)) <= nb_buckets)):
             #nodes with less than nb_buckets data will not get clustered (or at depth larger than 3 )
            return
        
        if(self.init == 'KMEANS'):
            km = sklearn.cluster.KMeans(n_clusters = int(nb_buckets), verbose=1, max_iter = 1, n_jobs=-1, n_init = 1, init = 'k-means++')
            #km = sklearn.cluster.MiniBatchKMeans(n_clusters = nb_buckets, verbose=0, init_size = 3*nb_buckets, batch_size=1024)
            km.fit(node.embeddings)
            '''
            find representative data points
            '''

            closest_embedding_to_centroid_with_idx, distances = pairwise_distances_argmin_min(km.cluster_centers_, node.embeddings)


            representative_points = node.embeddings[closest_embedding_to_centroid_with_idx]
            #representative_points = plus_plus(node.embeddings, nb_buckets)

            reps = closest_embedding_to_centroid_with_idx

            '''
            assign the data points to their representatives and get the distance
            '''

            topk_reps, topk_dists = pairwise_distances_argmin_min(node.embeddings, representative_points)
            topk_reps = topk_reps.reshape(-1,1)

            for i in range(len(topk_reps)):
                topk_reps[i] = embedding_indexes[reps[topk_reps[i]]]
        else:

            print("USING FPF ALGORITHM VERSION FOR HAIDES")
            print(len(node.embeddings))
            reps = np.full(nb_buckets, -1)                                                  #number of cluster centroids for current node
            reps[0] = np.random.randint(len(embeddings))                                    #local representative index, NOT global

            min_dists = np.full(len(embeddings), np.Inf, dtype=np.float32)
            dists = np.zeros((nb_buckets, len(embeddings)))
            dists[0, :] = get_and_update_dists(
                embeddings[reps[0]],
                embeddings,
                min_dists
            )
            
            for i in range(nb_buckets):
                reps[i] = np.argmax(min_dists)
                dists[i, :] = get_and_update_dists(
                    embeddings[reps[i]],
                    embeddings,
                    min_dists
                )

            dists = dists.transpose()                                              
            topk_reps = self.topk(max_k, dists, nb_buckets)
            topk_dists = np.zeros([len(topk_reps), max_k])
        
            for i in range(len(topk_dists)):
                topk_dists[i] = dists[i, topk_reps[i]]
            for i in range(len(topk_reps)):
                topk_reps[i] = embedding_indexes[reps[topk_reps[i]]]                   #get the global index for the closest representative of each data point

        recursive_bar_update(bar, depth)                                                 #update progress bar

        assignments = {}
        assignment_indexes = {}
        node.children = {}

        node.representatives = reps                                                         #local indexes of representative vectors in node
        self.global_reps = np.hstack([self.global_reps, embedding_indexes[reps]])

        for rep in node.representatives:
            global_index = embedding_indexes[rep]
            repembeddings = []
            repembedding_indexes = []
            for i in prange(len(embeddings)):
                if(topk_reps[i] == global_index):                                                
                    repembeddings.append(embeddings[i])                                     #append the embedding vector along with its global index
                    repembedding_indexes.append(embedding_indexes[i])


            assignments[rep] = np.array(repembeddings)
            assignment_indexes[rep] = np.array(repembedding_indexes)
            node.children[rep] = Node()

            self.bucket(assignments[rep], assignment_indexes[rep], max_k=1, depth=depth, node = node.children[rep], bar=bar)
        d = depth-1                                      
        for i in range(len(embedding_indexes)):
            global_index = embedding_indexes[i]
            self.topdepthreps[global_index][d] = topk_reps[i]
            self.topdepthdists[global_index][d] = topk_dists[i]
        
        node.assignments = assignments 
        node.topk_reps = topk_reps
        node.topk_dists = topk_dists
        return                                    
  
    def initialize_tree_of_coverings(self, node: Node, score_fn, dnn_cache):

        node.radius = 0  #set the radis of the root to 0
        self.active_nodes = []
        self.nodes_updated = 0
        self.queue_counter = itertools.count()
        self.col_indexes = []
        count = next(self.queue_counter)
        node.B_value = np.inf
        self.nodes_expanded_at_depth= {'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0,'9':0,'10':0,}
        self.predicate_satisfied_at_depth= {'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0,'9':0,'10':0,}
        self.data_with_rep_at_depth= {'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0,'9':0,'10':0,}
        queue = []
        queue.append(node)
        while(queue):
            node_to_visit = queue.pop(0)
            setattr(node_to_visit, "variance", 0)
            setattr(node_to_visit, "maj_mean", 0)
            setattr(node_to_visit, "min_mean", 0)
            setattr(node_to_visit, "bsize", len(node_to_visit.representatives))
            setattr(node_to_visit, "is_adm", False)
            setattr(node_to_visit, "sampled_reps", set())
            #node_to_visit.variance = 0
            if(node_to_visit.children):
                for rep in node_to_visit.representatives:
                    queue.append(node_to_visit.children[rep])
                    global_index = node_to_visit.embedding_indexes[rep]
                    node_to_visit.children[rep].rep_reward = float(score_fn(dnn_cache[global_index]))  #cache the dnn scores for speed                   
                    node_to_visit.children[rep].rep = global_index                                     #set the representative's index in its node  
   
    def get_and_update_BATCHED_representative(self, node: Node, t, depth_indexes, score_fn, dnn_cache, max_n):                    #select the next representative and update the depth indexes
        #new algorithm update 23-10-23
        queue = []
        traversed_path = []
        queue.append(node)
        root = node
        batch_size = self.nb_buckets
        node_to_visit = None
        depth = -1              #-1 means we start at the root
        #print("root:", root.counter)
        node_to_visit = None
        oracle_calls = 0

        
        while(queue):                            #1. loop that selects the next node to expand
            node_to_visit = queue.pop(0)         #pop the next node to visit
            traversed_path.append(node_to_visit) #add the node to the traversed path
            depth+=1

            if(node_to_visit.counter == 0): #or len(node_to_visit.sampled_reps) != batch_size):  
                #expand a node that hasnt been expanded before pr hasn't been fully expanded if not on batch-mode
                self.nodes_expanded_at_depth[str(depth)]+=1
                break
            else:                       
                #we descent the hierarchical index and select the next child to visit
                if(node_to_visit.children):
                    children_group = []
                    children_B_values = []
                    for rep in node_to_visit.representatives:
                            try:
                                children_group.append((node_to_visit.children[rep], rep))
                                children_B_values.append(node_to_visit.children[rep].B_value)
                            except:
                                print("Something went wrong when selecting a child node")
                                input()
                    max_B_value = max(children_B_values)
                    max_indexes = [i for i, j in enumerate(children_B_values) if j == max_B_value]

                    #rand_indexes = [i for i, j in enumerate(children_B_values)]
                    #print(children_B_values)
                    #print(len(max_indexes))
                    index_to_select = np.random.choice(max_indexes)
                    #USE THIS FOR RANDOM CHOICE EXPERIMENT
                    #print(rand_indexes)
                    #index_to_select = np.random.choice(rand_indexes)
                    best_child = children_group[index_to_select]


                    child_to_visit = best_child[0]
                    child_to_visit_rep =best_child[1]
                    queue.append(child_to_visit)
                
                    if(not(child_to_visit.children)): #delete leaf nodes that are set to be visited
                        rep_index_to_delete = np.argwhere(node_to_visit.representatives==child_to_visit_rep)
                        node_to_visit.representatives = np.delete(node_to_visit.representatives, rep_index_to_delete)

                        del node_to_visit.children[child_to_visit_rep]
                else:
                    #this should only happen with nodes that have had their children pruned
                    break

        self.active_nodes.append(node_to_visit)  #add the node to the list of expanded (active) nodes
        if(depth == -1):
            print("depth should be a non-negative number!")
            return
        
        children_rewards = []
        child_node_to_add_embeddings = None
        is_leaf = 0
        if(node_to_visit.children):
            #2. select all lower level centroids OR select a specified number of representatives
            '''
            if(node_to_visit == root):
                for rep in node_to_visit.representatives:
                    children_rewards.append(node_to_visit.children[rep].rep_reward)
                    #depth_indexes[node_to_visit.children[rep].embedding_indexes.tolist()] = depth
                    node_to_visit.sampled_reps.add(rep)
            else:
            '''
            for rep in node_to_visit.representatives:
                    #if(rep in node_to_visit.sampled_reps):
                        #continue
                try:
                        #3. get their representative rewards (this is the costly step that requires oracle calls)
                    children_rewards.append(node_to_visit.children[rep].rep_reward)                                         #change this accordingly if you need real time GPU inference
                    child_node_to_add_embeddings = node_to_visit.children[rep]
                        #depth_indexes[node_to_visit.children[rep].embedding_indexes.tolist()] = depth
                        #node_to_visit.sampled_reps.add(rep)
                        #break
                except:
                    print("Something went wrong during children reward collection")
                    input()
        else:
            #if it is a leaf node, we get the rewards for its embeddings directly
            is_leaf = 1
            if(node_to_visit.counter == 0):
                #input()
                limit = 0
                for index in node_to_visit.embedding_indexes:
                    children_rewards.append(float(score_fn(dnn_cache[index])))
                    self.col_indexes.append(index)
                    limit+=1
                    #if(limit > self.nb_buckets):
                    #    break             
            else:
                return 0 
        oracle_calls = len(children_rewards)
        #4. assign the node's data points to their lower level centroids

        depth_indexes[node_to_visit.embedding_indexes.tolist()] = depth
        #if(not(is_leaf)):
        #    depth_indexes[child_node_to_add_embeddings.embedding_indexes.tolist()] = depth

        #ENTROPY BASED REWARD

        #5. update the rewards of the traversed path (or active nodes if another version is desired)
        for node in traversed_path:
            for reward in children_rewards:
                node.counter +=1 
                node.mean = float(node.mean * (node.counter - 1) + reward) / (node.counter)
            
            emp_entropy = entropy(node.mean)
            if(len(node.embedding_indexes) <= self.nb_buckets):
                continue
            node.B_value = emp_entropy + 1*math.sqrt((2*math.log(max_n)) / node.counter)
            #node.B_value = np.inf  #random exploration
        #OVERHEAD ANALYSIS EXPERIMENT
        '''
        for node in self.active_nodes:
            node.B_value = entropy(node.mean) + 1*math.sqrt((2*math.log(max_n)) / node.counter)
        '''
        return oracle_calls

    def get_and_update_BATCHED_ONE_representative(self, node: Node, t, depth_indexes, score_fn, dnn_cache, max_n):                    #select the next representative and update the depth indexes
        #new algorithm update 23-10-23
        queue = []
        traversed_path = []
        queue.append(node)
        root = node
        batch_size = self.nb_buckets
        node_to_visit = None
        depth = -1              #-1 means we start at the root
        #print("root:", root.counter)
        node_to_visit = None
        oracle_calls = 0

        
        while(queue):                            #1. loop that selects the next node to expand
            node_to_visit = queue.pop(0)         #pop the next node to visit
            traversed_path.append(node_to_visit) #add the node to the traversed path
            depth+=1

            if(node_to_visit.counter == 0 or len(node_to_visit.sampled_reps) != node_to_visit.bsize):
                #expand a node that hasnt been expanded before pr hasn't been fully expanded if not on batch-mode
                self.nodes_expanded_at_depth[str(depth)]+=1
                break
            else:                       
                #we descent the hierarchical index and select the next child to visit
                if(node_to_visit.children):
                    children_group = []
                    children_B_values = []
                    for rep in node_to_visit.representatives:
                            try:
                                children_group.append((node_to_visit.children[rep], rep))
                                children_B_values.append(node_to_visit.children[rep].B_value)
                            except:
                                print("Something went wrong when selecting a child node")
                    max_B_value = max(children_B_values)
                    max_indexes = [i for i, j in enumerate(children_B_values) if j == max_B_value]

                    index_to_select = np.random.choice(max_indexes)
                    best_child = children_group[index_to_select]


                    child_to_visit = best_child[0]
                    child_to_visit_rep =best_child[1]
                    queue.append(child_to_visit)
                
                    if(not(child_to_visit.children)): #delete leaf nodes that are set to be visited
                        rep_index_to_delete = np.argwhere(node_to_visit.representatives==child_to_visit_rep)
                        node_to_visit.representatives = np.delete(node_to_visit.representatives, rep_index_to_delete)

                        del node_to_visit.children[child_to_visit_rep]
                else:
                    #this should only happen with nodes that have had their children pruned
                    break

        self.active_nodes.append(node_to_visit)  #add the node to the list of expanded (active) nodes
        if(depth == -1):
            print("depth should be a non-negative number!")
            return
        
        children_rewards = []
        child_node_to_add_embeddings = None
        is_leaf = 0
        if(node_to_visit.children):
            #2. select all lower level centroids OR select a specified number of representatives
            if(node_to_visit == root):
                for rep in node_to_visit.representatives:
                    children_rewards.append(node_to_visit.children[rep].rep_reward)
                    depth_indexes[node_to_visit.children[rep].embedding_indexes.tolist()] = depth
                    node_to_visit.sampled_reps.add(rep)
            else:
                for rep in node_to_visit.representatives:
                    if(rep in node_to_visit.sampled_reps):
                        continue
                    try:
                        #3. get their representative rewards (this is the costly step that requires oracle calls)
                        children_rewards.append(node_to_visit.children[rep].rep_reward)                                         #change this accordingly if you need real time GPU inference
                        child_node_to_add_embeddings = node_to_visit.children[rep]
                        depth_indexes[node_to_visit.children[rep].embedding_indexes.tolist()] = depth
                        node_to_visit.sampled_reps.add(rep)
                        break
                    except:
                        print("Something went wrong during children reward collection")
                        input()
        else:
            #if it is a leaf node, we get the rewards for its embeddings directly
            is_leaf = 1
            if(node_to_visit.counter == 0):
                limit = 0
                for index in node_to_visit.embedding_indexes:
                    children_rewards.append(float(score_fn(dnn_cache[index])))
                    self.col_indexes.append(index)
                    node_to_visit.sampled_reps.add(index)
                    limit+=1
                    if(limit > self.nb_buckets):
                        break             
            else:
                return 0 
        oracle_calls = len(children_rewards)
        #4. assign the node's data points to their lower level centroids

        #depth_indexes[node_to_visit.embedding_indexes.tolist()] = depth
        '''
        if(not(is_leaf)):
            depth_indexes[child_node_to_add_embeddings.embedding_indexes.tolist()] = depth
        '''

        #ENTROPY BASED REWARD

        #5. update the rewards of the traversed path (or active nodes if another version is desired)
        for node in traversed_path:
            for reward in children_rewards:
                node.counter +=1 
                node.mean = float(node.mean * (node.counter - 1) + reward) / (node.counter)
            
            emp_entropy = entropy(node.mean)
            if(len(node.embedding_indexes) <= self.nb_buckets):
                continue
            node.B_value = emp_entropy + 1.00*math.sqrt((2*math.log(max_n)) / node.counter)
            #node.B_value = np.inf  #random exploration
        #OVERHEAD ANALYSIS EXPERIMENT
        '''
        for node in self.active_nodes:
            node.B_value = entropy(node.mean) + 1*math.sqrt((2*math.log(max_n)) / node.counter)
        '''
        return oracle_calls
                                  
    def adaptive_index_descent(self, max_reps, embeddings, score, dnn_cache):
        #given a hierarchical index, a score_function indiciating whether a predicate is satisfied and oracle results in the form of dnn_cache, adapt the index to the predicate


        pbar = tqdm.tqdm(desc='Hierarchical Adaptive Index Descent', 
                     total = max_reps)


        embedding_depth_indexes = np.full((len(embeddings)), -2)
        total_reps = 1 #changed ot one for overhead analysis

        while (total_reps <= max_reps):
            result = self.get_and_update_BATCHED_representative(self.tree_root, total_reps, embedding_depth_indexes, score, dnn_cache, max_reps)
            #result = self.get_and_update_BATCHED_ONE_representative(self.tree_root, total_reps, embedding_depth_indexes, score, dnn_cache, max_reps)

            if(result != None):
                total_reps += result
                pbar.update(result)

        print("data points at each depth", Counter(embedding_depth_indexes))
        print("nodes expanded at each depth", self.nodes_expanded_at_depth)
        print("nodes expanded:", len(self.active_nodes))

        print(embedding_depth_indexes, total_reps)
        try:
            print(np.count_nonzero(embedding_depth_indexes == -1))
            print(np.count_nonzero(embedding_depth_indexes == -2))
        except:
            print("depth indexes should be a non-negative number")

        self.depth_indexes = embedding_depth_indexes

        self.depth_indexes = (np.rint(embedding_depth_indexes)).astype(int)

        hoo_result_reps= self.topdepthreps[np.arange(len(self.topdepthreps)), self.depth_indexes]
        hoo_result_dists= self.topdepthdists[np.arange(len(self.topdepthdists)), self.depth_indexes]

        for idx in self.col_indexes:
            hoo_result_reps[idx] = idx
            hoo_result_dists[idx] = 0.0

        print("leaf embeddings:", len(self.col_indexes))    
        print(hoo_result_reps[:5])
        print("mean hoo distance", np.mean(hoo_result_dists))
        
        print("total reps", len(np.unique(hoo_result_reps)))

        topk_reps = hoo_result_reps
        topk_dists = hoo_result_dists

        topk_dists = topk_dists.reshape(-1,1)
        topk_reps = topk_reps.reshape(-1,1)

        return topk_reps, topk_dists

    def saveIndex(self, path='./cache/', dataset='nightstreet', pkl_tree = 1):
        
        self.global_reps = np.rint(np.unique(np.sort(self.global_reps))).astype(int)
        print(self.global_reps[:5])
        save_path = path + dataset + '_' + str(len(self.original_embeddings)) + '_' + str(self.nb_buckets) + '_'

        if(pkl_tree == 1):
            np.save(save_path + 'haides_reps.npy', self.global_reps)
            save_path = save_path + 'hindex_tree.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump(self, f)
        else:
            np.save(save_path + 'haides_reps.npy', self.global_reps)
            np.save(save_path + 'reps_at_depth.npy', self.topdepthreps)
            np.save(save_path + 'dists_at_depth.npy', self.topdepthdists)

        return self.global_reps

    def partition_at_depth(self, budget):
        for row_depth in self.topdepthreps.T:
            reps_at_depth_s = len(np.rint(np.unique(np.sort(row_depth))).astype(int))
            if(reps_at_depth_s <= budget and reps_at_depth_s != 1):
                print(reps_at_depth_s)
                self.depth_indexes = (np.rint(row_depth)).astype(int)
            else:
                break

        hoo_result_reps= self.topdepthreps[np.arange(len(self.topdepthreps)), self.depth_indexes]
        hoo_result_dists= self.topdepthdists[np.arange(len(self.topdepthdists)), self.depth_indexes]

        return hoo_result_reps.reshape(-1,1), hoo_result_dists.reshape(-1,1)

    def count_reps_per_depth(self):
        for row_depth in self.topdepthreps.T:
            reps_at_depth_s = len(np.rint(np.unique(np.sort(row_depth))).astype(int))
            print(reps_at_depth_s)


           

