# HAIDES
Implementation for HAIDES: A Hierarchical Adaptive Index Descent Framework for Approximating Inference Queries over Unstructured Data

HAIDES is implemented on top of the TASTI repository

## Installation

To install, run 

```
pip install -r requirements.txt
pip install -e .
```

##Running HAIDES

We provide instructions to run HAIDES on the ```nightstreet``` dataset, similar instructions apply to the ```WikiSQL``` and ```CelebA```

To construct a HAIDES index, specify the embeddings file to load via ```self.embeddings```. 

Set the configuration as

```
self.do_bucketting = True

self.nb_buckets = 128
self.index_type = 'HAIDES'
self.max_oracle_calls = 20000
```
This will create a HAIDES index with K=128 and use 20000 calls during query execution

Finally, run HAIDES by executing

```
python night_street_offline.py 
```
