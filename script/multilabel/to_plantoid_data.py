# -*- coding: utf-8 -*-

import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import numpy as np
import pickle as pk
from collections import defaultdict

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

dataset="zf"
path="./"+dataset+"_data/" 
save_root="./"+dataset+"_data_ind"
idx_features_labels = np.genfromtxt("{}{}.content.norm".format(path, dataset),delimiter='\t',dtype=np.dtype(str))    ##delimiter默认值为连续空格
features = sp.csr_matrix(idx_features_labels[:, 1:-12], dtype=np.float32)
# features = normalize(features) # no normalization in plantoid

# labels = encode_onehot(idx_features_labels[:, -1])
labels = idx_features_labels[:, -12:]
print("features is: ", features)
print("idx_features_labels is: ", idx_features_labels)
print("labels is: ", labels)
print("labels shape is: ", labels.shape)
# build graph
idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
idx_map = {j: i for i, j in enumerate(idx)}
edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),dtype=np.int32)
edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                 dtype=np.int32).reshape(edges_unordered.shape)
adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                    shape=(labels.shape[0], labels.shape[0]),
                    dtype=np.float32)
print("111shape: ", adj.shape)
# build symmetric adjacency matrix
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
idx_train = range(56)
idx_val = range(100,250)
idx_test = range(250,550)


pk.dump(features[idx_train], open(save_root+"/ind."+dataset+".x", "wb" ) )
pk.dump(sp.vstack((features[:idx_test[0]], features[idx_test[-1]+1:])), open( save_root+"/ind."+dataset+".allx", "wb" ) )
pk.dump(features[idx_test], open(save_root+"/ind."+dataset+".tx", "wb") )

print("y shape is: ", labels[idx_train].shape)
print("ty shape is: ", labels[idx_test].shape)
print("ally shape is: ", labels[:idx_test[0]].shape)
pk.dump(labels[idx_train], open( save_root+"/ind."+dataset+".y", "wb" ) )
pk.dump(labels[idx_test], open( save_root+"/ind."+dataset+".ty", "wb" ) )
pk.dump(np.vstack((labels[:idx_test[0]],labels[idx_test[-1]+1:])), open( save_root+"/ind."+dataset+".ally", "wb" ) )

with open(save_root+"/ind."+dataset+".test.index", 'w') as f:
    for item in list(idx_test):
        f.write("%s\n" % item)


# ori_graph
array_adj = np.argwhere(adj.toarray())
ori_graph = defaultdict(list)
for edge in array_adj:
    ori_graph[edge[0]].append(edge[1])
pk.dump(ori_graph, open( save_root+"/ind."+dataset+".graph", "wb" ) )
print("Finished!")
