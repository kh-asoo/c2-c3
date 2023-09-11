#c4
#
import pandas as pd
import novosparc
from sklearn.manifold import TSNE
# import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.neighbors import kneighbors_graph
import time
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
import os
import ot

#constructing affinity matrix

# import TPM,label and LR data
TPM = pd.read_csv("path to TPM data",sep="\t")

LR = pd.read_csv( "path to LR data", sep="\t", header=None)

labelData = pd.read_csv("path to label data", sep="\t", header=0) 


genenames = TPM['X']
TPM = TPM.iloc[:, 1:]
TPM[TPM.isna()] = 0
cellnames = TPM.columns
# labels = pd.DataFrame(np.array(labelData['labels'])[np.where(np.in1d(labelData['cells'] , cellnames))])
labels = labelData['labels'][labelData['cells'].isin(cellnames)].values
labels[pd.isnull(labels)] = 'unlabeled'
standards = np.unique(labels)
labelIx = np.array([np.where(standards == label)[0][0] for label in labels])
cellCounts = pd.Series(labelIx).value_counts()
temp=  (pd.concat([LR[0], LR[1]], ignore_index=True))

temp=list(temp)
genenames=list(genenames)
ligandsIndex = []
for name in temp:
    if name in genenames:
        ligandsIndex.append(genenames.index(name))
    else:
        ligandsIndex.append(np.nan)

temp2=  (pd.concat([LR[1], LR[0]], ignore_index=True))
temp2=list(temp2)
genenames= list(genenames)
receptorIndex = []
for name in temp2:
    if name in genenames:
        receptorIndex.append(genenames.index(name))
    else:
        receptorIndex.append(np.nan)

receptorIndex=np.array(receptorIndex)

ligandsIndex= np.array(ligandsIndex, dtype='float32')

ligandsTPM = TPM.iloc[ligandsIndex[~np.isnan(ligandsIndex) & ~np.isnan(receptorIndex)],:].values

receptorTPM = TPM.iloc[receptorIndex[~np.isnan(ligandsIndex) & ~np.isnan(receptorIndex)],:].values

LRscores = np.array(LR.iloc[:,2].values.tolist()*2)[~np.isnan(ligandsIndex) & ~np.isnan(receptorIndex)]

affinityMat = np.transpose(ligandsTPM) @ np.diag(LRscores) @ receptorTPM

for i in range(affinityMat.shape[0]):
    affinityArray = affinityMat[i,:].copy()
    affinityArray[i] = 0
    affinityArraySorted = np.sort(affinityArray)[::-1]
    affinityArray[affinityArray <= affinityArraySorted[49]] = 0
    affinityMat[i,:] = affinityArray

eps = 2**(-52)
P = 0.5 * (affinityMat + affinityMat.T)
P[P < eps] = eps
P = P / np.sum(P)
affinity = P

#

seq =  pd.read_csv("path to TPM data",sep="\t")
seq = seq.iloc[:, 1:]
seeq= seq.T

X_embedded = TSNE(n_components=2, learning_rate='auto',
                   init='random', perplexity=30).fit_transform(seeq)

X = X_embedded 
y = affinity

p= ot.unif(X.shape[0])
q = ot.unif(X.shape[0])

mode= "connectivity"
metric="correlation"
k = 30
Xgraph=kneighbors_graph(X, k, mode=mode, metric=metric)
ygraph=kneighbors_graph(y, k, mode=mode, metric=metric)

X_shortestPath=dijkstra(csgraph= csr_matrix(Xgraph), directed=False, return_predecessors=False)
y_shortestPath=dijkstra(csgraph= csr_matrix(ygraph), directed=False, return_predecessors=False)

		# Deal with unconnected stuff (infinities):
X_max=np.nanmax(X_shortestPath[X_shortestPath != np.inf])
y_max=np.nanmax(y_shortestPath[y_shortestPath != np.inf])
X_shortestPath[X_shortestPath > X_max] = X_max
y_shortestPath[y_shortestPath > y_max] = y_max

		# Finnally, normalize the distance matrix:
Cx=X_shortestPath/X_shortestPath.max()
Cy=y_shortestPath/y_shortestPath.max()

#ot 
coupling= ot.gromov.entropic_gromov_wasserstein(C1= Cx,C2=affinity,p =p, q=q, loss_fun='square_loss', epsilon=0.1 , log=False)

#save the result
# np.savetxt("hc4.txt", coupling)
# np.savetxt("hc4loc.txt", X_embedded )

