
#c2
#
import pandas as pd
import novosparc
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
def setup_for_OT_reconstruction2(dge, locations, num_neighbors_source = 5, num_neighbors_target = 5,
                                locations_metric='minkowski', locations_metric_p=2,
                                expression_metric='minkowski', expression_metric_p=2, verbose=True):
    start_time = time.time()
    if verbose:
        print ('Setting up for reconstruction ... ', end='', flush=True)

    # Shortest paths matrices at target and source spaces
    num_neighbors_target = num_neighbors_target # number of neighbors for nearest neighbors graph at target
    A_locations = kneighbors_graph(locations, num_neighbors_target, mode='connectivity', include_self=True,
                                   metric=locations_metric, p=locations_metric_p)

    sp_locations = dijkstra(csgraph=csr_matrix(A_locations), directed=False, return_predecessors=False)
    # sp_locations = (1- sp_locations)
    sp_locations_max = np.nanmax(sp_locations[sp_locations != np.inf])
    sp_locations[sp_locations > sp_locations_max] = sp_locations_max #set threshold for shortest paths

    num_neighbors_source = num_neighbors_source # number of neighbors for nearest neighbors graph at source
    A_expression = kneighbors_graph(dge, num_neighbors_source, mode='connectivity', include_self=True,
                                    metric=expression_metric, p=expression_metric_p)
    sp_expression = dijkstra(csgraph=csr_matrix(A_expression), directed=False, return_predecessors=False)
    sp_expression_max = np.nanmax(sp_expression[sp_expression != np.inf])
    sp_expression[sp_expression > sp_expression_max] = sp_expression_max #set threshold for shortest paths

    # Set normalized cost matrices based on shortest paths matrices at target and source spaces
    cost_locations = sp_locations / sp_locations.max()
    
    cost_locations -= np.mean(cost_locations)
    cost_locations = (1- cost_locations)
  
    cost_expression = sp_expression / sp_expression.max()
    cost_expression -= np.mean(cost_expression)
    # set beta parameter 
    cost_expression = ((0.2* (1- cost_expression)) + (0.8 * affinity))

    if verbose:
        print('done (', round(time.time()-start_time, 2), 'seconds )')
    return cost_expression, cost_locations

def setup_smooth_costs2(dge_rep=None, num_neighbors_s=5, num_neighbors_t=5,
						   locations_metric='minkowski', locations_metric_p=2,
						   expression_metric='minkowski', expression_metric_p=2, verbose=True):
		"""
		Set cell-cell expression cost and location-location physical distance cost
		dge_rep             -- some representation of the expression matrix, e.g. pca, selected highly variable genes etc.
		num_neighbors_s     -- num neighbors for cell-cell expression cost
		num_neighbors_t     -- num neighbors for location-location physical distance cost
		locations_metric    -- discrepancy metric - physical distance cost
		locations_metric_p  -- power parameter of the Minkowski metric - locations distance cost
		expression_metric   -- discrepancy metric - expression distance cost
		expression_metric_p -- power parameter of the Minkowski metric - expression distance cost
		"""
		dge_rep = dge_rep if dge_rep is not None else tissue.dge
		tissue.costs['expression'], tissue.costs['locations'] = setup_for_OT_reconstruction2(dge_rep,
																									 tissue.locations,
																									 num_neighbors_source=num_neighbors_s,
																									 num_neighbors_target=num_neighbors_t,
																									 locations_metric=locations_metric, locations_metric_p=locations_metric_p,
																									 expression_metric=expression_metric, expression_metric_p=expression_metric_p,
																									 verbose=verbose)


def setup_linear_cost2( markers_to_use=None, atlas_matrix=None, markers_metric='minkowski', markers_metric_p=2):
		"""
		Set linear(=atlas) cost matrix
		markers_to_use   -- indices of the marker genes
		atlas_matrix     -- corresponding reference atlas
		markers_metric   -- discrepancy metric - cell-location distance cost
		markers_metric_p -- power parameter of the Minkowski metric - cell-location distance cost
		"""
		tissue.atlas_matrix = atlas_matrix if atlas_matrix is not None else tissue.atlas_matrix
		tissue.markers_to_use = markers_to_use if markers_to_use is not None else tissue.markers_to_use

		cell_expression = tissue.dge[:, tissue.markers_to_use] / np.amax(tissue.dge[:, tissue.markers_to_use])
		atlas_expression = tissue.atlas_matrix / np.amax(tissue.atlas_matrix)

		tissue.costs['markers'] = cdist(cell_expression, atlas_expression, metric=markers_metric, p=markers_metric_p)
		tissue.num_markers = len(tissue.markers_to_use)

def setup_reconstruction2( markers_to_use=None, atlas_matrix=None, num_neighbors_s=5, num_neighbors_t=5, verbose=True):
		"""
		Set cost matrices for reconstruction. If there are marker genes and an reference atlas matrix, these
		can be used as well.
		markers_to_use  -- indices of the marker genes
		atlas_matrix    -- reference atlas corresponding to markers_to_use
		num_neighbors_s -- num neighbors for cell-cell expression cost
		num_neighbors_t -- num neighbors for location-location physical distance cost
		"""
		tissue.atlas_matrix = atlas_matrix if atlas_matrix is not None else tissue.atlas_matrix
		tissue.markers_to_use = markers_to_use if markers_to_use is not None else tissue.markers_to_use

		# calculate cost matrices for OT
		if tissue.markers_to_use is not None:
			setup_linear_cost2(tissue.markers_to_use, tissue.atlas_matrix)
		setup_smooth_costs2(num_neighbors_s=num_neighbors_s, num_neighbors_t=num_neighbors_t,verbose=verbose)
 
 ##
 # this could be the dge file, or also can be a 10x mtx folder
dataset_path = "path to expression data"

    # target_space_path = 'path to location data' # location coordinates if exist
output_folder = 'path to output dir' # folder to save the results, plots etc.

    #######################################
    # 2. Read the dataset and subsample ###
    #######################################

    # Read the dge. this assumes the file formatted in a way that genes are columns and cells are rows.
    # If the data is the other way around, transpose the dataset object (e.g dataset=dataset.T)
dataset = novosparc.io.load_data(dataset_path)

    # Optional: downsample number of cells.
    #cells_selected, dataset = novosparc.pp.subsample_dataset(dataset, min_num_cells=500, max_num_cells=1000)
    
    # Optional: Subset to the highly variable genes
dataset.raw = dataset # this stores the current dataset with all the genes for future use
    #hvg_path = '/pathto/hvg.txt'

    # a file for a list of highly variable genes can be provided. or directly a gene list provided 
    # with the argument 'gene_list'. The whole process can be done also with scanpy
    #dataset, hvg = novosparc.pp.subset_to_hvg(dataset, hvg_file = hvg_path) 

    # Load the location coordinates from file if it exists
    #locations = novosparc.io.load_target_space(target_space_path, cells_selected, coords_cols=['xcoord', 'ycoord'])

    # Alternatively, construct a square target grid
    # locations = novosparc.geometry.construct_target_grid(num_cells)
    
#rectangle

# locations = novosparc.geometry.construct_target_grid(len(dataset.obs), ratio=8,random =True)
# locations = novosparc.rc.construct_target_grid(num_cells)

# sphere

# locations = novosparc.geometry.construct_sphere(len(dataset.obs))
#locations = novosparc.geometry.construct_target_grid(len(cells_selected))
    
#circle
locations = novosparc.geometry.construct_circle(len(dataset.obs))

#    
# locations = np.loadtxt("coordinates.txt",skiprows=1,usecols=(1,2,3))
    #########################################
    # 3. Setup and spatial reconstruction ###
    #########################################

tissue = novosparc.cm.Tissue(dataset=dataset, locations=locations, output_folder=output_folder) # create a tissue object

tissue.setup_reconstruction = setup_reconstruction2(num_neighbors_s=5, num_neighbors_t=5)
tissue.reconstruct(alpha_linear=0) # reconstruct with the given alpha value(de novo mode)

#save the result the optimal transport matrix
def result(tissue):
     gw = tissue.gw
    #  np.savetxt("T3b2.txt", gw) 
result(tissue=tissue)    

# save the locations data
# np.savetxt("locb5h.txt", locations)    
