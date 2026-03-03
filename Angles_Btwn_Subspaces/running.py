import functions as ABS
import numpy as np
np.random.seed(41)


X = np.array([[2., 0.,1.,0.,1.],[1.,1.,0.,2.,1.],[10.,10.,770.,2.,1.]]).T
#Y = np.array([[2.,1.,1.,1.,1.],[1.,1.,1.,0.,1.]]).T
#print('made it here')
#data_w_labels = ABS.generate_toy_data_classes(num_classes=3, dimension=2, num_samples=4, noise_std=0.2)
#print(data_w_labels)
# Plot
#ABS.plot_toy_data_classes(data_w_labels)

## Running it all
# generating classes for data (calls the toy_data function)
num_classes = 4
dimension = 15
num_samples = 100 # makes one more than num_samples,since one is made intially
noise_st_dev = 1.3
subspaces_size = 10
dist_type = 'chordal'
MDS_dim = 2
without_replace = True
# some of the above will not need to be used for actual, since data isnt being created
labels_data_list = ABS.generate_toy_data_classes(num_classes, dimension, num_samples, noise_std=noise_st_dev)
#ABS.toy_data(2, 5)
print(labels_data_list)

## Data needs to be in a list of lists, having classes as the first entry, and data with entries as rows ***
data, labels = ABS.stack_data_and_labels(labels_data_list) # only need labels, it isnt helpful to get data in this form

# this function takes the array of data and combines it. A corresponding label vector is also created
print(data)

subspace_list = []
for c in range(num_classes):
    class_data = data[labels == c]

    built_susbpace = ABS.build_subspaces_fast(class_data.T, subspaces_size, without_replace=without_replace) # transpose is necessary here
    #print(built_susbpaces.shape)
    print(built_susbpace)
    subspace_list.extend(built_susbpace)
# now we have a list of arrays that are the 'data' points. We can now get a distance matrix!
# Suppose subspace_tensor is a list of arrays of shape (dim, sbspc_size)
subspace_tensor_array = np.stack(subspace_list, axis=0)  # shape: (num_subspaces, dim, sbspc_size)
# compute distance matrix
dist_matrix = ABS.construct_dist_matrix(subspace_tensor_array, dist_type=dist_type)
config, evals = ABS.classical_mds(dist_matrix, MDS_dim,labels=labels, plot=True, subspace_size =subspaces_size, dist_type =dist_type)

ABS.plot_pca_and_energy(data, labels)
print(evals)