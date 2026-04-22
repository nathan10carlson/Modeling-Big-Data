import functions as ABS
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
np.random.seed(41)

is_this_the_cat_data = False #
num_classes = 2
subspaces_size = 10
dist_type = 'chordal'
MDS_dim = 2
without_replace = True

if is_this_the_cat_data == False:
    dimension = 15 # only for toy data
    num_samples = 40 # only for toy data, makes one more than num_samples,since one is made intially
    noise_st_dev = 1.3 # only for toy data

# performing Angles_Btwn_subspaces
if is_this_the_cat_data == True:
    load_path = r"/Users/nathancarlson/Desktop/programs/MATH 532/Angles_Btwn_Subspaces/CASE_CONTROL_ARRAY.pkl"

    with open(load_path, "rb") as f:
        labels_data_list = pickle.load(f)
else:
    labels_data_list = ABS.generate_toy_data_classes(num_classes, dimension, num_samples, noise_std=noise_st_dev)
    ABS.plot_toy_data_classes(labels_data_list)


data, labels = ABS.stack_data_and_labels(labels_data_list) # only need labels, it isnt helpful to get data in this form
if is_this_the_cat_data == True:
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

# this function takes the array of data and combines it. A corresponding label vector is also created

subspace_list = []
for c in range(num_classes):
    class_data = data[labels == c]

    built_susbpace = ABS.build_subspaces_fast(class_data.T, subspaces_size, without_replace=without_replace) # transpose is necessary here
#print(built_susbpaces.shape)
#print(built_susbpace)
    subspace_list.extend(built_susbpace)
# now we have a list of arrays that are the 'data' points. We can now get a distance matrix!
# Suppose subspace_tensor is a list of arrays of shape (dim, sbspc_size)
subspace_tensor_array = np.stack(subspace_list, axis=0)  # shape: (num_subspaces, dim, sbspc_size)
# compute distance matrix
dist_matrix = ABS.construct_dist_matrix(subspace_tensor_array, dist_type=dist_type)
config, evals = ABS.classical_mds(dist_matrix, MDS_dim,labels=labels, plot=True, subspace_size =subspaces_size, dist_type =dist_type)

ABS.plot_pca_and_energy(data, labels)
print(evals)