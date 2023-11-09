import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score

########################################################################
########### 1.2 Experiment ###################
#########################################################################

# evaluate K-means clustering and GMM on a simple 2-dimensional problem
# first, create a 2d synthetic dtaset of 300 points by sampling 100 points each from the three gaussian distributions given

# First implement K-means clustering and the expectation maximization algorithm for GMMs. Execute both meth-
# ods on five synthetic datasets, generated as shown above with σ ∈ {0.5, 1, 2, 4, 8}. Finally, evaluate both methods
# on (i) the clustering objective (1) and (ii) the clustering accuracy. For each of the two criteria, plot the value
# achieved by each method against σ

# Both algorithms are only guaranteed to find only a local optimum so we recommend trying multiple restarts
# and picking the one with the lowest objective value (This is (1) for K-means and the negative log likelihood
# GMMs). You may also experiment with a smart initialization strategy (such as kmeans++)

# o plot the clustering accuracy, you may treat the ‘label’ of points generated from distribution Pu as u,
# where u ∈ {a, b, c}. Assume that the cluster id i returned by a method is i ∈ {1, 2, 3}. Since clustering is
# an unsupervised learning problem, you should obtain the best possible mapping from {1, 2, 3} to {a, b, c}
# to compute the clustering objective. One way to do this is to compare the clustering centers returned by the
# method (centroids for K-means, means for GMMs) and map them to the distribution with the closest mean

# K-means clustering
def kmeans(data, k, max_iters=100):
    # Randomly initialize cluster centers
    np.random.seed(0)
    centroids = data[np.random.choice(len(data), k, replace=False)]
    for _ in range(max_iters):
        # Assign each data point to the nearest centroid
        labels = np.argmin(np.linalg.norm(data[:, np.newaxis] - centroids, axis=2), axis=1)
        # Update centroids
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        if np.all(new_centroids == centroids):
            break
        centroids = new_centroids
    return centroids, labels

# Gaussian Mixture Model (GMM)
def gmm(data, k, max_iters=100):
    n, d = data.shape
    # Randomly initialize parameters
    weights = np.ones(k) / k
    means = data[np.random.choice(n, k, replace=False)]
    covariances = [np.identity(d) for _ in range(k)]

    for _ in range(max_iters):
        # E-step: Calculate responsibilities
        responsibilities = np.array([weights[i] * multivariate_normal.pdf(data, means[i], covariances[i]) for i in range(k)]).T
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)

        # M-step: Update parameters
        Nk = responsibilities.sum(axis=0)
        weights = Nk / n
        means = np.array([np.sum(responsibilities[:, i:i + 1] * data, axis=0) / Nk[i] for i in range(k)])
        covariances = [np.dot((responsibilities[:, i:i + 1] * (data - means[i])).T, (data - means[i])) / Nk[i] for i in range(k)]

    return means, responsibilities.argmax(axis=1)

# Calculate clustering objective
def calculate_clustering_objective(data, labels, centroids):
    return np.sum([np.linalg.norm(data[i] - centroids[labels[i]])**2 for i in range(len(data))])


# Generate synthetic data for different values of sigma
sigmas = [0.5, 1, 2, 4, 8]
results_kmeans_obj = []
results_gmm_obj = []
results_kmeans_acc = []
results_gmm_acc = []

for sigma in sigmas:
    # Generate synthetic data
    np.random.seed(0)
    n_samples = 300
    means = np.array([[-1, -1], [1, -1], [0, 1]])
    covariances = [sigma * np.array([[2, 0.5], [0.5, 1]]),
                   sigma * np.array([[1, -0.5], [-0.5, 2]]),
                   sigma * np.array([[1, 0], [0, 2]])]
    data = []
    true_labels = []

    for i in range(len(means)):
        data_i = np.random.multivariate_normal(means[i], covariances[i], int(n_samples / 3))
        data.extend(data_i)
        true_labels.extend([chr(97 + i)] * len(data_i))

    data = np.array(data)

    # Perform K-means clustering
    kmeans_centers, kmeans_labels = kmeans(data, k=3)
    
    # Calculate clustering objective for K-means
    kmeans_obj = calculate_clustering_objective(data, kmeans_labels, kmeans_centers)
    results_kmeans_obj.append(kmeans_obj)

    # Perform GMM clustering
    gmm_means, gmm_labels = gmm(data, k=3)

    # Calculate clustering objective for GMM
    gmm_obj = calculate_clustering_objective(data, gmm_labels, gmm_means)
    results_gmm_obj.append(gmm_obj)

    # Calculate clustering accuracy for K-means
    kmeans_cluster_to_true_mapping = {}
    for i in range(3):
        distances = [np.linalg.norm(kmeans_centers[i] - means[j]) for j in range(3)]
        closest_distribution = chr(97 + np.argmin(distances))
        kmeans_cluster_to_true_mapping[i] = closest_distribution

    kmeans_mapped_labels = np.array([kmeans_cluster_to_true_mapping[label] for label in kmeans_labels])
    kmeans_acc = np.mean(kmeans_mapped_labels == true_labels)
    results_kmeans_acc.append(kmeans_acc)

    # Calculate clustering accuracy for GMM
    gmm_cluster_to_true_mapping = {}
    for i in range(3):
        distances = [np.linalg.norm(gmm_means[i] - means[j]) for j in range(3)]
        closest_distribution = chr(97 + np.argmin(distances))
        gmm_cluster_to_true_mapping[i] = closest_distribution

    gmm_mapped_labels = np.array([gmm_cluster_to_true_mapping[label] for label in gmm_labels])
    gmm_acc = np.mean(gmm_mapped_labels == true_labels)
    results_gmm_acc.append(gmm_acc)
    
    # Calculate precision, recall, and accuracy for K-means
    kmeans_precision = precision_score(true_labels, kmeans_mapped_labels, average='weighted')
    kmeans_recall = recall_score(true_labels, kmeans_mapped_labels, average='weighted')
    kmeans_accuracy = kmeans_acc
    
    # Calculate precision, recall, and accuracy for GMM
    gmm_precision = precision_score(true_labels, gmm_mapped_labels, average='weighted')
    gmm_recall = recall_score(true_labels, gmm_mapped_labels, average='weighted')
    gmm_accuracy = gmm_acc
    
    # Print results
    print(f"Sigma: {sigma}")
    print("K-means Precision: {:.2f}".format(kmeans_precision))
    print("K-means Recall: {:.2f}".format(kmeans_recall))
    print("K-means Accuracy: {:.2f}".format(kmeans_accuracy))
    print("GMM Precision: {:.2f}".format(gmm_precision))
    print("GMM Recall: {:.2f}".format(gmm_recall))
    print("GMM Accuracy: {:.2f}".format(gmm_accuracy))

# Plot the results
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(sigmas, results_kmeans_obj, marker='o', label='K-means')
plt.plot(sigmas, results_gmm_obj, marker='o', label='GMM')
plt.xlabel('Sigma')
plt.ylabel('Clustering Objective')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(sigmas, results_kmeans_acc, marker='o', label='K-means')
plt.plot(sigmas, results_gmm_acc, marker='o', label='GMM')
plt.xlabel('Sigma')
plt.ylabel('Clustering Accuracy')
plt.legend()

plt.show()

####################################################################
######## 2.3 Experiment ###############
###################################################################

# Here we will compare the above three methods on two data sets.
# • We will implement three variants of PCA:
# 1. ”buggy PCA”: PCA applied directly on the matrix X.
# 2. ”demeaned PCA”: We subtract the mean along each dimension before applying PCA.
# 3. ”normalized PCA”: Before applying PCA, we subtract the mean and scale each dimension so that the
# sample mean and standard deviation along each dimension is 0 and 1 respectively.
# • One way to study how well the low dimensional representation Z captures the linear structure in our data
# is to project Z back to D dimensions and look at the reconstruction error. For PCA, if we mapped it to
# d dimensions via z = V x then the reconstruction is V ⊤z. For the preprocessed versions, we first do this
# and then reverse the preprocessing steps as well. For DRO we just compute Az + b. We will compare all
# methods by the reconstruction error on the datasets.
# • Please implement code for the methods: Buggy PCA (just take the SVD of X) , Demeaned PCA, Normal-
# ized PCA, DRO. In all cases your function should take in an n × d data matrix and d as an argument. It
# should return the the d dimensional representations, the estimated parameters, and the reconstructions of
# these representations in D dimensions.
# • You are given two datasets: A two Dimensional dataset with 50 points data2D.csv and a thousand
# dimensional dataset with 500 points data1000D.csv.
# • For the 2D dataset use d = 1. For the 1000D dataset, you need to choose d. For this, observe the singular
# values in DRO and see if there is a clear “knee point” in the spectrum. Attach any figures/ Statistics you
# computed to justify your choice.
# • For the 2D dataset you need to attach the a plot comparing the orignal points with the reconstructed points
# for all 4 methods. For both datasets you should also report the reconstruction errors, that is the squared
# sum of differences ∑n
# i=1 ∥xi − r(zi)∥2, where xi’s are the original points and r(zi) are the D dimensional
# points reconstructed from the d dimensional representation zi

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data_2d = pd.read_csv(r'C:\Users\arneg\Dropbox\Classes\CS760\hw\hw5\data\data2D.csv').values
data_1000d = pd.read_csv(r'C:\Users\arneg\Dropbox\Classes\CS760\hw\hw5\data\data1000D.csv').values

############ plot 1000d dataset to visualize and choose d ###########
# Perform SVD
U, s, Vt = np.linalg.svd(data_1000d, full_matrices=False)

# Plot the singular values
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(s) + 1), s, marker='o', linestyle='-', color='b')
plt.title('Singular Value Spectrum')
plt.xlabel('Singular Value Index')
plt.ylabel('Singular Value')
plt.grid(True)

#plt.show()

# Set the x-axis and y-axis limits
plt.xlim(1, 60)  # Adjust the x-axis limits up to 60
plt.ylim(0, 2000)  # Adjust the y-axis limits up to 2000

plt.show()

########################################################################

def buggy_pca(X, d):
    # Perform PCA by taking SVD of X
    U, s, Vt = np.linalg.svd(X)
    Z = U[:, :d]  # Take the top d principal components
    V = Vt.T
    A = V[:, :d]  # Select the top d columns of V
    b = np.mean(X, axis=0)
    
    # Reconstruct the data in D dimensions
    reconstructed_data = np.dot(Z, A.T) + b
    return Z, (A, b), reconstructed_data

def demeaned_pca(X, d):
    # Subtract the mean from the data
    X_demeaned = X - np.mean(X, axis=0)
    
    # Perform PCA on demeaned data
    U, s, Vt = np.linalg.svd(X_demeaned)
    Z = U[:, :d]  # Take the top d principal components
    V = Vt.T
    A = V[:, :d]
    b = np.mean(X, axis=0)
    
    # Reconstruct the data in D dimensions
    reconstructed_data = np.dot(Z, A.T) + b
    return Z, (A, b), reconstructed_data

def normalized_pca(X, d):
    # Normalize the data 
    X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    # Perform PCA on normalized data
    U, s, Vt = np.linalg.svd(X_normalized)
    Z = U[:, :d]  # Take the top d principal components
    V = Vt.T
    A = V[:, :d]
    b = np.mean(X, axis=0)
    
    # Reconstruct the data in D dimensions
    reconstructed_data = np.dot(Z, A.T) + b
    return Z, (A, b), reconstructed_data

def dro(X, d):
    # Compute the DRO transformation directly
    A = np.random.randn(X.shape[1], d)  
    b = np.mean(X, axis=0)
    Z = np.dot(X - b, A)
    
    # Reconstruct the data in D dimensions
    reconstructed_data = np.dot(Z, A.T) + b
    return Z, (A, b), reconstructed_data

# Calculate reconstruction error
def reconstruction_error(original_data, reconstructed_data):
    return np.sum((original_data - reconstructed_data) ** 2)

# For 2D dataset, set d=1 and compare methods
d_2d = 1
original_data_2d = data_2d
results_2d = {}

methods_2d = ['Buggy PCA', 'Demeaned PCA', 'Normalized PCA', 'DRO']

for method in methods_2d:
    if method == 'Buggy PCA':
        Z, params, reconstructed_data = buggy_pca(original_data_2d, d_2d)
    elif method == 'Demeaned PCA':
        Z, params, reconstructed_data = demeaned_pca(original_data_2d, d_2d)
    elif method == 'Normalized PCA':
        Z, params, reconstructed_data = normalized_pca(original_data_2d, d_2d)
    elif method == 'DRO':
        Z, params, reconstructed_data = dro(original_data_2d, d_2d)
    
    error = reconstruction_error(original_data_2d, reconstructed_data)
    results_2d[method] = {'Z': Z, 'Params': params, 'ReconstructedData': reconstructed_data, 'Error': error}


# apply the selected d to the 1000D dataset
d_1000d = 2  # chosen d
original_data_1000d = data_1000d
results_1000d = {}

for method in methods_2d:
    if method == 'Buggy PCA':
        Z, params, reconstructed_data = buggy_pca(original_data_1000d, d_1000d)
    elif method == 'Demeaned PCA':
        Z, params, reconstructed_data = demeaned_pca(original_data_1000d, d_1000d)
    elif method == 'Normalized PCA':
        Z, params, reconstructed_data = normalized_pca(original_data_1000d, d_1000d)
    elif method == 'DRO':
        Z, params, reconstructed_data = dro(original_data_1000d, d_1000d)
        
    error = reconstruction_error(original_data_1000d, reconstructed_data)
    results_1000d[method] = {'Z': Z, 'Params': params, 'ReconstructedData': reconstructed_data, 'Error': error}
    
# Calculate reconstruction error
def reconstruction_error(original_data, reconstructed_data):
    return np.sum(np.square(original_data - reconstructed_data))

# Calculate the reconstruction errors for each method
for method in methods_2d:
    error = reconstruction_error(original_data_2d, results_2d[method]['ReconstructedData'])
    print(f'Reconstruction Error ({method}) for 2D dataset: {error}')
    
# Print reconstruction errors for 1000D dataset
for method in methods_2d:
    error = reconstruction_error(original_data_1000d, results_1000d[method]["ReconstructedData"])
    print(f'Reconstruction Error ({method}) for 1000D dataset: {error}')


############## Plots #########################
import matplotlib.pyplot as plt

# Iterate over the methods and create separate plots
for method in methods_2d:
    plt.figure(figsize=(10, 5))
    plt.scatter(original_data_2d[:, 0], original_data_2d[:, 1], label='Original Data', marker='o', s=20)
    plt.scatter(results_2d[method]['ReconstructedData'][:, 0], results_2d[method]['ReconstructedData'][:, 1], label=method, marker='x', s=20)
    plt.title(f'2D Dataset Reconstruction Comparison ({method})')
    plt.legend()
    plt.show()
    
