#medoid_shift.py
#!/usr/bin/env python3
import numpy as np
import os
from sklearn.cluster import estimate_bandwidth
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
from argparse import ArgumentParser


# add arguments
def parse_args():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--coord", help="Name of coord file")
    p.add_argument("-d", help="directory with npy")
   
    args = p.parse_args()
    return args

def compute_distance_matrix(data, metric):
    """Compute the distance between each pair of points.
    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input points.
    metric : string
        Metric used to compute the distance. See pairwise_distances doc to
        look at all the possible values.
    Returns
    -------
    distance_matrix : array-like, shape=[n_samples, n_samples]
        Distance between each pair of points.
    """

    return pairwise_distances(data, metric=metric)

def compute_weight_matrix(dist_matrix, window_type, bandwidth):
    """Compute the weight of each pair of points, according to the window
    chosen.
    Parameters
    ----------
    dist_matrix : array-like, shape=[n_samples, n_samples]
        Distance matrix.
    window_type : string
        Type of window to compute the weights matrix. Can be
        "flat" or "normal".
    bandwidth : float
        Value of the bandwidth for the window.
    Returns
    -------
    weight_matrix : array-like, shape=[n_samples, n_samples]
        Weight for each pair of points.
    """

    if window_type == 'flat':
        # 1* to convert boolean in int
        weight_matrix = 1*(dist_matrix <= bandwidth)
    elif window_type == 'normal':
        weight_matrix = np.exp(-dist_matrix**2 / (2 * bandwidth**2))
    else:
        raise ValueError("Unknown window type")
    return weight_matrix

def compute_medoids(dist_matrix, weight_matrix, lambd=None):
    """For each point, compute the associated medoid.
    Parameters
    ----------
    dist_matrix : array-like, shape=[n_samples, n_samples]
        Distance matrix.
    weight_matrix : array-like, shape=[n_samples, n_samples]
        Weight for each pair of points.
    lambd : array-like, shape=[n_samples, n_samples]
        If defined matrix, diagonal matrix with weights for each point.
    Returns
    -------
    medoids : array, shape=[n_samples]
        i-th value is the index of the medoid for i-th point.
    """
    if lambd is None:
        lambd = np.eye(len(dist_matrix))
    S = np.dot(np.dot(dist_matrix, lambd), weight_matrix)
    # new medoid for point i lowest coef in the i-th column of S from argmin
    return np.argmin(S, axis=0), S

def compute_stationary_medoids(data, window_type, bandwidth, metric,
                               lambd=None):
    """Return the indices of the own medoids.
    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input points.
    window_type : string
        Type of window to compute the weights matrix. Can be
        "flat" or "normal".
    bandwidth : float
        Value of the bandwidth for the window.
    metric : string
        Metric used to compute the distance. See pairwise_distances doc to
        look at all the possible values.
    lambd : array-like, shape=[n_samples, n_samples]
        If defined matrix, diagonal matrix with weights for each point.
    Returns
    -------
    medoids : array, shape=[n_samples]
        i-th value is the index of the medoid for i-th point.
    stationary_pts : array, shape=[n_stationary_pts]
        Indices of the points which are their own medoids.
    """
    dist_matrix = compute_distance_matrix(data, metric)
    weight_matrix = compute_weight_matrix(dist_matrix, window_type, bandwidth)
    medoids, score = compute_medoids(dist_matrix, weight_matrix, lambd=None)
    stationary_idx = []
    n_pts_attached = {}
    for i in range(len(medoids)):
        if medoids[i] == i:
            stationary_idx.append(i)
            n_pts_attached[i] = 0
    stationary_idx = np.asarray(stationary_idx)
    for i in range(len(medoids)):
        med = medoids[i]
        while med not in stationary_idx:
            med = medoids[med]
        n_pts_attached[med] += 1
    return medoids, stationary_idx, n_pts_attached

def medoid_shift(data, window_type, bandwidth, metric, lambd=None):
    """Perform medoid shiftclustering of data with corresponding parameters.
    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input points.
    window_type : string
        Type of window to compute the weights matrix. Can be
        "flat" or "normal".
    bandwidth : float
        Value of the bandwidth for the window.
    metric : string
        Metric used to compute the distance. See pairwise_distances doc to
        look at all the possible values.
    lambd : array-like, shape=[n_samples, n_samples]
        If defined matrix, diagonal matrix with weights for each point.
    Returns
    -------
    cluster_centers : array, shape=[n_clusters, n_features]
        Coordinates of cluster centers.
    labels : array, shape=[n_samples]
        Cluster labels for each point.
    cluster_centers_idx : array, shape=[n_clusters]
        Index in data of cluster centers.
    """
    if bandwidth is None:
        bandwidth = estimate_bandwidth(data)

    medoids, stat_idx, n_pts = compute_stationary_medoids(data, window_type,
                                                          bandwidth, metric,
                                                          lambd)
    new_data = data[stat_idx]
    if lambd is not None:
        for i in n_pts.keys():
            n_pts[i] += lambd[i]
    lambd = np.diag([i for i in n_pts])
    new_medoids, new_stat_idx, new_n_pts = compute_stationary_medoids(
        new_data, window_type, bandwidth, metric, lambd)
    if len(new_stat_idx) == len(new_data):
        cluster_centers = new_data
        labels = []
        labels_val = {}
        lab = 0
        for i in stat_idx:
            labels_val[i] = lab
            lab += 1
        for i in range(len(data)):
            next_med = medoids[i]
            while next_med not in stat_idx:
                next_med = medoids[next_med]
            labels.append(labels_val[next_med])
        return cluster_centers, np.asarray(labels), stat_idx

    else:
        cluster_centers, next_labels, next_clusters_centers_idx = \
            medoid_shift(new_data, window_type, bandwidth, metric, lambd)
        clusters_centers_idx = stat_idx[next_clusters_centers_idx]
        labels = []
        for i in range(len(data)):
            next_med = medoids[i]
            while next_med not in stat_idx:
                next_med = medoids[next_med]
            # center associated to the medoid in next iteration
            next_med_new_idx = np.where(stat_idx == next_med)[0][0]
            labels.append(next_labels[next_med_new_idx])
        return cluster_centers, np.asarray(labels), clusters_centers_idx


def main():
    args = parse_args()
    os.chdir(args.d)
    data = np.load(args.coord + '.npy').reshape(-1, 3)[::10][:]
    print(args.coord)
    cluster_center, labels, cluster_center_id = medoid_shift(data, 'normal', bandwidth=None, metric='euclidean', lambd=None) 
    os.chdir('/wynton/home/rotation/lisaz/fraser_water/water_placement/output_medoid_shift')
    np.save(args.coord + '_medoid_center.npy', cluster_center)

if __name__ == '__main__':
    main()
