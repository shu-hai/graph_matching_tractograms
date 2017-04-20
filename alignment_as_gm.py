"""Alignment of tractograms as graph matching.

See: Olivetti E, Sharmin N and Avesani P (2016) Alignment of
Tractograms As Graph Matching. Front. Neurosci. 10:554. 
doi:10.3389/fnins.2016.00554

"""

import numpy as np
from nibabel import trackvis
from dissimilarity import compute_dissimilarity, dissimilarity
from kmeans import mini_batch_kmeans, compute_labels, compute_centroids
from sklearn.neighbors import KDTree
from dipy.tracking.distances import bundles_distances_mam
from dipy.tracking.streamlinespeed import length
from DSPFP import DSPFP_faster, greedy_assignment
from joblib import  Parallel, delayed


def clustering(S_dr, k, b=100, t=100):
    """Wrapper of the mini-batch k-means algorithm that combines multiple
    basic functions into a convenient one.
    """
    # Generate k random centers:
    S_centers = S_dr[np.random.permutation(S_dr.shape[0])[:k]]

    # Improve the k centers with mini_batch_kmeans
    S_centers = mini_batch_kmeans(S_dr, S_centers, b=b, t=t)

    # Assign the cluster labels to each streamline. The label is the
    # index of the nearest center.
    S_cluster_labels = compute_labels(S_dr, S_centers)

    # Compute a cluster representive, for each cluster
    S_representatives_idx = compute_centroids(S_dr, S_centers)
    return S_representatives_idx, S_cluster_labels


def graph_matching(S_A, S_B, alpha=0.5, max_iter1=100, max_iter2=100,
                   initialization='NN', similarity='1/x',
                   epsilon=1.0e-8, verbose=True):
    """Wrapper of the DSPFP algorithm to deal with streamlines. In
    addition to calling DSPFP, this function adds initializations of
    the graph matching algorithm that are meaningful for streamlines,
    as well as some (optional) conversions from distances to
    similarities.
    """
    assert(len(S_A) >= len(S_B))  # required by DSPFP
    if verbose:
        print("Computing graph matching between streamlines.")
        print("Computing the distance matrix between streamlines in each set")

    dm_A = streamline_distance(S_A, S_A)
    dm_B = streamline_distance(S_B, S_B)

    if initialization == 'NN':
        X_init = streamline_distance(S_A, S_B)
    elif initialization == 'random':
        X_init = np.random.uniform(size=(len(S_A), len(S_B)))
    else:
        # flat initialization, default of DSPFP
        X_init = None

    # Wheter to use distances or similarities and, in case, which
    # similarity function
    if similarity == '1/x':
        sm_A = 1.0 / (1.0 + dm_A)
        sm_B = 1.0 / (1.0 + dm_B)
        if initialization == 'NN':
            X_init = 1.0 / (1.0 + X_init)

    elif similarity == 'exp(-x)':
        tmp = np.median(dm_A)
        sm_A = np.exp(-dm_A / tmp)
        tmp = np.median(dm_B)
        sm_B = np.exp(-dm_B / tmp)
        if initialization == 'NN':
            tmp = np.median(X_init)
            X_init = np.exp(-X_init * X_init / (tmp * tmp))

    else:  # Don't use similarity
        sm_A = dm_A
        sm_B = dm_B
        if initialization == 'NN':
            X_init = 1.0 / (1.0 + X_init)  # X_init needs similarity
                                           # when NN

    if verbose:
        print("Computing graph-matching via DSPFP")

    X = DSPFP_faster(sm_A, sm_B, alpha=alpha,
                     max_iter1=max_iter1,
                     max_iter2=max_iter2,
                     X=X_init, verbose=verbose)

    ga = greedy_assignment(X)
    corresponding_streamlines = ga.argmax(1)
    unassigned = ga.sum(1)==0
    corresponding_streamlines[unassigned] = -1
    return corresponding_streamlines


def streamline_distance(S_A, S_B=None, parallel=True):
    """Wrapper to decide what streamline distance function to use. The
    function computes the distance matrix between sets of
    streamlines. This implementation provides optimiztions like
    parallelization and avoiding useless computations when S_B is
    None.
    """
    distance_function = bundles_distances_mam
    if parallel:
        return dissimilarity(S_A, S_B, distance_function)
    else:
        return distance_function(S_A, S_B)


def distance_corresponding(A, B, correspondence):
    """Distance between streamlines in set A and the corresponding ones in
    B. The vector 'correspondence' has in position 'i' the index of
    the streamline in B that corresponds to A[i].

    """
    return np.array([streamline_distance([A[i]], [B[correspondence[i]]], parallel=False) for i in range(len(A))]).squeeze()


def graph_matching_two_clusters(cluster_A, cluster_B, alpha=0.5,
                                max_iter1=100, max_iter2=100):
    """Wrapper of graph_matching() between the streamlines of two
    clusters. This code is able two handle clusters of different sizes
    and to invert the result of corresponding_streamlines, if
    necessary.

    """
    if len(cluster_A) >= len(cluster_B):  # graph_matching(A, B)
        corresponding_streamlines = graph_matching(cluster_A,
                                                   cluster_B,
                                                   alpha=alpha,
                                                   max_iter1=max_iter1,
                                                   max_iter2=max_iter2,
                                                   verbose=False)
    else:  # graph_matching(B, A)
        corresponding_streamlines = graph_matching(cluster_B,
                                                   cluster_A,
                                                   alpha=alpha,
                                                   max_iter1=max_iter1,
                                                   max_iter2=max_iter2,
                                                   verbose=False)
        # invert result from B->A to A->B:
        tmp = -np.ones(len(cluster_A), dtype=np.int)
        for j, v in enumerate(corresponding_streamlines):
            if v != -1:
                tmp[v] = j

        corresponding_streamlines = tmp

    return corresponding_streamlines


if __name__ == '__main__':
    np.random.seed(0)

    T_A_filename = 'data/HCP_subject124422_100Kseeds/tracks_dti_100K.trk'
    T_B_filename = 'data/HCP_subject124422_100Kseeds/tracks_dti_100K.trk'

    # T_A_filename = 'data2/100307/Tractogram/tractogram_b1k_1.25mm_csd_wm_mask_eudx1M.trk'
    # T_B_filename = 'data2/100408/Tractogram/tractogram_b1k_1.25mm_csd_wm_mask_eudx1M.trk'
    # T_B_filename = T_A_filename


    # 1) load T_A and T_B
    print("Loading %s" % T_A_filename)
    T_A, hdr_A = trackvis.read(T_A_filename, as_generator=False)
    print("Loading %s" % T_B_filename)
    T_B, hdr_B = trackvis.read(T_B_filename, as_generator=False)
    T_A = np.array([s[0] for s in T_A], dtype=np.object)
    T_B = np.array([s[0] for s in T_B], dtype=np.object)
    print("T_A: %s streamlines" % len(T_A))
    print("T_B: %s streamlines" % len(T_B))

    # 1.1) Removing short artifactual streamlines
    threshold = 5.0
    print("Removing (presumably artifactual) streamlines shorter than %s" % threshold)
    T_A = np.array([s for s in T_A if length(s) >= threshold], dtype=np.object)
    T_B = np.array([s for s in T_B if length(s) >= threshold], dtype=np.object)
    print("T_A: %s streamlines" % len(T_A))
    print("T_B: %s streamlines" % len(T_B))

    if T_A_filename == T_B_filename:  # only if A and B are the same:
        # 1.2) Permuting the order of T_B and creating ground truth:
        print("Permuting the order of T_B and creating ground truth.")
        T_B_random_idx = np.random.permutation(len(T_B))
        correspondence_ground_truth = np.argsort(T_B_random_idx)
        T_B = T_B[T_B_random_idx]
        assert((T_A[0] == T_B[correspondence_ground_truth[0]]).all())

    # 2) Compute the dissimilarity representation of T_A and T_B
    print("Computing the dissimilarity representation of T_A")
    T_A_dr, prototypes_A = compute_dissimilarity(T_A)
    print("Computing the dissimilarity representation of T_B")
    T_B_dr, prototypes_B = compute_dissimilarity(T_B)

    # 3) Compute the k-means clustering of T_A and T_B
    k = 300  # number of clusters
    b = 100  # mini-batch size
    t = 100  # number of iterations
    print("Computing the k-means clustering of T_A and T_B, k=%s" % k)
    print("mini-batch k-means on T_A")
    T_A_representatives_idx, T_A_cluster_labels = clustering(T_A_dr, k=k, b=b, t=t)
    print("mini-batch k-means on T_B")
    T_B_representatives_idx, T_B_cluster_labels = clustering(T_B_dr, k=k, b=b, t=t)

    # 4) Compute graph matching between T_A_representatives and T_B_representatives
    alpha = 0.5
    max_iter1 = 100
    max_iter2 = 500
    corresponding_clusters = graph_matching(T_A[T_A_representatives_idx],
                                            T_B[T_B_representatives_idx],
                                            alpha=alpha, max_iter1=max_iter1,
                                            max_iter2=max_iter2)
    distance_clusters = distance_corresponding(T_A[T_A_representatives_idx],
                                               T_B[T_B_representatives_idx],
                                               corresponding_clusters)
    print("Mean distance between corresponding clusters: %s" % distance_clusters.mean())


    # 5) For each pair corresponding cluster, compute graph matching
    # between their streamlines
    print("Compute graph-matching between streamlines of corresponding clusters")
    correspondence_gm = -np.ones(len(T_A), dtype=np.int)
    for i in range(k):
        print("Graph matching between streamlines of corresponding clusters: cl_A=%s <-> cl_B=%s" % (i, corresponding_clusters[i]))
        cluster_A_idx = np.where(T_A_cluster_labels == i)[0]
        cluster_A = T_A[cluster_A_idx]
        cluster_B_idx = np.where(T_B_cluster_labels == corresponding_clusters[i])[0]
        cluster_B = T_B[cluster_B_idx]
        corresponding_streamlines = graph_matching_two_clusters(cluster_A,
                                                                cluster_B,
                                                                max_iter1=max_iter1,
                                                                max_iter2=max_iter2)

        tmp = corresponding_streamlines != -1
        correspondence_gm[cluster_A_idx[tmp]] = cluster_B_idx[corresponding_streamlines[tmp]]


    # 6) Filling the missing correspondences in T_A with the
    # corresponding streamline to their nearest neighbour in T_A
    print("Filling the missing correspondences in T_A with the corresponding to their nearest neighbour in T_A")
    correspondence = correspondence_gm.copy()
    T_A_corresponding_idx = np.where(correspondence != -1)[0]
    T_A_missing_idx = np.where(correspondence == -1)[0]
    T_A_corresponding_kdt = KDTree(T_A_dr[T_A_corresponding_idx])
    T_A_missing_NN = T_A_corresponding_kdt.query(T_A_dr[T_A_missing_idx], k=1, return_distance=False).squeeze()
    correspondence[T_A_missing_idx] = correspondence[T_A_corresponding_idx[T_A_missing_NN]]


    # 7) Compute the mean distance of corresponding streamlines, to
    # check the quality of the result
    distances = distance_corresponding(T_A, T_B, correspondence)
    print("Mean distance of corresponding streamlines: %s" % distances.mean())

    import matplotlib.pyplot as plt
    plt.interactive(True)
    plt.figure()
    plt.hist(distances, bins=50)
    plt.title("Distances between corresponding streamlines")
