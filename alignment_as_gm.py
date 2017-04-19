"""Alignment of tractograms as graph matching.

See: Olivetti E, Sharmin N and Avesani P (2016) Alignment of
Tractograms As Graph Matching. Front. Neurosci. 10:554. 
doi:10.3389/fnins.2016.00554

"""

import numpy as np
from nibabel import trackvis
from dissimilarity import compute_dissimilarity
from kmeans import mini_batch_kmeans, compute_labels, compute_centroids
from sklearn.neighbors import KDTree
from dipy.tracking.distances import bundles_distances_mam
from dipy.tracking.streamlinespeed import length
from DSPFP import DSPFP_faster, greedy_assignment
from joblib import  Parallel, delayed


def distance_corresponding(A, B, correspondence):
    """Distance between streamlines in set A and the corresponding ones in
    B. The vector 'correspondence' has in position 'i' the ID of the
    streamline in B corresponding to A[i].
    """
    return np.array([bundles_distances_mam([A[i]], [B[correspondence[i]]]) for i in range(len(A))]).squeeze()


def graph_matching(S_A, S_B, alpha=0.5, max_iter1=100, max_iter2=100,
                   initialization='NN', similarity='1/x',
                   epsilon=1.0e-8, verbose=True):
    """Wrapper of the DSPFP algorithm to deal with streamlines. This code
    add kinds of initializations meaningful for streamlines and
    conversion from distances to similarities.
    """
    if verbose:
        print("Compute graph matching between streamlines.")
        print("Compute the distance matrix between streamlines in each set")

    dm_A = bundles_distances_mam(S_A, S_A)
    dm_B = bundles_distances_mam(S_B, S_B)

    if initialization == 'NN':
        tmp = bundles_distances_mam(S_A, S_B)
        X_init = (tmp.max() + epsilon) / (tmp + epsilon)
    elif initialization == 'random':
        tmp = np.random.uniform(shape=(len(S_A), len(S_B)))
        X_init = (tmp.max() + epsilon) / (tmp + epsilon)
    else:
        # flat initialization, default of DSPFP
        X_init = None

    # Wheter to use distances or similarities and, in case, which
    # similarity function
    if similarity == '1/x':
        sm_A = (dm_A.max() + epsilon) / (dm_A + epsilon)
        sm_B = (dm_B.max() + epsilon) / (dm_B + epsilon)
    elif similarity == 'exp(-x)':
        tmp = np.median(dm_A)
        sm_A = np.exp(-dm_A * dm_A / tmp * tmp)
        tmp = np.median(dm_B)
        sm_B = np.exp(-dm_B * dm_B / tmp * tmp)
    else:
        # Don't use similarity
        sm_A = dm_A
        sm_B = dm_B

    if verbose:
        print("Compute graph-matching via DSPFP")

    X = DSPFP_faster(sm_A, sm_B, alpha=alpha,
                     max_iter1=max_iter1,
                     max_iter2=max_iter2,
                     X=X_init, verbose=verbose)
    corresponding_streamlines = greedy_assignment(X).argmax(1)
    return corresponding_streamlines


if __name__ == '__main__':
    np.random.seed(2)

    # T_A_filename = 'data/HCP_subject124422_100Kseeds/tracks_dti_100K.trk'
    # T_B_filename = 'data/HCP_subject124422_100Kseeds/tracks_dti_100K.trk'

    T_A_filename = 'data2/100307/Tractogram/tractogram_b1k_1.25mm_csd_wm_mask_eudx1M.trk'
    T_B_filename = 'data2/100408/Tractogram/tractogram_b1k_1.25mm_csd_wm_mask_eudx1M.trk'
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

    if T_A_filename == T_B_filename:
        # 1.2) Permuting the order of T_B and creating ground truth:
        print("Permuting the order of T_B and creating ground truth.")
        T_B_random_idx = np.random.permutation(len(T_B))
        correspondence_ground_truth = np.argsort(T_B_random_idx)
        T_B = T_B[T_B_random_idx]
        assert((T_A[0] == T_B[correspondence_ground_truth[0]]).all())

    # 2) Compute the dissimilarity representation of T_A and T_B
    print("Compute the dissimilarity representation of T_A")
    T_A_dr, prototypes_A = compute_dissimilarity(T_A)
    print("Compute the dissimilarity representation of T_B")
    T_B_dr, prototypes_B = compute_dissimilarity(T_B)

    # 3) Compute the k-means clustering of T_A and T_B
    k = 1000
    print("Compute the k-means clustering of T_A and T_B, k=%s" % k)
    # 3.1) Generate k initial random centers
    T_A_centers = T_A_dr[np.random.permutation(len(T_A))[:k]]
    T_B_centers = T_B_dr[np.random.permutation(len(T_B))[:k]]

    # 3.2) Improve the k centers with mini_batch_kmeans
    b = 100  # mini-batch size
    t = 100  # number of iterations
    print("MiniBatchKMeans on T_A")
    T_A_centers = mini_batch_kmeans(T_A_dr, T_A_centers, b=b, t=t)
    print("MiniBatchKMeans on T_B")
    T_B_centers = mini_batch_kmeans(T_B_dr, T_B_centers, b=b, t=t)

    # 3.3) Assign the cluster labels, i.e. the nearest center, for
    # each streamline.
    T_A_cluster_labels = compute_labels(T_A_dr, T_A_centers)
    T_B_cluster_labels = compute_labels(T_B_dr, T_B_centers)

    # 3.4) Compute a cluster representive, for each cluster
    T_A_representatives_idx = compute_centroids(T_A_dr, T_A_centers)
    T_B_representatives_idx = compute_centroids(T_B_dr, T_B_centers)

    # 4) Compute graph matching between T_A_representatives and T_B_representatives
    print("Compute graph matching between T_A_representatives and T_B_representatives.")

    # 4.1) Compute the distance matrix between T_A_representatives and T_B_representatives
    print("Compute the distance matrix between T_A_representatives")
    dm_TAr = bundles_distances_mam(T_A[T_A_representatives_idx], T_A[T_A_representatives_idx])
    print("Compute the distance matrix between T_B_representatives")
    dm_TBr = bundles_distances_mam(T_B[T_B_representatives_idx], T_B[T_B_representatives_idx])

    # tmp = np.median(dm_TAr)
    # sm_TAr = np.exp(-dm_TAr * dm_TAr / tmp * tmp)
    # tmp = np.median(dm_TBr)
    # sm_TBr = np.exp(-dm_TBr * dm_TBr / tmp * tmp)
    epsilon = 1.0e-8
    sm_TAr = (dm_TAr.max() + epsilon) / (dm_TAr + epsilon)
    sm_TBr = (dm_TBr.max() + epsilon) / (dm_TBr + epsilon)

    # 4.2) Compute graph-matching between representatives
    print("Compute graph-matching between representatives")
    alpha = 0.5
    max_iter1 = 100
    max_iter2 = 100
    tmp = bundles_distances_mam(T_A[T_A_representatives_idx], T_B[T_B_representatives_idx])
    X_init_NN = (tmp.max() + epsilon) / (tmp + epsilon)
    X_clusters = DSPFP_faster(sm_TAr, sm_TBr, alpha=alpha,
                              max_iter1=max_iter1,
                              max_iter2=max_iter2,
                              X=X_init_NN)
    corresponding_clusters = greedy_assignment(X_clusters).argmax(1)

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
        dm_clA = bundles_distances_mam(cluster_A, cluster_A)
        dm_clB = bundles_distances_mam(cluster_B, cluster_B)
        sm_clA = (dm_clA.max() + epsilon) / (dm_clA + epsilon)
        sm_clB = (dm_clB.max() + epsilon) / (dm_clB + epsilon)
        tmp = bundles_distances_mam(cluster_A, cluster_B)
        X_init_NN = (tmp.max() + epsilon) / (tmp + epsilon)
        if len(cluster_A) >= len(cluster_B):
            X = DSPFP_faster(sm_clA, sm_clB, alpha=alpha, verbose=False,
                             X=X_init_NN)
            corresponding_streamlines = greedy_assignment(X).argmax(0)
        else:
            X = DSPFP_faster(sm_clB, sm_clA, alpha=alpha, verbose=False,
                             X=X_init_NN.T)
            corresponding_streamlines = greedy_assignment(X).argmax(1)
            
        correspondence_gm[cluster_A_idx[corresponding_streamlines]] = cluster_B_idx


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
