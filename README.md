# graph_matching_tractograms
This is the code of the algorithm described in: Olivetti E, Sharmin N and Avesani P (2016) Alignment of Tractograms As Graph Matching. Front. Neurosci. 10:554. doi: 10.3389/fnins.2016.00554

This code takes two tractograms, T_A and T_B, as input and returns the correspondence between their streamlines. Technically, the algorithm returns a vector of streamlines IDs of T_B, such that the ID in position 'i' is the corresponding one of streamline with ID=i in T_A. The correspondence is computed by solving a graph-matching problem, as explained in the article above. As described in the article, even the fastest graph matching algorithm available today cannot address the size of a full tractogram. For this reason, the problem is solved in multiple steps:
1. Each tractogram is clustered in k=1000 clusters.
2. Graph-matching is done in order to find corresponding clusters across the two tractograms.
3. For each pair of corresponding clusters, e.g. c_A and c_B, graph-matching is executed on their streamlines in order to find the corresponding streamlines between c_A and c_B.

The code is built on the functionality available in other repositories:
- https://github.com/emanuele/dissimilarity   (dissimilarity representation)
- https://github.com/emanuele/minibatch_kmeans   (mini-batch k-means clustering)
- https://github.com/emanuele/DSPFP   (DSPFP graph-matching algorithm)

For sake of convenience, the necessary code from those repositories is copied in this repository, in order to provide a self-contained software.

Dependencies:
- numpy
- nibabel
- dipy
- sklearn
