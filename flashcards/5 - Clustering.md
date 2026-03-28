#flashcards/part1

## Unsupervised Learning: Clustering

What objective does K-Means minimize?
?
J = sum_k sum_{x in C_k} ||x - mu_k||^2, the total within-cluster variance. The algorithm alternates between assigning points to the nearest centroid and recomputing centroids as cluster means. Converges to a local minimum, not necessarily the global one.

How does K-Means++ improve on random initialization?
?
It chooses each new centroid with probability proportional to D(x)^2, the squared distance to the nearest existing centroid. This spreads initial centroids apart, reducing the chance of converging to a poor local minimum. It provides an O(log k) approximation guarantee.

What is the connection between K-Means and GMMs?
?
K-Means is a special case of EM on a GMM where all components have isotropic covariance (sigma*I) and sigma approaches zero. In this limit, soft probabilistic assignments (responsibilities) become hard 0/1 assignments, and the M-step reduces to computing cluster means.

Describe the E-step and M-step of EM for Gaussian Mixture Models.
?
E-step: compute responsibilities r_ik = pi_k * N(x_i | mu_k, Sigma_k) / sum_j pi_j * N(x_i | mu_j, Sigma_j). This is the posterior probability that point i belongs to component k. M-step: update pi_k (mixing weights), mu_k (means), and Sigma_k (covariances) using responsibilities as soft counts.

Why can GMMs have degenerate solutions?
?
If a component's mean lands exactly on a data point and its covariance shrinks to zero, the likelihood goes to infinity. This is a pathological maximum. Fixes include minimum covariance constraints, regularization, or careful initialization.

What are the four common linkage methods for hierarchical clustering?
?
Single: min distance between clusters (chains, sensitive to noise). Complete: max distance (compact clusters, sensitive to outliers). Average: mean pairwise distance (compromise). Ward's: minimizes total within-cluster variance increase (produces K-Means-like clusters).

How do you read a dendrogram to choose the number of clusters?
?
Look for the largest vertical gap between merges. Cut the dendrogram at that height. The number of vertical lines crossing the cut equals the number of clusters. Large gaps indicate natural cluster boundaries; small gaps suggest forced merges.

What does silhouette score measure, and how do you interpret it?
?
s(i) = (b(i) - a(i)) / max(a(i), b(i)), where a = average within-cluster distance, b = average distance to nearest other cluster. Range is [-1, +1]. Near +1: well clustered. Near 0: on a boundary between clusters. Negative: likely misassigned.

What are three methods for choosing K in K-Means?
?
1. Elbow method: plot J vs K, look for diminishing returns. 2. Silhouette score: pick K that maximizes average silhouette. 3. Gap statistic: compare log(J) to its expected value under a null reference distribution. Also: domain knowledge when the number of clusters has a natural meaning.
