#flashcards/part1

## Unsupervised Learning: Dimensionality Reduction

How does PCA work, and what are principal components?
?
PCA performs eigendecomposition of the covariance matrix S = (1/(n-1)) X^T X. The eigenvectors are the principal components (directions of maximum variance); the eigenvalues are the variance captured along each direction. Equivalently, it is the SVD of centered data: X = U Sigma V^T, where V's columns are the PCs.

How do you choose how many principal components to keep?
?
Scree plot: plot eigenvalues in decreasing order, look for an elbow. Variance threshold: retain enough components to explain 90-95% of total variance. The explained variance ratio for component j is lambda_j / sum(lambda_i).

Why must you standardize features before PCA?
?
PCA maximizes variance, so features with larger scales dominate the principal components. A feature measured in meters would dominate one measured in kilometers, even if the smaller-scale feature is more informative. Standardizing (zero mean, unit variance) puts all features on equal footing.

How does Kernel PCA extend standard PCA?
?
It applies the kernel trick to compute inner products in a high-dimensional feature space without explicit transformation. This allows PCA to capture nonlinear structure. Drawbacks: must choose a kernel, more expensive, and there is no simple inverse transform back to the original space.

What are the three things you should NOT interpret from a t-SNE plot?
?
1. Distances between clusters (they are not meaningful). 2. Cluster sizes (they are distorted). 3. Feature-space relationships (t-SNE only preserves local neighborhoods). t-SNE is for visualization only; do not use the embeddings as features for downstream models.

How does t-SNE work at a high level?
?
In high-D, compute pairwise Gaussian similarities (probability that points are neighbors). In low-D (2 or 3), compute pairwise Student-t similarities. Minimize KL divergence between the two distributions via gradient descent. The heavy-tailed Student-t in low-D prevents the crowding problem.

Why is UMAP generally preferred over t-SNE?
?
UMAP is faster, preserves more global structure, can embed into arbitrary dimensions (not just 2-3), and can be used for general dimensionality reduction, not just visualization. t-SNE is slower (O(n^2)), non-deterministic, and its embeddings should not be used as features.

How does Isolation Forest detect anomalies?
?
It builds random trees with random axis-aligned splits. Anomalies are points that can be isolated in fewer splits (shorter average path length), because they sit in sparse regions. Normal points require more splits because they are surrounded by similar points.

What is the connection between PCA and the bias-variance tradeoff?
?
Keeping more components preserves more information (lower bias) but retains noise (higher variance). Discarding low-variance components acts as denoising, increasing bias slightly but reducing variance. This is why PCA can improve downstream model performance even though it discards information.

What connects the covariance matrix, PCA, and LDA?
?
PCA eigendecomposes the total covariance matrix to find directions of maximum variance. LDA uses the within-class and between-class covariance matrices to find directions that maximize class separation. Both use the covariance matrix as their core object; they just optimize different objectives over it.
