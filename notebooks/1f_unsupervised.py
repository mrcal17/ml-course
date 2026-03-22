import marimo

app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.datasets import make_blobs, load_digits
    return (
        KMeans,
        PCA,
        StandardScaler,
        load_digits,
        make_blobs,
        np,
        plt,
        silhouette_score,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Unsupervised Learning

    Up to this point, every problem you've worked on has had the same structure: here are inputs $X$, here are labels $y$, learn the mapping. Supervised learning. The labels tell the algorithm what "right" looks like, and the loss function measures how far off you are. It's a well-defined optimization problem.

    Now we remove the labels entirely. You have a dataset $\{x_1, x_2, \ldots, x_N\}$ — no $y$, no "right answer," no explicit objective that says "you got it." The goal is to discover *structure* in the data: clusters, low-dimensional representations, density patterns, latent variables. This is unsupervised learning, and it is fundamentally harder and more ambiguous than the supervised case.

    Why harder? Because there's no single, universally agreed-upon loss function. In classification, you can measure accuracy. In regression, you can measure MSE. But what does it mean to "correctly" cluster a dataset? The answer depends on what you're looking for, which means unsupervised learning requires more judgment from the practitioner. The algorithms give you tools; you supply the interpretation.

    The two main families of tasks here are **dimensionality reduction** (represent high-dimensional data in fewer dimensions while preserving important structure) and **clustering** (group similar data points together). We'll also touch on density estimation and anomaly detection, which are closely related.

    For a broad overview, see [ISLR Ch 12](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf) and [ESL Ch 14](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 1. Principal Component Analysis (PCA)

    PCA is the most important dimensionality reduction technique, and it's one of those rare algorithms where the math is clean enough that you can actually understand *exactly* what it does and *why* it works. You already have all the tools — eigendecomposition and SVD from your linear algebra module.

    ### Two Equivalent Views

    PCA can be motivated from two completely different angles, and the fact that they give the same answer is part of what makes PCA elegant.

    **View 1: Maximum Variance.** You have data in $\mathbb{R}^D$. You want to find a direction $w_1 \in \mathbb{R}^D$ ($\|w_1\| = 1$) such that projecting the data onto $w_1$ retains as much variance as possible. Then find $w_2$ orthogonal to $w_1$ that captures the most remaining variance. And so on. The idea: variance is information, and you want to preserve as much as possible.

    **View 2: Minimum Reconstruction Error.** You want to find a $K$-dimensional linear subspace such that projecting the data onto it and then reconstructing back into $\mathbb{R}^D$ introduces the least error (measured by squared distance). You're looking for the best low-rank approximation to the data.

    These two formulations lead to the *exact same solution*. See [MML 10.2-10.3](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) for both derivations side by side.
    """)
    return


@app.cell
def _(mo):
    mo.image(src="../animations/rendered/PCAVarianceDirections.gif")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Derivation from Maximum Variance

    Assume the data is centered (mean-subtracted). The sample covariance matrix is:

    $$S = \frac{1}{N-1} X^\top X$$

    where $X$ is the $N \times D$ centered data matrix. The variance of the data projected onto a unit vector $w$ is:

    $$\text{Var} = w^\top S w$$

    We want to maximize $w^\top S w$ subject to $\|w\| = 1$. Using a Lagrange multiplier:

    $$\mathcal{L}(w, \lambda) = w^\top S w - \lambda(w^\top w - 1)$$

    Setting $\nabla_w \mathcal{L} = 0$:

    $$2Sw - 2\lambda w = 0 \implies Sw = \lambda w$$

    This is an eigenvalue equation. The direction that maximizes variance is the eigenvector of $S$ corresponding to the largest eigenvalue. The second principal component is the eigenvector with the second-largest eigenvalue, and so on.

    The eigenvalue $\lambda_k$ *is* the variance captured by the $k$-th principal component. This is why eigendecomposition isn't just a linear algebra curiosity — it's the core computational engine of PCA.

    ### Connection to SVD

    In practice, you almost never compute PCA by forming $S = X^\top X$ and then eigendecomposing it. This is numerically unstable for large or ill-conditioned data. Instead, you use the SVD of $X$ directly.

    The SVD gives $X = U \Sigma V^\top$, where $V$ contains the right singular vectors. Plugging in:

    $$S = \frac{1}{N-1} X^\top X = \frac{1}{N-1} V \Sigma^2 V^\top$$

    So the columns of $V$ are the eigenvectors of $S$ (i.e., the principal components), and the squared singular values $\sigma_k^2/(N-1)$ are the eigenvalues (i.e., the variances). The SVD computes PCA without ever forming $X^\top X$, which is both faster and more numerically stable for high-dimensional data.

    See [MML 10.4](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) for the SVD-based derivation.
    """)
    return


@app.cell
def _(np):
    # PCA from scratch: centering and covariance matrix
    # Generate simple 2D data to see each step
    rng_pca = np.random.RandomState(42)
    X_demo = rng_pca.randn(100, 2) @ np.array([[2, 0.8], [0.8, 1]])

    # Step 1: Center the data (subtract mean)
    X_centered = X_demo - X_demo.mean(axis=0)

    # Step 2: Covariance matrix S = X^T X / (N-1)
    N = X_centered.shape[0]
    S = X_centered.T @ X_centered / (N - 1)
    print("Covariance matrix S:")
    print(S)
    return (S, X_centered, X_demo, rng_pca)


@app.cell
def _(S, np):
    # PCA from scratch: eigendecomposition of the covariance matrix
    # Sw = lambda * w  =>  eigenvectors are principal directions
    eigenvalues, eigenvectors = np.linalg.eigh(S)

    # eigh returns in ascending order; flip to descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    print("Eigenvalues (variance per component):", eigenvalues)
    print("Explained variance ratio:", eigenvalues / eigenvalues.sum())
    print("PC1 direction:", eigenvectors[:, 0])
    return (eigenvalues, eigenvectors)


@app.cell
def _(X_centered, eigenvalues, eigenvectors, np, plt):
    # PCA from scratch: project data onto principal components
    # Project: Z = X_centered @ V  (each column of V is a PC direction)
    Z_pca = X_centered @ eigenvectors  # N x D projected coordinates

    # Reconstruct using only 1st component: X_hat = Z[:, :1] @ V[:, :1].T
    X_hat_1pc = Z_pca[:, :1] @ eigenvectors[:, :1].T

    fig_pca_demo, ax_pca_demo = plt.subplots(figsize=(6, 5))
    ax_pca_demo.scatter(X_centered[:, 0], X_centered[:, 1], alpha=0.4, s=15, label="Original")
    ax_pca_demo.scatter(X_hat_1pc[:, 0], X_hat_1pc[:, 1], alpha=0.4, s=15, label="1-PC reconstruction")
    # Draw PC directions scaled by sqrt(eigenvalue)
    for i in range(2):
        ax_pca_demo.arrow(0, 0, eigenvectors[0, i]*np.sqrt(eigenvalues[i]),
                          eigenvectors[1, i]*np.sqrt(eigenvalues[i]),
                          head_width=0.1, color='red', linewidth=2)
    ax_pca_demo.set_aspect('equal')
    ax_pca_demo.legend()
    ax_pca_demo.set_title("PCA from scratch: data, PCs, and 1-PC reconstruction")
    fig_pca_demo
    return (Z_pca,)


@app.cell
def _(X_centered, eigenvalues, eigenvectors, np):
    # PCA via SVD (the numerically stable way)
    # X = U Sigma V^T  =>  columns of V are eigenvectors of X^T X
    U_svd, sigma_svd, Vt_svd = np.linalg.svd(X_centered, full_matrices=False)

    # Eigenvalues from singular values: lambda_k = sigma_k^2 / (N-1)
    N_svd = X_centered.shape[0]
    eigenvalues_svd = sigma_svd**2 / (N_svd - 1)

    print("Eigenvalues (from eigh): ", eigenvalues)
    print("Eigenvalues (from SVD):  ", eigenvalues_svd)
    print("Match:", np.allclose(eigenvalues, eigenvalues_svd))

    # V from SVD should match eigenvectors (up to sign)
    print("PC1 from SVD:", Vt_svd[0])
    print("PC1 from eigh:", eigenvectors[:, 0])
    return (Vt_svd,)


@app.cell
def _(mo):
    mo.md(r"""
    ### How Many Components to Keep?

    This is a modeling decision, not a mathematical one. Common heuristics:

    - **Explained variance ratio.** The fraction of total variance captured by the first $K$ components is $\sum_{k=1}^K \lambda_k / \sum_{k=1}^D \lambda_k$. A common threshold is 90-95%.
    - **Scree plot.** Plot the eigenvalues in decreasing order. Look for an "elbow" — a point where the eigenvalues drop off sharply. Keep the components before the elbow.
    - **Domain knowledge.** Sometimes you know you want 2D or 3D for visualization, or you have a downstream task with specific constraints.

    None of these are rigorous. PCA doesn't have a built-in model selection criterion the way regularized models have cross-validation. That's part of the unsupervised ambiguity.

    ### Limitations

    PCA finds **linear** structure only. If your data lies on a curved manifold (like a Swiss roll), PCA will miss it entirely. The projections will mix together points that are far apart on the manifold.

    **Kernel PCA** addresses this by applying the kernel trick: implicitly map data into a high-dimensional feature space using a kernel function $k(x_i, x_j)$, then perform PCA in that space. You never compute the mapping explicitly — you only need the kernel matrix $K_{ij} = k(x_i, x_j)$. This can capture nonlinear structure, but it adds the complexity of choosing and tuning a kernel.

    See [Bishop 12.3](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) for kernel PCA, and [ISLR 12.2](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf) for a gentler treatment of standard PCA.
    """)
    return


@app.cell
def _(np):
    # Explained variance: choosing how many components to keep
    # Simulate eigenvalues from a 10D dataset
    eigenvalues_example = np.array([5.2, 3.1, 1.8, 0.9, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01])
    total_var = eigenvalues_example.sum()

    # Cumulative explained variance ratio
    cumulative_ratio = np.cumsum(eigenvalues_example) / total_var
    # Find K for 95% threshold
    K_95 = np.argmax(cumulative_ratio >= 0.95) + 1
    print(f"Eigenvalues: {eigenvalues_example}")
    print(f"Cumulative ratios: {np.round(cumulative_ratio, 3)}")
    print(f"Components for 95% variance: {K_95}")
    return (K_95,)


@app.cell
def _(PCA, StandardScaler, load_digits, np, plt):
    # PCA on the digits dataset
    digits = load_digits()
    X_digits = digits.data
    y_digits = digits.target

    # Always standardize before PCA
    X_scaled_digits = StandardScaler().fit_transform(X_digits)

    pca_full = PCA()
    X_pca_full = pca_full.fit_transform(X_scaled_digits)

    # Scree plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(range(1, len(pca_full.explained_variance_ratio_) + 1),
             np.cumsum(pca_full.explained_variance_ratio_), 'o-', markersize=3)
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel('Cumulative Explained Variance')
    ax1.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
    n_95 = np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= 0.95) + 1
    ax1.axvline(x=n_95, color='g', linestyle='--', alpha=0.5, label=f'{n_95} components for 95%')
    ax1.legend()
    ax1.set_title('PCA Explained Variance (Digits Dataset)')

    # 2D projection colored by digit
    ax2.scatter(X_pca_full[:, 0], X_pca_full[:, 1], c=y_digits, cmap='tab10', s=5, alpha=0.7)
    ax2.set_xlabel('PC 1')
    ax2.set_ylabel('PC 2')
    ax2.set_title('First Two Principal Components')
    plt.colorbar(ax2.collections[0], ax=ax2, label='Digit')

    plt.tight_layout()
    fig
    return (X_scaled_digits,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Interactive: PCA Components and Reconstruction

    Use the slider to control how many principal components are kept. Watch how the explained variance ratio changes and see the reconstruction quality.
    """)
    return


@app.cell
def _(mo):
    pca_slider = mo.ui.slider(start=1, stop=10, step=1, value=2, label="PCA components")
    pca_slider
    return (pca_slider,)


@app.cell
def _(PCA, X_scaled_digits, load_digits, np, pca_slider, plt):
    # PCA with selected number of components
    pca_k = PCA(n_components=pca_slider.value)
    X_reduced = pca_k.fit_transform(X_scaled_digits)
    X_reconstructed = pca_k.inverse_transform(X_reduced)

    total_var = pca_k.explained_variance_ratio_.sum()

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle(f"PCA with {pca_slider.value} components — {total_var*100:.1f}% variance explained\n"
                 f"Top: Original | Bottom: Reconstructed", fontsize=12)

    digits_data = load_digits()
    # Show 5 sample digits
    indices = [0, 100, 200, 500, 1000]
    scaler_mean = X_scaled_digits.mean(axis=0)
    scaler_std = X_scaled_digits.std(axis=0)

    for i, idx in enumerate(indices):
        axes[0, i].imshow(digits_data.images[idx], cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(f"Label: {digits_data.target[idx]}")

        # Reconstruct (undo standardization for display)
        recon = X_reconstructed[idx] * scaler_std + scaler_mean
        recon_img = recon.reshape(8, 8)
        axes[1, i].imshow(recon_img, cmap='gray')
        axes[1, i].axis('off')
        mse = np.mean((digits_data.data[idx] - recon) ** 2)
        axes[1, i].set_title(f"MSE: {mse:.1f}")

    plt.tight_layout()
    fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 2. K-Means Clustering

    K-means is the simplest and most widely used clustering algorithm. It partitions $N$ data points into $K$ clusters by minimizing the total within-cluster sum of squares.

    ### The Algorithm

    The objective (also called *inertia*) is:

    $$J = \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$

    where $C_k$ is the set of points assigned to cluster $k$ and $\mu_k$ is the centroid of cluster $k$.

    ```
    Algorithm: K-Means
    ─────────────────────
    Input: data {x_1, ..., x_N}, number of clusters K
    Initialize centroids mu_1, ..., mu_K
    repeat:
        Assignment step:
            for each x_i: assign to cluster k* = argmin_k ||x_i - mu_k||^2
        Update step:
            for each k: mu_k <- mean of all points assigned to cluster k
    until assignments don't change
    return assignments, centroids
    ```

    **Convergence guarantee.** Each step either decreases $J$ or leaves it unchanged. The assignment step decreases $J$ by assigning each point to its nearest centroid. The update step decreases $J$ because the mean minimizes sum-of-squared distances (you proved this flavor of result in your statistical estimation module). Since $J$ is bounded below by 0 and strictly decreases at each step, the algorithm must converge in a finite number of iterations.

    But convergence is to a **local** minimum, not necessarily the global one. K-means is non-convex.
    """)
    return


@app.cell
def _(np):
    # K-Means from scratch
    def kmeans_scratch(X, K, max_iters=100, seed=42):
        rng_km = np.random.RandomState(seed)
        N_km = X.shape[0]

        # Initialize: pick K random data points as centroids
        centroids_km = X[rng_km.choice(N_km, K, replace=False)]

        for iteration in range(max_iters):
            # Assignment step: argmin_k ||x_i - mu_k||^2
            # Compute distances: (N, K) matrix
            dists = np.linalg.norm(X[:, None] - centroids_km[None, :], axis=2)
            labels_km = np.argmin(dists, axis=1)

            # Update step: mu_k = mean of points in cluster k
            new_centroids = np.array([X[labels_km == k].mean(axis=0) for k in range(K)])

            # Check convergence
            if np.allclose(new_centroids, centroids_km):
                break
            centroids_km = new_centroids

        # Inertia: J = sum of squared distances to assigned centroid
        inertia = sum(np.sum((X[labels_km == k] - centroids_km[k])**2) for k in range(K))
        return labels_km, centroids_km, inertia, iteration + 1

    return (kmeans_scratch,)


@app.cell
def _(kmeans_scratch, make_blobs, np, plt):
    # Run our from-scratch K-Means on synthetic data
    X_ks, y_ks_true = make_blobs(n_samples=200, centers=3, cluster_std=1.0, random_state=0)

    labels_ks, centroids_ks, inertia_ks, iters_ks = kmeans_scratch(X_ks, K=3)

    fig_ks, ax_ks = plt.subplots(figsize=(6, 5))
    for c in range(3):
        mask = labels_ks == c
        ax_ks.scatter(X_ks[mask, 0], X_ks[mask, 1], s=20, alpha=0.7, label=f"Cluster {c}")
    ax_ks.scatter(centroids_ks[:, 0], centroids_ks[:, 1], c='black', marker='X',
                  s=200, edgecolors='white', linewidths=2, label='Centroids')
    ax_ks.set_title(f"K-Means from scratch | Inertia={inertia_ks:.1f} | {iters_ks} iterations")
    ax_ks.legend(fontsize=8)
    fig_ks
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Initialization Matters: K-Means++

    Bad initialization can lead to terrible clusterings. K-means++ is a principled initialization that spreads out the initial centroids:

    1. Choose the first centroid uniformly at random from the data.
    2. For each subsequent centroid, choose a point with probability proportional to $D(x)^2$, where $D(x)$ is the distance from $x$ to the nearest already-chosen centroid.

    This gives an $O(\log K)$-competitive approximation to the optimal clustering. In practice, it almost always gives much better results than random initialization. Scikit-learn uses k-means++ by default.

    ### Choosing K

    - **Elbow method.** Plot $J$ vs. $K$. Look for a bend — the point where adding more clusters stops giving much improvement. This is subjective.
    - **Silhouette score.** For each point, compute $s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$, where $a(i)$ is the mean distance to points in the same cluster and $b(i)$ is the mean distance to the nearest other cluster. Ranges from $-1$ (badly clustered) to $+1$ (well clustered). Choose $K$ that maximizes the average silhouette score.
    - **Gap statistic.** Compares the within-cluster dispersion to what you'd expect under a null reference distribution (uniform). Choose the smallest $K$ where the gap is within one standard error of the maximum.

    ### Limitations

    K-means assumes **spherical, equal-size clusters**. It uses Euclidean distance, so it can't detect elongated or irregularly shaped clusters. It's sensitive to feature scaling (always standardize). And you have to specify $K$ — which, in unsupervised learning, is a parameter you often don't know.

    See [ISLR 12.4.1](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf) and [ESL 14.3.6](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf) for K-means, and [Bishop 9.1](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) for the connection between K-means and EM.
    """)
    return


@app.cell
def _(kmeans_scratch, make_blobs, np, plt):
    # Silhouette score from scratch for one K value
    X_sil, _ = make_blobs(n_samples=150, centers=3, cluster_std=1.0, random_state=42)
    labels_sil, _, _, _ = kmeans_scratch(X_sil, K=3, seed=42)

    def silhouette_scratch(X, labels):
        """Compute mean silhouette score: s(i) = (b(i) - a(i)) / max(a(i), b(i))"""
        n = len(X)
        sil_vals = np.zeros(n)
        for i in range(n):
            # a(i): mean distance to same-cluster points
            same = labels == labels[i]
            same[i] = False
            a_i = np.mean(np.linalg.norm(X[same] - X[i], axis=1)) if same.sum() > 0 else 0
            # b(i): min over other clusters of mean distance
            b_i = np.inf
            for k in set(labels):
                if k == labels[i]:
                    continue
                other = labels == k
                b_i = min(b_i, np.mean(np.linalg.norm(X[other] - X[i], axis=1)))
            sil_vals[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0
        return np.mean(sil_vals)

    print(f"Silhouette score (from scratch): {silhouette_scratch(X_sil, labels_sil):.3f}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Interactive: K-Means Clustering

    Use the slider to select different values of $k$. Watch how the clustering changes and how the data points are assigned to different clusters.
    """)
    return


@app.cell
def _(mo):
    k_slider = mo.ui.slider(start=2, stop=10, step=1, value=3, label="Number of clusters k")
    k_slider
    return (k_slider,)


@app.cell
def _(KMeans, k_slider, make_blobs, np, plt, silhouette_score):
    # Generate synthetic data with known structure
    X_km, y_km_true = make_blobs(
        n_samples=300, centers=4, cluster_std=1.0, random_state=42
    )

    # Run K-means with selected K
    km = KMeans(n_clusters=k_slider.value, init='k-means++', n_init=10, random_state=42)
    labels_km = km.fit_predict(X_km)
    centroids = km.cluster_centers_

    sil_score = silhouette_score(X_km, labels_km)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Cluster plot
    for c in range(k_slider.value):
        mask = labels_km == c
        ax1.scatter(X_km[mask, 0], X_km[mask, 1], s=20, alpha=0.7, label=f'Cluster {c}')
    ax1.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=200,
                edgecolors='white', linewidths=2, label='Centroids')
    ax1.set_title(f"K-Means with k={k_slider.value}\n"
                  f"Silhouette: {sil_score:.3f} | Inertia: {km.inertia_:.0f}")
    ax1.legend(fontsize=8)
    ax1.set_xlabel("$x_1$")
    ax1.set_ylabel("$x_2$")

    # Elbow + Silhouette across K values
    K_range = range(2, 11)
    inertias = []
    silhouettes = []
    for k_val in K_range:
        km_temp = KMeans(n_clusters=k_val, init='k-means++', n_init=10, random_state=42)
        labs = km_temp.fit_predict(X_km)
        inertias.append(km_temp.inertia_)
        silhouettes.append(silhouette_score(X_km, labs))

    ax2_twin = ax2.twinx()
    l1 = ax2.plot(list(K_range), inertias, 'bo-', label='Inertia')
    l2 = ax2_twin.plot(list(K_range), silhouettes, 'rs-', label='Silhouette')
    ax2.axvline(x=k_slider.value, color='green', linestyle='--', alpha=0.5, label=f'Current k={k_slider.value}')
    ax2.set_xlabel('K')
    ax2.set_ylabel('Inertia', color='b')
    ax2_twin.set_ylabel('Silhouette Score', color='r')
    ax2.set_title('Elbow & Silhouette Analysis')
    lines = l1 + l2
    labels_legend = [l.get_label() for l in lines]
    ax2.legend(lines, labels_legend, fontsize=8)

    plt.tight_layout()
    fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 3. Hierarchical Clustering

    Hierarchical clustering builds a tree (a *dendrogram*) of nested clusters instead of producing a flat partition. The most common variant is **agglomerative** (bottom-up): start with each point as its own cluster, then iteratively merge the two closest clusters until everything is in one cluster.

    ### Linkage Criteria

    "Distance between two clusters" isn't well-defined unless you specify a **linkage** rule:

    | Linkage | Distance between clusters $A$ and $B$ | Character |
    |---|---|---|
    | **Single** | $\min_{a \in A, b \in B} d(a, b)$ | Can find elongated/irregular clusters; prone to "chaining" |
    | **Complete** | $\max_{a \in A, b \in B} d(a, b)$ | Prefers compact clusters; sensitive to outliers |
    | **Average** | $\frac{1}{|A||B|} \sum_{a \in A} \sum_{b \in B} d(a,b)$ | Compromise between single and complete |
    | **Ward's** | Increase in total within-cluster variance after merging | Tends to produce equal-size clusters; most similar to K-means |

    Ward's linkage is the default in many implementations and usually gives the most interpretable results for "blob-like" data.

    ### Dendrograms

    The dendrogram is the main output. The x-axis shows individual data points (or clusters); the y-axis shows the distance at which merges happen. To get $K$ clusters, you "cut" the dendrogram at an appropriate height — drawing a horizontal line and counting the number of vertical lines it crosses.

    Key things to look for: large jumps in merge distance (suggesting natural cluster boundaries), and the overall shape (balanced trees suggest roughly equal-size clusters; lopsided trees suggest chaining or outliers).

    ### Tradeoffs

    **Pros:** No need to specify $K$ upfront — the dendrogram reveals the full hierarchy, and you choose where to cut. Reveals nested structure (sub-clusters within clusters). Works with any distance metric.

    **Cons:** Naive implementation is $O(N^3)$ time and $O(N^2)$ space (you need the full distance matrix). Not feasible for very large datasets. The result is sensitive to the choice of linkage and distance metric, and there's no principled way to choose these.

    See [ISLR 12.4.2](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf) and [ESL 14.3.12](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf) for hierarchical clustering.
    """)
    return


@app.cell
def _(make_blobs, plt):
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

    X_hc, _ = make_blobs(n_samples=50, centers=3, cluster_std=1.0, random_state=42)

    # Compute linkage matrix (Ward's method)
    Z = linkage(X_hc, method='ward')

    # Plot dendrogram
    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(Z, truncate_mode='lastp', p=30, leaf_rotation=90, ax=ax)
    ax.set_xlabel('Cluster Size')
    ax.set_ylabel('Distance')
    ax.set_title("Hierarchical Clustering Dendrogram (Ward's Linkage)")
    plt.tight_layout()
    fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 4. Gaussian Mixture Models (GMMs) and the EM Algorithm

    This is where unsupervised learning gets genuinely deep. GMMs introduce the idea of **latent variables** — unobserved quantities that explain the data — and the **EM algorithm**, which is *the* general-purpose tool for fitting models with latent variables.

    ### From K-Means to GMMs

    K-means makes **hard** assignments: each point belongs to exactly one cluster. GMMs make **soft** assignments: each point has a *probability* of belonging to each cluster. K-means is, in fact, a special case of GMMs — you get K-means by taking the EM algorithm for GMMs and fixing all covariance matrices to $\sigma^2 I$ and letting $\sigma^2 \to 0$.

    ### The Generative Model

    A GMM assumes the data was generated by the following process:

    1. Pick a component $k$ with probability $\pi_k$ (the **mixing weight**, where $\sum_k \pi_k = 1$).
    2. Sample $x$ from a Gaussian $\mathcal{N}(\mu_k, \Sigma_k)$.

    The resulting density is:

    $$p(x) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k)$$

    The parameters to estimate are: the means $\mu_k$, covariance matrices $\Sigma_k$, and mixing weights $\pi_k$.

    The problem: you can't just maximize the log-likelihood $\sum_i \log p(x_i)$ directly because the log of a sum doesn't decompose nicely. This is where EM comes in.

    ### The EM Algorithm

    EM alternates between two steps:

    **E-step (Expectation).** Compute the *responsibilities* — the posterior probability that component $k$ generated point $x_i$:

    $$r_{ik} = \frac{\pi_k \, \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \, \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}$$

    This is just Bayes' theorem. The responsibility $r_{ik}$ is the soft assignment of point $i$ to cluster $k$.

    **M-step (Maximization).** Update the parameters using the responsibilities as weights:

    $$\mu_k = \frac{\sum_i r_{ik} \, x_i}{\sum_i r_{ik}}, \qquad \Sigma_k = \frac{\sum_i r_{ik} \, (x_i - \mu_k)(x_i - \mu_k)^\top}{\sum_i r_{ik}}, \qquad \pi_k = \frac{\sum_i r_{ik}}{N}$$

    Repeat until convergence.
    """)
    return


@app.cell
def _(np):
    # EM for GMM from scratch — helper: multivariate Gaussian PDF
    def gaussian_pdf(X, mu, Sigma):
        """N(x | mu, Sigma) for each row of X. Returns (N,) array."""
        D = X.shape[1]
        diff = X - mu  # (N, D)
        Sigma_inv = np.linalg.inv(Sigma)
        # Mahalanobis: (x-mu)^T Sigma^{-1} (x-mu) for each row
        maha = np.sum(diff @ Sigma_inv * diff, axis=1)
        log_norm = -0.5 * (D * np.log(2 * np.pi) + np.log(np.linalg.det(Sigma)))
        return np.exp(log_norm - 0.5 * maha)

    return (gaussian_pdf,)


@app.cell
def _(gaussian_pdf, np):
    # EM for GMM from scratch — full algorithm
    def em_gmm(X, K, max_iters=100, seed=42):
        rng_em = np.random.RandomState(seed)
        N_em, D_em = X.shape

        # Initialize: random means from data, identity covariances, uniform weights
        idx_em = rng_em.choice(N_em, K, replace=False)
        mus = X[idx_em].copy()
        Sigmas = np.array([np.eye(D_em) for _ in range(K)])
        pis = np.ones(K) / K

        for _ in range(max_iters):
            # --- E-step: compute responsibilities r_ik = pi_k * N(x_i|mu_k,Sig_k) / sum_j ---
            resp = np.zeros((N_em, K))
            for k in range(K):
                resp[:, k] = pis[k] * gaussian_pdf(X, mus[k], Sigmas[k])
            resp /= resp.sum(axis=1, keepdims=True)  # normalize rows

            # --- M-step: update parameters using responsibilities as weights ---
            N_k = resp.sum(axis=0)  # effective number of points per cluster
            for k in range(K):
                mus[k] = (resp[:, k:k+1] * X).sum(axis=0) / N_k[k]
                diff = X - mus[k]
                Sigmas[k] = (resp[:, k:k+1] * diff).T @ diff / N_k[k]
            pis = N_k / N_em

        labels_em = np.argmax(resp, axis=1)
        return mus, Sigmas, pis, resp, labels_em

    return (em_gmm,)


@app.cell
def _(em_gmm, make_blobs, plt):
    # Test our EM-GMM on synthetic data
    X_em_test, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=0)
    mus_fit, Sigmas_fit, pis_fit, resp_fit, labels_em_fit = em_gmm(X_em_test, K=3)

    print("Recovered mixing weights:", pis_fit.round(3))
    print("Recovered means:\n", mus_fit.round(2))

    fig_em, ax_em = plt.subplots(figsize=(6, 5))
    ax_em.scatter(X_em_test[:, 0], X_em_test[:, 1], c=labels_em_fit, cmap='viridis', s=15, alpha=0.6)
    ax_em.scatter(mus_fit[:, 0], mus_fit[:, 1], c='red', marker='*', s=300, edgecolors='black', label='Fitted means')
    ax_em.set_title("EM-GMM from scratch")
    ax_em.legend()
    fig_em
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Why EM Works

    EM maximizes the log-likelihood indirectly by constructing and maximizing a lower bound — the **evidence lower bound (ELBO)**. At each iteration, the E-step tightens the bound (makes it touch the log-likelihood at the current parameters), and the M-step maximizes this tight bound. The log-likelihood is guaranteed to increase (or stay the same) at every iteration.

    Formally, for any distribution $q(z)$ over the latent variables $z$:

    $$\log p(x) \geq \mathbb{E}_{q(z)}[\log p(x, z)] - \mathbb{E}_{q(z)}[\log q(z)] = \text{ELBO}$$

    The E-step sets $q(z) = p(z \mid x, \theta^{\text{old}})$, which makes the bound tight. The M-step maximizes the ELBO with respect to $\theta$. This is a framework you'll see again in variational inference and variational autoencoders.

    ### Connection to K-Means

    If you fix $\Sigma_k = \sigma^2 I$ for all $k$ and take $\sigma^2 \to 0$, the responsibilities become hard: $r_{ik} \to 1$ for the nearest centroid and $r_{ik} \to 0$ otherwise. The M-step for $\mu_k$ becomes an ordinary mean of the assigned points. You recover K-means exactly. K-means is the "hard, isotropic" limit of EM for GMMs.

    ### Model Selection

    Use the **Bayesian Information Criterion (BIC)** to choose the number of components:

    $$\text{BIC} = -2 \ln \hat{L} + p \ln N$$

    where $\hat{L}$ is the maximized likelihood and $p$ is the number of free parameters. BIC penalizes complexity (more components = more parameters). Choose the $K$ that minimizes BIC.

    See [Bishop 9.2-9.3](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) for the full GMM/EM treatment, [ESL 8.5](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf) for EM in a broader context, and [Murphy PML1 11.3](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf) for a modern presentation.
    """)
    return


@app.cell
def _(StandardScaler, make_blobs, np, plt):
    from sklearn.mixture import GaussianMixture

    X_gmm, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)
    X_gmm_scaled = StandardScaler().fit_transform(X_gmm)

    # Fit GMMs with different numbers of components
    bics = []
    K_range = range(1, 11)
    for k in K_range:
        gmm = GaussianMixture(n_components=k, covariance_type='full',
                               n_init=5, random_state=42)
        gmm.fit(X_gmm_scaled)
        bics.append(gmm.bic(X_gmm_scaled))

    best_k = list(K_range)[np.argmin(bics)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(list(K_range), bics, 'o-')
    ax1.axvline(x=best_k, color='r', linestyle='--', label=f'Best K={best_k}')
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel('BIC')
    ax1.set_title('BIC for GMM Model Selection')
    ax1.legend()

    # Fit the best model
    gmm_best = GaussianMixture(n_components=best_k, covariance_type='full',
                                n_init=5, random_state=42)
    gmm_best.fit(X_gmm_scaled)
    labels_gmm = gmm_best.predict(X_gmm_scaled)
    responsibilities = gmm_best.predict_proba(X_gmm_scaled)

    for c in range(best_k):
        mask = labels_gmm == c
        ax2.scatter(X_gmm_scaled[mask, 0], X_gmm_scaled[mask, 1], s=20, alpha=0.7, label=f'Component {c}')
    ax2.set_title(f'GMM Clustering (K={best_k})')
    ax2.legend(fontsize=8)
    ax2.set_xlabel("$x_1$")
    ax2.set_ylabel("$x_2$")

    plt.tight_layout()
    fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 5. Dimensionality Reduction Beyond PCA

    PCA is linear and optimizes for variance preservation globally. For visualization and for data that lives on nonlinear manifolds, you need different tools.

    ### t-SNE (t-distributed Stochastic Neighbor Embedding)

    t-SNE is designed for one specific task: producing informative 2D or 3D visualizations of high-dimensional data. It does this by preserving **local neighborhood structure** — points that are close in high-D should be close in low-D.

    **How it works (conceptual):**

    1. In high-D: for each pair of points, compute a similarity based on Gaussian distances. This creates a probability distribution $P$ over pairs — nearby points get high probability, distant points get low probability.
    2. In low-D: define a similar probability distribution $Q$ over the same pairs, but using a Student-$t$ distribution (heavier tails than Gaussian).
    3. Minimize the KL divergence $D_{KL}(P \| Q)$ by gradient descent, adjusting the low-D coordinates.

    The heavy-tailed $t$-distribution in low-D is the key innovation — it alleviates the "crowding problem" where moderately distant points in high-D all get crushed together in low-D.

    **Perplexity** is the main hyperparameter, roughly controlling the effective number of neighbors considered. Typical values are 5-50. Low perplexity emphasizes very local structure; high perplexity gives a more global view.

    **Critical limitations:**
    - **Only for visualization.** Do not use t-SNE embeddings as features for downstream ML tasks.
    - **Non-deterministic.** Different random seeds give different results.
    - **Distances between clusters are meaningless.** Two well-separated clusters in a t-SNE plot are not necessarily far apart in the original space.
    - **Cluster sizes are meaningless.** t-SNE expands dense clusters and contracts sparse ones.
    - **Slow.** $O(N^2)$ naive, $O(N \log N)$ with Barnes-Hut approximation.

    ### UMAP (Uniform Manifold Approximation and Projection)

    UMAP is the newer alternative. It's based on a different mathematical framework (fuzzy topological structures), but the practical effect is similar to t-SNE with several advantages:

    - **Faster.** Scales better to large datasets.
    - **Better global structure preservation.** Relative positions of clusters are more meaningful.
    - **Can be used for general dimensionality reduction**, not just visualization — you can embed into arbitrary dimensions and use the result for downstream tasks.

    ### When to Use Which

    | Method | Use case | Preserves |
    |---|---|---|
    | **PCA** | Preprocessing, denoising, feature extraction | Global linear structure, variance |
    | **t-SNE** | Visualization of clusters/classes in 2D | Local neighborhood structure |
    | **UMAP** | Visualization or general-purpose nonlinear DR | Local + some global structure |

    A common pipeline: PCA to reduce from, say, 1000 dimensions to 50, then t-SNE or UMAP from 50 to 2 for visualization.

    See [Murphy PML1 Ch 20](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf) for a survey of dimensionality reduction methods and [ESL 14.8](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf) for nonlinear methods.
    """)
    return


@app.cell
def _(PCA, StandardScaler, load_digits, plt):
    from sklearn.manifold import TSNE

    digits_tsne = load_digits()
    X_tsne = StandardScaler().fit_transform(digits_tsne.data)

    # Reduce dimensions with PCA first (speeds up t-SNE significantly)
    pca_50 = PCA(n_components=50)
    X_pca50 = pca_50.fit_transform(X_tsne)

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    X_tsne_2d = tsne.fit_transform(X_pca50)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_tsne_2d[:, 0], X_tsne_2d[:, 1],
                         c=digits_tsne.target, cmap='tab10', s=5, alpha=0.7)
    ax.set_title('t-SNE Visualization of Handwritten Digits')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=ax, label='Digit')
    plt.tight_layout()
    fig

    # UMAP (requires: pip install umap-learn)
    # import umap
    # reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
    # X_umap = reducer.fit_transform(X_pca50)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 6. Anomaly / Outlier Detection

    Anomaly detection is the problem of identifying data points that are "unusual" relative to the majority. This comes up everywhere: fraud detection, manufacturing defect detection, network intrusion, medical diagnosis.

    ### Density-Based Approaches

    The simplest idea: fit a density model to the data, then flag points in low-density regions. If you've fit a GMM, any point with low $p(x)$ under the model is a candidate anomaly. You set a threshold $\tau$ and flag points where $p(x) < \tau$.

    This connects directly to what you learned about GMMs — the density $p(x) = \sum_k \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)$ gives you a natural anomaly score for free.

    ### Isolation Forest

    Isolation Forest takes a completely different approach based on a clever insight: **anomalies are easier to isolate**. If you randomly partition the feature space with axis-aligned splits (like building a random tree), anomalous points will be isolated in fewer splits than normal points.

    The algorithm builds an ensemble of random trees on random subsamples. For each point, the **anomaly score** is the average path length to isolate that point across all trees. Short average path length = anomaly.

    This is fast, scales well, and doesn't require density estimation. It works particularly well in high dimensions where density estimation becomes unreliable.

    See [Murphy PML1 21.2](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf) for anomaly detection methods.
    """)
    return


@app.cell
def _(gaussian_pdf, np):
    # Anomaly detection via GMM density — from scratch
    # Compute p(x) = sum_k pi_k * N(x|mu_k, Sigma_k), flag low-density points
    rng_anom = np.random.RandomState(0)
    X_normal = rng_anom.randn(200, 2)  # normal data
    X_anom_pts = rng_anom.uniform(-5, 5, size=(10, 2))  # outliers
    X_all = np.vstack([X_normal, X_anom_pts])

    # Fit a single Gaussian (simplest density model)
    mu_fit = X_all.mean(axis=0)
    Sigma_fit = np.cov(X_all.T)
    densities = gaussian_pdf(X_all, mu_fit, Sigma_fit)

    # Flag bottom 5% as anomalies
    threshold_pct = np.percentile(densities, 5)
    is_anomaly = densities < threshold_pct
    print(f"Threshold density: {threshold_pct:.6f}")
    print(f"Anomalies detected: {is_anomaly.sum()} / {len(X_all)}")
    return


@app.cell
def _(StandardScaler, make_blobs, np, plt):
    from sklearn.ensemble import IsolationForest
    from sklearn.mixture import GaussianMixture as GM_anom

    # Create data with some outliers
    X_anom, _ = make_blobs(n_samples=280, centers=3, cluster_std=0.8, random_state=42)
    rng = np.random.RandomState(42)
    X_outliers = rng.uniform(low=-10, high=10, size=(20, 2))
    X_anom_full = np.vstack([X_anom, X_outliers])
    X_anom_scaled = StandardScaler().fit_transform(X_anom_full)

    # Fit Isolation Forest
    iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    anomaly_labels = iso.fit_predict(X_anom_scaled)  # -1 for anomalies, 1 for normal

    # Anomaly scores (lower = more anomalous)
    scores = iso.decision_function(X_anom_scaled)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Isolation Forest results
    normal = anomaly_labels == 1
    anomalous = anomaly_labels == -1
    ax1.scatter(X_anom_full[normal, 0], X_anom_full[normal, 1], c='blue', s=20, alpha=0.5, label='Normal')
    ax1.scatter(X_anom_full[anomalous, 0], X_anom_full[anomalous, 1], c='red', s=50, marker='x',
                linewidths=2, label='Anomaly')
    ax1.set_title(f'Isolation Forest\n({anomalous.sum()} anomalies detected)')
    ax1.legend()
    ax1.set_xlabel("$x_1$")
    ax1.set_ylabel("$x_2$")

    # GMM-based anomaly detection
    gmm_anom = GM_anom(n_components=3, covariance_type='full', random_state=42)
    gmm_anom.fit(X_anom_scaled)
    log_probs = gmm_anom.score_samples(X_anom_scaled)  # log-likelihood per sample
    threshold = np.percentile(log_probs, 5)  # flag bottom 5%
    gmm_anomalies = log_probs < threshold

    normal_gmm = ~gmm_anomalies
    ax2.scatter(X_anom_full[normal_gmm, 0], X_anom_full[normal_gmm, 1], c='blue', s=20, alpha=0.5, label='Normal')
    ax2.scatter(X_anom_full[gmm_anomalies, 0], X_anom_full[gmm_anomalies, 1], c='red', s=50, marker='x',
                linewidths=2, label='Anomaly')
    ax2.set_title(f'GMM Density-Based\n({gmm_anomalies.sum()} anomalies detected)')
    ax2.legend()
    ax2.set_xlabel("$x_1$")
    ax2.set_ylabel("$x_2$")

    plt.tight_layout()
    fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 7. The Big Picture

    Here's how these methods relate to each other and to what you've already learned:

    **K-means** is a special case of **GMMs** (hard assignment, isotropic covariance). GMMs are fit by the **EM algorithm**, which is a general framework for latent variable models. EM itself is an instance of optimizing the **ELBO**, which you'll encounter again in variational autoencoders.

    **PCA** is the optimal linear dimensionality reduction under squared reconstruction error. It connects to the **SVD**, which you learned in linear algebra. **Kernel PCA** extends this to nonlinear settings via the kernel trick, the same idea behind kernel SVMs from earlier modules.

    **t-SNE** and **UMAP** are nonlinear dimensionality reduction methods designed for visualization. They don't optimize the same objective as PCA — they preserve *neighborhoods* rather than variance.

    **Anomaly detection** connects back to density estimation and to the tree-based methods from Module 1E (Isolation Forest is just a random forest variant).

    The unifying theme: in unsupervised learning, the "right" method depends on your assumptions about the data's structure and your downstream goals. There is no free lunch here even more so than in supervised learning.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Reading List

    | Topic | Primary | Supplementary |
    |---|---|---|
    | PCA | [MML Ch 10](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) | [Bishop 12.1](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf), [ISLR 12.2](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf) |
    | K-Means | [ISLR 12.4.1](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf) | [ESL 14.3.6](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf), [Bishop 9.1](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) |
    | Hierarchical Clustering | [ISLR 12.4.2](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf) | [ESL 14.3.12](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf) |
    | GMMs and EM | [Bishop 9.2-9.3](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) | [ESL 8.5](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf), [Murphy PML1 11.3](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf) |
    | t-SNE / UMAP | [Murphy PML1 Ch 20](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf) | [ESL 14.8](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf) |
    | Anomaly Detection | [Murphy PML1 21.2](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf) | |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Practice

    ### Conceptual

    1. **PCA geometry.** You have 3D data that lies on a flat plane (a 2D subspace). PCA with $K = 2$ captures 100% of the variance. Now tilt the plane slightly and add a tiny amount of Gaussian noise perpendicular to it. What happens to the explained variance of the third component? What if the noise is large?

    2. **K-means failure modes.** Sketch (or describe) a 2D dataset where K-means with $K = 2$ will consistently find the wrong clustering. What property of the data causes this?

    3. **EM intuition.** In the GMM E-step, if one component has a much larger mixing weight $\pi_k$ than the others, how does this affect the responsibilities? What happens in the extreme case where $\pi_1 = 0.99$?

    4. **t-SNE traps.** You run t-SNE on a dataset and see 5 clearly separated clusters. Your colleague concludes there are 5 groups. What questions should you ask before accepting this?

    5. **EM and K-means.** Prove informally that if you take the EM updates for a GMM with $\Sigma_k = \sigma^2 I$ for all $k$ and let $\sigma^2 \to 0$, the E-step reduces to hard assignment and the M-step reduces to the K-means centroid update.

    ### Applied

    6. **PCA on real data.** Load the `sklearn.datasets.load_digits` dataset (8x8 pixel images of handwritten digits, 64 features). Standardize, run PCA, and plot the explained variance curve. How many components capture 95% of the variance? Visualize the first two principal components colored by digit label.

    7. **Clustering comparison.** On the digits data (or the Iris dataset), run K-means, hierarchical clustering (Ward's linkage), and a GMM. Compare the results using the adjusted Rand index (`sklearn.metrics.adjusted_rand_score`) against the true labels. Which method performs best? Why?

    8. **Full pipeline.** Load a high-dimensional dataset (try `sklearn.datasets.fetch_olivetti_faces` — 400 face images, 4096 features). Reduce to 50 dimensions with PCA, then: (a) cluster with K-means and GMM, using BIC/silhouette to choose $K$, (b) visualize with t-SNE, (c) identify potential outliers with Isolation Forest. Write up your findings.

    9. **EM from scratch.** Implement the EM algorithm for a 2D GMM with $K = 3$ components from scratch in NumPy. Generate synthetic data from a known mixture, fit your model, and verify that the recovered parameters are close to the true ones. Compare with `sklearn.mixture.GaussianMixture`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Code It

    Implementation exercises to solidify your understanding. Each exercise gives you a problem and skeleton code — fill in the missing parts.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Exercise 1: PCA from Scratch on Real Data

    Implement PCA on the digits dataset using only numpy. Do not use sklearn's PCA. Steps:
    1. Center the data (subtract mean)
    2. Compute the covariance matrix
    3. Eigendecompose it
    4. Project onto the top K components
    5. Compute the explained variance ratio

    Verify your result matches sklearn's PCA.
    """)
    return


@app.cell
def _(np, load_digits, StandardScaler, PCA):
    # EXERCISE 1: PCA from scratch on digits
    _digits_ex = load_digits()
    X_ex1 = StandardScaler().fit_transform(_digits_ex.data)

    # Step 1: Center the data
    # X_c = ...

    # Step 2: Covariance matrix S = X^T X / (N-1)
    # S_ex1 = ...

    # Step 3: Eigendecompose (use np.linalg.eigh, then sort descending)
    # evals_ex1, evecs_ex1 = ...

    # Step 4: Project onto top 2 components
    # Z_ex1 = ...

    # Step 5: Explained variance ratio
    # evr_ex1 = evals_ex1 / evals_ex1.sum()

    # Verify: compare with sklearn
    # pca_check = PCA(n_components=2).fit(X_ex1)
    # print("Your explained variance ratio (top 2):", evr_ex1[:2])
    # print("Sklearn explained variance ratio:     ", pca_check.explained_variance_ratio_)

    # Placeholder so cell runs
    print("Fill in the code above, then uncomment the verification block.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Exercise 2: PCA via SVD

    Redo Exercise 1, but use `np.linalg.svd` instead of eigendecomposing the covariance matrix. Recall:
    - $X = U \Sigma V^T$
    - Principal components are rows of $V^T$ (or columns of $V$)
    - Eigenvalues of the covariance matrix are $\sigma_k^2 / (N-1)$

    Verify that both approaches give the same eigenvalues and the same projected coordinates (up to sign flips).
    """)
    return


@app.cell
def _(np, load_digits, StandardScaler):
    # EXERCISE 2: PCA via SVD
    _digits_ex2 = load_digits()
    X_ex2 = StandardScaler().fit_transform(_digits_ex2.data)
    X_c_ex2 = X_ex2 - X_ex2.mean(axis=0)
    N_ex2 = X_c_ex2.shape[0]

    # SVD: X = U @ diag(sigma) @ Vt
    # U_ex2, sigma_ex2, Vt_ex2 = ...

    # Eigenvalues from singular values
    # evals_svd = sigma_ex2**2 / (N_ex2 - 1)

    # Project onto top 2: Z = X_c @ V[:, :2]  (V = Vt.T)
    # Z_svd = ...

    # Compare with eigendecomposition approach from Exercise 1
    # print("Top 2 eigenvalues (SVD):", evals_svd[:2])

    print("Fill in the SVD-based PCA code above.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 3: K-Means with K-Means++ Initialization

    Implement K-means++ initialization from scratch, then plug it into a K-means loop.

    K-means++ rule:
    1. Pick first centroid uniformly at random from the data
    2. For each subsequent centroid, pick point $x$ with probability $\propto D(x)^2$, where $D(x)$ = distance to nearest existing centroid
    """)
    return


@app.cell
def _(np, make_blobs, plt):
    # EXERCISE 3: K-Means++ initialization + K-Means
    X_ex3, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

    def kmeans_pp_init(X, K, seed=42):
        """K-means++ initialization. Returns (K, D) array of centroids."""
        rng = np.random.RandomState(seed)
        N, D = X.shape
        centroids = np.empty((K, D))

        # Pick first centroid uniformly at random
        centroids[0] = X[rng.randint(N)]

        for k in range(1, K):
            # Compute D(x)^2 = min distance^2 to any existing centroid
            # dists_sq = ...
            # probs = dists_sq / dists_sq.sum()
            # centroids[k] = X[rng.choice(N, p=probs)]
            pass  # Replace with your implementation

        return centroids

    # Then run K-Means using these centroids
    # centroids_init = kmeans_pp_init(X_ex3, K=4)
    # ... assignment + update loop ...

    print("Implement kmeans_pp_init, then run K-Means with it.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Exercise 4: Reconstruction Error vs. Number of Components

    Write code that:
    1. Runs PCA from scratch (using SVD) for K = 1, 2, ..., 20
    2. For each K, reconstructs the data: $\hat{X} = Z_K V_K^T + \bar{x}$
    3. Computes the mean squared reconstruction error
    4. Plots MSE vs. K

    You should see MSE decrease and eventually flatten — this is the "elbow" in reconstruction space.
    """)
    return


@app.cell
def _(np, load_digits, StandardScaler, plt):
    # EXERCISE 4: Reconstruction error curve
    _digits_ex4 = load_digits()
    X_ex4 = StandardScaler().fit_transform(_digits_ex4.data)
    X_mean_ex4 = X_ex4.mean(axis=0)
    X_c_ex4 = X_ex4 - X_mean_ex4

    # SVD once
    # U4, s4, Vt4 = np.linalg.svd(X_c_ex4, full_matrices=False)

    # mse_list = []
    # for K in range(1, 21):
    #     # Project to K dims and reconstruct
    #     # Z_k = X_c_ex4 @ Vt4[:K].T
    #     # X_hat = Z_k @ Vt4[:K]
    #     # mse = np.mean((X_c_ex4 - X_hat)**2)
    #     # mse_list.append(mse)

    # plt.plot(range(1, 21), mse_list, 'o-')
    # plt.xlabel("Number of components K")
    # plt.ylabel("Mean Squared Reconstruction Error")
    # plt.title("Reconstruction Error vs. K")

    print("Uncomment and fill in the reconstruction error computation.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Exercise 5: EM for a 1D Gaussian Mixture

    Implement EM for a simple 1D, 2-component Gaussian mixture. This is easier than the 2D case and lets you focus on the algorithm logic.

    Generate data: 100 points from $N(-3, 1)$ and 100 points from $N(3, 1.5^2)$. Then recover the parameters.
    """)
    return


@app.cell
def _(np, plt):
    # EXERCISE 5: 1D EM for a 2-component Gaussian mixture
    rng_ex5 = np.random.RandomState(42)
    # True parameters
    X1_ex5 = rng_ex5.normal(-3, 1.0, size=100)
    X2_ex5 = rng_ex5.normal(3, 1.5, size=100)
    X_ex5 = np.concatenate([X1_ex5, X2_ex5])

    # Initialize parameters
    # mu1, mu2 = -1.0, 1.0         # initial guesses for means
    # sig1, sig2 = 1.0, 1.0        # initial guesses for std devs
    # pi1, pi2 = 0.5, 0.5          # initial mixing weights

    # for _ in range(50):
    #     # E-step: compute responsibilities
    #     # r1 = pi1 * N(X_ex5 | mu1, sig1^2) / (pi1*N(...) + pi2*N(...))
    #     # r2 = 1 - r1
    #
    #     # M-step: update parameters
    #     # N1, N2 = r1.sum(), r2.sum()
    #     # mu1 = (r1 * X_ex5).sum() / N1
    #     # mu2 = (r2 * X_ex5).sum() / N2
    #     # sig1 = np.sqrt((r1 * (X_ex5 - mu1)**2).sum() / N1)
    #     # sig2 = np.sqrt((r2 * (X_ex5 - mu2)**2).sum() / N2)
    #     # pi1, pi2 = N1/len(X_ex5), N2/len(X_ex5)

    # print(f"Recovered: mu1={mu1:.2f}, mu2={mu2:.2f}, sig1={sig1:.2f}, sig2={sig2:.2f}")
    # print(f"True:      mu1=-3.00, mu2=3.00, sig1=1.00, sig2=1.50")

    print("Uncomment the EM loop and fill in the E-step computation.")
    return


if __name__ == "__main__":
    app.run()
