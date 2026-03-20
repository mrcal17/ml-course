import marimo

app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Linear Algebra Refresh

    You already know linear algebra. You can multiply matrices, find eigenvalues, maybe even compute an SVD by hand if someone put a gun to your head. Good. This module is not about re-teaching those mechanics -- it is about reframing everything you learned through the lens of machine learning, so that when you encounter a covariance matrix or a projection operator in a paper, you see it for what it actually is rather than treating it as abstract formalism.

    Linear algebra is not a prerequisite for ML in the way that, say, knowing how to read is a prerequisite for studying literature. It is more fundamental than that. Linear algebra *is* the language ML is written in. Every dataset is a matrix. Every model parameter is a vector. Every prediction is a linear (or locally linear) transformation. Every optimization landscape has curvature described by eigenvalues. If your linear algebra is rusty, everything downstream -- PCA, SVD-based recommenders, neural network weight matrices, attention mechanisms -- will feel like arbitrary symbol manipulation instead of geometry.

    So let us sharpen the blade.

    > **Primary reference:** [MML Chapter 2: Linear Algebra](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) and [MML Chapter 3: Analytic Geometry](file:///C:/Users/landa/ml-course/textbooks/MML.pdf). These two chapters are excellent and I will reference specific pages throughout. Skim them alongside this lecture.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 1. Vectors and Vector Spaces

    ### Vectors: Points and Directions

    A vector $\mathbf{x} \in \mathbb{R}^n$ is simultaneously a point in $n$-dimensional space and a direction with magnitude. You know this. What matters for ML is the interpretation:

    - **A data point** is a vector. A grayscale image of 28x28 pixels is a vector in $\mathbb{R}^{784}$. A customer described by 50 features lives in $\mathbb{R}^{50}$.
    - **Model weights** are vectors. A linear model $f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b$ has a weight vector $\mathbf{w}$ that defines a hyperplane in feature space.
    - **Embeddings** are vectors. Word2Vec, BERT embeddings, graph node embeddings -- all of these map discrete objects into continuous vector spaces where geometry (distance, angle) carries semantic meaning.

    ### Norms

    You will encounter three norms constantly:

    | Norm | Definition | ML Usage |
    |------|-----------|----------|
    | $L^1$ (Manhattan) | $\|\mathbf{x}\|_1 = \sum_i |x_i|$ | Lasso regularization, promotes sparsity |
    | $L^2$ (Euclidean) | $\|\mathbf{x}\|_2 = \sqrt{\sum_i x_i^2}$ | Ridge regularization, Euclidean distance |
    | $L^\infty$ (Max) | $\|\mathbf{x}\|_\infty = \max_i |x_i|$ | Adversarial robustness (perturbation budgets) |

    The $L^1$ norm gives you sparse solutions because its unit ball has corners aligned with the axes -- optimization tends to land on those corners, zeroing out coordinates. The $L^2$ norm shrinks all coordinates toward zero but rarely to exactly zero. This geometric distinction between $L^1$ and $L^2$ regularization is one of the most important practical insights in ML. See [MML Section 2.7: Norms](file:///C:/Users/landa/ml-course/textbooks/MML.pdf).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Inner Products

    The inner product $\langle \mathbf{x}, \mathbf{y} \rangle = \mathbf{x}^\top \mathbf{y} = \sum_i x_i y_i$ measures similarity. Specifically:

    $$\mathbf{x}^\top \mathbf{y} = \|\mathbf{x}\| \|\mathbf{y}\| \cos\theta$$

    When you see "cosine similarity" in NLP or recommendation systems, it is just the inner product of unit-normalized vectors. When you see a kernel function $k(\mathbf{x}, \mathbf{y})$, it is a generalized inner product in some (possibly infinite-dimensional) feature space. The inner product is the fundamental operation that quantifies how much two vectors "agree."

    ### Linear Independence, Span, Basis, Dimension

    Quick recap:

    - Vectors $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ are **linearly independent** if no vector can be written as a linear combination of the others.
    - The **span** of those vectors is the set of all their linear combinations -- a subspace.
    - A **basis** is a linearly independent set that spans the space. Its size is the **dimension**.

    **Why this matters:** If your $n$ features are linearly dependent, your data actually lives in a lower-dimensional subspace. This is exactly what dimensionality reduction exploits. PCA finds a new basis aligned with the directions of maximum variance. Autoencoders learn a nonlinear analog of the same idea.

    See [MML Section 2.4: Vector Spaces and Subspaces](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) for the formal treatment.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 2. Matrices as Linear Transformations

    A matrix $A \in \mathbb{R}^{m \times n}$ defines a linear map $f: \mathbb{R}^n \to \mathbb{R}^m$ via $f(\mathbf{x}) = A\mathbf{x}$. Every linear transformation between finite-dimensional spaces can be represented as a matrix (once you fix bases). This is the single most important idea in linear algebra.

    ### Geometric View

    In $\mathbb{R}^2$, the standard transformations are:

    - **Rotation** by angle $\theta$: $\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$ -- preserves lengths and angles
    - **Scaling**: $\begin{pmatrix} s_1 & 0 \\ 0 & s_2 \end{pmatrix}$ -- stretches along axes
    - **Shearing**: $\begin{pmatrix} 1 & k \\ 0 & 1 \end{pmatrix}$ -- skews one axis
    - **Projection** onto the $x$-axis: $\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$ -- collapses a dimension

    Neural network layers are just these kinds of transformations composed with nonlinearities. A single layer computes $\mathbf{h} = \sigma(W\mathbf{x} + \mathbf{b})$: a linear transformation ($W\mathbf{x}$), a translation ($+\mathbf{b}$), and a pointwise nonlinearity ($\sigma$). Without $\sigma$, stacking layers would just give you another linear transformation -- composition of linear maps is linear. The nonlinearity is what gives depth its power.
    """)
    return


@app.cell
def _(mo):
    mo.image(src="../animations/rendered/LinearTransformation2D.gif")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Matrix Multiplication as Composition

    If $A$ represents transformation $f$ and $B$ represents transformation $g$, then $AB$ represents $f \circ g$ (apply $g$ first, then $f$). This is why matrix multiplication is not commutative -- applying a rotation then a scaling is different from scaling then rotating.

    ### Rank

    The **rank** of a matrix $A$ is the dimension of its column space (equivalently, its row space). It tells you the dimensionality of the output of the transformation. Key facts:

    - $\text{rank}(A) \leq \min(m, n)$
    - $A$ is **full rank** if $\text{rank}(A) = \min(m, n)$
    - A rank-deficient matrix maps to a lower-dimensional subspace -- information is lost

    **ML connection:** If your data matrix $X \in \mathbb{R}^{n \times d}$ has rank $r < d$, your features are redundant. Low-rank matrix factorization (used in collaborative filtering, topic models, etc.) assumes the data matrix is approximately low-rank: $X \approx UV^\top$ where $U \in \mathbb{R}^{n \times r}$, $V \in \mathbb{R}^{d \times r}$, and $r \ll d$.

    See [MML Section 2.5: Linear Mappings](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) and [MML Section 2.6: Matrix Representations](file:///C:/Users/landa/ml-course/textbooks/MML.pdf).

    ### Inverse and Pseudo-Inverse

    The inverse $A^{-1}$ exists only for square, full-rank matrices. In ML, we rarely have square matrices -- our data matrix $X$ is typically $n \times d$ with $n \neq d$. Enter the **Moore-Penrose pseudo-inverse**:

    $$A^+ = (A^\top A)^{-1} A^\top \quad \text{(left pseudo-inverse, when } n > d\text{)}$$

    This gives the least-squares solution to overdetermined systems. When you see the OLS formula $\hat{\mathbf{w}} = (X^\top X)^{-1} X^\top \mathbf{y}$, you are looking at $X^+ \mathbf{y}$. We will return to this below.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 3. Systems of Linear Equations

    The equation $A\mathbf{x} = \mathbf{b}$ with $A \in \mathbb{R}^{m \times n}$ falls into three cases:

    | Case | Condition | Meaning |
    |------|-----------|---------|
    | **Overdetermined** | $m > n$ (more equations than unknowns) | Typically no exact solution; find best approximation |
    | **Underdetermined** | $m < n$ (fewer equations than unknowns) | Infinitely many solutions; need regularization to pick one |
    | **Exact** | $m = n$, $A$ full rank | Unique solution $\mathbf{x} = A^{-1}\mathbf{b}$ |

    **The ML connection is immediate.** Linear regression with $n$ data points and $d$ features gives you the system $X\mathbf{w} = \mathbf{y}$ where $X \in \mathbb{R}^{n \times d}$:

    - $n > d$: overdetermined. More data points than features. No exact solution -- find $\mathbf{w}$ that minimizes $\|X\mathbf{w} - \mathbf{y}\|^2$. This is ordinary least squares.
    - $n < d$: underdetermined. More features than data points. Infinitely many perfect solutions -- regularization (Ridge, Lasso) picks one with desirable properties.
    - $n = d$: you can interpolate exactly, but you will probably overfit.

    Understanding these three regimes is essential for understanding why regularization exists and when you need it. See [MML Section 2.7: Solving Systems of Linear Equations](file:///C:/Users/landa/ml-course/textbooks/MML.pdf).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 4. Orthogonality and Projections

    ### Orthogonal Vectors and Matrices

    Two vectors are **orthogonal** if $\mathbf{x}^\top \mathbf{y} = 0$. An **orthogonal matrix** $Q$ satisfies $Q^\top Q = QQ^\top = I$, meaning its columns form an orthonormal basis. Key property: orthogonal transformations preserve lengths and angles ($\|Q\mathbf{x}\| = \|\mathbf{x}\|$). Rotations and reflections are orthogonal.

    **Why you care:** Orthogonal matrices are numerically stable. They do not amplify rounding errors. This is why algorithms like QR decomposition and SVD, which factor matrices using orthogonal components, are the workhorses of numerical linear algebra.

    ### Projection onto a Subspace

    This is the most important geometric idea in this module.

    Given a subspace spanned by the columns of $A$, the **orthogonal projection** of $\mathbf{b}$ onto this subspace is:

    $$\hat{\mathbf{b}} = A(A^\top A)^{-1} A^\top \mathbf{b}$$

    The matrix $P = A(A^\top A)^{-1} A^\top$ is the **projection matrix** (also called the hat matrix in statistics because it "puts the hat on $\mathbf{y}$").

    Now here is the punchline: **least squares regression is projection.** When you solve $\min_\mathbf{w} \|X\mathbf{w} - \mathbf{y}\|^2$, you are finding the point in the column space of $X$ that is closest to $\mathbf{y}$. The predicted values $\hat{\mathbf{y}} = X(X^\top X)^{-1} X^\top \mathbf{y}$ are exactly the orthogonal projection of $\mathbf{y}$ onto the column space of $X$. The residual vector $\mathbf{y} - \hat{\mathbf{y}}$ is orthogonal to that column space.

    This geometric picture -- $\mathbf{y}$ decomposed into a component in the column space of $X$ and a perpendicular residual -- is worth burning into your brain. It makes the normal equations ($X^\top X \mathbf{w} = X^\top \mathbf{y}$) obvious: multiply both sides of $X\mathbf{w} = \mathbf{y}$ by $X^\top$ to project the problem into a solvable square system.

    See [MML Section 3.8: Orthogonal Projections](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) and [ISLR Section 3.1: Simple Linear Regression](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf) for the statistical side.

    ### QR Decomposition

    Any matrix $A$ can be factored as $A = QR$ where $Q$ is orthogonal and $R$ is upper triangular. This is numerically superior to directly computing $(A^\top A)^{-1} A^\top$ for solving least squares, because $A^\top A$ can be ill-conditioned (squaring the condition number). With QR:

    $$A\mathbf{x} = \mathbf{b} \implies QR\mathbf{x} = \mathbf{b} \implies R\mathbf{x} = Q^\top \mathbf{b}$$

    and back-substitution on the triangular system $R$ is fast and stable. In practice, `numpy.linalg.lstsq` uses this (or SVD) under the hood.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 5. Eigendecomposition

    ### Eigenvectors and Eigenvalues

    A nonzero vector $\mathbf{v}$ is an **eigenvector** of $A$ with **eigenvalue** $\lambda$ if:

    $$A\mathbf{v} = \lambda \mathbf{v}$$

    Geometrically: $A$ acts on $\mathbf{v}$ by simply scaling it, not changing its direction. The eigenvectors are the "natural axes" of the transformation, and the eigenvalues tell you the scaling factor along each axis.

    ### Diagonalization

    If $A$ has $n$ linearly independent eigenvectors (collected as columns of $P$) with eigenvalues $\lambda_1, \ldots, \lambda_n$ (on the diagonal of $D$):

    $$A = PDP^{-1}$$

    This decomposition says: change to the eigenvector basis ($P^{-1}$), scale along each axis ($D$), change back ($P$). Matrix powers become trivial: $A^k = PD^kP^{-1}$.

    ### Symmetric Matrices

    Symmetric matrices ($A = A^\top$) are everywhere in ML -- covariance matrices, kernel matrices, Hessians of scalar functions. They have two crucial properties:

    1. **All eigenvalues are real.**
    2. **Eigenvectors corresponding to distinct eigenvalues are orthogonal.** You can always find an orthonormal eigenbasis.

    This gives the **spectral decomposition**: $A = Q\Lambda Q^\top$ where $Q$ is orthogonal and $\Lambda$ is diagonal. No need for $Q^{-1}$ -- it is just $Q^\top$. Clean, numerically stable, and geometrically transparent.

    See [MML Section 4.2: Eigenvalues and Eigenvectors](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) and [MML Section 4.3: Eigendecomposition](file:///C:/Users/landa/ml-course/textbooks/MML.pdf).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Positive Definite and Semi-Definite Matrices

    A symmetric matrix $A$ is:
    - **Positive definite (PD)** if $\mathbf{x}^\top A \mathbf{x} > 0$ for all $\mathbf{x} \neq 0$ (equivalently, all eigenvalues $> 0$)
    - **Positive semi-definite (PSD)** if $\mathbf{x}^\top A \mathbf{x} \geq 0$ for all $\mathbf{x}$ (equivalently, all eigenvalues $\geq 0$)

    **The connection to convexity is direct:** A twice-differentiable function $f$ is convex if and only if its Hessian $H$ is positive semi-definite everywhere. Convex functions have no local minima other than the global minimum -- gradient descent is guaranteed to converge. This is why the PSD condition shows up when we analyze loss function landscapes.

    Covariance matrices are always PSD (by construction). Kernel matrices (Gram matrices) in SVMs must be PSD -- this is what makes a kernel function valid.

    See [Boyd Section 3.1: Convex Functions](file:///C:/Users/landa/ml-course/textbooks/Boyd-ConvexOptimization.pdf) for the convexity connection and [Bishop Section 2.3: The Gaussian Distribution](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) for covariance matrices in context.

    ### The Spectral Theorem

    For any real symmetric matrix $A$, there exists an orthogonal matrix $Q$ and diagonal matrix $\Lambda$ such that $A = Q\Lambda Q^\top$. The columns of $Q$ are orthonormal eigenvectors. This is the spectral theorem, and it is the theoretical foundation for PCA: the covariance matrix of your data is symmetric, so it has an orthogonal eigenbasis, and the eigenvectors with the largest eigenvalues point in the directions of greatest variance.
    """)
    return


@app.cell
def _():
    import numpy as np

    # Demonstrate eigendecomposition of a symmetric matrix
    A = np.array([[4, 2], [2, 3]], dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(A)  # eigh for symmetric matrices

    print(f"A = \n{A}\n")
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Eigenvectors (columns):\n{eigenvectors}\n")

    # Verify spectral decomposition: A = Q Lambda Q^T
    Q = eigenvectors
    Lambda = np.diag(eigenvalues)
    A_reconstructed = Q @ Lambda @ Q.T
    print(f"Q @ Lambda @ Q^T = \n{A_reconstructed}")
    print(f"Reconstruction matches: {np.allclose(A, A_reconstructed)}")

    # Verify positive definiteness
    print(f"\nAll eigenvalues > 0: {np.all(eigenvalues > 0)} (positive definite)")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 6. Singular Value Decomposition (SVD)

    The SVD is arguably the single most useful matrix factorization in all of applied mathematics. Every matrix -- any shape, any rank -- has an SVD.

    ### The Decomposition

    For any $A \in \mathbb{R}^{m \times n}$ with rank $r$:

    $$A = U \Sigma V^\top$$

    where:
    - $U \in \mathbb{R}^{m \times m}$ -- orthogonal, columns are **left singular vectors** (eigenvectors of $AA^\top$)
    - $\Sigma \in \mathbb{R}^{m \times n}$ -- diagonal (in the generalized sense), entries $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$ are **singular values** (square roots of eigenvalues of $A^\top A$)
    - $V \in \mathbb{R}^{n \times n}$ -- orthogonal, columns are **right singular vectors** (eigenvectors of $A^\top A$)

    Think of it this way: any linear transformation can be decomposed into (1) a rotation/reflection ($V^\top$), then (2) a scaling along axes ($\Sigma$), then (3) another rotation/reflection ($U$). Every matrix is, in disguise, just a scaling operation in the right coordinate system.
    """)
    return


@app.cell
def _(mo):
    mo.image(src="../animations/rendered/SVDDecomposition.gif")
    return


@app.cell
def _():
    import numpy as np

    # SVD demonstration
    A = np.array([[3, 0], [0, 2], [0, 0]], dtype=float)
    U, S, Vt = np.linalg.svd(A, full_matrices=True)

    print(f"A = \n{A}\n")
    print(f"U = \n{np.round(U, 4)}\n")
    print(f"Singular values: {S}")
    print(f"V^T = \n{np.round(Vt, 4)}\n")

    # Reconstruct A from SVD
    Sigma = np.zeros_like(A, dtype=float)
    np.fill_diagonal(Sigma, S)
    A_reconstructed = U @ Sigma @ Vt
    print(f"U @ Sigma @ V^T = \n{np.round(A_reconstructed, 4)}")
    print(f"Reconstruction matches: {np.allclose(A, A_reconstructed)}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Low-Rank Approximation

    The **Eckart-Young theorem** states that the best rank-$k$ approximation to $A$ (in both Frobenius and spectral norms) is obtained by keeping only the top $k$ singular values:

    $$A_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$$

    This is not just a theorem -- it is a design principle. When you truncate the SVD, you keep the $k$ most important "modes" of the matrix and discard the rest as noise. The fraction of "energy" retained is $\sum_{i=1}^{k} \sigma_i^2 / \sum_{i=1}^{r} \sigma_i^2$.
    """)
    return


@app.cell
def _():
    import numpy as np

    # Low-rank approximation demo
    rng = np.random.default_rng(42)
    # Create a rank-2 matrix with some noise
    true_U = rng.standard_normal((10, 2))
    true_V = rng.standard_normal((8, 2))
    A = true_U @ true_V.T + 0.1 * rng.standard_normal((10, 8))

    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    print(f"Singular values: {np.round(S, 3)}")

    # Energy captured by top-k singular values
    total_energy = np.sum(S**2)
    for k in range(1, min(len(S), 6) + 1):
        energy_k = np.sum(S[:k]**2) / total_energy
        print(f"  Rank-{k}: {energy_k:.4f} of total energy")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Connection to PCA

    PCA on centered data matrix $X$ (rows are data points) works as follows:

    1. Compute the covariance matrix $C = \frac{1}{n-1} X^\top X$
    2. Find eigenvectors of $C$ -- these are the principal components
    3. Project data onto the top $k$ eigenvectors

    But you can also compute the SVD of $X = U\Sigma V^\top$. The columns of $V$ are eigenvectors of $X^\top X$, which are proportional to eigenvectors of $C$. The singular values satisfy $\sigma_i^2 / (n-1) = \lambda_i$ (the $i$-th eigenvalue of $C$). So **PCA is just the SVD of the centered data matrix.** In practice, you should always compute PCA via SVD -- it is more numerically stable than explicitly forming $X^\top X$.

    See [MML Section 4.5: Singular Value Decomposition](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) and [MML Section 10.2: PCA via SVD](file:///C:/Users/landa/ml-course/textbooks/MML.pdf).

    ### Truncated SVD for Dimensionality Reduction

    In practice, for large matrices you do not compute the full SVD and then truncate -- you use **truncated SVD** algorithms (e.g., randomized SVD) that directly compute only the top $k$ singular triplets. Scikit-learn's `TruncatedSVD` does this. It is the method behind latent semantic analysis (LSA) in NLP: take a term-document matrix, compute its truncated SVD, and the resulting low-dimensional representation captures latent semantic structure.
    """)
    return


@app.cell
def _():
    import numpy as np

    # PCA via eigendecomposition vs SVD -- verify they give the same result
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 5))
    X_centered = X - X.mean(axis=0)

    # Method 1: Eigendecomposition of covariance matrix
    C = (X_centered.T @ X_centered) / (X_centered.shape[0] - 1)
    eig_vals, eig_vecs = np.linalg.eigh(C)
    # Sort by descending eigenvalue
    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]

    # Method 2: SVD of centered data matrix
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    svd_eig_vals = S**2 / (X_centered.shape[0] - 1)

    print("Eigenvalues from covariance matrix:")
    print(f"  {np.round(eig_vals, 6)}")
    print("Eigenvalues from SVD (sigma^2 / (n-1)):")
    print(f"  {np.round(svd_eig_vals, 6)}")
    print(f"\nMatch: {np.allclose(eig_vals, svd_eig_vals)}")

    # Principal components match (up to sign)
    print("\nPrincipal components match (up to sign flips):")
    for i in range(5):
        match = np.allclose(np.abs(eig_vecs[:, i]), np.abs(Vt[i, :]))
        print(f"  PC{i+1}: {match}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 7. Special Matrices and Operations

    ### Matrix Types You Will See Repeatedly

    | Matrix Type | Property | Where It Appears |
    |------------|----------|-----------------|
    | **Symmetric** | $A = A^\top$ | Covariance matrices, kernels, Hessians |
    | **Diagonal** | Nonzero only on diagonal | Eigenvalue matrices, scaling |
    | **Orthogonal** | $Q^\top Q = I$ | SVD components, rotations, QR |
    | **Positive definite** | $\mathbf{x}^\top A\mathbf{x} > 0$ | Covariance, convexity conditions |
    | **Sparse** | Most entries are zero | Graph adjacency, NLP feature matrices |

    ### Trace

    The trace $\text{tr}(A) = \sum_i a_{ii}$ is the sum of diagonal entries. Key properties:

    - $\text{tr}(A) = \sum_i \lambda_i$ (sum of eigenvalues)
    - $\text{tr}(AB) = \text{tr}(BA)$ (cyclic permutation)
    - $\|A\|_F^2 = \text{tr}(A^\top A)$ (Frobenius norm via trace)

    You will see the trace in loss functions (e.g., $\text{tr}(\Sigma)$ as the total variance), in matrix calculus identities, and in the formulation of various ML objectives. The cyclic property is especially useful for simplifying derivatives involving matrix products.

    ### Determinant

    The determinant $\det(A)$ is the signed volume scaling factor of the transformation. Key facts:

    - $\det(A) = \prod_i \lambda_i$ (product of eigenvalues)
    - $\det(A) = 0$ iff $A$ is singular (rank-deficient)
    - $\det(AB) = \det(A)\det(B)$

    In ML, determinants appear in:
    - **Gaussian distributions:** $\mathcal{N}(\mathbf{x} | \boldsymbol{\mu}, \Sigma) \propto |\Sigma|^{-1/2} \exp(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu}))$. The $|\Sigma|^{-1/2}$ normalizing factor is a determinant.
    - **Log-likelihoods:** We often work with $\log\det(\Sigma)$ to avoid numerical overflow. The identity $\log\det(A) = \text{tr}(\log A)$ connects determinants and traces.
    - **Change of variables** in normalizing flows: computing the Jacobian determinant of a transformation.

    See [Bishop Section 2.3](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) and [Murphy Section 7.2: Multivariate Gaussians](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf).
    """)
    return


@app.cell
def _():
    import numpy as np

    # Trace and determinant demonstrations
    A = np.array([[4, 2], [2, 3]], dtype=float)
    eigenvalues = np.linalg.eigvalsh(A)

    print(f"A = \n{A}\n")
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Trace(A) = {np.trace(A):.4f}")
    print(f"Sum of eigenvalues = {np.sum(eigenvalues):.4f}")
    print(f"Det(A) = {np.linalg.det(A):.4f}")
    print(f"Product of eigenvalues = {np.prod(eigenvalues):.4f}")

    # Cyclic property: tr(AB) = tr(BA)
    B = np.array([[1, 3], [2, 4]], dtype=float)
    print(f"\ntr(AB) = {np.trace(A @ B):.4f}")
    print(f"tr(BA) = {np.trace(B @ A):.4f}")

    # Frobenius norm via trace
    frob_norm_sq = np.linalg.norm(A, 'fro')**2
    trace_ata = np.trace(A.T @ A)
    print(f"\n||A||_F^2 = {frob_norm_sq:.4f}")
    print(f"tr(A^T A) = {trace_ata:.4f}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Kronecker and Hadamard Products

    Two products you will encounter occasionally:

    **Kronecker product** $A \otimes B$: For $A \in \mathbb{R}^{m \times n}$, $B \in \mathbb{R}^{p \times q}$, produces an $mp \times nq$ block matrix. Shows up in multi-task learning (modeling correlations between tasks and features simultaneously) and in vectorizing matrix equations: $\text{vec}(AXB) = (B^\top \otimes A)\text{vec}(X)$.

    **Hadamard (elementwise) product** $A \odot B$: Same-shaped matrices, multiply entry by entry. Ubiquitous in neural networks -- gating mechanisms in LSTMs and GRUs use elementwise products, and attention mechanisms use elementwise operations extensively.
    """)
    return


@app.cell
def _():
    import numpy as np

    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])

    # Kronecker product
    kron = np.kron(A, B)
    print(f"A = \n{A}\n")
    print(f"B = \n{B}\n")
    print(f"Kronecker product A (x) B = \n{kron}\n")

    # Hadamard (elementwise) product
    hadamard = A * B  # elementwise in NumPy
    print(f"Hadamard product A (*) B = \n{hadamard}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 8. Putting It Together: The Linear Algebra of Learning

    Let me connect the threads. A typical supervised learning pipeline, viewed through linear algebra:

    1. **Your data** is a matrix $X \in \mathbb{R}^{n \times d}$ -- each row a data point, each column a feature. This matrix defines a cloud of points in $\mathbb{R}^d$.

    2. **Preprocessing** often involves centering ($X - \bar{X}$) and scaling (dividing by standard deviations). The covariance matrix $\frac{1}{n-1}X^\top X$ is symmetric PSD.

    3. **Dimensionality reduction** (PCA) finds the eigenvectors of this covariance matrix -- equivalently, the right singular vectors of $X$. You project onto the top $k$ to get $X_k \in \mathbb{R}^{n \times k}$.

    4. **Linear regression** finds $\mathbf{w}$ such that $X\mathbf{w} \approx \mathbf{y}$. The solution $\hat{\mathbf{w}} = (X^\top X)^{-1}X^\top\mathbf{y}$ is projection. Ridge regression adds $\lambda I$ to make $(X^\top X + \lambda I)$ invertible and better-conditioned.

    5. **Gradient descent** updates $\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla L$. The Hessian $H = \nabla^2 L$ (a symmetric matrix) determines the curvature. Its eigenvalues tell you the curvature in each direction -- if eigenvalues vary wildly (ill-conditioning), gradient descent oscillates and converges slowly. This is why preconditioning (Newton's method uses $H^{-1}$) helps.

    6. **Neural networks** stack linear transformations with nonlinearities: $\mathbf{h}_\ell = \sigma(W_\ell \mathbf{h}_{\ell-1} + \mathbf{b}_\ell)$. Each $W_\ell$ is a learned linear map. The singular values of these weight matrices control information flow -- if they are all near 1, gradients propagate cleanly (orthogonal initialization). If they decay or explode, you get vanishing or exploding gradients.

    Every one of these steps is linear algebra. Master it, and the rest of this course will feel like variations on a theme.
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    # End-to-end demo: OLS as projection
    rng = np.random.default_rng(42)

    # Generate data: y = 2x + 1 + noise
    n = 50
    x = rng.uniform(0, 5, n)
    y = 2 * x + 1 + rng.standard_normal(n) * 0.8

    # Design matrix with intercept
    X = np.column_stack([np.ones(n), x])

    # OLS solution: w = (X^T X)^{-1} X^T y
    w_hat = np.linalg.solve(X.T @ X, X.T @ y)
    y_hat = X @ w_hat
    residuals = y - y_hat

    print(f"Estimated coefficients: intercept={w_hat[0]:.3f}, slope={w_hat[1]:.3f}")
    print(f"Residuals orthogonal to X columns:")
    print(f"  X[:,0]^T @ residuals = {X[:, 0] @ residuals:.10f}")
    print(f"  X[:,1]^T @ residuals = {X[:, 1] @ residuals:.10f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, alpha=0.6, label='Data')
    x_line = np.linspace(0, 5, 100)
    ax.plot(x_line, w_hat[0] + w_hat[1] * x_line, 'r-', linewidth=2,
            label=f'OLS: y = {w_hat[1]:.2f}x + {w_hat[0]:.2f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Ordinary Least Squares as Projection')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Practice Exercises

    These exercises are designed to refresh your computational skills while building ML intuition.

    **Exercise 1 -- Norms and Regularization Geometry.**
    Plot the unit balls for $L^1$, $L^2$, and $L^\infty$ norms in $\mathbb{R}^2$. Now draw a family of concentric ellipses representing level sets of a quadratic loss function. Visually confirm that the $L^1$ ball touches the ellipse at a corner (on an axis), giving a sparse solution, while the $L^2$ ball touches it at a generic point.

    **Exercise 2 -- Projection and Least Squares.**
    Let $X = \begin{pmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{pmatrix}$ and $\mathbf{y} = \begin{pmatrix} 1 \\ 2 \\ 4 \end{pmatrix}$.
    (a) Compute $\hat{\mathbf{w}} = (X^\top X)^{-1}X^\top \mathbf{y}$.
    (b) Compute $\hat{\mathbf{y}} = X\hat{\mathbf{w}}$.
    (c) Verify that the residual $\mathbf{y} - \hat{\mathbf{y}}$ is orthogonal to both columns of $X$.
    """)
    return


@app.cell
def _():
    import numpy as np

    # Exercise 2 solution scaffold
    X = np.array([[1, 1], [1, 2], [1, 3]], dtype=float)
    y = np.array([1, 2, 4], dtype=float)

    # (a) Compute w_hat
    w_hat = np.linalg.solve(X.T @ X, X.T @ y)
    print(f"w_hat = {w_hat}")

    # (b) Compute y_hat
    y_hat = X @ w_hat
    print(f"y_hat = {y_hat}")

    # (c) Verify orthogonality
    residual = y - y_hat
    print(f"residual = {residual}")
    print(f"X[:,0] . residual = {X[:, 0] @ residual:.10f}")
    print(f"X[:,1] . residual = {X[:, 1] @ residual:.10f}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Exercise 3 -- Eigenvalues and PD Matrices.**
    Consider $A = \begin{pmatrix} 4 & 2 \\ 2 & 3 \end{pmatrix}$.
    (a) Find the eigenvalues and eigenvectors.
    (b) Is $A$ positive definite? Verify both by checking eigenvalue signs and by evaluating $\mathbf{x}^\top A \mathbf{x}$ for a few test vectors.
    (c) Write the spectral decomposition $A = Q\Lambda Q^\top$ and verify by multiplication.

    **Exercise 4 -- SVD and Low-Rank Approximation.**
    Let $B = \begin{pmatrix} 3 & 0 \\ 0 & 2 \\ 0 & 0 \end{pmatrix}$.
    (a) Compute the SVD $B = U\Sigma V^\top$.
    (b) What is the best rank-1 approximation of $B$?
    (c) What fraction of the Frobenius norm is captured by this rank-1 approximation?

    **Exercise 5 -- PCA as SVD.**
    Generate a 100x5 random data matrix in Python (use `np.random.randn`). Center it. Compute PCA two ways: (i) eigendecompose the covariance matrix, (ii) compute the SVD of the centered data matrix. Verify that both methods yield the same principal components (up to sign flips).

    **Exercise 6 -- Condition Number.**
    For the matrix $X^\top X$ from Exercise 2, compute the condition number $\kappa = \sigma_{\max}/\sigma_{\min}$. Now add $\lambda I$ for $\lambda = 0.1, 1, 10$ and recompute. How does regularization affect conditioning?
    """)
    return


@app.cell
def _():
    import numpy as np

    # Exercise 6: Condition number and regularization
    X = np.array([[1, 1], [1, 2], [1, 3]], dtype=float)
    XtX = X.T @ X

    S = np.linalg.svd(XtX, compute_uv=False)
    kappa = S[0] / S[-1]
    print(f"X^T X = \n{XtX}")
    print(f"Condition number: {kappa:.4f}\n")

    for lam in [0.1, 1, 10]:
        XtX_reg = XtX + lam * np.eye(2)
        S_reg = np.linalg.svd(XtX_reg, compute_uv=False)
        kappa_reg = S_reg[0] / S_reg[-1]
        print(f"lambda={lam:5.1f} -> condition number: {kappa_reg:.4f}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Further Reading

    - [MML Chapter 2: Linear Algebra](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) -- comprehensive treatment of vector spaces, linear maps, and matrix decompositions
    - [MML Chapter 3: Analytic Geometry](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) -- norms, inner products, projections, rotations
    - [MML Chapter 4: Matrix Decompositions](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) -- eigendecomposition, SVD, Cholesky
    - [Boyd Section 2.5-3.1](file:///C:/Users/landa/ml-course/textbooks/Boyd-ConvexOptimization.pdf) -- positive semidefiniteness and convexity
    - [ESL Section 3.2: Linear Regression](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf) -- the projection perspective on OLS
    - [Murphy Chapter 7: Linear Algebra Background](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf) -- concise reference with ML focus
    """)
    return


if __name__ == "__main__":
    app.run()
