import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np

    return (np,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Linear Algebra Refresh

    You already know linear algebra. You can multiply matrices, find eigenvalues, maybe even compute an SVD by hand if someone put a gun to your head. Good. This module is not about re-teaching those mechanics -- it is about reframing everything you learned through the lens of machine learning, so that when you encounter a covariance matrix or a projection operator in a paper, you see it for what it actually is rather than treating it as abstract formalism.

    Linear algebra is not a prerequisite for ML in the way that, say, knowing how to read is a prerequisite for studying literature. It is more fundamental than that. Linear algebra *is* the language ML is written in. Every dataset is a matrix. Every model parameter is a vector. Every prediction is a linear (or locally linear) transformation. Every optimization landscape has curvature described by eigenvalues. If your linear algebra is rusty, everything downstream -- PCA, SVD-based recommenders, neural network weight matrices, attention mechanisms -- will feel like arbitrary symbol manipulation instead of geometry.

    So let us sharpen the blade.

    > **Primary reference:** [MML Chapter 2: Linear Algebra](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) and [MML Chapter 3: Analytic Geometry](file:///C:/Users/landa/ml-course/textbooks/MML.pdf). These two chapters are excellent and I will reference specific pages throughout. Skim them alongside this lecture.
    """)
    return


@app.cell(hide_code=True)
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
def _(np):
    # Norms in numpy: ||x||_1, ||x||_2, ||x||_inf
    x = np.array([3.0, -4.0, 1.0, -2.0])

    l1 = np.linalg.norm(x, ord=1)       # sum of absolute values
    l2 = np.linalg.norm(x, ord=2)       # Euclidean length
    linf = np.linalg.norm(x, ord=np.inf) # max absolute value

    print(f"x = {x}")
    print(f"L1  norm: {l1:.4f}  (= |3|+|4|+|1|+|2| = 10)")
    print(f"L2  norm: {l2:.4f}  (= sqrt(9+16+1+4) = sqrt(30))")
    print(f"Linf norm: {linf:.4f}  (= max(3,4,1,2) = 4)")

    # Unit-normalizing a vector (used everywhere: cosine similarity, etc.)
    x_unit = x / np.linalg.norm(x)
    print(f"\nUnit vector: {np.round(x_unit, 4)}, norm = {np.linalg.norm(x_unit):.4f}")
    return


@app.cell(hide_code=True)
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
def _(np):
    # Inner product and cosine similarity: x^T y = ||x|| ||y|| cos(theta)
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])

    dot_product = a @ b  # x^T y
    cos_sim = dot_product / (np.linalg.norm(a) * np.linalg.norm(b))  # cosine similarity
    angle_rad = np.arccos(np.clip(cos_sim, -1, 1))

    print(f"a = {a}, b = {b}")
    print(f"a^T b = {dot_product:.4f}")
    print(f"Cosine similarity = {cos_sim:.4f}")
    print(f"Angle between a and b = {np.degrees(angle_rad):.2f} degrees")

    # Orthogonal vectors have cos(theta) = 0
    c = np.array([1.0, 0.0])
    d = np.array([0.0, 1.0])
    print(f"\nc = {c}, d = {d}")
    print(f"c^T d = {c @ d:.1f}  (orthogonal)")
    return


@app.cell(hide_code=True)
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
def _(np):
    # Matrix transformations in R^2: rotation, scaling, shearing, projection
    theta = np.pi / 4  # 45 degrees

    # Rotation matrix
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])

    # Scaling matrix
    S = np.array([[2, 0],
                  [0, 0.5]])

    # Apply to a vector
    v = np.array([1.0, 0.0])
    print(f"Original vector:  {v}")
    print(f"After rotation:   {np.round(R @ v, 4)}  (45 deg)")
    print(f"After scaling:    {S @ v}  (stretch x by 2, shrink y by 0.5)")

    # Composition: rotate then scale vs scale then rotate
    print(f"\nS @ R @ v = {np.round(S @ R @ v, 4)}")
    print(f"R @ S @ v = {np.round(R @ S @ v, 4)}")
    print("These differ -- matrix multiplication is not commutative!")
    return


@app.cell(hide_code=True)
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
def _(np):
    # Rank and pseudo-inverse
    # Full-rank matrix
    M_full = np.array([[1, 2], [3, 4]])
    print(f"Full-rank matrix:\n{M_full}")
    print(f"Rank = {np.linalg.matrix_rank(M_full)}")

    # Rank-deficient matrix (row 2 = 2 * row 1)
    M_deficient = np.array([[1, 2], [2, 4]])
    print(f"\nRank-deficient matrix:\n{M_deficient}")
    print(f"Rank = {np.linalg.matrix_rank(M_deficient)}")

    # Pseudo-inverse: A+ = (A^T A)^{-1} A^T for overdetermined systems
    A_tall = np.array([[1, 1], [1, 2], [1, 3]], dtype=float)
    A_pinv = np.linalg.pinv(A_tall)  # Moore-Penrose pseudo-inverse
    print(f"\nA (3x2):\n{A_tall}")
    print(f"A+ (2x3):\n{np.round(A_pinv, 4)}")
    print(f"A+ @ A = I? {np.allclose(A_pinv @ A_tall, np.eye(2))}")
    return


@app.cell(hide_code=True)
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
def _(np):
    # Three cases of Ax = b

    # Case 1: Exact (square, full rank) -- unique solution
    A_exact = np.array([[2, 1], [1, 3]], dtype=float)
    b_exact = np.array([5, 7], dtype=float)
    x_exact = np.linalg.solve(A_exact, b_exact)  # x = A^{-1} b
    print(f"Exact system (2x2): x = {x_exact}")

    # Case 2: Overdetermined (more equations than unknowns) -- least squares
    A_over = np.array([[1, 1], [1, 2], [1, 3]], dtype=float)
    b_over = np.array([1.1, 1.9, 3.2], dtype=float)
    x_ls, residuals, rank, sv = np.linalg.lstsq(A_over, b_over, rcond=None)
    print(f"\nOverdetermined (3x2): x_ls = {np.round(x_ls, 4)}")
    print(f"  Residual norm = {np.linalg.norm(A_over @ x_ls - b_over):.6f}")

    # Case 3: Underdetermined (fewer equations than unknowns) -- min-norm solution
    A_under = np.array([[1, 1, 1]], dtype=float)
    b_under = np.array([3.0])
    x_minnorm = np.linalg.pinv(A_under) @ b_under  # minimum-norm solution
    print(f"\nUnderdetermined (1x3): x_minnorm = {np.round(x_minnorm, 4)}")
    print(f"  A @ x = {A_under @ x_minnorm}  (satisfies equation)")
    print(f"  ||x|| = {np.linalg.norm(x_minnorm):.4f}  (smallest possible norm)")
    return


@app.cell(hide_code=True)
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
def _(np):
    # Projection: P = A (A^T A)^{-1} A^T
    A_proj = np.array([[1, 1], [1, 2], [1, 3]], dtype=float)
    y_proj = np.array([1.0, 2.0, 4.0])

    # Projection matrix (hat matrix)
    P = A_proj @ np.linalg.solve(A_proj.T @ A_proj, A_proj.T)
    y_hat_proj = P @ y_proj
    residual_proj = y_proj - y_hat_proj

    print(f"y     = {y_proj}")
    print(f"y_hat = {np.round(y_hat_proj, 4)}  (projection onto col(X))")
    print(f"resid = {np.round(residual_proj, 4)}")
    print(f"\nResidual orthogonal to col(X)?")
    print(f"  X[:,0] . resid = {A_proj[:, 0] @ residual_proj:.1e}")
    print(f"  X[:,1] . resid = {A_proj[:, 1] @ residual_proj:.1e}")

    # QR decomposition: A = QR, solve Rx = Q^T b
    Q, R_qr = np.linalg.qr(A_proj)
    w_qr = np.linalg.solve(R_qr, Q.T @ y_proj)  # back-substitution
    w_ols = np.linalg.solve(A_proj.T @ A_proj, A_proj.T @ y_proj)
    print(f"\nOLS via normal equations: {np.round(w_ols, 6)}")
    print(f"OLS via QR:              {np.round(w_qr, 6)}")
    return


@app.cell(hide_code=True)
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
def _(np):
    # Eigendecomposition: A v = lambda v
    # Diagonalization: A = P D P^{-1}, so A^k = P D^k P^{-1}
    A_eig = np.array([[2, 1], [1, 2]], dtype=float)
    eigenvalues_demo, P_eig = np.linalg.eig(A_eig)
    D_eig = np.diag(eigenvalues_demo)

    print(f"A = \n{A_eig}")
    print(f"Eigenvalues: {eigenvalues_demo}")
    print(f"Eigenvectors (cols of P):\n{np.round(P_eig, 4)}")

    # Verify: A v = lambda v
    for i in range(2):
        lhs = A_eig @ P_eig[:, i]
        rhs = eigenvalues_demo[i] * P_eig[:, i]
        print(f"\nA @ v{i} = {np.round(lhs, 4)},  lambda*v{i} = {np.round(rhs, 4)}")

    # Matrix power via diagonalization: A^5 = P D^5 P^{-1}
    k = 5
    Ak_direct = np.linalg.matrix_power(A_eig, k)
    Ak_diag = P_eig @ np.diag(eigenvalues_demo**k) @ np.linalg.inv(P_eig)
    print(f"\nA^{k} (direct):\n{Ak_direct}")
    print(f"A^{k} (via P D^k P^-1):\n{np.round(Ak_diag, 4)}")
    return


@app.cell(hide_code=True)
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
def _(np):
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
def _(np):
    # Positive definiteness: test x^T A x > 0 for random vectors
    A_pd = np.array([[4, 2], [2, 3]], dtype=float)
    rng_pd = np.random.default_rng(0)
    test_vectors = rng_pd.standard_normal((5, 2))

    print("Testing x^T A x > 0 for random vectors:")
    for i in range(5):
        x_test = test_vectors[i]
        quad_form = x_test @ A_pd @ x_test
        print(f"  x={np.round(x_test, 3)} -> x^T A x = {quad_form:.4f} > 0? {quad_form > 0}")

    # A PSD matrix (not PD): has a zero eigenvalue
    A_psd = np.array([[1, 1], [1, 1]], dtype=float)
    eigs_psd = np.linalg.eigvalsh(A_psd)
    print(f"\nPSD matrix eigenvalues: {eigs_psd}  (one is zero -> PSD, not PD)")
    return


@app.cell(hide_code=True)
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
def _(np):
    # SVD: A = U Sigma V^T
    A_svd = np.array([[3, 0], [0, 2], [0, 0]], dtype=float)
    U_svd, S_svd, Vt_svd = np.linalg.svd(A_svd, full_matrices=True)

    print(f"A (3x2) = \n{A_svd}\n")
    print(f"U (3x3) = \n{np.round(U_svd, 4)}\n")
    print(f"Singular values: {S_svd}")
    print(f"V^T (2x2) = \n{np.round(Vt_svd, 4)}\n")

    # Reconstruct: U @ Sigma @ V^T
    Sigma_full = np.zeros_like(A_svd, dtype=float)
    np.fill_diagonal(Sigma_full, S_svd)
    A_svd_recon = U_svd @ Sigma_full @ Vt_svd
    print(f"Reconstruction matches: {np.allclose(A_svd, A_svd_recon)}")

    # Verify: singular values = sqrt(eigenvalues of A^T A)
    eigs_ATA = np.linalg.eigvalsh(A_svd.T @ A_svd)[::-1]
    print(f"\nsqrt(eig(A^T A)) = {np.round(np.sqrt(eigs_ATA), 4)}")
    print(f"Singular values  = {S_svd}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Low-Rank Approximation

    The **Eckart-Young theorem** states that the best rank-$k$ approximation to $A$ (in both Frobenius and spectral norms) is obtained by keeping only the top $k$ singular values:

    $$A_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$$

    This is not just a theorem -- it is a design principle. When you truncate the SVD, you keep the $k$ most important "modes" of the matrix and discard the rest as noise. The fraction of "energy" retained is $\sum_{i=1}^{k} \sigma_i^2 / \sum_{i=1}^{r} \sigma_i^2$.
    """)
    return


@app.cell
def _(np):
    # Low-rank approximation: A_k = sum of top-k singular triplets
    rng_lr = np.random.default_rng(42)
    # Create a rank-2 matrix with some noise
    true_U_lr = rng_lr.standard_normal((10, 2))
    true_V_lr = rng_lr.standard_normal((8, 2))
    A_lr = true_U_lr @ true_V_lr.T + 0.1 * rng_lr.standard_normal((10, 8))

    U_lr, S_lr, Vt_lr = np.linalg.svd(A_lr, full_matrices=False)
    print(f"Singular values: {np.round(S_lr, 3)}")

    # Energy captured by top-k singular values
    total_energy = np.sum(S_lr**2)
    for k in range(1, min(len(S_lr), 6) + 1):
        energy_k = np.sum(S_lr[:k]**2) / total_energy
        # Rank-k approximation: A_k = U[:,:k] @ diag(S[:k]) @ Vt[:k,:]
        A_k = U_lr[:, :k] @ np.diag(S_lr[:k]) @ Vt_lr[:k, :]
        approx_err = np.linalg.norm(A_lr - A_k, 'fro') / np.linalg.norm(A_lr, 'fro')
        print(f"  Rank-{k}: {energy_k:.4f} energy, {approx_err:.4f} relative error")
    return


@app.cell(hide_code=True)
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
def _(np):
    # PCA via eigendecomposition vs SVD -- verify they give the same result
    rng_pca = np.random.default_rng(42)
    X_pca = rng_pca.standard_normal((100, 5))
    X_centered = X_pca - X_pca.mean(axis=0)

    # Method 1: Eigendecomposition of covariance matrix
    C = (X_centered.T @ X_centered) / (X_centered.shape[0] - 1)
    eig_vals, eig_vecs = np.linalg.eigh(C)
    # Sort by descending eigenvalue
    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]

    # Method 2: SVD of centered data matrix
    U_pca, S_pca, Vt_pca = np.linalg.svd(X_centered, full_matrices=False)
    svd_eig_vals = S_pca**2 / (X_centered.shape[0] - 1)

    print("Eigenvalues from covariance matrix:")
    print(f"  {np.round(eig_vals, 6)}")
    print("Eigenvalues from SVD (sigma^2 / (n-1)):")
    print(f"  {np.round(svd_eig_vals, 6)}")
    print(f"\nMatch: {np.allclose(eig_vals, svd_eig_vals)}")

    # Principal components match (up to sign)
    print("\nPrincipal components match (up to sign flips):")
    for i in range(5):
        match = np.allclose(np.abs(eig_vecs[:, i]), np.abs(Vt_pca[i, :]))
        print(f"  PC{i+1}: {match}")
    return


@app.cell(hide_code=True)
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
def _(np):
    # Trace and determinant demonstrations
    A_td = np.array([[4, 2], [2, 3]], dtype=float)
    eigenvalues_td = np.linalg.eigvalsh(A_td)

    print(f"A = \n{A_td}\n")
    print(f"Eigenvalues: {eigenvalues_td}")
    print(f"Trace(A) = {np.trace(A_td):.4f}")
    print(f"Sum of eigenvalues = {np.sum(eigenvalues_td):.4f}")
    print(f"Det(A) = {np.linalg.det(A_td):.4f}")
    print(f"Product of eigenvalues = {np.prod(eigenvalues_td):.4f}")

    # Cyclic property: tr(AB) = tr(BA)
    B_td = np.array([[1, 3], [2, 4]], dtype=float)
    print(f"\ntr(AB) = {np.trace(A_td @ B_td):.4f}")
    print(f"tr(BA) = {np.trace(B_td @ A_td):.4f}")

    # Frobenius norm via trace: ||A||_F^2 = tr(A^T A)
    frob_norm_sq = np.linalg.norm(A_td, 'fro')**2
    trace_ata = np.trace(A_td.T @ A_td)
    print(f"\n||A||_F^2 = {frob_norm_sq:.4f}")
    print(f"tr(A^T A) = {trace_ata:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Kronecker and Hadamard Products

    Two products you will encounter occasionally:

    **Kronecker product** $A \otimes B$: For $A \in \mathbb{R}^{m \times n}$, $B \in \mathbb{R}^{p \times q}$, produces an $mp \times nq$ block matrix. Shows up in multi-task learning (modeling correlations between tasks and features simultaneously) and in vectorizing matrix equations: $\text{vec}(AXB) = (B^\top \otimes A)\text{vec}(X)$.

    **Hadamard (elementwise) product** $A \odot B$: Same-shaped matrices, multiply entry by entry. Ubiquitous in neural networks -- gating mechanisms in LSTMs and GRUs use elementwise products, and attention mechanisms use elementwise operations extensively.
    """)
    return


@app.cell
def _(np):
    A_kh = np.array([[1, 2], [3, 4]])
    B_kh = np.array([[5, 6], [7, 8]])

    # Kronecker product
    kron = np.kron(A_kh, B_kh)
    print(f"A = \n{A_kh}\n")
    print(f"B = \n{B_kh}\n")
    print(f"Kronecker product A (x) B = \n{kron}\n")

    # Hadamard (elementwise) product
    hadamard = A_kh * B_kh  # elementwise in NumPy
    print(f"Hadamard product A (*) B = \n{hadamard}")
    return


@app.cell(hide_code=True)
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
def _(np):
    import matplotlib.pyplot as plt

    # End-to-end demo: OLS as projection
    rng_ols = np.random.default_rng(42)

    # Generate data: y = 2x + 1 + noise
    n_ols = 50
    x_ols = rng_ols.uniform(0, 5, n_ols)
    y_ols = 2 * x_ols + 1 + rng_ols.standard_normal(n_ols) * 0.8

    # Design matrix with intercept
    X_ols = np.column_stack([np.ones(n_ols), x_ols])

    # OLS solution: w = (X^T X)^{-1} X^T y
    w_hat_ols = np.linalg.solve(X_ols.T @ X_ols, X_ols.T @ y_ols)
    y_hat_ols = X_ols @ w_hat_ols
    residuals_ols = y_ols - y_hat_ols

    print(f"Estimated coefficients: intercept={w_hat_ols[0]:.3f}, slope={w_hat_ols[1]:.3f}")
    print(f"Residuals orthogonal to X columns:")
    print(f"  X[:,0]^T @ residuals = {X_ols[:, 0] @ residuals_ols:.10f}")
    print(f"  X[:,1]^T @ residuals = {X_ols[:, 1] @ residuals_ols:.10f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_ols, y_ols, alpha=0.6, label='Data')
    x_line = np.linspace(0, 5, 100)
    ax.plot(x_line, w_hat_ols[0] + w_hat_ols[1] * x_line, 'r-', linewidth=2,
            label=f'OLS: y = {w_hat_ols[1]:.2f}x + {w_hat_ols[0]:.2f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Ordinary Least Squares as Projection')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig
    return


@app.cell(hide_code=True)
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
def _(np):
    # Exercise 2 solution scaffold
    X_ex2 = np.array([[1, 1], [1, 2], [1, 3]], dtype=float)
    y_ex2 = np.array([1, 2, 4], dtype=float)

    # (a) Compute w_hat
    w_hat_ex2 = np.linalg.solve(X_ex2.T @ X_ex2, X_ex2.T @ y_ex2)
    print(f"w_hat = {w_hat_ex2}")

    # (b) Compute y_hat
    y_hat_ex2 = X_ex2 @ w_hat_ex2
    print(f"y_hat = {y_hat_ex2}")

    # (c) Verify orthogonality
    residual_ex2 = y_ex2 - y_hat_ex2
    print(f"residual = {residual_ex2}")
    print(f"X[:,0] . residual = {X_ex2[:, 0] @ residual_ex2:.10f}")
    print(f"X[:,1] . residual = {X_ex2[:, 1] @ residual_ex2:.10f}")
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
def _(np):
    # Exercise 6: Condition number and regularization
    X_ex6 = np.array([[1, 1], [1, 2], [1, 3]], dtype=float)
    XtX_ex6 = X_ex6.T @ X_ex6

    S_ex6 = np.linalg.svd(XtX_ex6, compute_uv=False)
    kappa_ex6 = S_ex6[0] / S_ex6[-1]
    print(f"X^T X = \n{XtX_ex6}")
    print(f"Condition number: {kappa_ex6:.4f}\n")

    for lam in [0.1, 1, 10]:
        XtX_reg = XtX_ex6 + lam * np.eye(2)
        S_reg = np.linalg.svd(XtX_reg, compute_uv=False)
        kappa_reg = S_reg[0] / S_reg[-1]
        print(f"lambda={lam:5.1f} -> condition number: {kappa_reg:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Code It: Implementation Exercises

    The exercises below test your ability to translate linear algebra concepts into working numpy code. Each problem gives a brief description; fill in the skeleton.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Code It 1: Gram-Schmidt Orthogonalization

    Implement the classical Gram-Schmidt process. Given a matrix whose columns are linearly independent vectors, produce an orthonormal basis for their span.

    **Input:** Matrix $A$ with linearly independent columns.
    **Output:** Matrix $Q$ with orthonormal columns spanning the same space.

    Recall the algorithm: for each vector $\mathbf{a}_k$, subtract its projections onto all previously computed basis vectors, then normalize.
    """)
    return


@app.cell
def _(np):
    def gram_schmidt(A):
        """Classical Gram-Schmidt: returns Q with orthonormal columns."""
        m, n = A.shape
        Q = np.zeros((m, n))
        for k in range(n):
            v = A[:, k].copy()
            # Subtract projections onto previous basis vectors
            for j in range(k):
                v -= (Q[:, j] @ A[:, k]) * Q[:, j]
            Q[:, k] = v / np.linalg.norm(v)
        return Q

    # Test it
    A_gs = np.array([[1, 1, 0],
                     [1, 0, 1],
                     [0, 1, 1]], dtype=float)
    Q_gs = gram_schmidt(A_gs)
    print(f"Q =\n{np.round(Q_gs, 4)}\n")
    print(f"Q^T Q (should be I):\n{np.round(Q_gs.T @ Q_gs, 10)}")
    return (gram_schmidt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Code It 2: Power Iteration for Dominant Eigenvector

    Implement the power iteration method to find the largest eigenvalue and its eigenvector for a symmetric matrix. This is the simplest eigensolver and the idea behind Google's original PageRank.

    **Algorithm:** Start with a random vector $\mathbf{b}_0$. Repeat: $\mathbf{b}_{k+1} = A\mathbf{b}_k / \|A\mathbf{b}_k\|$. The vector converges to the eigenvector with the largest absolute eigenvalue, and $\lambda \approx \mathbf{b}^\top A \mathbf{b}$ (Rayleigh quotient).
    """)
    return


@app.cell
def _(np):
    def power_iteration(A, num_iters=100):
        """Find dominant eigenvector via power iteration."""
        rng = np.random.default_rng(42)
        b = rng.standard_normal(A.shape[0])
        b = b / np.linalg.norm(b)

        for _ in range(num_iters):
            Ab = A @ b
            b = Ab / np.linalg.norm(Ab)

        # Rayleigh quotient: eigenvalue estimate
        eigenvalue = b @ A @ b
        return eigenvalue, b

    # Test: compare with numpy
    A_pow = np.array([[4, 2], [2, 3]], dtype=float)
    lam_pow, v_pow = power_iteration(A_pow)
    lam_true, V_true = np.linalg.eigh(A_pow)

    print(f"Power iteration:  lambda = {lam_pow:.6f}, v = {np.round(v_pow, 6)}")
    print(f"numpy eigh:       lambda = {lam_true[-1]:.6f}, v = {np.round(V_true[:, -1], 6)}")
    print(f"Eigenvalue match: {np.isclose(lam_pow, lam_true[-1])}")
    return (power_iteration,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Code It 3: Ridge Regression from Scratch

    Implement ridge regression: $\hat{\mathbf{w}} = (X^\top X + \lambda I)^{-1} X^\top \mathbf{y}$.

    Compare the ridge solution for several values of $\lambda$ against OLS ($\lambda=0$). Observe how increasing $\lambda$ shrinks the coefficients toward zero and improves the condition number.
    """)
    return


@app.cell
def _(np):
    def ridge_regression(X, y, lam):
        """Ridge regression: w = (X^T X + lambda I)^{-1} X^T y"""
        d = X.shape[1]
        # Solve (X^T X + lambda I) w = X^T y
        w = np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)
        return w

    # Generate synthetic data
    rng_ridge = np.random.default_rng(42)
    n_r, d_r = 50, 5
    X_ridge = rng_ridge.standard_normal((n_r, d_r))
    w_true = np.array([3.0, -1.0, 0.5, 0.0, 0.0])  # last two are zero
    y_ridge = X_ridge @ w_true + 0.5 * rng_ridge.standard_normal(n_r)

    print(f"True weights:       {w_true}")
    print(f"{'lambda':>10s} | {'||w||':>8s} | {'cond(X^TX+lI)':>14s} | weights")
    print("-" * 75)
    for lam_r in [0, 0.01, 0.1, 1.0, 10.0]:
        w_r = ridge_regression(X_ridge, y_ridge, lam_r)
        cond = np.linalg.cond(X_ridge.T @ X_ridge + lam_r * np.eye(d_r))
        print(f"{lam_r:10.2f} | {np.linalg.norm(w_r):8.4f} | {cond:14.2f} | {np.round(w_r, 3)}")
    return (ridge_regression,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Code It 4: Image Compression via Truncated SVD

    Use SVD to compress a synthetic "image" (matrix). Reconstruct it with increasing rank $k$ and measure the relative Frobenius error $\|A - A_k\|_F / \|A\|_F$.
    """)
    return


@app.cell
def _(np):
    # SVD image compression: reconstruct matrix with top-k singular values
    rng_img = np.random.default_rng(42)

    # Create a structured 50x50 "image" (low-rank + noise)
    rows = np.arange(50).reshape(-1, 1)
    cols = np.arange(50).reshape(1, -1)
    image = np.sin(rows * 0.2) * np.cos(cols * 0.15) + 0.05 * rng_img.standard_normal((50, 50))

    U_img, S_img, Vt_img = np.linalg.svd(image, full_matrices=False)
    norm_original = np.linalg.norm(image, 'fro')

    print(f"Image shape: {image.shape}, rank: {np.linalg.matrix_rank(image)}")
    print(f"Top 10 singular values: {np.round(S_img[:10], 3)}\n")
    print(f"{'k':>3s} | {'rel error':>10s} | {'energy':>8s} | {'compression':>12s}")
    print("-" * 45)
    total_energy_img = np.sum(S_img**2)
    for k_img in [1, 2, 5, 10, 25]:
        # Rank-k reconstruction: U[:,:k] @ diag(S[:k]) @ Vt[:k,:]
        recon = U_img[:, :k_img] @ np.diag(S_img[:k_img]) @ Vt_img[:k_img, :]
        rel_err = np.linalg.norm(image - recon, 'fro') / norm_original
        energy = np.sum(S_img[:k_img]**2) / total_energy_img
        # Storage: k*(50+50+1) vs 50*50
        storage_ratio = k_img * (50 + 50 + 1) / (50 * 50)
        print(f"{k_img:3d} | {rel_err:10.6f} | {energy:8.4f} | {storage_ratio:11.1%}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Code It 5: Cosine Similarity Search

    Given a "database" of unit-normalized vectors (think word embeddings), implement a function that finds the $k$ most similar vectors to a query using cosine similarity. This is the core operation behind nearest-neighbor search in embedding spaces.
    """)
    return


@app.cell
def _(np):
    def cosine_knn(database, query, k=3):
        """Find k most similar vectors by cosine similarity.

        database: (n, d) array of unit-normalized vectors
        query: (d,) unit-normalized vector
        Returns indices and similarities of top-k matches.
        """
        # Cosine similarity = dot product for unit vectors
        similarities = database @ query
        top_k_idx = np.argsort(similarities)[::-1][:k]
        return top_k_idx, similarities[top_k_idx]

    # Simulate a small embedding database
    rng_cos = np.random.default_rng(42)
    n_vecs, dim = 1000, 50
    db = rng_cos.standard_normal((n_vecs, dim))
    db = db / np.linalg.norm(db, axis=1, keepdims=True)  # unit normalize

    query_vec = db[42] + 0.1 * rng_cos.standard_normal(dim)  # perturbed version of vector 42
    query_vec = query_vec / np.linalg.norm(query_vec)

    indices, sims = cosine_knn(db, query_vec, k=5)
    print("Top-5 most similar vectors to perturbed version of vector 42:")
    for rank, (idx, sim) in enumerate(zip(indices, sims)):
        print(f"  #{rank+1}: index={idx}, cosine_sim={sim:.4f}")
    return (cosine_knn,)


@app.cell(hide_code=True)
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
