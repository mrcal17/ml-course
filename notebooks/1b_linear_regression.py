import marimo

app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV, LassoCV
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error
    return (mo, np, plt, PolynomialFeatures, LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, train_test_split, r2_score, mean_squared_error)


@app.cell
def _(mo):
    mo.md(r"""
    # Linear Regression

    ## 1. The Simplest ML Model — And the Most Important to Understand Deeply

    Linear regression is the hydrogen atom of machine learning. It is the simplest system that exhibits nearly every phenomenon you will encounter in more complex models: optimization, overfitting, regularization, the bias-variance tradeoff, probabilistic interpretation, and generalization. If you understand linear regression at the level we are going to cover it, you will have a structural advantage when encountering neural networks, kernel methods, and Bayesian models later. Every one of those builds on ideas that appear here first.

    The model is:

    $$y = X\mathbf{w} + \boldsymbol{\varepsilon}$$

    where:
    - $y \in \mathbb{R}^n$ is the vector of observations (targets),
    - $X \in \mathbb{R}^{n \times p}$ is the design matrix — $n$ data points, each with $p$ features,
    - $\mathbf{w} \in \mathbb{R}^p$ is the weight vector (parameters we want to learn),
    - $\boldsymbol{\varepsilon} \in \mathbb{R}^n$ is the noise vector.

    We assume:
    1. **Linearity**: the true relationship between inputs and output is linear in the parameters $\mathbf{w}$ (not necessarily linear in the raw inputs — we will exploit this later with basis functions).
    2. **Noise structure**: $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$, i.i.d. The noise is Gaussian, has zero mean, constant variance (homoscedasticity), and observations are independent.

    These assumptions may seem restrictive, but they buy us closed-form solutions and deep geometric insight. And when we relax them, we do so deliberately, understanding exactly what we gain and what we lose.

    > **Reading**: [MML S9.1-9.2](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) for the model setup, [ISLR S3.1](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf) for motivation and examples, [Bishop S3.1](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) for the pattern recognition perspective.
    """)
    return


@app.cell
def _(np):
    # y = Xw + epsilon: the linear model in numpy
    _rng = np.random.default_rng(7)
    n_demo, p_demo = 50, 2
    X_demo = _rng.standard_normal((n_demo, p_demo))             # design matrix (n x p)
    X_demo = np.c_[np.ones(n_demo), X_demo]               # prepend 1s column for intercept
    w_demo_true = np.array([5.0, -2.0, 3.0])              # true weights [intercept, w1, w2]
    epsilon_demo = 0.4 * _rng.standard_normal(n_demo)           # Gaussian noise
    y_demo = X_demo @ w_demo_true + epsilon_demo           # y = Xw + eps

    print(f"X shape: {X_demo.shape}  (n={n_demo}, p={p_demo}+1 with intercept)")
    print(f"y shape: {y_demo.shape}")
    print(f"True weights: {w_demo_true}")
    return (X_demo, y_demo, w_demo_true, n_demo, p_demo)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2. Ordinary Least Squares (OLS)

    ### The Loss Function

    We want to find $\mathbf{w}$ that makes our predictions $\hat{y} = X\mathbf{w}$ as close to $y$ as possible. The most natural measure of "closeness" is the sum of squared residuals:

    $$\mathcal{L}(\mathbf{w}) = \|y - X\mathbf{w}\|^2 = (y - X\mathbf{w})^\top(y - X\mathbf{w})$$

    Why squared error? Three reasons that reinforce each other: (1) it is differentiable everywhere, (2) it has a unique global minimum when $X^\top X$ is invertible, and (3) it falls out naturally from the probabilistic view under Gaussian noise, which we will derive in Section 4.

    ### Deriving the Normal Equation

    Let us expand the loss and differentiate. This is a derivation you should be able to reproduce from memory.

    $$\mathcal{L}(\mathbf{w}) = (y - X\mathbf{w})^\top(y - X\mathbf{w})$$

    Expand:

    $$= y^\top y - y^\top X\mathbf{w} - \mathbf{w}^\top X^\top y + \mathbf{w}^\top X^\top X \mathbf{w}$$

    Since $y^\top X\mathbf{w}$ is a scalar, it equals its transpose $\mathbf{w}^\top X^\top y$. So:

    $$\mathcal{L}(\mathbf{w}) = y^\top y - 2\mathbf{w}^\top X^\top y + \mathbf{w}^\top X^\top X \mathbf{w}$$

    Take the gradient with respect to $\mathbf{w}$. Recall the matrix calculus identities: $\nabla_\mathbf{w}(\mathbf{a}^\top \mathbf{w}) = \mathbf{a}$ and $\nabla_\mathbf{w}(\mathbf{w}^\top A\mathbf{w}) = 2A\mathbf{w}$ when $A$ is symmetric (and $X^\top X$ is always symmetric):

    $$\nabla_\mathbf{w}\mathcal{L} = -2X^\top y + 2X^\top X\mathbf{w}$$

    Set the gradient to zero:

    $$X^\top X\mathbf{w}^* = X^\top y$$

    This is the **normal equation**. If $X^\top X$ is invertible:

    $$\boxed{\mathbf{w}^* = (X^\top X)^{-1} X^\top y}$$

    This is one of the most important equations in all of machine learning.
    """)
    return


@app.cell
def _(np, X_demo, y_demo, w_demo_true):
    # Normal equation: w* = (X^T X)^{-1} X^T y
    # Method 1: explicit inverse (conceptual, avoid in practice)
    w_normal_inv = np.linalg.inv(X_demo.T @ X_demo) @ X_demo.T @ y_demo

    # Method 2: np.linalg.solve (numerically stable — solves X^T X w = X^T y)
    w_normal_solve = np.linalg.solve(X_demo.T @ X_demo, X_demo.T @ y_demo)

    print(f"True weights:       {w_demo_true}")
    print(f"Normal eq (inv):    {w_normal_inv}")
    print(f"Normal eq (solve):  {w_normal_solve}")
    print(f"Max diff inv vs solve: {np.max(np.abs(w_normal_inv - w_normal_solve)):.2e}")
    return (w_normal_solve,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Geometric Interpretation

    Here is where linear algebra gives you real intuition. The predicted vector $\hat{y} = X\mathbf{w}^*$ is the **orthogonal projection** of $y$ onto the column space of $X$, denoted $\text{Col}(X)$.

    Think about it: $\text{Col}(X)$ is the set of all vectors of the form $X\mathbf{w}$ for any $\mathbf{w}$. We want the point in this subspace closest to $y$. From Module 0C, you know the closest point is the orthogonal projection — the residual vector $y - \hat{y}$ must be perpendicular to $\text{Col}(X)$:

    $$X^\top(y - X\mathbf{w}^*) = \mathbf{0}$$

    Rearrange and you recover the normal equation. The "normal" in "normal equation" literally means perpendicular. The residual is normal to the column space.

    The projection matrix (or "hat matrix," because it puts the hat on $y$) is:

    $$H = X(X^\top X)^{-1}X^\top, \qquad \hat{y} = Hy$$

    This is an idempotent, symmetric matrix — exactly the properties of an orthogonal projector from linear algebra.
    """)
    return


@app.cell
def _(np, X_demo, y_demo, w_normal_solve):
    # Hat matrix: H = X (X^T X)^{-1} X^T
    H_demo = X_demo @ np.linalg.inv(X_demo.T @ X_demo) @ X_demo.T

    # y_hat = H y  (projection onto Col(X))
    y_hat_demo = H_demo @ y_demo
    residual_demo = y_demo - y_hat_demo

    # Verify: residual is orthogonal to column space (X^T r ≈ 0)
    ortho_check = X_demo.T @ residual_demo
    print(f"X^T @ residual (should be ~0): {ortho_check}")

    # Verify: H is idempotent (H^2 = H) and symmetric (H^T = H)
    print(f"||H^2 - H|| = {np.linalg.norm(H_demo @ H_demo - H_demo):.2e}")
    print(f"||H^T - H|| = {np.linalg.norm(H_demo.T - H_demo):.2e}")

    # Verify: Hw = Xw gives same predictions
    print(f"Max |Hy - Xw|: {np.max(np.abs(y_hat_demo - X_demo @ w_normal_solve)):.2e}")
    return (y_hat_demo,)


@app.cell
def _(mo):
    mo.image(src="../animations/rendered/RegressionProjection.gif")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### When $X^\top X$ Is Not Invertible

    $X^\top X$ is singular when:
    - **$p > n$**: more features than observations. The system is underdetermined — infinitely many solutions exist.
    - **Multicollinearity**: some columns of $X$ are linearly dependent (or nearly so). The matrix is rank-deficient.

    In practice, near-singularity (high condition number) causes numerical instability: small changes in the data produce wildly different $\mathbf{w}^*$. This is a direct motivation for regularization (Section 7).

    > **Reading**: [ISLR S3.2-3.3](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf), [ESL S3.2](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf), [MML S9.2](file:///C:/Users/landa/ml-course/textbooks/MML.pdf).
    """)
    return


@app.cell
def _(np):
    def _run():
        # Demonstrating multicollinearity and condition number
        _rng = np.random.default_rng(0)
        X_good = _rng.standard_normal((50, 3))
        X_bad = np.column_stack([X_good, X_good[:, 0] + 1e-8 * _rng.standard_normal(50)])  # near-duplicate column

        cond_good = np.linalg.cond(X_good.T @ X_good)
        cond_bad = np.linalg.cond(X_bad.T @ X_bad)

        print(f"Condition number (independent features):  {cond_good:.1f}")
        print(f"Condition number (collinear features):    {cond_bad:.1e}")
        print("High condition number => small data changes cause large weight changes")
        return ()


    _run()
    return

@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 3. Gradient Descent for Linear Regression

    ### When the Closed-Form Is Too Expensive

    The normal equation requires computing $(X^\top X)^{-1}$, which involves $O(p^3)$ operations for the inversion and $O(np^2)$ to form $X^\top X$. When $p$ is very large (high-dimensional data) or $n$ is enormous (millions of data points), this becomes prohibitive. Gradient descent offers an iterative alternative.

    ### Deriving the Gradient

    We already computed it:

    $$\nabla_\mathbf{w}\mathcal{L} = -2X^\top(y - X\mathbf{w})$$

    The gradient descent update rule is:

    $$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla_\mathbf{w}\mathcal{L} = \mathbf{w}_t + 2\eta X^\top(y - X\mathbf{w}_t)$$

    where $\eta$ is the learning rate. Each iteration costs $O(np)$ — a matrix-vector product. For the squared-error loss, $\mathcal{L}$ is convex (the Hessian is $2X^\top X$, which is positive semidefinite), so gradient descent converges to the global minimum for a sufficiently small $\eta$.

    ### Comparing GD to the Closed-Form

    With enough iterations and a proper learning rate, gradient descent and the normal equation produce the same $\mathbf{w}^*$. This is worth verifying empirically — it builds confidence that the two approaches are equivalent views of the same optimization problem.

    ### Stochastic Gradient Descent (SGD)

    Full gradient descent computes $X^\top(y - X\mathbf{w})$ using all $n$ data points. SGD approximates the gradient using a single random data point (or a mini-batch of $B$ points):

    $$\mathbf{w}_{t+1} = \mathbf{w}_t + 2\eta \, x_i(y_i - x_i^\top \mathbf{w}_t)$$

    The stochastic gradient is an unbiased estimator of the true gradient. SGD introduces noise into the updates, but the per-iteration cost drops from $O(np)$ to $O(p)$ (or $O(Bp)$ for mini-batches). For large datasets this is transformative — you can make progress without touching the entire dataset.

    > **Reading**: [MML S9.2.3-9.2.4](file:///C:/Users/landa/ml-course/textbooks/MML.pdf), [Boyd S9.3](file:///C:/Users/landa/ml-course/textbooks/Boyd-ConvexOptimization.pdf) for convergence analysis.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Code: OLS and Gradient Descent From Scratch""")
    return


@app.cell
def _(np):
    # Generate synthetic data
    _rng = np.random.default_rng(42)
    n, p = 100, 3
    X_synth = _rng.standard_normal((n, p))
    X_synth = np.c_[np.ones(n), X_synth]  # Add intercept column
    w_true = np.array([2.0, -1.0, 0.5, 3.0])
    y_synth = X_synth @ w_true + 0.5 * _rng.standard_normal(n)

    # --- Closed-form (Normal Equation) ---
    w_ols = np.linalg.solve(X_synth.T @ X_synth, X_synth.T @ y_synth)   # numerically stabler than inv()
    print("OLS weights:", w_ols)

    # --- Gradient Descent ---
    w_gd = np.zeros(X_synth.shape[1])
    lr = 0.001
    for epoch in range(5000):
        residual = y_synth - X_synth @ w_gd
        grad = -2 * X_synth.T @ residual
        w_gd -= lr * grad

    print("GD  weights:", w_gd)
    print("Max difference:", np.max(np.abs(w_ols - w_gd)))

    # --- Stochastic Gradient Descent ---
    w_sgd = np.zeros(X_synth.shape[1])
    lr_sgd = 0.005
    for epoch in range(50):
        indices = _rng.permutation(n)
        for i in indices:
            xi = X_synth[i:i+1]
            yi = y_synth[i:i+1]
            grad_i = -2 * xi.T @ (yi - xi @ w_sgd)
            w_sgd -= lr_sgd * grad_i.flatten()
        lr_sgd *= 0.98  # learning rate decay

    print("SGD weights:", w_sgd)
    return (X_synth, y_synth, w_true, w_ols, w_gd, w_sgd)


@app.cell
def _(np, X_synth, y_synth, w_ols):
    # Gradient descent with loss tracking — watch convergence
    # grad L = -2 X^T (y - Xw)
    w_gd_track = np.zeros(X_synth.shape[1])
    lr_track = 0.001
    losses_gd = []
    for _t in range(2000):
        residual_t = y_synth - X_synth @ w_gd_track
        loss_t = residual_t @ residual_t                   # ||y - Xw||^2
        losses_gd.append(loss_t)
        grad_t = -2 * X_synth.T @ residual_t               # gradient
        w_gd_track -= lr_track * grad_t                     # update

    loss_ols = np.sum((y_synth - X_synth @ w_ols)**2)
    print(f"OLS loss (optimal):         {loss_ols:.6f}")
    print(f"GD loss after 2000 steps:   {losses_gd[-1]:.6f}")
    print(f"GD converged to within:     {losses_gd[-1] - loss_ols:.2e}")
    return (losses_gd,)


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 4. The Probabilistic View

    This section ties together Module 0E (MLE) and regression. It is one of the most elegant connections in machine learning.

    ### Linear Regression as Maximum Likelihood

    Under the Gaussian noise assumption, each observation follows:

    $$y_i \mid x_i, \mathbf{w} \sim \mathcal{N}(x_i^\top \mathbf{w}, \, \sigma^2)$$

    The likelihood of the entire dataset (i.i.d. observations) is:

    $$L(\mathbf{w}) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - x_i^\top \mathbf{w})^2}{2\sigma^2}\right)$$

    Take the log:

    $$\ell(\mathbf{w}) = -\frac{n}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (y_i - x_i^\top \mathbf{w})^2$$

    To maximize $\ell(\mathbf{w})$ with respect to $\mathbf{w}$, the first term is a constant. We need to minimize:

    $$\sum_{i=1}^n (y_i - x_i^\top \mathbf{w})^2 = \|y - X\mathbf{w}\|^2$$

    **This is exactly the OLS objective.** Minimizing squared error is equivalent to maximizing the Gaussian log-likelihood. The least-squares solution *is* the maximum likelihood estimator under Gaussian noise. This is not a coincidence — it is the reason squared error is the default loss for regression.

    > **Reading**: [Bishop S3.1.1](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf), [Murphy S11.2](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf), [MML S9.2.1](file:///C:/Users/landa/ml-course/textbooks/MML.pdf).
    """)
    return


@app.cell
def _(np, X_synth, y_synth, w_ols):
    # MLE = OLS: verify they give the same w*
    # Log-likelihood: l(w) = -n/2 ln(2*pi*sigma^2) - 1/(2*sigma^2) * ||y - Xw||^2
    # Maximizing l(w) w.r.t. w is equivalent to minimizing ||y - Xw||^2

    sigma_mle = 1.0  # assumed noise std

    def neg_log_likelihood(w, X, y, sigma):
        n = len(y)
        residuals = y - X @ w
        return (n / 2) * np.log(2 * np.pi * sigma**2) + np.sum(residuals**2) / (2 * sigma**2)

    # The MLE minimizes NLL; OLS minimizes ||y - Xw||^2 — same argmin
    nll_at_ols = neg_log_likelihood(w_ols, X_synth, y_synth, sigma_mle)
    nll_at_zero = neg_log_likelihood(np.zeros(4), X_synth, y_synth, sigma_mle)
    print(f"NLL at w=0:    {nll_at_zero:.4f}")
    print(f"NLL at w_OLS:  {nll_at_ols:.4f}")

    # MLE estimate of sigma^2: unbiased is 1/(n-p) * ||y - Xw||^2
    residuals_mle = y_synth - X_synth @ w_ols
    sigma2_hat = np.sum(residuals_mle**2) / (len(y_synth) - X_synth.shape[1])
    print(f"MLE sigma^2 estimate: {sigma2_hat:.4f}  (true: 0.25)")
    return ()


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 5. Evaluating Regression Models

    ### $R^2$ — Coefficient of Determination

    $$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

    $R^2$ measures the fraction of variance in $y$ explained by the model. An $R^2$ of 0.85 means the model captures 85% of the variability. Key points:
    - $R^2 = 1$: perfect fit (dangerous — likely overfitting unless data is truly noiseless).
    - $R^2 = 0$: the model is no better than predicting the mean $\bar{y}$.
    - $R^2$ can be negative on test data — the model is actively worse than predicting the mean.
    - Adding more features always increases $R^2$ on training data (adjusted $R^2$ corrects for this).

    ### Residual Analysis

    After fitting, plot the residuals $e_i = y_i - \hat{y}_i$. Check:
    - **Residuals vs. fitted values**: should show no pattern. A funnel shape indicates heteroscedasticity. A curve indicates the linearity assumption is violated.
    - **Q-Q plot**: residuals should fall on the diagonal if the Gaussian assumption holds.
    - **Residuals vs. each feature**: reveals nonlinear relationships the model is missing.

    ### Train/Test Split

    Never evaluate a model only on the data it was trained on. Hold out a portion (typically 20-30%) as a test set and report the test MSE:

    $$\text{MSE}_{\text{test}} = \frac{1}{n_{\text{test}}} \sum_{i \in \text{test}} (y_i - \hat{y}_i)^2$$

    A large gap between train MSE and test MSE signals overfitting. We will formalize this with cross-validation in Module 1D.
    """)
    return


@app.cell
def _(np, X_synth, y_synth, w_ols):
    def _run():
        # R^2 from scratch: R^2 = 1 - SS_res / SS_tot
        y_hat_r2 = X_synth @ w_ols
        ss_res = np.sum((y_synth - y_hat_r2)**2)        # sum of squared residuals
        ss_tot = np.sum((y_synth - np.mean(y_synth))**2) # total sum of squares
        r2_manual = 1 - ss_res / ss_tot

        # MSE from scratch: MSE = (1/n) * ||y - y_hat||^2
        mse_manual = np.mean((y_synth - y_hat_r2)**2)

        print(f"R^2 (from scratch): {r2_manual:.6f}")
        print(f"MSE (from scratch): {mse_manual:.6f}")
        print(f"SS_res = {ss_res:.4f},  SS_tot = {ss_tot:.4f}")
        return ()


    _run()
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Code: Train/Test Evaluation""")
    return


@app.cell
def _(X_synth, y_synth, LinearRegression, train_test_split, r2_score, mean_squared_error):
    def _run():
        X_train_eval, X_test_eval, y_train_eval, y_test_eval = train_test_split(
            X_synth[:, 1:], y_synth, test_size=0.2, random_state=42   # drop intercept col; sklearn adds it
        )

        model_eval = LinearRegression()
        model_eval.fit(X_train_eval, y_train_eval)

        y_pred_train_eval = model_eval.predict(X_train_eval)
        y_pred_test_eval = model_eval.predict(X_test_eval)

        print(f"Train R2: {r2_score(y_train_eval, y_pred_train_eval):.4f}")
        print(f"Test  R2: {r2_score(y_test_eval, y_pred_test_eval):.4f}")
        print(f"Train MSE: {mean_squared_error(y_train_eval, y_pred_train_eval):.4f}")
        print(f"Test  MSE: {mean_squared_error(y_test_eval, y_pred_test_eval):.4f}")


    _run()
    return

@app.cell
def _(mo):
    mo.md(r"""
    > **Reading**: [ISLR S3.1.3](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf) for RSE and $R^2$, [ESL S3.2.2](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf) for the Gauss-Markov theorem.

    ---

    ## 6. Polynomial Regression and Basis Functions

    Linear regression is linear in the **parameters**, not necessarily in the inputs. This is a crucial distinction. We can model nonlinear relationships by transforming the input through basis functions $\phi(x)$ and then fitting a linear model in the transformed space:

    $$y = \phi(x)^\top \mathbf{w} + \varepsilon$$

    For polynomial regression with a single input $x$:

    $$\phi(x) = [1, \; x, \; x^2, \; \ldots, \; x^d]$$

    The design matrix $\Phi$ replaces $X$, but the normal equation is identical:

    $$\mathbf{w}^* = (\Phi^\top \Phi)^{-1} \Phi^\top y$$

    This is still a linear model — "linear" refers to linearity in $\mathbf{w}$, not in $x$. The entire OLS machinery applies unchanged.

    ### The Danger: Overfitting

    A degree-$d$ polynomial with $n$ data points and $d \geq n-1$ can interpolate all training points perfectly ($R^2_{\text{train}} = 1$). But it will oscillate wildly between points and generalize terribly. This is overfitting in its purest form: the model fits the noise, not the signal.

    More features (higher $d$) give the model more capacity. More capacity without more data means more overfitting. This tension is the central problem of machine learning, and regularization is the primary tool for resolving it.
    """)
    return


@app.cell
def _(np):
    def _run():
        # Basis functions: build polynomial design matrix from scratch
        # phi(x) = [1, x, x^2, ..., x^d]  =>  Phi is (n x (d+1))
        x_basis = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        degree_basis = 3

        # Build Phi manually: each row is [1, x_i, x_i^2, ..., x_i^d]
        Phi_manual = np.column_stack([x_basis**k for k in range(degree_basis + 1)])
        print(f"Phi (degree {degree_basis}) for x = {x_basis}:")
        print(Phi_manual)

        # Normal equation still applies: w* = (Phi^T Phi)^{-1} Phi^T y
        y_basis = 2 - x_basis + 0.5 * x_basis**2  # true quadratic
        w_basis = np.linalg.solve(Phi_manual.T @ Phi_manual, Phi_manual.T @ y_basis)
        print(f"\nFitted weights: {w_basis}")
        print(f"(True: [2, -1, 0.5, 0])")
        return ()


    _run()
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Code: Polynomial Fits of Increasing Degree""")
    return


@app.cell
def _(np, PolynomialFeatures, LinearRegression, mean_squared_error):
    # True function: y = 0.5x^2 - x + 2 + noise
    _rng = np.random.default_rng(0)
    x_poly = np.sort(_rng.uniform(-3, 3, 30))
    y_poly = 0.5 * x_poly**2 - x_poly + 2 + _rng.standard_normal(30) * 0.8

    # Fit polynomials of increasing degree
    for degree in [1, 2, 5, 15]:
        poly = PolynomialFeatures(degree)
        X_poly = poly.fit_transform(x_poly.reshape(-1, 1))
        model_poly = LinearRegression().fit(X_poly, y_poly)
        y_pred_poly = model_poly.predict(X_poly)
        print(f"Degree {degree:2d}: Train MSE = {mean_squared_error(y_poly, y_pred_poly):.4f}, "
              f"Coefficients = {model_poly.coef_.shape[0]}")
    return (x_poly, y_poly)


@app.cell
def _(mo):
    mo.md(r"""
    > **Reading**: [Bishop S3.1](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) for basis functions and polynomial fits, [MML S9.1](file:///C:/Users/landa/ml-course/textbooks/MML.pdf).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ### Interactive: Polynomial Degree and Overfitting

    Use the slider below to change the polynomial degree and observe the transition from underfitting to good fit to overfitting.
    """)
    return


@app.cell
def _(mo):
    degree_slider = mo.ui.slider(start=1, stop=15, step=1, value=1, label="Polynomial degree")
    degree_slider
    return (degree_slider,)


@app.cell
def _(degree_slider, np, plt, PolynomialFeatures, LinearRegression, mean_squared_error, x_poly, y_poly):
    def _run():
        _degree = degree_slider.value

        # Fit polynomial of chosen degree
        _poly = PolynomialFeatures(_degree)
        _X_poly = _poly.fit_transform(x_poly.reshape(-1, 1))
        _model = LinearRegression().fit(_X_poly, y_poly)

        # Generate smooth curve for plotting
        _x_smooth = np.linspace(-3.2, 3.2, 300)
        _X_smooth = _poly.transform(_x_smooth.reshape(-1, 1))
        _y_smooth = _model.predict(_X_smooth)

        # True function
        _y_true_smooth = 0.5 * _x_smooth**2 - _x_smooth + 2

        _train_mse = mean_squared_error(y_poly, _model.predict(_X_poly))

        _fig, _ax = plt.subplots(1, 1, figsize=(9, 5))
        _ax.scatter(x_poly, y_poly, color="steelblue", s=40, zorder=5, label="Training data")
        _ax.plot(_x_smooth, _y_true_smooth, "k--", alpha=0.5, label="True function")
        _ax.plot(_x_smooth, _y_smooth, "r-", linewidth=2, label=f"Degree {_degree} fit")
        _ax.set_ylim(-5, 20)
        _ax.set_xlabel("x")
        _ax.set_ylabel("y")
        _ax.set_title(f"Polynomial Degree = {_degree}  |  Train MSE = {_train_mse:.4f}")
        _ax.legend()
        plt.tight_layout()
        _fig


    _run()
    return

@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 7. Regularization — Controlling Complexity

    ### Ridge Regression (L2 Regularization)

    Add a penalty on the magnitude of $\mathbf{w}$:

    $$\mathcal{L}_{\text{ridge}}(\mathbf{w}) = \|y - X\mathbf{w}\|^2 + \lambda\|\mathbf{w}\|^2_2$$

    where $\lambda > 0$ is the regularization strength. Taking the gradient and setting it to zero:

    $$\nabla_\mathbf{w}\mathcal{L} = -2X^\top(y - X\mathbf{w}) + 2\lambda\mathbf{w} = 0$$

    $$\boxed{\mathbf{w}^*_{\text{ridge}} = (X^\top X + \lambda I)^{-1} X^\top y}$$

    Compare this to the OLS solution: the only difference is the $\lambda I$ term added to $X^\top X$. This has two profound effects:

    1. **Numerical stability**: $X^\top X + \lambda I$ is always invertible for $\lambda > 0$, even when $X^\top X$ is singular. The regularization "fixes" the rank deficiency.
    2. **Shrinkage**: the coefficients are pulled toward zero. Larger $\lambda$ means more shrinkage. At $\lambda = 0$ we recover OLS; as $\lambda \to \infty$, $\mathbf{w}^* \to \mathbf{0}$.

    **Geometric interpretation**: The OLS objective is an elliptical bowl. The ridge penalty $\|\mathbf{w}\|^2 \leq t$ is a sphere centered at the origin. The ridge solution is the point where the smallest contour of the loss touches the sphere. This pushes $\mathbf{w}^*$ toward the origin.

    **Bayesian interpretation**: Ridge regression is equivalent to MAP estimation with a Gaussian prior $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \tau^2 I)$. The posterior mode is:

    $$\mathbf{w}_{\text{MAP}} = \arg\max_\mathbf{w} \left[\ln p(y \mid X, \mathbf{w}) + \ln p(\mathbf{w})\right]$$

    The log-prior $\ln p(\mathbf{w}) = -\frac{\|\mathbf{w}\|^2}{2\tau^2} + \text{const}$ contributes a quadratic penalty with $\lambda = \sigma^2 / \tau^2$. This directly connects to MAP estimation from Module 0E: the regularization term *is* the prior. A strong prior (small $\tau^2$, large $\lambda$) pulls the weights toward zero; a weak prior (large $\tau^2$, small $\lambda$) lets the data dominate.

    ### Lasso Regression (L1 Regularization)

    Replace the L2 penalty with L1:

    $$\mathcal{L}_{\text{lasso}}(\mathbf{w}) = \|y - X\mathbf{w}\|^2 + \lambda\|\mathbf{w}\|_1$$

    There is no closed-form solution because $|\cdot|$ is not differentiable at zero. Optimization requires coordinate descent or subgradient methods. But the key property of lasso is what makes it worth the trouble: **it produces sparse solutions**.

    Why does L1 induce sparsity while L2 does not? Consider the constraint geometry:
    - The L2 constraint set $\|\mathbf{w}\|^2_2 \leq t$ is a **circle** (sphere in higher dimensions). The elliptical loss contours are likely to touch the sphere at a point where no component of $\mathbf{w}$ is exactly zero.
    - The L1 constraint set $\|\mathbf{w}\|_1 \leq t$ is a **diamond** (cross-polytope). Its corners lie on the axes, where one or more components are exactly zero. The loss contours are much more likely to first contact the diamond at a corner.

    This is one of the most beautiful geometric arguments in statistics. In high-dimensional spaces, the corners of the L1 ball vastly outnumber the smooth surface of the L2 ball, so sparsity becomes even more likely.

    **Practical consequence**: Lasso performs automatic feature selection. Features with zero coefficients are effectively removed from the model.

    ### Elastic Net

    Combine both penalties:

    $$\mathcal{L}_{\text{elastic}}(\mathbf{w}) = \|y - X\mathbf{w}\|^2 + \lambda_1\|\mathbf{w}\|_1 + \lambda_2\|\mathbf{w}\|^2_2$$

    Elastic Net gets the sparsity of lasso while handling correlated features better (lasso tends to arbitrarily select one from a group of correlated features; elastic net keeps them all or drops them all).

    ### The Regularization Path

    Fix the data and vary $\lambda$ from $0$ to $\infty$. Plot each coefficient $w_j$ as a function of $\lambda$. For ridge, the coefficients smoothly shrink toward zero. For lasso, they hit zero at different values of $\lambda$ and stay there — this is the sparsity pattern revealing which features matter most.

    ### Choosing $\lambda$ with Cross-Validation

    $\lambda$ is a **hyperparameter** — not learned by the optimization, but chosen externally. The standard approach is $k$-fold cross-validation: split the training data into $k$ folds, train on $k-1$ folds, evaluate on the held-out fold, and average the error over all $k$ splits. Choose the $\lambda$ with the lowest average validation error. We will cover this in full detail in Module 1D.
    """)
    return


@app.cell
def _(np, X_synth, y_synth, w_ols):
    def _run():
        # Ridge from scratch: w_ridge = (X^T X + lambda * I)^{-1} X^T y
        lam_demo = 5.0
        I_p = np.eye(X_synth.shape[1])
        w_ridge_scratch = np.linalg.solve(
            X_synth.T @ X_synth + lam_demo * I_p,  # regularized normal equation
            X_synth.T @ y_synth
        )

        # Compare norms: ridge shrinks weights toward zero
        print(f"OLS   weights: {w_ols},  ||w|| = {np.linalg.norm(w_ols):.4f}")
        print(f"Ridge weights: {w_ridge_scratch},  ||w|| = {np.linalg.norm(w_ridge_scratch):.4f}")
        print(f"Ridge shrinks ||w|| by {(1 - np.linalg.norm(w_ridge_scratch)/np.linalg.norm(w_ols))*100:.1f}%")
        return ()


    _run()
    return

@app.cell
def _(np, X_synth, y_synth):
    def _run():
        # Lasso via coordinate descent (from scratch)
        # For each j: minimize over w_j holding others fixed
        # Soft-thresholding: w_j = sign(rho_j) * max(|rho_j| - lambda/2, 0) / (X_j^T X_j)
        def lasso_coordinate_descent(X, y, lam, n_iters=1000):
            n, p = X.shape
            w = np.zeros(p)
            for _ in range(n_iters):
                for j in range(p):
                    # Partial residual excluding feature j
                    r_j = y - X @ w + X[:, j] * w[j]
                    rho_j = X[:, j] @ r_j
                    z_j = X[:, j] @ X[:, j]
                    # Soft-thresholding operator
                    w[j] = np.sign(rho_j) * max(abs(rho_j) - lam / 2, 0) / z_j
            return w

        w_lasso_scratch = lasso_coordinate_descent(X_synth, y_synth, lam=5.0)
        print(f"Lasso weights (lambda=5): {w_lasso_scratch}")
        print(f"Non-zero: {np.sum(np.abs(w_lasso_scratch) > 1e-8)} / {len(w_lasso_scratch)}")
        print(f"L1 norm: {np.sum(np.abs(w_lasso_scratch)):.4f}")
        return ()


    _run()
    return

@app.cell
def _(mo):
    mo.image(src="../animations/rendered/RegularizationPath.gif")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Code: Ridge, Lasso, and Elastic Net""")
    return


@app.cell
def _(np, PolynomialFeatures, RidgeCV, LassoCV):
    # Using polynomial features (degree 10, easy to overfit)
    _rng = np.random.default_rng(0)
    x_reg = np.sort(_rng.uniform(-3, 3, 30))
    y_true_reg = 0.5 * x_reg**2 - x_reg + 2
    y_reg = y_true_reg + _rng.standard_normal(30) * 0.8

    poly_reg = PolynomialFeatures(10)
    X_poly_reg = poly_reg.fit_transform(x_reg.reshape(-1, 1))

    # --- Ridge with cross-validated lambda ---
    ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 100), scoring="neg_mean_squared_error")
    ridge_cv.fit(X_poly_reg, y_reg)
    print(f"Ridge: best alpha = {ridge_cv.alpha_:.4f}")
    print(f"Ridge coefficients:\n{ridge_cv.coef_}")

    # --- Lasso with cross-validated lambda ---
    lasso_cv = LassoCV(alphas=np.logspace(-3, 1, 100), cv=5, random_state=42)
    lasso_cv.fit(X_poly_reg, y_reg)
    print(f"\nLasso: best alpha = {lasso_cv.alpha_:.4f}")
    print(f"Lasso coefficients:\n{lasso_cv.coef_}")
    print(f"Non-zero coefficients: {np.sum(lasso_cv.coef_ != 0)} / {len(lasso_cv.coef_)}")

    # --- From scratch: Ridge closed-form ---
    lam = 1.0
    w_ridge_manual = np.linalg.solve(X_poly_reg.T @ X_poly_reg + lam * np.eye(X_poly_reg.shape[1]),
                               X_poly_reg.T @ y_reg)
    print(f"\nRidge (from scratch, lambda=1): {w_ridge_manual}")
    return (x_reg, y_reg, X_poly_reg, poly_reg)


@app.cell
def _(np, X_synth, y_synth, w_ols):
    def _run():
        # Comparing penalty terms: L1 vs L2 vs Elastic Net
        # Ridge penalty: lambda * ||w||_2^2
        # Lasso penalty: lambda * ||w||_1
        # Elastic Net:   lambda1 * ||w||_1 + lambda2 * ||w||_2^2
        lam_compare = 2.0
        ols_loss = np.sum((y_synth - X_synth @ w_ols)**2)

        print(f"OLS loss (no penalty):    {ols_loss:.4f}")
        print(f"+ Ridge penalty (L2):     {ols_loss + lam_compare * np.sum(w_ols**2):.4f}")
        print(f"+ Lasso penalty (L1):     {ols_loss + lam_compare * np.sum(np.abs(w_ols)):.4f}")
        print(f"+ Elastic Net (L1+L2):    {ols_loss + lam_compare * np.sum(np.abs(w_ols)) + lam_compare * np.sum(w_ols**2):.4f}")
        print(f"\n||w||_1 = {np.sum(np.abs(w_ols)):.4f}")
        print(f"||w||_2^2 = {np.sum(w_ols**2):.4f}")
        return ()


    _run()
    return

@app.cell
def _(mo):
    mo.md(r"""
    > **Reading**: [ISLR S6.2](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf) for Ridge and Lasso, [ESL S3.4](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf) for shrinkage methods, [Bishop S3.1.4](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) for regularized least squares, [MML S9.2.3](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) for MAP estimation as regularization.

    ---

    ### Interactive: Regularization Strength ($\lambda$) and Coefficient Shrinkage

    Use the slider below to control the Ridge regularization parameter $\lambda = 10^{\text{slider}}$ and observe how the coefficients shrink.
    """)
    return


@app.cell
def _(mo):
    lambda_slider = mo.ui.slider(start=-4, stop=2, step=0.1, value=0, label=r"log₁₀(λ)")
    lambda_slider
    return (lambda_slider,)


@app.cell
def _(lambda_slider, np, plt, Ridge, PolynomialFeatures, x_reg, y_reg):
    def _run():
        _lam = 10 ** lambda_slider.value

        # Fit degree-10 polynomial with Ridge
        _poly = PolynomialFeatures(10)
        _X = _poly.fit_transform(x_reg.reshape(-1, 1))
        _model = Ridge(alpha=_lam).fit(_X, y_reg)

        # Smooth curve
        _x_sm = np.linspace(-3.2, 3.2, 300)
        _X_sm = _poly.transform(_x_sm.reshape(-1, 1))
        _y_sm = _model.predict(_X_sm)
        _y_true_sm = 0.5 * _x_sm**2 - _x_sm + 2

        _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left panel: fit
        _ax1.scatter(x_reg, y_reg, color="steelblue", s=40, zorder=5, label="Training data")
        _ax1.plot(_x_sm, _y_true_sm, "k--", alpha=0.5, label="True function")
        _ax1.plot(_x_sm, _y_sm, "r-", linewidth=2, label=f"Ridge fit")
        _ax1.set_ylim(-5, 20)
        _ax1.set_xlabel("x")
        _ax1.set_ylabel("y")
        _ax1.set_title(f"Ridge Regression  |  λ = {_lam:.4f}")
        _ax1.legend()

        # Right panel: coefficients
        _coefs = _model.coef_[1:]  # skip intercept feature coef
        _ax2.bar(range(1, len(_coefs)+1), _coefs, color="steelblue", alpha=0.8)
        _ax2.set_xlabel("Coefficient index")
        _ax2.set_ylabel("Coefficient value")
        _ax2.set_title(f"Coefficient magnitudes  |  ||w||₂ = {np.linalg.norm(_model.coef_):.2f}")
        _ax2.axhline(y=0, color="k", linewidth=0.5)

        plt.tight_layout()
        _fig


    _run()
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 8. Bias-Variance Tradeoff in Action

    In Module 1A we introduced the bias-variance decomposition abstractly. Linear regression with regularization makes it concrete and visible.

    For a given model complexity (say, polynomial degree $d$ and regularization parameter $\lambda$), the expected test error decomposes as:

    $$\mathbb{E}[\text{test error}] = \text{Bias}^2 + \text{Variance} + \sigma^2$$

    The noise term $\sigma^2$ is irreducible — no model can eliminate it. The other two terms trade off:

    | Regime | Bias | Variance | What Happens |
    |---|---|---|---|
    | No regularization ($\lambda = 0$), high degree | Low | High | Model fits training data closely (including noise). Predictions vary wildly across different training sets. |
    | Heavy regularization ($\lambda \gg 1$) or low degree | High | Low | Model is too constrained to capture the true signal. Predictions are stable but systematically wrong. |

    Between these extremes lies the sweet spot — the value of $\lambda$ (or $d$) that minimizes total test error. The resulting curve is **U-shaped**: test error decreases as you increase model complexity (reducing bias), hits a minimum, then increases again (variance takes over).

    This U-shaped test error curve is the single most important concept in applied machine learning. Every model selection technique — cross-validation, regularization, early stopping, architecture search — is ultimately about finding the bottom of this curve.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Code: Bias-Variance U-Curve""")
    return


@app.cell
def _(np, Ridge, PolynomialFeatures, mean_squared_error):
    def _run():
        _rng = np.random.default_rng(42)

        def _true_fn(x):
            return 0.5 * x**2 - x + 2

        # Test set (fixed)
        x_test_bv = np.linspace(-3, 3, 200)
        y_test_bv = _true_fn(x_test_bv)

        _degree = 10
        alphas_bv = np.logspace(-2, 4, 20)
        n_trials = 50
        train_errors_bv = []
        test_errors_bv = []

        for _alpha in alphas_bv:
            _trial_train, _trial_test = [], []
            for _ in range(n_trials):
                _x_train = _rng.uniform(-3, 3, 30)
                _y_train = _true_fn(_x_train) + _rng.standard_normal(30) * 0.8

                _poly = PolynomialFeatures(_degree)
                _X_tr = _poly.fit_transform(_x_train.reshape(-1, 1))
                _X_te = _poly.transform(x_test_bv.reshape(-1, 1))

                _model = Ridge(alpha=_alpha).fit(_X_tr, _y_train)
                _trial_train.append(mean_squared_error(_y_train, _model.predict(_X_tr)))
                _trial_test.append(mean_squared_error(y_test_bv, _model.predict(_X_te)))

            train_errors_bv.append(np.mean(_trial_train))
            test_errors_bv.append(np.mean(_trial_test))

        best_idx = np.argmin(test_errors_bv)
        print(f"Best alpha: {alphas_bv[best_idx]:.4f}")
        print(f"Best test MSE: {test_errors_bv[best_idx]:.4f}")
        print(f"Train MSE at best alpha: {train_errors_bv[best_idx]:.4f}")
        # Observe: train error increases with alpha, test error is U-shaped


    _run()
    return

@app.cell
def _(mo):
    mo.md(r"""
    > **Reading**: [ISLR S2.2](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf) for bias-variance tradeoff, [ESL S2.9](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf) for the mathematical decomposition, [Bishop S3.2](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) for the Bayesian perspective on model complexity.

    ---

    ## Summary

    What you have learned in this module:

    | Concept | Key Equation/Idea |
    |---|---|
    | OLS solution | $\mathbf{w}^* = (X^\top X)^{-1}X^\top y$ |
    | Geometry | $\hat{y}$ is the projection of $y$ onto $\text{Col}(X)$ |
    | Gradient | $\nabla\mathcal{L} = -2X^\top(y - X\mathbf{w})$ |
    | Probabilistic view | OLS = MLE under Gaussian noise |
    | Ridge | $\mathbf{w}^* = (X^\top X + \lambda I)^{-1}X^\top y$ = MAP with Gaussian prior |
    | Lasso | L1 penalty induces sparsity via diamond geometry |
    | Bias-variance | Regularization trades bias for variance; test error is U-shaped |

    Every advanced model you will encounter — logistic regression, neural networks, Gaussian processes, SVMs — reuses the concepts from this module. The loss function changes, the optimization becomes harder, but the structure is the same: define a model, define a loss, optimize, regularize, evaluate.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Code It: Implementation Exercises

    Work through these exercises to build your own linear regression toolkit from scratch. Each exercise gives you a problem statement followed by skeleton code to complete.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Exercise 1: Normal Equation Solver

    Implement a function that solves the normal equation $\mathbf{w}^* = (X^\top X)^{-1} X^\top y$ using `np.linalg.solve` (not `np.linalg.inv`). Your function should automatically prepend an intercept column of ones. Test it on the provided data and verify against sklearn.
    """)
    return


@app.cell
def _(np):
    def ols_fit(X, y):
        """Fit OLS via normal equation. X is (n, p) without intercept."""
        # TODO: prepend a column of ones to X
        X_aug = np.c_[np.ones(X.shape[0]), X]
        # TODO: solve (X^T X) w = X^T y
        w = np.linalg.solve(X_aug.T @ X_aug, X_aug.T @ y)
        return w

    def ols_predict(X, w):
        """Predict y = [1|X] @ w."""
        X_aug = np.c_[np.ones(X.shape[0]), X]
        return X_aug @ w

    # Test data
    _rng = np.random.default_rng(99)
    _X_ex = _rng.standard_normal((80, 3))
    _w_ex_true = np.array([1.5, -0.8, 2.1, 0.3])  # intercept + 3 features
    _y_ex = np.c_[np.ones(80), _X_ex] @ _w_ex_true + 0.2 * _rng.standard_normal(80)

    w_fit = ols_fit(_X_ex, _y_ex)
    print(f"True weights:  {_w_ex_true}")
    print(f"Fitted weights: {w_fit}")
    print(f"Max error: {np.max(np.abs(w_fit - _w_ex_true)):.4f}")
    return (ols_fit, ols_predict)


@app.cell
def _(mo):
    mo.md(r"""
    ### Exercise 2: Gradient Descent From Scratch

    Implement gradient descent for linear regression. Track the loss at each iteration and verify that it converges to the OLS solution. Experiment with different learning rates.

    Recall: $\nabla_\mathbf{w}\mathcal{L} = -2X^\top(y - X\mathbf{w})$
    """)
    return


@app.cell
def _(np):
    def gd_fit(X, y, lr=0.01, n_iters=1000):
        """Gradient descent for linear regression. Returns (weights, loss_history)."""
        X_aug = np.c_[np.ones(X.shape[0]), X]
        w = np.zeros(X_aug.shape[1])
        losses = []
        for _ in range(n_iters):
            residual = y - X_aug @ w
            loss = np.sum(residual**2)
            losses.append(loss)
            # gradient: -2 X^T (y - Xw)
            grad = -2 * X_aug.T @ residual
            w = w - lr * grad
        return w, losses

    # Test: compare GD to OLS on same data
    _rng = np.random.default_rng(99)
    _X_gd = _rng.standard_normal((80, 3))
    _w_gd_true = np.array([1.5, -0.8, 2.1, 0.3])
    _y_gd = np.c_[np.ones(80), _X_gd] @ _w_gd_true + 0.2 * _rng.standard_normal(80)

    w_gd_ex, loss_hist = gd_fit(_X_gd, _y_gd, lr=0.001, n_iters=3000)
    print(f"True:  {_w_gd_true}")
    print(f"GD:    {w_gd_ex}")
    print(f"Loss: {loss_hist[0]:.2f} -> {loss_hist[-1]:.4f} ({len(loss_hist)} iters)")
    return (gd_fit,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 3: Ridge Regression From Scratch

    Implement the ridge closed-form solution: $\mathbf{w}^*_{\text{ridge}} = (X^\top X + \lambda I)^{-1} X^\top y$

    Then sweep $\lambda$ from $10^{-3}$ to $10^{3}$ and plot the coefficient norms $\|\mathbf{w}\|_2$ versus $\lambda$. You should see the norm shrink monotonically.
    """)
    return


@app.cell
def _(np, plt):
    def ridge_fit(X, y, lam):
        """Ridge regression via closed-form. X is (n, p) without intercept."""
        X_aug = np.c_[np.ones(X.shape[0]), X]
        p = X_aug.shape[1]
        # w_ridge = (X^T X + lambda * I)^{-1} X^T y
        w = np.linalg.solve(X_aug.T @ X_aug + lam * np.eye(p), X_aug.T @ y)
        return w

    # Sweep lambda and plot ||w||_2
    _rng = np.random.default_rng(99)
    _X_r = _rng.standard_normal((80, 5))
    _w_r_true = np.array([1.0, -2.0, 3.0, 0.5, -1.5, 2.5])
    _y_r = np.c_[np.ones(80), _X_r] @ _w_r_true + 0.3 * _rng.standard_normal(80)

    lambdas = np.logspace(-3, 3, 50)
    norms = [np.linalg.norm(ridge_fit(_X_r, _y_r, l)) for l in lambdas]

    _fig_r, _ax_r = plt.subplots(figsize=(8, 4))
    _ax_r.semilogx(lambdas, norms, "b-", linewidth=2)
    _ax_r.set_xlabel("lambda")
    _ax_r.set_ylabel("||w||_2")
    _ax_r.set_title("Ridge: coefficient norm shrinks with increasing lambda")
    _ax_r.grid(True, alpha=0.3)
    plt.tight_layout()
    _fig_r
    return (ridge_fit,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 4: Lasso via Coordinate Descent

    Implement the lasso using coordinate descent with the soft-thresholding operator. For each feature $j$, the update is:

    $$w_j \leftarrow \frac{S(\rho_j, \, \lambda/2)}{z_j}, \quad \text{where } \rho_j = X_j^\top(y - X_{-j}w_{-j}), \; z_j = X_j^\top X_j$$

    and $S(a, b) = \text{sign}(a)\max(|a| - b, 0)$ is the soft-thresholding operator. Run on data with some irrelevant features and verify lasso zeros them out.
    """)
    return


@app.cell
def _(np):
    def lasso_fit(X, y, lam, n_iters=500):
        """Lasso via coordinate descent with soft-thresholding."""
        X_aug = np.c_[np.ones(X.shape[0]), X]
        n, p = X_aug.shape
        w = np.zeros(p)
        for _ in range(n_iters):
            for j in range(p):
                # Partial residual: r_j = y - Xw + X_j * w_j
                r_j = y - X_aug @ w + X_aug[:, j] * w[j]
                rho_j = X_aug[:, j] @ r_j
                z_j = X_aug[:, j] @ X_aug[:, j]
                # Soft-thresholding (skip penalty on intercept j=0)
                if j == 0:
                    w[j] = rho_j / z_j
                else:
                    w[j] = np.sign(rho_j) * max(abs(rho_j) - lam / 2, 0) / z_j
        return w

    # Data: 3 real features + 4 irrelevant ones
    _rng = np.random.default_rng(42)
    _X_l = _rng.standard_normal((100, 7))
    _w_l_true = np.array([1.0, 2.0, -1.5, 3.0, 0, 0, 0, 0])  # last 4 are zero
    _y_l = np.c_[np.ones(100), _X_l] @ _w_l_true + 0.3 * _rng.standard_normal(100)

    w_lasso_ex = lasso_fit(_X_l, _y_l, lam=10.0)
    print(f"True:  {_w_l_true}")
    print(f"Lasso: {np.round(w_lasso_ex, 3)}")
    print(f"Non-zero (excl. intercept): {np.sum(np.abs(w_lasso_ex[1:]) > 1e-6)} / {len(w_lasso_ex)-1}")
    return (lasso_fit,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 5: Full Pipeline — Fit, Predict, Evaluate

    Combine your implementations into a complete pipeline: generate data, split into train/test, fit OLS + Ridge + Lasso, compute train and test MSE, and compare. Does regularization help when the model is overparameterized?
    """)
    return


@app.cell
def _(np, ols_fit, ols_predict, ridge_fit, lasso_fit):
    def _run():
        _rng = np.random.default_rng(123)
        # Overparameterized: 20 features, only 5 matter, 60 training samples
        _n_pipe, _p_pipe = 60, 20
        _X_pipe = _rng.standard_normal((_n_pipe, _p_pipe))
        _w_pipe_true = np.zeros(_p_pipe + 1)
        _w_pipe_true[:6] = [3.0, -2.0, 1.5, 0.8, -1.2, 2.0]  # intercept + 5 features
        _y_pipe = np.c_[np.ones(_n_pipe), _X_pipe] @ _w_pipe_true + 0.5 * _rng.standard_normal(_n_pipe)

        # Train/test split
        _split = 45
        X_tr, X_te = _X_pipe[:_split], _X_pipe[_split:]
        y_tr, y_te = _y_pipe[:_split], _y_pipe[_split:]

        def mse(y, yhat):
            return np.mean((y - yhat)**2)

        # OLS
        w_o = ols_fit(X_tr, y_tr)
        # Ridge
        w_r = ridge_fit(X_tr, y_tr, lam=5.0)
        # Lasso
        w_l = lasso_fit(X_tr, y_tr, lam=5.0)

        for name, w in [("OLS", w_o), ("Ridge", w_r), ("Lasso", w_l)]:
            yhat_tr = ols_predict(X_tr, w)
            yhat_te = ols_predict(X_te, w)
            print(f"{name:6s} | Train MSE: {mse(y_tr, yhat_tr):.4f} | Test MSE: {mse(y_te, yhat_te):.4f} | "
                  f"||w||_1: {np.sum(np.abs(w)):.2f} | nonzero: {np.sum(np.abs(w) > 0.01)}")
        return ()


    _run()
    return

@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Exercises

    ### Pencil and Paper

    1. **Derive the OLS solution** from scratch without looking at the notes. Start from $\mathcal{L}(\mathbf{w}) = \|y - X\mathbf{w}\|^2$, expand, differentiate, solve.

    2. **Show that the hat matrix $H = X(X^\top X)^{-1}X^\top$ is idempotent** ($H^2 = H$) and symmetric ($H^\top = H$). Why do these properties make it a valid projection matrix?

    3. **Derive the ridge regression solution** by adding $\lambda\|\mathbf{w}\|^2$ to the OLS loss and differentiating. Verify that setting $\lambda = 0$ recovers OLS.

    4. **Probabilistic derivation**: Starting from $y_i \sim \mathcal{N}(x_i^\top\mathbf{w}, \sigma^2)$ and a prior $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \tau^2 I)$, show that the MAP estimate equals the ridge solution with $\lambda = \sigma^2/\tau^2$.

    5. **Gradient descent convergence**: For the quadratic loss $\mathcal{L}(\mathbf{w}) = \frac{1}{2}\mathbf{w}^\top A\mathbf{w} - \mathbf{b}^\top\mathbf{w}$, show that gradient descent converges if and only if the learning rate satisfies $0 < \eta < 2/\lambda_{\max}$, where $\lambda_{\max}$ is the largest eigenvalue of $A$. *(Hint: diagonalize $A$ and analyze each component independently.)*

    ### Coding

    6. **Implement linear regression from scratch** using only NumPy. Your class should have `.fit(X, y)` and `.predict(X)` methods. Support both the closed-form and gradient descent solutions. Verify they produce the same weights on a synthetic dataset.

    7. **Regularization path plot**: For a dataset of your choice (or synthetic), fit lasso regression for 100 values of $\lambda$ ranging from $10^{-3}$ to $10^2$. Plot all coefficients as a function of $\log(\lambda)$. Identify which features are "selected" first (last to be zeroed out).

    8. **Bias-variance visualization**: Generate data from $y = \sin(x) + \varepsilon$ where $\varepsilon \sim \mathcal{N}(0, 0.3^2)$. For polynomial degrees $d = 1, 3, 5, 10, 15$, generate 100 training sets of size 20, fit a polynomial on each, and compute the average prediction at each test point. Decompose the test error into bias^2 and variance empirically. Plot the U-shaped curve.

    9. **Comparison experiment**: Load the `diabetes` dataset from `sklearn.datasets`. Compare the test MSE (5-fold CV) of: OLS, Ridge (with CV-tuned $\lambda$), Lasso (with CV-tuned $\lambda$), and Elastic Net. Which performs best? How many features does Lasso select?

    > **Next**: 1C - Classification Foundations — logistic regression, where we take the linear model and bend it to predict categories instead of continuous values.
    """)
    return


if __name__ == "__main__":
    app.run()
