import marimo


app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    rng = np.random.default_rng(42)
    return (np,)


@app.cell
def _(mo):
    mo.md(r"""
    # Path D --- Bayesian Machine Learning

    Most of the ML you've learned so far is frequentist in spirit: you find a single best parameter vector $\hat{\theta}$ (via MLE or MAP), make predictions using that point estimate, and call it a day. Bayesian ML takes a fundamentally different stance. Instead of asking "what is the best $\theta$?", it asks "given the data I've seen, what is the full probability distribution over $\theta$?"

    This isn't just philosophical navel-gazing. The posterior distribution $p(\theta \mid \mathcal{D})$ tells you not only what the model thinks, but also **how uncertain it is** --- and that uncertainty information turns out to be enormously useful. It gives you calibrated predictions, principled model comparison, and automatic regularization. The cost? Computation. Computing the posterior is usually intractable, and the entire field of Bayesian ML is essentially a catalog of clever approximations.

    This guide maps out the territory. You should already be comfortable with Bayes' theorem, priors, posteriors, and likelihoods from 0D --- Probability Foundations and 0E --- Statistical Estimation. Here we go much further.

    > **Primary references:** [Murphy PML1](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf) is the best single reference for this entire guide. [Bishop PRML](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) Chapters 3, 6, and 8-13 cover most of these topics with exceptional clarity. Rasmussen & Williams, *Gaussian Processes for Machine Learning* (2006), is freely available at [gaussianprocess.org](http://www.gaussianprocess.org/gpml/) and is the definitive GP reference.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 1. The Bayesian Philosophy

    Recall Bayes' rule applied to model parameters:

    $$p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta) \, p(\theta)}{p(\mathcal{D})}$$

    - $p(\theta)$ --- the **prior**: what you believe about $\theta$ before seeing data.
    - $p(\mathcal{D} \mid \theta)$ --- the **likelihood**: how probable the data is under each $\theta$.
    - $p(\theta \mid \mathcal{D})$ --- the **posterior**: your updated beliefs after seeing data.
    - $p(\mathcal{D}) = \int p(\mathcal{D} \mid \theta) p(\theta) \, d\theta$ --- the **marginal likelihood** (or model evidence): a normalizing constant that plays a starring role in model comparison.

    The MLE and MAP estimates that you already know are special cases. MLE maximizes $p(\mathcal{D} \mid \theta)$, which is equivalent to Bayesian inference with a flat (uniform) prior, keeping only the mode. MAP maximizes $p(\theta \mid \mathcal{D})$, which includes the prior but again keeps only the mode. Full Bayesian inference keeps the entire distribution.

    **Why keep the whole distribution?** Because for prediction, you should average over all plausible parameters:

    $$p(y^* \mid x^*, \mathcal{D}) = \int p(y^* \mid x^*, \theta) \, p(\theta \mid \mathcal{D}) \, d\theta$$

    This **posterior predictive distribution** automatically accounts for parameter uncertainty. Where the posterior is spread out (you're unsure about $\theta$), the predictive distribution is wide (you're unsure about your prediction). This is uncertainty quantification, and it comes for free from the Bayesian framework.

    See [Bishop S1.2.3--1.2.6](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) and [Murphy PML1 Ch 4](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf) for thorough treatments.
    """)
    return


@app.cell
def _(np):
    def _run():
        # --- Bayes' rule for a coin flip (conjugate Beta-Binomial) ---
        # Prior: Beta(alpha, beta) -> observe data -> Posterior: Beta(alpha + heads, beta + tails)
        from scipy.stats import beta as beta_dist

        alpha_prior, beta_prior = 2, 5  # prior: believe coin is biased toward tails
        data = np.array([1, 1, 1, 0, 1, 1, 0, 1, 1, 1])  # 1=heads, 0=tails
        heads, tails = data.sum(), len(data) - data.sum()

        # Posterior parameters (conjugate update -- no integral needed!)
        alpha_post = alpha_prior + heads  # 2 + 8 = 10
        beta_post = beta_prior + tails    # 5 + 2 = 7

        theta_grid = np.linspace(0, 1, 200)
        prior_pdf = beta_dist.pdf(theta_grid, alpha_prior, beta_prior)
        posterior_pdf = beta_dist.pdf(theta_grid, alpha_post, beta_post)

        print(f"Prior:     Beta({alpha_prior}, {beta_prior})  ->  E[theta] = {alpha_prior/(alpha_prior+beta_prior):.3f}")
        print(f"Posterior: Beta({alpha_post}, {beta_post})  ->  E[theta] = {alpha_post/(alpha_post+beta_post):.3f}")
        print(f"MLE:       {heads/len(data):.3f}")
        return alpha_post, alpha_prior, beta_post, beta_prior, posterior_pdf, prior_pdf, theta_grid


    _run()
    return

@app.cell
def _(mo):
    mo.md(r"""
    Above: the prior said $\theta \approx 0.29$, the MLE said $0.80$, and the posterior compromises at $0.59$. The posterior is pulled toward the prior --- this is **shrinkage**, and it provides automatic regularization.
    """)
    return


@app.cell
def _(alpha_post, beta_post, np):
    # --- Posterior predictive: "what is P(next flip = heads)?" ---
    # For Beta-Binomial, the posterior predictive P(y=1 | data) = E[theta | data]
    p_next_heads = alpha_post / (alpha_post + beta_post)
    print(f"Posterior predictive P(next flip = heads) = {p_next_heads:.3f}")

    # Monte Carlo version: sample thetas from posterior, then flip coins
    theta_samples = rng.beta(alpha_post, beta_post, size=10000)
    mc_pred = np.mean(theta_samples)  # average prediction over posterior
    mc_std = np.std(theta_samples)    # uncertainty in theta
    print(f"Monte Carlo estimate: {mc_pred:.3f} +/- {mc_std:.3f}")
    return (theta_samples,)


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 2. Gaussian Processes

    Gaussian Processes are the jewel of Bayesian nonparametrics. Instead of putting a prior on a finite set of parameters, you put a prior directly on **functions**. This is not just a metaphor --- a GP defines a probability distribution over the space of all functions $f: \mathcal{X} \to \mathbb{R}$.

    ### Definition

    A GP is fully specified by a mean function $m(x)$ and a covariance (kernel) function $k(x, x')$:

    $$f \sim \mathcal{GP}(m(x), k(x, x'))$$

    For any finite set of inputs $\{x_1, \ldots, x_n\}$, the function values $[f(x_1), \ldots, f(x_n)]$ are jointly Gaussian with mean vector $\mu_i = m(x_i)$ and covariance matrix $K_{ij} = k(x_i, x_j)$.

    ### The Kernel Determines Everything

    The kernel function $k(x, x')$ controls what kinds of functions the GP considers plausible. Common choices:

    - **Squared exponential (RBF):** $k(x, x') = \sigma^2 \exp\left(-\frac{\|x - x'\|^2}{2\ell^2}\right)$. Produces infinitely smooth functions. The lengthscale $\ell$ controls wiggliness; the signal variance $\sigma^2$ controls amplitude.
    - **Matern class:** A family parameterized by a smoothness parameter $\nu$. Matern-3/2 and Matern-5/2 are the most common. Less smooth than the RBF, and often more realistic for physical processes.
    - **Periodic kernel:** For data with periodic structure.

    Choosing and combining kernels is the primary modeling decision in GP work. You can add kernels (sum of patterns), multiply them (interaction of patterns), and compose them to express rich structural assumptions. See Rasmussen & Williams Ch 4 for a comprehensive catalog.
    """)
    return


@app.cell
def _(np):
    def _run():
        # --- Kernel functions in numpy ---
        def rbf_kernel(X1, X2, length_scale=1.0, signal_var=1.0):
            """k(x,x') = sigma^2 * exp(-||x-x'||^2 / (2*l^2))"""
            sq_dist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * X1 @ X2.T
            return signal_var * np.exp(-0.5 * sq_dist / length_scale**2)

        def matern32_kernel(X1, X2, length_scale=1.0, signal_var=1.0):
            """Matern-3/2: k = sigma^2 * (1 + sqrt(3)*r/l) * exp(-sqrt(3)*r/l)"""
            sq_dist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * X1 @ X2.T
            r = np.sqrt(np.maximum(sq_dist, 1e-12))
            scaled = np.sqrt(3) * r / length_scale
            return signal_var * (1 + scaled) * np.exp(-scaled)

        # Compare: sample from GP prior with different kernels
        X_grid = np.linspace(-5, 5, 200).reshape(-1, 1)
        K_rbf = rbf_kernel(X_grid, X_grid, length_scale=1.0)
        K_mat = matern32_kernel(X_grid, X_grid, length_scale=1.0)

        # Draw 3 function samples from each prior: f ~ N(0, K)
        L_rbf = np.linalg.cholesky(K_rbf + 1e-6 * np.eye(200))
        L_mat = np.linalg.cholesky(K_mat + 1e-6 * np.eye(200))
        samples_rbf = L_rbf @ rng.standard_normal((200, 3))
        samples_mat = L_mat @ rng.standard_normal((200, 3))

        print("RBF samples are infinitely smooth; Matern-3/2 samples are rougher.")
        print(f"RBF kernel matrix shape: {K_rbf.shape}, trace: {np.trace(K_rbf):.1f}")
        return X_grid, matern32_kernel, rbf_kernel, samples_mat, samples_rbf


    _run()
    return

@app.cell
def _(mo):
    mo.md(r"""
    ### GP Regression

    Given training data $(X, y)$ where $y = f(X) + \epsilon$ with $\epsilon \sim \mathcal{N}(0, \sigma_n^2)$, the posterior over $f$ at test points $X_*$ is available in **closed form**:

    $$f_* \mid X, y, X_* \sim \mathcal{N}(\bar{f}_*, \text{cov}(f_*))$$

    where:

    $$\bar{f}_* = K(X_*, X)[K(X, X) + \sigma_n^2 I]^{-1} y$$
    $$\text{cov}(f_*) = K(X_*, X_*) - K(X_*, X)[K(X, X) + \sigma_n^2 I]^{-1} K(X, X_*)$$

    You get a full predictive distribution --- mean and variance at every test point. The variance is small near training data (where you have information) and grows as you move away (where you're uncertain). This is not a post-hoc calibration trick; it's a direct consequence of Bayesian inference.

    See [Bishop S6.4](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) and [Murphy PML2 Ch 18](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML2.pdf) for derivations.
    """)
    return


@app.cell
def _(X_grid, np, rbf_kernel):
    def _run():
        # --- GP regression in numpy (the key equations from above) ---
        # Training data: noisy observations of sin(x)
        X_train = np.array([-4, -3, -1, 0, 2, 4]).reshape(-1, 1)
        y_train = np.sin(X_train).ravel() + 0.1 * rng.standard_normal((len(X_train)))

        noise_var = 0.1**2  # sigma_n^2
        length_scale, signal_var = 1.0, 1.0

        # Compute kernel matrices
        K = rbf_kernel(X_train, X_train, length_scale, signal_var) + noise_var * np.eye(len(X_train))
        K_s = rbf_kernel(X_train, X_grid, length_scale, signal_var)
        K_ss = rbf_kernel(X_grid, X_grid, length_scale, signal_var)

        # Posterior mean and covariance (the two key GP equations)
        K_inv = np.linalg.inv(K)
        mu_post = K_s.T @ K_inv @ y_train                    # f_* mean
        cov_post = K_ss - K_s.T @ K_inv @ K_s                # f_* covariance
        std_post = np.sqrt(np.diag(cov_post))                 # marginal std at each point

        print(f"Posterior mean range: [{mu_post.min():.2f}, {mu_post.max():.2f}]")
        print(f"Uncertainty (std) near data: {std_post[100]:.3f}, far from data: {std_post[0]:.3f}")
        return K, K_inv, K_s, X_train, cov_post, mu_post, noise_var, std_post, y_train


    _run()
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice how the uncertainty (std) is small near training points and large far from data --- this is the hallmark of GP predictions.

    ### Hyperparameter Selection via Marginal Likelihood

    The kernel has hyperparameters (lengthscale, signal variance, noise variance). Instead of cross-validation, GPs use the **log marginal likelihood**:

    $$\log p(y \mid X) = -\frac{1}{2} y^\top (K + \sigma_n^2 I)^{-1} y - \frac{1}{2} \log |K + \sigma_n^2 I| - \frac{n}{2} \log 2\pi$$

    This decomposes into a data fit term and a complexity penalty term (the log determinant). You maximize it with gradient-based optimization --- no validation set needed. This is one of the most elegant aspects of the GP framework.
    """)
    return


@app.cell
def _(K, K_inv, np, y_train):
    # --- Log marginal likelihood computation ---
    n = len(y_train)
    # Three terms: data fit, complexity penalty, constant
    data_fit = -0.5 * y_train @ K_inv @ y_train
    complexity = -0.5 * np.log(np.linalg.det(K))  # penalizes complex models
    constant = -0.5 * n * np.log(2 * np.pi)
    log_marg_lik = data_fit + complexity + constant

    print(f"Log marginal likelihood: {log_marg_lik:.3f}")
    print(f"  Data fit term:         {data_fit:.3f}")
    print(f"  Complexity penalty:    {complexity:.3f}")
    print(f"  Constant:              {constant:.3f}")
    return (log_marg_lik,)


@app.cell
def _(mo):
    mo.md(r"""
    ### The Computational Bottleneck

    The main limitation: GP regression requires inverting an $n \times n$ matrix, costing $O(n^3)$ time and $O(n^2)$ memory. For $n > 10{,}000$ or so, this becomes prohibitive.

    **Sparse GP methods** approximate the full GP using $m \ll n$ **inducing points** --- a small set of pseudo-data points that summarize the training data. The cost drops to $O(nm^2)$. Key methods include FITC, VFE (Titsias, 2009; [JMLR paper](http://proceedings.mlr.press/v5/titsias09a.html)), and the variational sparse GP framework. The GPyTorch library makes these practical for datasets in the hundreds of thousands.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3. Bayesian Neural Networks

    Can we apply Bayesian inference to deep networks? In principle, yes: put a prior $p(W)$ over the weights, observe data, compute the posterior $p(W \mid \mathcal{D})$. In practice, this is incredibly hard because neural networks are high-dimensional, non-linear, and multi-modal --- the posterior is nothing like a Gaussian.

    ### The Intractability Problem

    For a neural network with $d$ parameters, the posterior $p(W \mid \mathcal{D})$ lives in $\mathbb{R}^d$, where $d$ can be millions or billions. The marginal likelihood integral $p(\mathcal{D}) = \int p(\mathcal{D} \mid W) p(W) \, dW$ is utterly intractable. We need approximations.

    ### Variational Inference (VI)

    The most popular approach: approximate the intractable posterior $p(W \mid \mathcal{D})$ with a simpler distribution $q_\phi(W)$ from a tractable family (typically a diagonal Gaussian --- meaning each weight has its own independent mean and variance). You then minimize the KL divergence $D_\text{KL}(q_\phi \| p(W \mid \mathcal{D}))$, which is equivalent to maximizing the **Evidence Lower Bound (ELBO)**:

    $$\text{ELBO}(\phi) = \mathbb{E}_{q_\phi(W)}[\log p(\mathcal{D} \mid W)] - D_\text{KL}(q_\phi(W) \| p(W))$$

    The first term is the expected data fit; the second is a regularizer pulling the approximate posterior toward the prior. The reparameterization trick (same one used in VAEs) makes this differentiable.
    """)
    return


@app.cell
def _(np):
    # --- ELBO and the reparameterization trick (minimal demo) ---
    # Approximate posterior: q(w) = N(mu, sigma^2)
    # Prior: p(w) = N(0, 1)
    # KL(q || p) for two Gaussians has closed form:
    def kl_gaussian(mu_q, sigma_q, mu_p=0.0, sigma_p=1.0):
        """KL(N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2))"""
        return (np.log(sigma_p / sigma_q)
                + (sigma_q**2 + (mu_q - mu_p)**2) / (2 * sigma_p**2)
                - 0.5)

    # Example: approximate posterior q(w) = N(0.5, 0.3^2), prior p(w) = N(0, 1)
    mu_q, sigma_q = 0.5, 0.3
    kl = kl_gaussian(mu_q, sigma_q)
    print(f"KL(q || prior) = {kl:.4f} nats")

    # Reparameterization trick: sample w = mu + sigma * epsilon, epsilon ~ N(0,1)
    epsilon = rng.standard_normal(5)
    w_samples = mu_q + sigma_q * epsilon  # differentiable w.r.t. mu, sigma!
    print(f"Weight samples via reparam trick: {w_samples.round(3)}")
    return (kl_gaussian,)


@app.cell
def _(mo):
    mo.md(r"""
    **Bayes by Backprop** (Blundell et al., 2015; [arXiv:1505.05424](https://arxiv.org/abs/1505.05424)) implements this for neural networks. Each forward pass samples weights from $q_\phi$, making the network stochastic. Uncertainty estimates come from running multiple forward passes and measuring the spread of predictions.

    See [Murphy PML2 Ch 12](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML2.pdf) and [Bishop Ch 10](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) for variational inference foundations.

    ### Monte Carlo Dropout

    Gal & Ghahramani (2016; [arXiv:1506.02142](https://arxiv.org/abs/1506.02142)) showed that a network trained with dropout, when dropout is also applied at test time, is mathematically equivalent to an approximate variational inference scheme. This means **any dropout network is already a Bayesian approximation** --- you just need to run multiple forward passes with dropout enabled and look at the variance across predictions.

    This is the cheapest way to get uncertainty estimates from a neural network. The quality of the uncertainty estimates is debated, but for practical purposes it's often surprisingly useful.
    """)
    return


@app.cell
def _(np):
    # --- MC Dropout in numpy: a tiny 1-hidden-layer network ---
    # Forward pass with dropout at test time
    def mc_dropout_predict(X, W1, b1, W2, b2, drop_rate=0.5, n_samples=50):
        """Run multiple forward passes with dropout ON -> get mean & std."""
        preds = []
        for _ in range(n_samples):
            mask = (rng.random(W1.shape) > drop_rate) / (1 - drop_rate)
            h = np.maximum(0, X @ (W1 * mask) + b1)  # ReLU hidden layer
            y_hat = h @ W2 + b2
            preds.append(y_hat)
        preds = np.array(preds)
        return preds.mean(axis=0), preds.std(axis=0)  # predictive mean & uncertainty

    # Tiny demo: random weights, 1D input
    W1 = rng.standard_normal((1, 20)) * 0.5
    b1 = np.zeros(20)
    W2 = rng.standard_normal((20, 1)) * 0.5
    b2 = np.zeros(1)
    x_test = np.linspace(-3, 3, 5).reshape(-1, 1)
    mean, std = mc_dropout_predict(x_test, W1, b1, W2, b2)
    print("MC Dropout predictions (mean +/- std):")
    for i in range(len(x_test)):
        print(f"  x={x_test[i,0]:+.1f}:  {mean[i,0]:.3f} +/- {std[i,0]:.3f}")
    return (mc_dropout_predict,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Laplace Approximation

    Fit a Gaussian centered at the MAP estimate $\hat{W}$, with covariance equal to the inverse Hessian of the loss at $\hat{W}$:

    $$q(W) = \mathcal{N}(\hat{W}, \, [\nabla^2 \mathcal{L}(\hat{W})]^{-1})$$

    You train the network normally, then compute the Hessian post-hoc. For modern networks, the full Hessian is too large, so you use approximations: diagonal, KFAC, or last-layer-only Laplace. The `laplace-torch` library makes this easy. See Daxberger et al., 2021 ([arXiv:2106.14806](https://arxiv.org/abs/2106.14806)) for a practical guide.
    """)
    return


@app.cell
def _(np):
    # --- Laplace approximation for 1D logistic regression ---
    # Find MAP, then approximate posterior as Gaussian using the Hessian
    # Model: p(y=1|x,w) = sigmoid(w*x), prior: w ~ N(0, 1)
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    # Data
    x_data = np.array([-2, -1, -0.5, 0.5, 1, 2])
    y_data = np.array([0,  0,   0,   1,  1, 1])

    # MAP estimate via Newton's method (logistic regression is convex)
    w = 0.0  # init
    for _ in range(20):
        p = sigmoid(w * x_data)
        grad = np.sum((p - y_data) * x_data) + w  # + w is prior gradient (N(0,1))
        hess = np.sum(p * (1 - p) * x_data**2) + 1  # + 1 is prior Hessian
        w -= grad / hess  # Newton step

    # Laplace: posterior ~ N(w_map, 1/hessian)
    w_map = w
    w_std = 1 / np.sqrt(hess)
    print(f"MAP estimate: w = {w_map:.3f}")
    print(f"Laplace posterior: N({w_map:.3f}, {w_std:.3f}^2)")
    print(f"95% credible interval: [{w_map - 1.96*w_std:.3f}, {w_map + 1.96*w_std:.3f}]")
    return (sigmoid,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4. Markov Chain Monte Carlo (MCMC)

    Variational inference is fast but biased --- it only finds the best approximation within its chosen family. MCMC is asymptotically exact: given enough time, it produces samples from the true posterior. The tradeoff is speed.

    ### Metropolis-Hastings

    The foundational MCMC algorithm. Propose a new point $\theta'$ from a proposal distribution $q(\theta' | \theta)$. Accept it with probability $\min(1, \frac{p(\theta'|\mathcal{D}) q(\theta|\theta')}{p(\theta|\mathcal{D}) q(\theta'|\theta)})$. Repeat. The resulting chain of samples converges to the posterior distribution. The problem: in high dimensions, random proposals are almost always rejected because they move in the wrong direction. Mixing is painfully slow.
    """)
    return


@app.cell
def _(np):
    # --- Metropolis-Hastings for a 1D posterior ---
    # Target: posterior of mu given data ~ N(mu, 1), prior mu ~ N(0, 5^2)
    observed_data = np.array([2.1, 1.8, 2.5, 1.9, 2.3])
    data_mean = observed_data.mean()
    n_obs = len(observed_data)

    def log_posterior(mu):
        """Log posterior: log-likelihood + log-prior"""
        log_lik = -0.5 * np.sum((observed_data - mu)**2)  # N(mu, 1) likelihood
        log_prior = -0.5 * mu**2 / 25                      # N(0, 5^2) prior
        return log_lik + log_prior

    # MH sampling
    n_samples_mh = 5000
    samples_mh = np.zeros(n_samples_mh)
    samples_mh[0] = 0.0  # init
    accepted = 0
    for i in range(1, n_samples_mh):
        proposal = samples_mh[i-1] + 0.5 * rng.standard_normal()  # symmetric Gaussian proposal
        log_alpha = log_posterior(proposal) - log_posterior(samples_mh[i-1])
        if np.log(rng.random()) < log_alpha:  # accept?
            samples_mh[i] = proposal
            accepted += 1
        else:
            samples_mh[i] = samples_mh[i-1]  # reject -> stay

    burnin = 500
    posterior_samples_mh = samples_mh[burnin:]
    print(f"Acceptance rate: {accepted/n_samples_mh:.1%}")
    print(f"Posterior mean (MCMC): {posterior_samples_mh.mean():.3f}")
    print(f"Posterior std  (MCMC): {posterior_samples_mh.std():.3f}")
    # Analytic posterior: N(n*xbar/(n+1/25), 1/(n+1/25))
    prec = n_obs + 1/25
    print(f"Posterior mean (exact): {n_obs * data_mean / prec:.3f}")
    return (posterior_samples_mh,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Hamiltonian Monte Carlo (HMC)

    HMC (Neal, 2011; [arXiv:1206.1901](https://arxiv.org/abs/1206.1901)) fixes this by using gradient information. It treats the parameter space as a physical system: $\theta$ is position, and it introduces a momentum variable $p$. The system evolves according to Hamiltonian dynamics, which naturally follows the contours of the posterior. The result: proposals that are far away in parameter space but still have high acceptance probability.

    **NUTS** (No-U-Turn Sampler; Hoffman & Gelman, 2014; [arXiv:1111.4246](https://arxiv.org/abs/1111.4246)) automatically tunes the trajectory length in HMC, removing the most sensitive hyperparameter. NUTS is the default sampler in Stan and PyMC.
    """)
    return


@app.cell
def _(np, observed_data):
    # --- Hamiltonian Monte Carlo (simplified, 1D) ---
    def grad_log_posterior(mu):
        """Gradient of log posterior w.r.t. mu"""
        grad_lik = np.sum(observed_data - mu)  # d/dmu of -0.5*sum((x-mu)^2)
        grad_prior = -mu / 25                   # d/dmu of -0.5*mu^2/25
        return grad_lik + grad_prior

    def hmc_sample(current, step_size=0.1, n_leapfrog=20):
        """One HMC step: leapfrog integration + accept/reject."""
        q = current
        p = rng.standard_normal()  # sample momentum
        current_p = p
        # Leapfrog integration (simulates Hamiltonian dynamics)
        p -= 0.5 * step_size * (-grad_log_posterior(q))  # half step momentum
        for _ in range(n_leapfrog - 1):
            q += step_size * p                             # full step position
            p -= step_size * (-grad_log_posterior(q))      # full step momentum
        q += step_size * p
        p -= 0.5 * step_size * (-grad_log_posterior(q))   # half step momentum
        # Accept/reject using total energy (Hamiltonian)
        def log_post(mu):
            return -0.5 * np.sum((observed_data - mu)**2) - 0.5 * mu**2 / 25
        current_H = -log_post(current) + 0.5 * current_p**2
        proposed_H = -log_post(q) + 0.5 * p**2
        if np.log(rng.random()) < current_H - proposed_H:
            return q, True
        return current, False

    # Run HMC
    n_hmc = 2000
    hmc_chain = np.zeros(n_hmc)
    hmc_accepted = 0
    for i in range(1, n_hmc):
        hmc_chain[i], acc = hmc_sample(hmc_chain[i-1])
        hmc_accepted += acc

    print(f"HMC acceptance rate: {hmc_accepted/n_hmc:.1%} (much higher than MH!)")
    print(f"HMC posterior mean: {hmc_chain[200:].mean():.3f}")
    print(f"HMC posterior std:  {hmc_chain[200:].std():.3f}")
    return (hmc_chain,)


@app.cell
def _(mo):
    mo.md(r"""
    ### When MCMC is Practical

    MCMC works well for models with tens to thousands of parameters: Bayesian regression, hierarchical models, Gaussian mixture models, small Bayesian neural networks. For models with millions of parameters (modern deep networks), MCMC is generally too slow. This is the regime where variational methods dominate despite their bias.

    ### Probabilistic Programming

    Modern probabilistic programming languages let you specify Bayesian models and run inference automatically:

    - **PyMC** --- Python-native, uses NUTS by default, excellent for applied work
    - **Stan** --- its own language, highly optimized HMC/NUTS, the gold standard for statistical modeling
    - **NumPyro** --- JAX-based, very fast, supports both HMC and VI

    These tools have democratized Bayesian inference. You specify the model (priors, likelihood), and the framework handles the inference machinery.

    See [Murphy PML1 Ch 11--12](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf) for a comprehensive treatment of MCMC and VI.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 5. Bayesian Model Comparison

    One of the most powerful (and underappreciated) features of the Bayesian framework: principled model comparison without a held-out test set.

    ### Marginal Likelihood

    Given two models $\mathcal{M}_1$ and $\mathcal{M}_2$, Bayes says:

    $$\frac{p(\mathcal{M}_1 \mid \mathcal{D})}{p(\mathcal{M}_2 \mid \mathcal{D})} = \frac{p(\mathcal{D} \mid \mathcal{M}_1)}{p(\mathcal{D} \mid \mathcal{M}_2)} \cdot \frac{p(\mathcal{M}_1)}{p(\mathcal{M}_2)}$$

    The ratio $p(\mathcal{D} \mid \mathcal{M}_1) / p(\mathcal{D} \mid \mathcal{M}_2)$ is the **Bayes factor**. Each marginal likelihood is:

    $$p(\mathcal{D} \mid \mathcal{M}) = \int p(\mathcal{D} \mid \theta, \mathcal{M}) \, p(\theta \mid \mathcal{M}) \, d\theta$$

    This integral naturally penalizes complexity. A complex model spreads its prior probability thinly across many possible datasets, so $p(\mathcal{D} \mid \mathcal{M})$ is low unless the data strongly supports the extra complexity. A simple model concentrates its prior on fewer datasets --- if the data matches, the marginal likelihood is high. This is **Bayesian Occam's razor**: the automatic preference for simpler models, with no need for explicit regularization or a validation set.

    See [Bishop S3.4](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) and [Murphy PML1 S5.2](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf) for detailed discussion.
    """)
    return


@app.cell
def _(np):
    # --- Bayesian model comparison: polynomial regression ---
    # Compare degree-1 vs degree-3 polynomial using marginal likelihood
    # For linear regression with known noise, marginal likelihood is analytic

    rng = np.random.default_rng(7)
    x_mc = np.linspace(-1, 1, 15)
    y_mc = 0.5 * x_mc + 0.1 * rng.standard_normal(15)  # true model is linear

    def log_marginal_likelihood_linear(X_design, y, noise_var=0.01, prior_var=1.0):
        """Analytic log marginal likelihood for Bayesian linear regression.
        y ~ N(X @ w, noise_var * I), w ~ N(0, prior_var * I)"""
        n, d = X_design.shape
        # Marginal: y ~ N(0, noise_var * I + prior_var * X @ X.T)
        Sigma_y = noise_var * np.eye(n) + prior_var * X_design @ X_design.T
        sign, logdet = np.linalg.slogdet(Sigma_y)
        return -0.5 * (y @ np.linalg.solve(Sigma_y, y) + logdet + n * np.log(2 * np.pi))

    # Design matrices for degree 1 and degree 3
    X1 = np.column_stack([np.ones_like(x_mc), x_mc])
    X3 = np.column_stack([np.ones_like(x_mc), x_mc, x_mc**2, x_mc**3])

    lml_1 = log_marginal_likelihood_linear(X1, y_mc)
    lml_3 = log_marginal_likelihood_linear(X3, y_mc)
    log_bf = lml_1 - lml_3  # log Bayes factor in favor of model 1

    print(f"Log marginal likelihood (degree 1): {lml_1:.2f}")
    print(f"Log marginal likelihood (degree 3): {lml_3:.2f}")
    print(f"Log Bayes factor (degree 1 vs 3):   {log_bf:.2f}")
    print(f"Bayes factor: {np.exp(log_bf):.1f}x in favor of {'linear' if log_bf > 0 else 'cubic'} model")
    return (log_marginal_likelihood_linear,)


@app.cell
def _(mo):
    mo.md(r"""
    The Bayes factor correctly favors the simpler linear model --- Bayesian Occam's razor in action.

    ---

    ## 6. Bayesian Optimization

    Here's a beautiful application where everything comes together: GPs, uncertainty quantification, and the explore-exploit tradeoff.

    **Problem:** you have an expensive black-box function $f(x)$ (e.g., the validation loss of a neural network as a function of its hyperparameters). Each evaluation costs minutes to hours. You want to find the minimum using as few evaluations as possible.

    **Solution:** Fit a GP surrogate model to the evaluations you've done so far. The GP gives you both a prediction and an uncertainty estimate at every point. Use an **acquisition function** to decide where to evaluate next:

    - **Expected Improvement (EI):** How much improvement over the current best do we expect at this point? Balances exploitation (high predicted value) with exploration (high uncertainty).
    - **Upper Confidence Bound (UCB):** $\mu(x) - \kappa \sigma(x)$ for minimization. The parameter $\kappa$ controls the explore-exploit tradeoff.

    Evaluate $f$ at the point that maximizes the acquisition function, update the GP, repeat.

    This is exactly the hyperparameter tuning strategy hinted at in 1D --- Model Selection & Evaluation. Libraries like BoTorch (built on GPyTorch) and Optuna implement Bayesian optimization. It consistently outperforms grid search and random search when evaluations are expensive.

    See [Murphy PML2 S6.6](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML2.pdf) and Snoek et al., 2012 ([arXiv:1206.2944](https://arxiv.org/abs/1206.2944)) for the foundational paper on Bayesian optimization for hyperparameters.
    """)
    return


@app.cell
def _(np, rbf_kernel):
    # --- Bayesian optimization loop with Expected Improvement ---
    from scipy.stats import norm

    def gp_predict(X_train_bo, y_train_bo, X_test, l=0.3, sig=1.0, noise=1e-4):
        """GP posterior mean and std at test points."""
        K = rbf_kernel(X_train_bo, X_train_bo, l, sig) + noise * np.eye(len(X_train_bo))
        Ks = rbf_kernel(X_train_bo, X_test, l, sig)
        Kss = rbf_kernel(X_test, X_test, l, sig)
        K_inv = np.linalg.inv(K)
        mu = Ks.T @ K_inv @ y_train_bo
        var = np.diag(Kss - Ks.T @ K_inv @ Ks)
        return mu, np.sqrt(np.maximum(var, 1e-8))

    def expected_improvement(mu, std, best_y):
        """EI(x) = E[max(f(x) - best_y, 0)] -- for maximization."""
        z = (mu - best_y) / (std + 1e-8)
        return (mu - best_y) * norm.cdf(z) + std * norm.pdf(z)

    # Objective: find the max of f(x) = -x^2 + sin(3x) on [-2, 2]
    f_obj = lambda x: -x**2 + np.sin(3 * x)
    X_bo = np.array([[-1.5], [0.5]])  # 2 initial evaluations
    y_bo = np.array([f_obj(x[0]) for x in X_bo])

    x_cand = np.linspace(-2, 2, 200).reshape(-1, 1)
    for step in range(5):
        mu_bo, std_bo = gp_predict(X_bo, y_bo, x_cand)
        ei = expected_improvement(mu_bo, std_bo, y_bo.max())
        x_next = x_cand[np.argmax(ei)]
        y_next = f_obj(x_next[0])
        X_bo = np.vstack([X_bo, x_next.reshape(1, -1)])
        y_bo = np.append(y_bo, y_next)
        print(f"Step {step+1}: evaluate x={x_next[0]:+.3f}, f(x)={y_next:.3f}, best so far={y_bo.max():.3f}")
    return (expected_improvement, gp_predict)


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Recommended Resources

    **Textbooks:**
    - [Murphy, Probabilistic Machine Learning: An Introduction](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf) --- Chapters 4, 5, 11, 12, 15, 18
    - [Murphy, Probabilistic Machine Learning: Advanced Topics](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML2.pdf) --- Chapters 6, 12, 17, 18, 30
    - [Bishop, PRML](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) --- Chapters 3, 6, 8, 9, 10, 13
    - Rasmussen & Williams, *Gaussian Processes for Machine Learning* --- free at [gaussianprocess.org](http://www.gaussianprocess.org/gpml/)

    **Key papers:**
    - Rasmussen & Williams, 2006 --- GP textbook (the canonical reference)
    - Blundell et al., 2015 --- Bayes by Backprop ([arXiv:1505.05424](https://arxiv.org/abs/1505.05424))
    - Gal & Ghahramani, 2016 --- MC Dropout as Bayesian approximation ([arXiv:1506.02142](https://arxiv.org/abs/1506.02142))
    - Snoek et al., 2012 --- Practical Bayesian Optimization ([arXiv:1206.2944](https://arxiv.org/abs/1206.2944))
    - Hoffman & Gelman, 2014 --- NUTS ([arXiv:1111.4246](https://arxiv.org/abs/1111.4246))
    - Daxberger et al., 2021 --- Laplace Redux ([arXiv:2106.14806](https://arxiv.org/abs/2106.14806))
    - Wilson & Izmailov, 2020 --- Bayesian Deep Learning and a Probabilistic Perspective of Generalization ([arXiv:2002.08791](https://arxiv.org/abs/2002.08791))
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Code It: Implementation Exercises

    Test your understanding by implementing Bayesian ML concepts from scratch. Each exercise gives you a skeleton --- fill in the TODOs.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Exercise 1: Bayesian Linear Regression

    Implement the posterior update for Bayesian linear regression with a Gaussian prior.

    Given data $(X, y)$ with known noise variance $\sigma_n^2$ and prior $w \sim \mathcal{N}(0, \alpha^{-1} I)$:
    - Posterior precision: $S_N^{-1} = \alpha I + \sigma_n^{-2} X^\top X$
    - Posterior mean: $m_N = \sigma_n^{-2} S_N X^\top y$
    - Posterior predictive at $x_*$: $\mathcal{N}(m_N^\top x_*, \sigma_n^2 + x_*^\top S_N x_*)$
    """)
    return


@app.cell
def _(np):
    def bayesian_linear_regression(X_train_blr, y_train_blr, X_test_blr, noise_var_blr=0.1, alpha_blr=1.0):
        """
        Bayesian linear regression with Gaussian prior.

        Returns:
            m_N: posterior mean of weights
            S_N: posterior covariance of weights
            pred_mean: predictive mean at test points
            pred_std: predictive std at test points
        """
        n, d = X_train_blr.shape

        # TODO: Compute posterior precision matrix S_N_inv = alpha * I + (1/noise_var) * X^T X
        S_N_inv = None

        # TODO: Compute posterior covariance S_N = inv(S_N_inv)
        S_N = None

        # TODO: Compute posterior mean m_N = (1/noise_var) * S_N @ X^T @ y
        m_N = None

        # TODO: Compute predictive mean and variance at test points
        # pred_mean = X_test @ m_N
        # pred_var = noise_var + diag(X_test @ S_N @ X_test^T)
        pred_mean = None
        pred_std = None

        return m_N, S_N, pred_mean, pred_std

    # Test data
    _x = np.linspace(0, 2, 20)
    _X_train = np.column_stack([np.ones_like(_x), _x])
    _y_train = 1.5 * _x + 0.3 + 0.2 * rng.standard_normal(20)
    _X_test = np.column_stack([np.ones(5), np.linspace(0, 3, 5)])

    _m, _S, _pm, _ps = bayesian_linear_regression(_X_train, _y_train, _X_test)
    if _pm is not None:
        print("Posterior weights:", _m.round(3))
        print("Predictive mean:", _pm.round(3))
        print("Predictive std:", _ps.round(3))
    else:
        print("Fill in the TODOs to see results!")
    return (bayesian_linear_regression,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Exercise 2: GP Regression from Scratch

    Implement GP regression end-to-end. Use the RBF kernel, compute the posterior, and evaluate the log marginal likelihood for hyperparameter selection.
    """)
    return


@app.cell
def _(np):
    def gp_regression(X_tr, y_tr, X_te, length_scale_gp=1.0, signal_var_gp=1.0, noise_var_gp=0.01):
        """
        Full GP regression: posterior mean, std, and log marginal likelihood.

        Returns:
            mu: predictive mean at X_te
            std: predictive std at X_te
            lml: log marginal likelihood
        """
        n = len(X_tr)

        # TODO: Compute the RBF kernel matrix K(X_tr, X_tr) + noise * I
        # Hint: use k(x,x') = signal_var * exp(-||x-x'||^2 / (2*length_scale^2))
        K = None

        # TODO: Compute cross-kernel K(X_tr, X_te)
        K_s = None

        # TODO: Compute K(X_te, X_te)
        K_ss = None

        # TODO: Posterior mean = K_s^T @ K^{-1} @ y_tr
        mu = None

        # TODO: Posterior variance = diag(K_ss - K_s^T @ K^{-1} @ K_s)
        std = None

        # TODO: Log marginal likelihood
        # = -0.5 * y^T K^{-1} y - 0.5 * log|K| - n/2 * log(2*pi)
        lml = None

        return mu, std, lml

    # Test
    _X_gp = np.array([-3, -2, -1, 0, 1, 2, 3]).reshape(-1, 1).astype(float)
    _y_gp = np.sin(_X_gp).ravel() + 0.05 * rng.standard_normal(7)
    _X_gp_test = np.linspace(-4, 4, 50).reshape(-1, 1)

    _mu_gp, _std_gp, _lml_gp = gp_regression(_X_gp, _y_gp, _X_gp_test)
    if _mu_gp is not None:
        print(f"Log marginal likelihood: {_lml_gp:.3f}")
        print(f"Pred mean range: [{_mu_gp.min():.3f}, {_mu_gp.max():.3f}]")
    else:
        print("Fill in the TODOs to see results!")
    return (gp_regression,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Exercise 3: Metropolis-Hastings Sampler

    Implement a general-purpose Metropolis-Hastings sampler. Then use it to sample from a Bayesian posterior for estimating the mean of a Gaussian.
    """)
    return


@app.cell
def _(np):
    def metropolis_hastings(log_target, x0, proposal_std_mh=0.5, n_samples_ex=5000):
        """
        General Metropolis-Hastings sampler with Gaussian proposals.

        Args:
            log_target: function that computes log of the (unnormalized) target density
            x0: initial value
            proposal_std_mh: std of Gaussian proposal distribution
            n_samples_ex: number of samples to draw

        Returns:
            samples: array of samples
            acceptance_rate: fraction of proposals accepted
        """
        samples = np.zeros(n_samples_ex)
        samples[0] = x0
        accepted = 0

        for i in range(1, n_samples_ex):
            # TODO: Draw proposal from N(current, proposal_std^2)
            proposal = None

            # TODO: Compute log acceptance ratio = log_target(proposal) - log_target(current)
            log_alpha = None

            # TODO: Accept with probability min(1, exp(log_alpha))
            # If accepted, samples[i] = proposal and increment accepted counter
            # If rejected, samples[i] = samples[i-1]
            pass

        acceptance_rate = accepted / n_samples_ex
        return samples, acceptance_rate

    # Test: sample posterior of mu given data ~ N(mu, 1), prior mu ~ N(0, 10)
    _data_ex = np.array([4.2, 3.8, 4.1, 3.9, 4.5])
    _log_target = lambda mu: -0.5 * np.sum((_data_ex - mu)**2) - 0.5 * mu**2 / 100

    _samples_ex, _ar = metropolis_hastings(_log_target, x0=0.0)
    if _samples_ex[-1] != 0.0:  # check if anything was filled in
        _burnin = 500
        print(f"Acceptance rate: {_ar:.1%}")
        print(f"Posterior mean: {_samples_ex[_burnin:].mean():.3f}  (analytic: ~{5*np.mean(_data_ex)/(5+0.01):.3f})")
        print(f"Posterior std:  {_samples_ex[_burnin:].std():.3f}")
    else:
        print("Fill in the TODOs to see results!")
    return (metropolis_hastings,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Exercise 4: Expected Improvement for Bayesian Optimization

    Implement the Expected Improvement acquisition function and use it in a simple 1D optimization loop. The GP surrogate is provided for you --- just implement EI and the optimization loop.
    """)
    return


@app.cell
def _(np, rbf_kernel):
    from scipy.stats import norm as norm_dist

    def compute_ei(mu_ei, std_ei, best_so_far):
        """
        Compute Expected Improvement for maximization.

        EI(x) = (mu - best) * Phi(z) + std * phi(z)
        where z = (mu - best) / std

        Args:
            mu_ei: predicted means (array)
            std_ei: predicted stds (array)
            best_so_far: best observed value so far

        Returns:
            ei: expected improvement at each point
        """
        # TODO: Compute z = (mu - best_so_far) / std (handle std=0)
        z = None

        # TODO: EI = (mu - best_so_far) * Phi(z) + std * phi(z)
        # Use norm_dist.cdf(z) for Phi and norm_dist.pdf(z) for phi
        ei = None

        return ei

    def bayesopt_loop(objective, x_init, y_init, x_candidates, n_iters=5):
        """
        Run Bayesian optimization for n_iters steps.

        Args:
            objective: function to maximize
            x_init: initial X points (n, 1)
            y_init: initial y values (n,)
            x_candidates: candidate points to evaluate EI over (m, 1)
            n_iters: number of BO iterations

        Returns:
            X_all, y_all: all evaluated points and values
        """
        X_all = x_init.copy()
        y_all = y_init.copy()

        for step in range(n_iters):
            # GP predict (provided)
            _K = rbf_kernel(X_all, X_all, 0.5, 1.0) + 1e-4 * np.eye(len(X_all))
            _Ks = rbf_kernel(X_all, x_candidates, 0.5, 1.0)
            _Kss = rbf_kernel(x_candidates, x_candidates, 0.5, 1.0)
            _Ki = np.linalg.inv(_K)
            _mu = _Ks.T @ _Ki @ y_all
            _var = np.diag(_Kss - _Ks.T @ _Ki @ _Ks)
            _std = np.sqrt(np.maximum(_var, 1e-8))

            # TODO: Compute EI using compute_ei
            ei = None

            # TODO: Select x_next = candidate with highest EI
            # TODO: Evaluate objective at x_next
            # TODO: Append to X_all and y_all
            if ei is None:
                break

        return X_all, y_all

    # Test
    _f_test = lambda x: -(x - 0.7)**2 + np.sin(5 * x)
    _x0 = np.array([[-1.0], [1.0]])
    _y0 = np.array([_f_test(_x0[0, 0]), _f_test(_x0[1, 0])])
    _cands = np.linspace(-2, 2, 300).reshape(-1, 1)

    _Xr, _yr = bayesopt_loop(_f_test, _x0, _y0, _cands)
    if len(_yr) > 2:
        print(f"Best found: f({_Xr[np.argmax(_yr), 0]:.3f}) = {_yr.max():.3f}")
    else:
        print("Fill in the TODOs to see results!")
    return (bayesopt_loop, compute_ei)


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Project Ideas

    1. **GP regression from scratch.** Implement GP regression in NumPy: prior sampling, posterior computation, marginal likelihood optimization. Visualize the prior and posterior distributions over functions for different kernels (RBF, Matern, periodic). This builds deep intuition for what GPs actually do.

    2. **Bayesian neural network with uncertainty.** Use either Bayes by Backprop (via `torchbnn` or manual implementation) or MC Dropout to train a network on a regression task. Plot the predictive mean and uncertainty bands. Show that uncertainty grows in regions with no training data --- and that a standard (non-Bayesian) network gives overconfident predictions in those same regions.

    3. **Bayesian optimization for hyperparameter tuning.** Use BoTorch or Optuna to tune the hyperparameters of a model from an earlier module (e.g., the random forest from 1E --- Trees & Ensembles or the CNN from 2D --- Convolutional Neural Networks). Compare the number of evaluations needed versus random search. Visualize the surrogate GP model and the acquisition function.

    4. **Probabilistic programming with PyMC.** Build a Bayesian hierarchical model for a real dataset --- student test scores across schools, batting averages across baseball players, or medical trial results across hospitals. Use partial pooling and visualize how the hierarchical structure shares information between groups.

    5. **Bayesian model comparison.** Take a dataset and fit models of varying complexity (e.g., polynomial regression of degrees 1 through 10). Compute the marginal likelihood for each and show that Bayesian model comparison automatically selects the right complexity --- without cross-validation. Compare the result to AIC/BIC and cross-validation.
    """)
    return


if __name__ == "__main__":
    app.run()
