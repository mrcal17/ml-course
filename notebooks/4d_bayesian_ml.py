import marimo

app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Path D --- Bayesian Machine Learning

    Most of the ML you've learned so far is frequentist in spirit: you find a single best parameter vector $\hat{\theta}$ (via MLE or MAP), make predictions using that point estimate, and call it a day. Bayesian ML takes a fundamentally different stance. Instead of asking "what is the best $\theta$?", it asks "given the data I've seen, what is the full probability distribution over $\theta$?"

    This isn't just philosophical navel-gazing. The posterior distribution $p(\theta \mid \mathcal{D})$ tells you not only what the model thinks, but also **how uncertain it is** --- and that uncertainty information turns out to be enormously useful. It gives you calibrated predictions, principled model comparison, and automatic regularization. The cost? Computation. Computing the posterior is usually intractable, and the entire field of Bayesian ML is essentially a catalog of clever approximations.

    This guide maps out the territory. You should already be comfortable with Bayes' theorem, priors, posteriors, and likelihoods from 0D --- Probability Foundations and 0E --- Statistical Estimation. Here we go much further.

    > **Primary references:** [Murphy PML1](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf) is the best single reference for this entire guide. [Bishop PRML](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) Chapters 3, 6, and 8-13 cover most of these topics with exceptional clarity. Rasmussen & Williams, *Gaussian Processes for Machine Learning* (2006), is freely available at [gaussianprocess.org](http://www.gaussianprocess.org/gpml/) and is the definitive GP reference.
    """)
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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

    ### GP Regression

    Given training data $(X, y)$ where $y = f(X) + \epsilon$ with $\epsilon \sim \mathcal{N}(0, \sigma_n^2)$, the posterior over $f$ at test points $X_*$ is available in **closed form**:

    $$f_* \mid X, y, X_* \sim \mathcal{N}(\bar{f}_*, \text{cov}(f_*))$$

    where:

    $$\bar{f}_* = K(X_*, X)[K(X, X) + \sigma_n^2 I]^{-1} y$$
    $$\text{cov}(f_*) = K(X_*, X_*) - K(X_*, X)[K(X, X) + \sigma_n^2 I]^{-1} K(X, X_*)$$

    You get a full predictive distribution --- mean and variance at every test point. The variance is small near training data (where you have information) and grows as you move away (where you're uncertain). This is not a post-hoc calibration trick; it's a direct consequence of Bayesian inference.

    See [Bishop S6.4](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) and [Murphy PML2 Ch 18](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML2.pdf) for derivations.

    ### Hyperparameter Selection via Marginal Likelihood

    The kernel has hyperparameters (lengthscale, signal variance, noise variance). Instead of cross-validation, GPs use the **log marginal likelihood**:

    $$\log p(y \mid X) = -\frac{1}{2} y^\top (K + \sigma_n^2 I)^{-1} y - \frac{1}{2} \log |K + \sigma_n^2 I| - \frac{n}{2} \log 2\pi$$

    This decomposes into a data fit term and a complexity penalty term (the log determinant). You maximize it with gradient-based optimization --- no validation set needed. This is one of the most elegant aspects of the GP framework.

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

    **Bayes by Backprop** (Blundell et al., 2015; [arXiv:1505.05424](https://arxiv.org/abs/1505.05424)) implements this for neural networks. Each forward pass samples weights from $q_\phi$, making the network stochastic. Uncertainty estimates come from running multiple forward passes and measuring the spread of predictions.

    See [Murphy PML2 Ch 12](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML2.pdf) and [Bishop Ch 10](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) for variational inference foundations.

    ### Monte Carlo Dropout

    Gal & Ghahramani (2016; [arXiv:1506.02142](https://arxiv.org/abs/1506.02142)) showed that a network trained with dropout, when dropout is also applied at test time, is mathematically equivalent to an approximate variational inference scheme. This means **any dropout network is already a Bayesian approximation** --- you just need to run multiple forward passes with dropout enabled and look at the variance across predictions.

    This is the cheapest way to get uncertainty estimates from a neural network. The quality of the uncertainty estimates is debated, but for practical purposes it's often surprisingly useful.

    ### Laplace Approximation

    Fit a Gaussian centered at the MAP estimate $\hat{W}$, with covariance equal to the inverse Hessian of the loss at $\hat{W}$:

    $$q(W) = \mathcal{N}(\hat{W}, \, [\nabla^2 \mathcal{L}(\hat{W})]^{-1})$$

    You train the network normally, then compute the Hessian post-hoc. For modern networks, the full Hessian is too large, so you use approximations: diagonal, KFAC, or last-layer-only Laplace. The `laplace-torch` library makes this easy. See Daxberger et al., 2021 ([arXiv:2106.14806](https://arxiv.org/abs/2106.14806)) for a practical guide.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4. Markov Chain Monte Carlo (MCMC)

    Variational inference is fast but biased --- it only finds the best approximation within its chosen family. MCMC is asymptotically exact: given enough time, it produces samples from the true posterior. The tradeoff is speed.

    ### Metropolis-Hastings

    The foundational MCMC algorithm. Propose a new point $\theta'$ from a proposal distribution $q(\theta' | \theta)$. Accept it with probability $\min(1, \frac{p(\theta'|\mathcal{D}) q(\theta|\theta')}{p(\theta|\mathcal{D}) q(\theta'|\theta)})$. Repeat. The resulting chain of samples converges to the posterior distribution. The problem: in high dimensions, random proposals are almost always rejected because they move in the wrong direction. Mixing is painfully slow.

    ### Hamiltonian Monte Carlo (HMC)

    HMC (Neal, 2011; [arXiv:1206.1901](https://arxiv.org/abs/1206.1901)) fixes this by using gradient information. It treats the parameter space as a physical system: $\theta$ is position, and it introduces a momentum variable $p$. The system evolves according to Hamiltonian dynamics, which naturally follows the contours of the posterior. The result: proposals that are far away in parameter space but still have high acceptance probability.

    **NUTS** (No-U-Turn Sampler; Hoffman & Gelman, 2014; [arXiv:1111.4246](https://arxiv.org/abs/1111.4246)) automatically tunes the trajectory length in HMC, removing the most sensitive hyperparameter. NUTS is the default sampler in Stan and PyMC.

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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
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


@app.cell(hide_code=True)
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
