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
    return (np, plt)


@app.cell
def _(mo):
    mo.md(r"""
    # Module 0E: Statistical Estimation

    You now have solid probability. You know about random variables, distributions, expectations, Bayes' theorem. Good. This module is where we pivot from "here's how randomness works" to "here's how we learn from data." That pivot is the entire foundation of machine learning, and I mean that literally — almost every ML algorithm you'll encounter is doing some form of statistical estimation under the hood.

    I know you've seen MLE and hypothesis testing before. You got through the exams. But you didn't really internalize it — and that's fine, because this time we're going to build it from the ground up, connecting every concept directly to what you'll need for ML.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 1. The Estimation Problem

    Here's the setup. Nature has some process that generates data. You don't get to see the process — you only see the data it spits out. Your job is to figure out what that process looks like, using only the data you've observed.

    That's it. That's the entire game of statistics, and it's the entire game of machine learning.

    When you train a neural network on images, you're estimating the function that maps pixels to labels. When you fit a linear regression, you're estimating slope and intercept parameters. When you cluster data points, you're estimating which group each point belongs to. It's all estimation.

    More formally: we assume the data $x_1, x_2, \ldots, x_n$ were drawn from some distribution $p(x \mid \theta)$, where $\theta$ represents the unknown parameters of the process. Our goal is to infer $\theta$ from the data.

    > This framing is developed carefully in [Chan 8.1 — Estimation: The Big Picture](file:///C:/Users/landa/ml-course/textbooks/Chan-Probability.pdf) and [Bishop 1.2 — Probability Theory](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 2. Point Estimation

    ### What Is an Estimator?

    An **estimator** $\hat{\theta}$ is a function of the data. Nothing more. You feed in your observed data $x_1, \ldots, x_n$, and the estimator outputs a guess for $\theta$:

    $$\hat{\theta} = g(x_1, x_2, \ldots, x_n)$$

    The sample mean $\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$ is an estimator for the population mean $\mu$. The sample variance is an estimator for the population variance. These are functions of the data — they produce a single number (a "point") as your best guess.

    Because the data are random, the estimator is also random. Different datasets give you different estimates. This is crucial: an estimator has its own distribution, called the **sampling distribution**, and we can talk about its properties.

    ### Properties of Estimators

    **Bias.** An estimator is **unbiased** if, on average, it hits the true value:

    $$\text{Bias}(\hat{\theta}) = E[\hat{\theta}] - \theta$$

    If $\text{Bias} = 0$, the estimator is unbiased. The sample mean is unbiased for $\mu$. The naive sample variance $\frac{1}{n}\sum(x_i - \bar{x})^2$ is actually biased — it systematically underestimates $\sigma^2$, which is why we divide by $n-1$ instead of $n$ (Bessel's correction).

    **Variance.** How much does $\hat{\theta}$ jump around from sample to sample?

    $$\text{Var}(\hat{\theta}) = E[(\hat{\theta} - E[\hat{\theta}])^2]$$

    Low variance means the estimator is stable. But low variance alone doesn't mean good — you could have an estimator that always returns 42 regardless of the data. Zero variance, completely useless.

    **Mean Squared Error.** The MSE captures both sins:

    $$\text{MSE}(\hat{\theta}) = \text{Bias}(\hat{\theta})^2 + \text{Var}(\hat{\theta})$$

    This is the bias-variance decomposition for estimators. You'll see this exact same structure again in Module 1A when we talk about model prediction error — it's the same fundamental tradeoff operating at a different level. Sometimes a slightly biased estimator with much lower variance beats an unbiased estimator. This is the core intuition behind regularization.

    **Consistency.** As you get more data, does $\hat{\theta}$ converge to the true $\theta$? Formally, $\hat{\theta}_n \xrightarrow{P} \theta$ as $n \to \infty$. An estimator can be biased but still consistent (the bias shrinks with more data).

    **Efficiency.** Among all unbiased estimators, the one with the smallest variance is called **efficient**. The Cramer-Rao lower bound gives us the minimum possible variance for any unbiased estimator — if an estimator achieves that bound, it's efficient.

    > [Chan 8.2 — Properties of Point Estimators](file:///C:/Users/landa/ml-course/textbooks/Chan-Probability.pdf) covers bias, variance, MSE, and consistency with worked examples.
    """)
    return


@app.cell
def _(np):
    def _run():
        # Demonstrate bias of sample variance with 1/n vs 1/(n-1)
        # True parameters: mu=5, sigma^2=4
        true_mu, true_sigma2 = 5.0, 4.0
        n_samples, n_trials = 10, 50_000

        rng = np.random.default_rng(0)
        biased_vars, unbiased_vars = [], []
        for _ in range(n_trials):
            x = rng.normal(true_mu, np.sqrt(true_sigma2), size=n_samples)
            biased_vars.append(np.var(x))          # divides by n (MLE)
            unbiased_vars.append(np.var(x, ddof=1)) # divides by n-1

        print(f"True sigma^2 = {true_sigma2}")
        print(f"E[biased estimator]   = {np.mean(biased_vars):.4f}  (bias = {np.mean(biased_vars) - true_sigma2:.4f})")
        print(f"E[unbiased estimator] = {np.mean(unbiased_vars):.4f}  (bias = {np.mean(unbiased_vars) - true_sigma2:.4f})")


    _run()
    return

@app.cell
def _(np, plt):
    def _run():
        # Simulate the bias-variance tradeoff: shrinkage estimator vs sample mean
        # Estimator family: theta_hat_c = c * x_bar  (c=1 is unbiased, c<1 adds bias but reduces variance)
        true_mean = 5.0
        n_obs = 10
        n_sim = 20_000
        rng_bv = np.random.default_rng(42)

        c_values = np.linspace(0.5, 1.2, 50)
        bias2_arr, var_arr, mse_arr = [], [], []

        samples = rng_bv.normal(true_mean, 2.0, size=(n_sim, n_obs))
        x_bars = samples.mean(axis=1)

        for c in c_values:
            estimates = c * x_bars
            bias2 = (estimates.mean() - true_mean) ** 2
            var = estimates.var()
            bias2_arr.append(bias2)
            var_arr.append(var)
            mse_arr.append(bias2 + var)

        fig_bv, ax_bv = plt.subplots(figsize=(7, 4))
        ax_bv.plot(c_values, bias2_arr, label="Bias$^2$")
        ax_bv.plot(c_values, var_arr, label="Variance")
        ax_bv.plot(c_values, mse_arr, "k--", lw=2, label="MSE = Bias$^2$ + Var")
        ax_bv.axvline(1.0, color="gray", ls=":", label="c=1 (unbiased)")
        ax_bv.set_xlabel("Shrinkage factor c")
        ax_bv.set_ylabel("Value")
        ax_bv.set_title("Bias-Variance Tradeoff for Shrinkage Estimator")
        ax_bv.legend()
        ax_bv.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_bv


    _run()
    return

@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 3. Maximum Likelihood Estimation (MLE)

    This is the single most important estimation method you'll learn. If you understand MLE deeply, you understand the mathematical foundation of most ML algorithms. I'm not exaggerating.

    ### The Likelihood Function

    You have data $\mathbf{x} = (x_1, \ldots, x_n)$ and a model $p(x \mid \theta)$. The **likelihood function** flips the perspective:

    $$L(\theta) = p(\mathbf{x} \mid \theta) = \prod_{i=1}^n p(x_i \mid \theta)$$

    The product comes from assuming the observations are independent. Notice what happened: the probability function $p(x \mid \theta)$ treats $\theta$ as fixed and asks "how probable is this data?" The likelihood function $L(\theta)$ treats the data as fixed and asks "how plausible is this parameter value?"

    The **maximum likelihood estimate** is the $\theta$ that makes the observed data most probable:

    $$\hat{\theta}_{\text{MLE}} = \arg\max_\theta \; L(\theta)$$

    ### Why We Take the Log

    In practice, we almost always work with the **log-likelihood**:

    $$\ell(\theta) = \log L(\theta) = \sum_{i=1}^n \log p(x_i \mid \theta)$$

    Two reasons. First, products become sums, which are easier to differentiate. Second, multiplying many small probabilities together causes numerical underflow — computers can't represent numbers like $10^{-3000}$. Sums of log-probabilities don't have this problem.

    Since $\log$ is monotonically increasing, maximizing $\ell(\theta)$ gives the same answer as maximizing $L(\theta)$.

    ### The MLE Recipe

    1. Write down the likelihood $L(\theta) = \prod_{i=1}^n p(x_i \mid \theta)$
    2. Take the log: $\ell(\theta) = \sum_{i=1}^n \log p(x_i \mid \theta)$
    3. Differentiate with respect to $\theta$ and set to zero: $\frac{\partial \ell}{\partial \theta} = 0$
    4. Solve for $\hat{\theta}$

    Let's do this three times.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Worked Example: MLE for Bernoulli

    Data: $x_1, \ldots, x_n \in \{0, 1\}$, coin flips with unknown probability $p$.

    **Step 1 — Likelihood:**
    $$L(p) = \prod_{i=1}^n p^{x_i}(1-p)^{1-x_i}$$

    **Step 2 — Log-likelihood:**
    $$\ell(p) = \sum_{i=1}^n \left[ x_i \log p + (1-x_i)\log(1-p) \right]$$

    Let $k = \sum x_i$ (number of heads). Then:
    $$\ell(p) = k \log p + (n - k)\log(1 - p)$$

    **Step 3 — Differentiate and set to zero:**
    $$\frac{d\ell}{dp} = \frac{k}{p} - \frac{n-k}{1-p} = 0$$

    **Step 4 — Solve:**
    $$\hat{p}_{\text{MLE}} = \frac{k}{n}$$

    The MLE for a coin's bias is just the fraction of heads. Intuitively obvious, but now you've derived it rigorously.
    """)
    return


@app.cell
def _(np, plt):
    def _run():
        # MLE for Bernoulli: compute and visualize the log-likelihood
        rng_bern = np.random.default_rng(7)
        true_p = 0.7
        flips = rng_bern.binomial(1, true_p, size=20)  # 20 coin flips
        k = flips.sum()
        n_bern = len(flips)

        # MLE = k/n
        p_mle_bern = k / n_bern
        print(f"Observed: {k} heads in {n_bern} flips")
        print(f"MLE for p: {p_mle_bern:.3f}  (true p = {true_p})")

        # Plot log-likelihood as function of p
        p_grid = np.linspace(0.01, 0.99, 200)
        log_lik_bern = k * np.log(p_grid) + (n_bern - k) * np.log(1 - p_grid)

        fig_bern, ax_bern = plt.subplots(figsize=(6, 3.5))
        ax_bern.plot(p_grid, log_lik_bern)
        ax_bern.axvline(p_mle_bern, color="r", ls="--", label=f"MLE = {p_mle_bern:.2f}")
        ax_bern.set_xlabel("p"); ax_bern.set_ylabel("Log-likelihood")
        ax_bern.set_title("Bernoulli Log-Likelihood")
        ax_bern.legend(); ax_bern.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_bern


    _run()
    return

@app.cell
def _(mo):
    mo.md(r"""
    ### Worked Example: MLE for Gaussian Parameters

    Data: $x_1, \ldots, x_n \sim \mathcal{N}(\mu, \sigma^2)$ with both $\mu$ and $\sigma^2$ unknown.

    **Log-likelihood:**
    $$\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n(x_i - \mu)^2$$

    **Differentiate w.r.t. $\mu$:**
    $$\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^n(x_i - \mu) = 0 \implies \hat{\mu}_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n x_i = \bar{x}$$

    **Differentiate w.r.t. $\sigma^2$:**
    $$\frac{\partial \ell}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_{i=1}^n(x_i - \mu)^2 = 0 \implies \hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n(x_i - \bar{x})^2$$

    The MLE for the mean is the sample mean. The MLE for the variance is the sample variance with $n$ in the denominator (not $n-1$). This is biased — but it's consistent, and the bias is tiny for large $n$. MLE doesn't guarantee unbiasedness.

    > [Chan 8.3 — Maximum Likelihood Estimation](file:///C:/Users/landa/ml-course/textbooks/Chan-Probability.pdf) works through these derivations step by step with additional examples.
    """)
    return


@app.cell
def _(np, plt):
    def _run():
        # MLE for Gaussian: compute estimates and visualize log-likelihood surface
        rng_gauss = np.random.default_rng(42)
        data_gauss = rng_gauss.normal(loc=3.0, scale=2.0, size=50)

        # Analytical MLE
        mu_hat = data_gauss.mean()           # sample mean
        sigma2_hat = data_gauss.var()        # 1/n variance (MLE)
        print(f"MLE: mu = {mu_hat:.3f}, sigma^2 = {sigma2_hat:.3f}")
        print(f"True: mu = 3.0, sigma^2 = 4.0")

        # Log-likelihood on a grid of mu values (sigma^2 fixed at MLE)
        mu_grid = np.linspace(1, 5, 200)
        n_g = len(data_gauss)
        ll_mu = -0.5 * n_g * np.log(2 * np.pi * sigma2_hat) - \
                0.5 * np.sum((data_gauss[:, None] - mu_grid[None, :]) ** 2, axis=0) / sigma2_hat

        fig_g, ax_g = plt.subplots(figsize=(6, 3.5))
        ax_g.plot(mu_grid, ll_mu)
        ax_g.axvline(mu_hat, color="r", ls="--", label=f"MLE mu = {mu_hat:.2f}")
        ax_g.set_xlabel("mu"); ax_g.set_ylabel("Log-likelihood")
        ax_g.set_title("Gaussian Log-Likelihood (sigma^2 at MLE)")
        ax_g.legend(); ax_g.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_g


    _run()
    return

@app.cell
def _(mo):
    mo.md(r"""
    ### Worked Example: MLE for Linear Regression

    This one is important. Assume:

    $$y_i = \mathbf{w}^T \mathbf{x}_i + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2)$$

    So $y_i \mid \mathbf{x}_i \sim \mathcal{N}(\mathbf{w}^T\mathbf{x}_i, \sigma^2)$. The log-likelihood is:

    $$\ell(\mathbf{w}) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n(y_i - \mathbf{w}^T\mathbf{x}_i)^2$$

    Maximizing this with respect to $\mathbf{w}$ is equivalent to minimizing:

    $$\sum_{i=1}^n (y_i - \mathbf{w}^T\mathbf{x}_i)^2$$

    That's the sum of squared residuals. **MLE for linear regression under Gaussian noise is ordinary least squares (OLS).** The loss function you've been using since your first stats class has a probabilistic justification — it comes from assuming Gaussian errors and doing MLE. This connection between probabilistic models and loss functions is one of the deepest ideas in ML.

    > [Bishop 3.1 — Linear Basis Function Models](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) develops this connection between MLE and least squares in full generality.
    """)
    return


@app.cell
def _(np):
    def _run():
        # MLE for linear regression = OLS via the normal equation
        rng_lr = np.random.default_rng(99)
        n_lr = 80
        X_lr = rng_lr.standard_normal((n_lr, 2))
        X_lr_b = np.column_stack([np.ones(n_lr), X_lr])  # add intercept column
        w_true_lr = np.array([1.0, 2.0, -0.5])
        y_lr = X_lr_b @ w_true_lr + rng_lr.normal(0, 0.5, n_lr)

        # Normal equation: w_hat = (X^T X)^{-1} X^T y
        w_ols_lr = np.linalg.solve(X_lr_b.T @ X_lr_b, X_lr_b.T @ y_lr)
        print(f"True weights:     {w_true_lr}")
        print(f"OLS/MLE weights:  {np.round(w_ols_lr, 4)}")

        # MLE for noise variance
        resid = y_lr - X_lr_b @ w_ols_lr
        sigma2_lr = np.mean(resid ** 2)
        print(f"MLE sigma^2: {sigma2_lr:.4f}  (true = 0.25)")


    _run()
    return

@app.cell
def _(mo):
    mo.md(r"""
    ### Properties of MLE

    - **Consistent:** $\hat{\theta}_{\text{MLE}} \xrightarrow{P} \theta$ as $n \to \infty$
    - **Asymptotically efficient:** as $n \to \infty$, MLE achieves the Cramer-Rao lower bound
    - **Asymptotically normal:** $\hat{\theta}_{\text{MLE}} \approx \mathcal{N}(\theta, I(\theta)^{-1}/n)$ for large $n$, where $I(\theta)$ is the Fisher information
    - **Equivariant:** if $\hat{\theta}$ is the MLE for $\theta$, then $g(\hat{\theta})$ is the MLE for $g(\theta)$

    ### When MLE Goes Wrong

    MLE is not a universal solution. It can fail when:

    - **Small samples:** MLE overfits. With 3 coin flips that all land heads, the MLE says $p = 1$. That's a terrible estimate.
    - **Multimodal likelihood:** the optimization can get stuck at local maxima.
    - **Degenerate solutions:** in Gaussian mixture models, if a component collapses to a single data point, the likelihood goes to infinity. MLE gives you a useless answer.

    These failures motivate the Bayesian approach.
    """)
    return


@app.cell
def _(np, plt):
    def _run():
        # MLE overfitting with small samples: repeated experiments
        rng_small = np.random.default_rng(10)
        true_p_demo = 0.6
        sample_sizes = [3, 10, 50, 200]
        n_repeats = 5_000

        fig_mle_fail, axes_mle = plt.subplots(1, 4, figsize=(14, 3), sharey=True)
        for ax_m, n_s in zip(axes_mle, sample_sizes):
            mle_estimates = rng_small.binomial(n_s, true_p_demo, n_repeats) / n_s
            ax_m.hist(mle_estimates, bins=30, density=True, alpha=0.7, edgecolor="k")
            ax_m.axvline(true_p_demo, color="r", ls="--", lw=2, label=f"true p={true_p_demo}")
            ax_m.set_title(f"n = {n_s}")
            ax_m.set_xlabel("MLE estimate")
            if n_s == 3:
                ax_m.set_ylabel("Density")
            ax_m.legend(fontsize=8)
        fig_mle_fail.suptitle("MLE Sampling Distribution: Small vs Large Samples", y=1.02)
        plt.tight_layout()
        fig_mle_fail


    _run()
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4. Maximum A Posteriori (MAP) Estimation

    ### The Bayesian Fix

    MLE treats $\theta$ as a fixed unknown. The Bayesian approach treats $\theta$ as a random variable with a **prior distribution** $p(\theta)$ that encodes your beliefs before seeing data.

    By Bayes' theorem:

    $$p(\theta \mid \mathbf{x}) = \frac{p(\mathbf{x} \mid \theta) \, p(\theta)}{p(\mathbf{x})} \propto p(\mathbf{x} \mid \theta) \, p(\theta)$$

    $$\text{Posterior} \propto \text{Likelihood} \times \text{Prior}$$

    The **MAP estimate** is the mode of the posterior:

    $$\hat{\theta}_{\text{MAP}} = \arg\max_\theta \; p(\theta \mid \mathbf{x}) = \arg\max_\theta \left[ \log p(\mathbf{x} \mid \theta) + \log p(\theta) \right]$$

    Compare this to MLE, which maximizes just $\log p(\mathbf{x} \mid \theta)$. MAP adds the $\log p(\theta)$ term — a penalty or bonus depending on how well $\theta$ aligns with your prior beliefs. **MAP = MLE + regularization.** This is one of the most important conceptual connections in ML.
    """)
    return


@app.cell
def _(np, plt):
    def _run():
        # MAP vs MLE for Bernoulli with Beta prior
        # Observe 7 heads in 10 flips — compare MLE and MAP under different priors
        from scipy.stats import beta as beta_dist_demo
        k_demo, n_demo = 7, 10
        p_mle_demo = k_demo / n_demo

        priors = [(1, 1, "Beta(1,1) — uniform"), (2, 5, "Beta(2,5) — skeptical of high p")]
        p_range = np.linspace(0.01, 0.99, 300)

        fig_map, axes_map = plt.subplots(1, 2, figsize=(12, 4))
        for ax_mp, (a, b, label) in zip(axes_map, priors):
            # MAP = (k + a - 1) / (n + a + b - 2)
            p_map_demo = (k_demo + a - 1) / (n_demo + a + b - 2)
            prior_pdf = beta_dist_demo.pdf(p_range, a, b)
            post_pdf = beta_dist_demo.pdf(p_range, a + k_demo, b + n_demo - k_demo)
            lik = p_range ** k_demo * (1 - p_range) ** (n_demo - k_demo)
            lik = lik / lik.max() * prior_pdf.max()  # scale for plotting
            ax_mp.plot(p_range, prior_pdf, "g--", lw=2, label="Prior")
            ax_mp.plot(p_range, lik, "r:", lw=2, label="Likelihood (scaled)")
            ax_mp.plot(p_range, post_pdf, "b-", lw=2, label="Posterior")
            ax_mp.axvline(p_mle_demo, color="r", alpha=0.4, label=f"MLE={p_mle_demo:.2f}")
            ax_mp.axvline(p_map_demo, color="b", alpha=0.4, label=f"MAP={p_map_demo:.2f}")
            ax_mp.set_title(label); ax_mp.legend(fontsize=8); ax_mp.set_xlabel("p")
        plt.tight_layout()
        fig_map


    _run()
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Worked Example: MAP with Gaussian Prior -> Ridge Regression

    Take linear regression with the MLE setup from before, and add a Gaussian prior on the weights:

    $$\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \tau^2 \mathbf{I})$$

    The log-posterior (up to constants) is:

    $$\log p(\mathbf{w} \mid \mathbf{X}, \mathbf{y}) \propto -\frac{1}{2\sigma^2}\sum_{i=1}^n(y_i - \mathbf{w}^T\mathbf{x}_i)^2 - \frac{1}{2\tau^2}\|\mathbf{w}\|_2^2$$

    Maximizing this is equivalent to minimizing:

    $$\sum_{i=1}^n(y_i - \mathbf{w}^T\mathbf{x}_i)^2 + \lambda\|\mathbf{w}\|_2^2$$

    where $\lambda = \sigma^2/\tau^2$. That's **Ridge regression**. The L2 regularization penalty that you might have thought was just an ad-hoc trick to prevent overfitting is actually the log of a Gaussian prior on the weights. When $\tau^2$ is large (weak prior, "I don't have strong beliefs about $\mathbf{w}$"), $\lambda$ is small and you're close to MLE. When $\tau^2$ is small (strong prior, "I believe $\mathbf{w}$ should be near zero"), $\lambda$ is large and regularization dominates.

    ### Worked Example: MAP with Laplace Prior -> Lasso Regression

    Now use a Laplace (double exponential) prior instead:

    $$p(w_j) = \frac{1}{2b}\exp\left(-\frac{|w_j|}{b}\right)$$

    The log-prior is $-\frac{1}{b}\sum_j |w_j|$ plus a constant. The MAP objective becomes:

    $$\sum_{i=1}^n(y_i - \mathbf{w}^T\mathbf{x}_i)^2 + \lambda\|\mathbf{w}\|_1$$

    That's the **Lasso**. L1 regularization = Laplace prior. The sharp peak of the Laplace distribution at zero is what drives some weights to exactly zero, producing sparse solutions. The geometry of this — why L1 gives sparsity but L2 doesn't — is worth thinking about visually (imagine the contours of the loss function intersecting the constraint region).

    > [Bishop 3.3 — Bayesian Linear Regression](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) develops the full Bayesian treatment. [Murphy 11.6 — MAP estimation](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf) connects MAP to regularization explicitly.

    ### Choosing Priors

    The prior is both the strength and the criticism of Bayesian methods. Some guidelines:

    - **Uninformative priors** (e.g., very wide Gaussians) let the data speak. As $n \to \infty$, the prior becomes irrelevant and MAP -> MLE.
    - **Weakly informative priors** encode soft beliefs — "I think the weights are probably small but I'm not sure." This is what regularization is in practice.
    - **Conjugate priors** make the math tractable (more on this below).
    - **Empirical Bayes** estimates the prior from data — a practical compromise.

    The choice of prior matters most when you have little data. With abundant data, the likelihood overwhelms any reasonable prior.
    """)
    return


@app.cell
def _(np, plt):
    def _run():
        # Ridge vs Lasso: compare weight shrinkage paths
        rng_reg = np.random.default_rng(0)
        n_reg = 100
        X_reg = rng_reg.standard_normal((n_reg, 5))
        # Only features 0 and 1 matter; rest are noise
        w_true_reg = np.array([3.0, -2.0, 0.0, 0.0, 0.0])
        y_reg = X_reg @ w_true_reg + rng_reg.normal(0, 1, n_reg)

        lambdas_reg = np.logspace(-1, 3, 60)
        ridge_path = np.zeros((len(lambdas_reg), 5))
        lasso_path = np.zeros((len(lambdas_reg), 5))

        XtX = X_reg.T @ X_reg
        Xty = X_reg.T @ y_reg
        for i, lam in enumerate(lambdas_reg):
            # Ridge closed form: (X^T X + lambda I)^{-1} X^T y
            ridge_path[i] = np.linalg.solve(XtX + lam * np.eye(5), Xty)

        # Lasso via coordinate descent (simple implementation)
        from sklearn.linear_model import Lasso
        for i, lam in enumerate(lambdas_reg):
            model_l = Lasso(alpha=lam / (2 * n_reg), fit_intercept=False, max_iter=5000)
            model_l.fit(X_reg, y_reg)
            lasso_path[i] = model_l.coef_

        fig_reg, (ax_r, ax_l) = plt.subplots(1, 2, figsize=(12, 4))
        for j in range(5):
            ax_r.plot(lambdas_reg, ridge_path[:, j], label=f"w{j} (true={w_true_reg[j]:.0f})")
            ax_l.plot(lambdas_reg, lasso_path[:, j], label=f"w{j} (true={w_true_reg[j]:.0f})")
        for ax_rl, title_rl in [(ax_r, "Ridge (L2 / Gaussian prior)"), (ax_l, "Lasso (L1 / Laplace prior)")]:
            ax_rl.set_xscale("log"); ax_rl.set_xlabel("lambda"); ax_rl.set_ylabel("Weight")
            ax_rl.set_title(title_rl); ax_rl.legend(fontsize=7); ax_rl.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_reg


    _run()
    return

@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 5. Bayesian Inference — The Full Picture

    MAP gives you a single point estimate (the posterior mode). Full Bayesian inference uses the **entire posterior distribution** $p(\theta \mid \mathbf{x})$.

    ### Why the Full Posterior Matters

    A point estimate tells you "my best guess for $\theta$ is 0.7." The full posterior tells you "my best guess is 0.7, but there's a 95% chance $\theta$ is between 0.5 and 0.9, and the distribution is slightly skewed right." That uncertainty information is extremely valuable — it tells you how confident you should be in your model's predictions.

    ### Posterior Predictive Distribution

    To predict a new data point $x_{\text{new}}$, the Bayesian approach integrates over all possible parameter values:

    $$p(x_{\text{new}} \mid \mathbf{x}) = \int p(x_{\text{new}} \mid \theta) \, p(\theta \mid \mathbf{x}) \, d\theta$$

    Instead of committing to a single $\theta$, you average predictions across all $\theta$ values, weighted by how plausible they are given the data. This naturally accounts for parameter uncertainty and tends to give better-calibrated predictions.

    ### Conjugate Priors

    A prior $p(\theta)$ is **conjugate** to a likelihood $p(\mathbf{x} \mid \theta)$ if the posterior $p(\theta \mid \mathbf{x})$ is in the same distribution family as the prior. Examples:

    | Likelihood | Conjugate Prior | Posterior |
    |---|---|---|
    | Bernoulli / Binomial | Beta | Beta |
    | Gaussian (known variance) | Gaussian | Gaussian |
    | Poisson | Gamma | Gamma |
    | Multinomial | Dirichlet | Dirichlet |

    Conjugacy gives you closed-form posteriors — no numerical integration needed. Outside of these nice cases, you need approximation methods like Markov Chain Monte Carlo (MCMC) or variational inference, which are powerful but computationally expensive. This computational cost is the main practical limitation of full Bayesian methods.

    > [Bishop 2.1 — Binary Variables](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) walks through the Beta-Bernoulli conjugate pair in beautiful detail. [Chan 8.5 — Bayesian Estimation](file:///C:/Users/landa/ml-course/textbooks/Chan-Probability.pdf) covers the conceptual foundations.
    """)
    return


@app.cell
def _(np, plt):
    def _run():
        # Bayesian updating with conjugate Beta-Bernoulli model
        # Watch the posterior sharpen as we see more data
        from scipy.stats import beta as beta_conj
        rng_bayes = np.random.default_rng(3)
        true_p_bayes = 0.65
        all_flips = rng_bayes.binomial(1, true_p_bayes, size=100)

        prior_a, prior_b = 2, 2  # mild prior centered at 0.5
        checkpoints = [0, 1, 5, 20, 100]  # after this many observations
        p_vals = np.linspace(0, 1, 300)

        fig_bu, ax_bu = plt.subplots(figsize=(7, 4))
        for cp in checkpoints:
            k_cp = all_flips[:cp].sum() if cp > 0 else 0
            a_post = prior_a + k_cp
            b_post = prior_b + (cp - k_cp)
            ax_bu.plot(p_vals, beta_conj.pdf(p_vals, a_post, b_post),
                       label=f"n={cp} (k={k_cp})")
        ax_bu.axvline(true_p_bayes, color="k", ls=":", label=f"true p={true_p_bayes}")
        ax_bu.set_xlabel("p"); ax_bu.set_ylabel("Density")
        ax_bu.set_title("Bayesian Updating: Posterior Sharpens with More Data")
        ax_bu.legend(); ax_bu.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_bu


    _run()
    return

@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 6. Confidence Intervals and Uncertainty

    ### What a Confidence Interval Actually Means

    This is one of the most misunderstood concepts in all of statistics, so pay attention.

    A 95% confidence interval does **not** mean "there is a 95% probability that $\theta$ is in this interval." That's a common and completely wrong interpretation.

    What it actually means: if you repeated the experiment many times and computed a 95% CI each time, about 95% of those intervals would contain the true $\theta$. The probability is about the **procedure**, not about any particular interval.

    Once you've computed a specific CI — say, $[0.4, 0.8]$ — the true $\theta$ is either in it or it's not. There's no probability statement to be made. This is unsatisfying, I know.

    ### Credible Intervals — What People Think CIs Are

    The Bayesian **credible interval** actually is what people intuitively want. A 95% credible interval is a region $[a, b]$ such that:

    $$P(\theta \in [a, b] \mid \mathbf{x}) = 0.95$$

    This says: given the data, there's a 95% probability that $\theta$ is in this interval. It's a direct probability statement about $\theta$, made possible by the Bayesian framework where $\theta$ is a random variable.

    ### Bootstrap

    The **bootstrap** is an elegant trick. You can't repeatedly sample from the true population, but you can repeatedly resample from your data. The idea:

    1. From your $n$ data points, draw $n$ samples **with replacement** — this is a bootstrap sample.
    2. Compute your statistic $\hat{\theta}^*$ on the bootstrap sample.
    3. Repeat steps 1-2 many times (e.g., 10,000).
    4. The distribution of $\hat{\theta}^*$ approximates the sampling distribution of $\hat{\theta}$.

    You can then construct confidence intervals from the bootstrap distribution (e.g., take the 2.5th and 97.5th percentiles for a 95% CI). The beauty is that this works for almost any estimator — you don't need to derive its sampling distribution analytically.

    ### Why This Matters for ML

    Uncertainty quantification is becoming increasingly important. If your model predicts "this patient has an 80% chance of the disease," you want to know whether the model is confident in that 80% or whether it might easily be 50% or 95%. Point predictions without uncertainty can be dangerously misleading.

    > [Chan 8.4 — Confidence Intervals](file:///C:/Users/landa/ml-course/textbooks/Chan-Probability.pdf) gives a clear treatment with the correct interpretation.
    """)
    return


@app.cell
def _(np, plt):
    def _run():
        # Simulate the frequentist interpretation of confidence intervals
        # Generate many 95% CIs and count how many contain the true mean
        from scipy import stats as stats_ci
        rng_ci = np.random.default_rng(42)
        true_mu_ci = 5.0
        n_ci = 25
        n_experiments = 100

        fig_ci, ax_ci = plt.subplots(figsize=(8, 6))
        contains_count = 0
        for i in range(n_experiments):
            sample = rng_ci.normal(true_mu_ci, 2.0, size=n_ci)
            xbar = sample.mean()
            se = sample.std(ddof=1) / np.sqrt(n_ci)
            t_crit = stats_ci.t.ppf(0.975, df=n_ci - 1)
            lo, hi = xbar - t_crit * se, xbar + t_crit * se
            contains = lo <= true_mu_ci <= hi
            contains_count += contains
            color = "steelblue" if contains else "red"
            ax_ci.plot([lo, hi], [i, i], color=color, lw=1.2)
        ax_ci.axvline(true_mu_ci, color="k", ls="--", lw=1.5, label=f"true mu={true_mu_ci}")
        ax_ci.set_xlabel("Value"); ax_ci.set_ylabel("Experiment")
        ax_ci.set_title(f"100 Confidence Intervals (95%): {contains_count} contain true mean")
        ax_ci.legend()
        plt.tight_layout()
        fig_ci


    _run()
    return

@app.cell
def _(np, plt):
    def _run():
        # Bootstrap confidence intervals for the mean and median
        rng_boot = np.random.default_rng(7)
        data_boot = rng_boot.normal(4.0, 1.5, size=40)
        n_boot_samples = 10_000

        # Bootstrap: resample with replacement, compute statistic each time
        boot_means = np.array([
            rng_boot.choice(data_boot, size=len(data_boot), replace=True).mean()
            for _ in range(n_boot_samples)
        ])
        boot_medians = np.array([
            np.median(rng_boot.choice(data_boot, size=len(data_boot), replace=True))
            for _ in range(n_boot_samples)
        ])

        ci_mean = np.percentile(boot_means, [2.5, 97.5])
        ci_median = np.percentile(boot_medians, [2.5, 97.5])
        print(f"Bootstrap 95% CI for mean:   [{ci_mean[0]:.3f}, {ci_mean[1]:.3f}]")
        print(f"Bootstrap 95% CI for median: [{ci_median[0]:.3f}, {ci_median[1]:.3f}]")

        fig_boot, (ax_b1, ax_b2) = plt.subplots(1, 2, figsize=(11, 3.5))
        ax_b1.hist(boot_means, bins=50, edgecolor="k", alpha=0.7)
        ax_b1.axvline(ci_mean[0], color="r", ls="--"); ax_b1.axvline(ci_mean[1], color="r", ls="--")
        ax_b1.set_title("Bootstrap Distribution of Mean")
        ax_b2.hist(boot_medians, bins=50, edgecolor="k", alpha=0.7)
        ax_b2.axvline(ci_median[0], color="r", ls="--"); ax_b2.axvline(ci_median[1], color="r", ls="--")
        ax_b2.set_title("Bootstrap Distribution of Median")
        plt.tight_layout()
        fig_boot


    _run()
    return

@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 7. Hypothesis Testing

    I'll keep this brief because hypothesis testing is less central to ML than estimation. But you should know the framework.

    ### The Setup

    - **Null hypothesis** $H_0$: some default claim (e.g., "the coin is fair," "this feature has no effect")
    - **Alternative hypothesis** $H_1$: what you're trying to show (e.g., "the coin is biased," "this feature matters")

    ### p-Values

    The **p-value** is the probability of observing data as extreme as (or more extreme than) what you got, assuming $H_0$ is true.

    $$p\text{-value} = P(\text{data this extreme} \mid H_0 \text{ is true})$$

    What p-values are **NOT**:
    - Not the probability that $H_0$ is true
    - Not the probability that you made an error
    - Not a measure of effect size

    A small p-value (conventionally < 0.05) means the data would be surprising if $H_0$ were true, so you reject $H_0$. But "statistically significant" doesn't mean "practically important."

    ### Type I and Type II Errors

    |  | $H_0$ true | $H_0$ false |
    |---|---|---|
    | Reject $H_0$ | **Type I error** (false positive) | Correct |
    | Fail to reject $H_0$ | Correct | **Type II error** (false negative) |

    - $\alpha$ = P(Type I error) — the significance level (usually 0.05)
    - $\beta$ = P(Type II error)
    - $1 - \beta$ = **power** — the probability of detecting a real effect

    ### Connection to ML

    In ML, you'll encounter hypothesis testing mostly in:
    - **Feature selection:** is this feature significantly associated with the target?
    - **Model comparison:** is model A significantly better than model B?
    - **A/B testing:** does the new recommendation algorithm perform better?

    But in practice, cross-validation and information criteria (next section) are more commonly used for model selection than classical hypothesis tests.
    """)
    return


@app.cell
def _(np):
    def _run():
        # Simulate p-values and Type I / Type II error rates
        from scipy import stats as stats_hyp
        rng_hyp = np.random.default_rng(5)
        n_hyp = 30
        n_tests = 10_000
        alpha = 0.05

        # Type I error: H0 is true (mu=0), test if mu != 0
        type1_rejections = 0
        for _ in range(n_tests):
            sample_null = rng_hyp.normal(0, 1, n_hyp)  # H0 true
            _, p_val = stats_hyp.ttest_1samp(sample_null, 0)
            if p_val < alpha:
                type1_rejections += 1

        # Type II error: H0 is false (mu=0.5), test if mu != 0
        type2_failures = 0
        for _ in range(n_tests):
            sample_alt = rng_hyp.normal(0.5, 1, n_hyp)  # H0 false, true mu=0.5
            _, p_val = stats_hyp.ttest_1samp(sample_alt, 0)
            if p_val >= alpha:
                type2_failures += 1

        print(f"Type I error rate:  {type1_rejections / n_tests:.3f}  (expected ~ {alpha})")
        print(f"Type II error rate: {type2_failures / n_tests:.3f}")
        print(f"Power:              {1 - type2_failures / n_tests:.3f}")


    _run()
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 8. Information Criteria

    When comparing models, you face a tension: more complex models fit the training data better, but they might overfit. Information criteria balance fit against complexity.

    ### AIC — Akaike Information Criterion

    $$\text{AIC} = -2\ell(\hat{\theta}) + 2k$$

    where $\ell(\hat{\theta})$ is the maximized log-likelihood and $k$ is the number of parameters. The first term rewards fit; the second penalizes complexity. **Lower AIC is better.**

    AIC is derived from an approximation to the expected out-of-sample prediction error — it's trying to estimate how well the model will perform on new data, without actually needing a test set.

    ### BIC — Bayesian Information Criterion

    $$\text{BIC} = -2\ell(\hat{\theta}) + k \log n$$

    Similar to AIC, but the penalty grows with $\log n$. For large datasets, BIC penalizes complexity more heavily than AIC, favoring simpler models. BIC is derived from an approximation to the Bayesian model evidence $p(\mathbf{x} \mid \text{model})$.

    ### AIC vs BIC

    - **AIC** is better when your goal is prediction — it tends to select models that predict well.
    - **BIC** is better when your goal is identifying the true model — it's consistent (selects the true model as $n \to \infty$, if it's in your candidate set).
    - Both are asymptotic approximations to cross-validation, with slightly different targets.

    > [Bishop 1.3 — Model Selection](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) introduces BIC in the context of polynomial curve fitting. [Murphy 5.2 — Bayesian Model Comparison](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf) places both criteria in the Bayesian model selection framework.
    """)
    return


@app.cell
def _(np, plt):
    def _run():
        # AIC and BIC for polynomial model selection
        rng_ic = np.random.default_rng(12)
        n_ic = 80
        x_ic = np.sort(rng_ic.uniform(-3, 3, n_ic))
        y_ic_true = 0.5 * x_ic ** 3 - x_ic + 2  # true model is degree 3
        y_ic = y_ic_true + rng_ic.normal(0, 3, n_ic)

        degrees_ic = range(1, 10)
        aic_vals, bic_vals = [], []
        for d in degrees_ic:
            # Fit polynomial of degree d
            coeffs = np.polyfit(x_ic, y_ic, d)
            y_pred_ic = np.polyval(coeffs, x_ic)
            resid_ic = y_ic - y_pred_ic
            sigma2_ic = np.mean(resid_ic ** 2)
            k_ic = d + 2  # coefficients + sigma^2
            # Log-likelihood for Gaussian model
            ll = -0.5 * n_ic * np.log(2 * np.pi * sigma2_ic) - 0.5 * n_ic
            aic_vals.append(-2 * ll + 2 * k_ic)
            bic_vals.append(-2 * ll + k_ic * np.log(n_ic))

        fig_ic, ax_ic = plt.subplots(figsize=(7, 4))
        ax_ic.plot(list(degrees_ic), aic_vals, "bo-", label=f"AIC (best: deg {list(degrees_ic)[np.argmin(aic_vals)]})")
        ax_ic.plot(list(degrees_ic), bic_vals, "rs-", label=f"BIC (best: deg {list(degrees_ic)[np.argmin(bic_vals)]})")
        ax_ic.set_xlabel("Polynomial Degree"); ax_ic.set_ylabel("Criterion Value")
        ax_ic.set_title("AIC vs BIC for Model Selection")
        ax_ic.legend(); ax_ic.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_ic


    _run()
    return

@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 9. Connecting Estimation to ML

    Everything in this module maps directly to machine learning. Let me make the connections explicit:

    **Training a model = finding parameter estimates.** When you call `model.fit(X, y)` in scikit-learn, you are computing $\hat{\theta}$ from data. That's estimation.

    **Loss function = negative log-likelihood (often).** Cross-entropy loss for classification? That's the negative log-likelihood of a Bernoulli/Categorical model. Mean squared error for regression? Negative log-likelihood of a Gaussian model (up to constants). When you minimize a loss function, you're usually doing MLE without knowing it.

    | ML Loss Function | Equivalent Probabilistic Model |
    |---|---|
    | Mean Squared Error | Gaussian likelihood (MLE) |
    | Cross-Entropy | Bernoulli / Categorical likelihood (MLE) |
    | MSE + L2 penalty | Gaussian likelihood + Gaussian prior (MAP) |
    | MSE + L1 penalty | Gaussian likelihood + Laplace prior (MAP) |

    **Regularization = prior beliefs about parameters.** L2 regularization says "I believe the weights are probably small" (Gaussian prior). L1 says the same but expects many weights to be exactly zero (Laplace prior). Dropout can be interpreted as approximate Bayesian inference. The regularization strength $\lambda$ controls how much you trust your prior vs. your data.

    **The bias-variance tradeoff in estimation is the bias-variance tradeoff in ML.** More complex models have low bias but high variance. Regularization introduces bias but reduces variance. Finding the sweet spot is the central challenge.

    This framework — likelihood + prior + inference method — unifies nearly everything in ML. Once you internalize it, new models stop looking like arbitrary collections of tricks and start looking like different instantiations of the same principle.

    > [Bishop 1.1 — Polynomial Curve Fitting](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) beautifully illustrates overfitting, regularization, and the Bayesian approach all on one simple example. Read this section — it's one of the best 15 pages in any ML textbook.
    """)
    return


@app.cell
def _(np):
    def _run():
        # Cross-entropy loss IS negative log-likelihood of Bernoulli
        rng_ce = np.random.default_rng(0)
        y_true_ce = rng_ce.integers(0, 2, size=50).astype(float)
        p_pred = np.clip(rng_ce.uniform(0.2, 0.8, 50), 1e-8, 1 - 1e-8)

        # Cross-entropy loss (as used in ML)
        ce_loss = -np.mean(y_true_ce * np.log(p_pred) + (1 - y_true_ce) * np.log(1 - p_pred))

        # Negative log-likelihood of Bernoulli (same formula!)
        nll = -np.sum(y_true_ce * np.log(p_pred) + (1 - y_true_ce) * np.log(1 - p_pred)) / len(y_true_ce)

        print(f"Cross-entropy loss:            {ce_loss:.6f}")
        print(f"Bernoulli negative log-lik:    {nll:.6f}")
        print(f"They are identical:            {np.isclose(ce_loss, nll)}")


    _run()
    return

@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Textbook Reading Assignments

    **Primary reading:**
    - [Chan Chapter 8: Estimation](file:///C:/Users/landa/ml-course/textbooks/Chan-Probability.pdf) — complete chapter. This is your main text.
    - [Chan Chapter 9: Hypothesis Testing](file:///C:/Users/landa/ml-course/textbooks/Chan-Probability.pdf) — read for conceptual understanding.
    - [Bishop 1.1–1.3 (pp. 21–33)](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) — polynomial curve fitting example that ties everything together.
    - [Bishop 2.1 (pp. 67–76)](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) — Beta-Bernoulli conjugate example.

    **Supplementary reading:**
    - [Bishop 3.1–3.3 (pp. 137–162)](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) — Bayesian linear regression (full treatment).
    - [Murphy 4.2 — Maximum Likelihood Estimation](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf) — concise treatment with ML focus.
    - [Murphy 5.2 — Bayesian Model Comparison](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf) — AIC, BIC, and model evidence.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Practice Exercises

    ### Derivations (Pen and Paper)

    1. **MLE for Exponential distribution.** Given $x_1, \ldots, x_n \sim \text{Exp}(\lambda)$ with density $f(x) = \lambda e^{-\lambda x}$, derive the MLE for $\lambda$. Verify that it's unbiased.

    2. **MLE for Uniform distribution.** Given $x_1, \ldots, x_n \sim \text{Uniform}(0, \theta)$, derive the MLE for $\theta$. This one is tricky — you can't just take derivatives. Is the MLE biased?

    3. **Bias of MLE variance estimator.** Show that $\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum(x_i - \bar{x})^2$ has bias $-\sigma^2/n$, confirming the need for Bessel's correction.

    4. **MAP for Bernoulli with Beta prior.** Given $x_1, \ldots, x_n \sim \text{Bernoulli}(p)$ and prior $p \sim \text{Beta}(\alpha, \beta)$, derive the MAP estimate for $p$. Show that it reduces to MLE when $\alpha = \beta = 1$ (uniform prior).

    5. **Ridge regression as MAP.** Starting from $y = \mathbf{X}\mathbf{w} + \epsilon$ with $\epsilon \sim \mathcal{N}(0, \sigma^2\mathbf{I})$ and $\mathbf{w} \sim \mathcal{N}(0, \tau^2\mathbf{I})$, derive the closed-form MAP estimate $\hat{\mathbf{w}}_{\text{MAP}} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$ and identify $\lambda$ in terms of $\sigma^2$ and $\tau^2$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Python Implementation

    **Exercise 6: MLE by hand and by optimization.** Generate 100 samples from $\mathcal{N}(\mu=3, \sigma^2=4)$. (a) Compute the MLE analytically using your derived formulas. (b) Compute it numerically using `scipy.optimize.minimize` on the negative log-likelihood. Verify they match.
    """)
    return


@app.cell
def _():
    def _run():
        from scipy.optimize import minimize

        rng = np.random.default_rng(42)
        data = rng.normal(loc=3, scale=2, size=100)

        # (a) Analytical MLE
        mu_mle = np.mean(data)
        sigma2_mle = np.var(data)  # Note: np.var uses 1/n by default

        # (b) Numerical MLE
        def neg_log_lik(params):
            mu, log_sigma2 = params  # optimize log(sigma^2) to ensure positivity
            sigma2 = np.exp(log_sigma2)
            return 0.5 * len(data) * np.log(2 * np.pi * sigma2) + \
                   0.5 * np.sum((data - mu)**2) / sigma2

        result = minimize(neg_log_lik, x0=[0, 0])
        mu_num, sigma2_num = result.x[0], np.exp(result.x[1])
        print(f"Analytical: mu={mu_mle:.4f}, sigma2={sigma2_mle:.4f}")
        print(f"Numerical:  mu={mu_num:.4f}, sigma2={sigma2_num:.4f}")


    _run()
    return

@app.cell
def _(mo):
    mo.md(r"""
    **Exercise 7: MAP vs MLE with small samples.** Generate 5 coin flips from a coin with $p = 0.7$. Suppose all 5 come up heads. (a) Compute the MLE. (b) Compute the MAP with a $\text{Beta}(2, 2)$ prior. (c) Repeat with a $\text{Beta}(10, 10)$ prior. (d) Plot the prior, likelihood, and posterior for each case.
    """)
    return


@app.cell
def _():
    def _run():
        from scipy.stats import beta as beta_dist

        # All 5 flips are heads
        n_flips = 5
        k_heads = 5

        # MLE
        p_mle = k_heads / n_flips
        print(f"MLE: p = {p_mle:.2f}")

        # MAP with Beta(2,2) prior
        alpha1, beta1 = 2, 2
        p_map1 = (k_heads + alpha1 - 1) / (n_flips + alpha1 + beta1 - 2)
        print(f"MAP with Beta(2,2): p = {p_map1:.4f}")

        # MAP with Beta(10,10) prior
        alpha2, beta2 = 10, 10
        p_map2 = (k_heads + alpha2 - 1) / (n_flips + alpha2 + beta2 - 2)
        print(f"MAP with Beta(10,10): p = {p_map2:.4f}")

        # Plot
        x = np.linspace(0.001, 0.999, 200)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax, (a, b, title) in zip(axes, [
            (alpha1, beta1, f"Beta({alpha1},{beta1}) prior"),
            (alpha2, beta2, f"Beta({alpha2},{beta2}) prior")
        ]):
            prior = beta_dist.pdf(x, a, b)
            # Likelihood (unnormalized)
            likelihood = x**k_heads * (1-x)**(n_flips - k_heads)
            likelihood = likelihood / likelihood.max() * prior.max()
            # Posterior
            posterior = beta_dist.pdf(x, a + k_heads, b + n_flips - k_heads)

            ax.plot(x, prior, 'g--', lw=2, label='Prior')
            ax.plot(x, likelihood, 'r:', lw=2, label='Likelihood (scaled)')
            ax.plot(x, posterior, 'b-', lw=2, label='Posterior')
            ax.axvline(p_mle, color='r', alpha=0.3, label=f'MLE = {p_mle:.2f}')
            ax.set_title(title)
            ax.legend()
            ax.set_xlabel('p')

        plt.tight_layout()
        fig


    _run()
    return

@app.cell
def _(mo):
    mo.md(r"""
    **Exercise 8: Bootstrap confidence intervals.** Using the Gaussian data from exercise 6: (a) Compute the bootstrap 95% CI for the mean using 10,000 bootstrap samples. (b) Compare with the analytical CI using the t-distribution. (c) Now compute a bootstrap CI for the median.
    """)
    return


@app.cell
def _():
    def _run():
        from scipy import stats

        rng = np.random.default_rng(42)
        data = rng.normal(loc=3, scale=2, size=100)

        # (a) Bootstrap CI for the mean
        n_bootstrap = 10000
        boot_means = np.array([
            rng.choice(data, size=len(data), replace=True).mean()
            for _ in range(n_bootstrap)
        ])
        ci_boot_mean = np.percentile(boot_means, [2.5, 97.5])
        print(f"Bootstrap 95% CI for mean: [{ci_boot_mean[0]:.4f}, {ci_boot_mean[1]:.4f}]")

        # (b) Analytical CI using t-distribution
        n = len(data)
        se = data.std(ddof=1) / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n-1)
        ci_analytical = [data.mean() - t_crit * se, data.mean() + t_crit * se]
        print(f"Analytical 95% CI for mean: [{ci_analytical[0]:.4f}, {ci_analytical[1]:.4f}]")

        # (c) Bootstrap CI for the median
        boot_medians = np.array([
            np.median(rng.choice(data, size=len(data), replace=True))
            for _ in range(n_bootstrap)
        ])
        ci_boot_median = np.percentile(boot_medians, [2.5, 97.5])
        print(f"Bootstrap 95% CI for median: [{ci_boot_median[0]:.4f}, {ci_boot_median[1]:.4f}]")


    _run()
    return

@app.cell
def _(mo):
    mo.md(r"""
    **Exercise 9: MLE for linear regression.** Generate data from $y = 2x_1 + 3x_2 + 1 + \epsilon$ where $\epsilon \sim \mathcal{N}(0, 1)$. (a) Compute OLS estimates using the normal equation. (b) Compute MLE by numerically maximizing the Gaussian log-likelihood. (c) Verify they give the same weights. (d) Add an L2 penalty and compute the MAP (Ridge) solution. Plot how the weights change as $\lambda$ varies from 0 to 100.
    """)
    return


@app.cell
def _():
    def _run():
        from scipy.optimize import minimize

        rng = np.random.default_rng(42)
        n = 200
        X = rng.standard_normal((n, 2))
        X_bias = np.column_stack([np.ones(n), X])  # add intercept
        true_w = np.array([1.0, 2.0, 3.0])
        y = X_bias @ true_w + rng.standard_normal(n)

        # (a) OLS via normal equation
        w_ols = np.linalg.solve(X_bias.T @ X_bias, X_bias.T @ y)
        print(f"OLS weights: {w_ols}")

        # (b) Numerical MLE
        def neg_log_lik_lr(params):
            w = params[:3]
            log_sigma2 = params[3]
            sigma2 = np.exp(log_sigma2)
            residuals = y - X_bias @ w
            return 0.5 * n * np.log(2 * np.pi * sigma2) + 0.5 * np.sum(residuals**2) / sigma2

        result = minimize(neg_log_lik_lr, x0=[0, 0, 0, 0])
        w_mle = result.x[:3]
        print(f"MLE weights: {w_mle}")

        # (c) Verify they match
        print(f"Max absolute difference: {np.max(np.abs(w_ols - w_mle)):.6f}")

        # (d) Ridge regression for varying lambda
        lambdas = np.logspace(-2, 2, 50)
        ridge_weights = []
        for lam in lambdas:
            penalty = lam * np.eye(3)
            penalty[0, 0] = 0  # don't regularize intercept
            w_ridge = np.linalg.solve(X_bias.T @ X_bias + penalty, X_bias.T @ y)
            ridge_weights.append(w_ridge)

        ridge_weights = np.array(ridge_weights)

        fig, ax = plt.subplots(figsize=(8, 5))
        for i, label in enumerate(['Intercept', 'w1', 'w2']):
            ax.plot(lambdas, ridge_weights[:, i], label=f'{label} (true={true_w[i]:.0f})')
        ax.set_xscale('log')
        ax.set_xlabel('Lambda (regularization strength)')
        ax.set_ylabel('Weight value')
        ax.set_title('Ridge Regression: Weight Shrinkage with Regularization')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig


    _run()
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 10: AIC/BIC model selection.** Generate data from a degree-3 polynomial. Fit polynomials of degrees 1 through 10. For each, compute AIC and BIC. Plot both criteria against polynomial degree. Which degree does each criterion select? Compare with the result from 5-fold cross-validation.
    """)
    return


@app.cell
def _():
    def _run():
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import cross_val_score

        rng = np.random.default_rng(42)
        n = 100
        x = np.sort(rng.uniform(-3, 3, n))
        # True model: degree 3
        y_true = 0.5 * x**3 - 2 * x**2 + x + 3
        y = y_true + rng.standard_normal(n) * 3

        degrees = range(1, 11)
        aics = []
        bics = []
        cv_scores = []

        for d in degrees:
            poly = PolynomialFeatures(d, include_bias=False)
            X_poly = poly.fit_transform(x.reshape(-1, 1))

            model = LinearRegression()
            model.fit(X_poly, y)
            y_pred = model.predict(X_poly)

            # Compute log-likelihood
            residuals = y - y_pred
            sigma2 = np.mean(residuals**2)
            k = d + 2  # polynomial coefficients + intercept + sigma^2
            log_lik = -0.5 * n * np.log(2 * np.pi * sigma2) - 0.5 * n

            aic = -2 * log_lik + 2 * k
            bic = -2 * log_lik + k * np.log(n)
            aics.append(aic)
            bics.append(bic)

            # 5-fold CV (negative MSE)
            cv_mse = -cross_val_score(LinearRegression(), X_poly, y,
                                       cv=5, scoring='neg_mean_squared_error').mean()
            cv_scores.append(cv_mse)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(list(degrees), aics, 'bo-')
        axes[0].axvline(list(degrees)[np.argmin(aics)], color='r', linestyle='--',
                        label=f'Best: degree {list(degrees)[np.argmin(aics)]}')
        axes[0].set_xlabel('Polynomial Degree')
        axes[0].set_ylabel('AIC')
        axes[0].set_title('AIC Model Selection')
        axes[0].legend()

        axes[1].plot(list(degrees), bics, 'go-')
        axes[1].axvline(list(degrees)[np.argmin(bics)], color='r', linestyle='--',
                        label=f'Best: degree {list(degrees)[np.argmin(bics)]}')
        axes[1].set_xlabel('Polynomial Degree')
        axes[1].set_ylabel('BIC')
        axes[1].set_title('BIC Model Selection')
        axes[1].legend()

        axes[2].plot(list(degrees), cv_scores, 'ro-')
        axes[2].axvline(list(degrees)[np.argmin(cv_scores)], color='b', linestyle='--',
                        label=f'Best: degree {list(degrees)[np.argmin(cv_scores)]}')
        axes[2].set_xlabel('Polynomial Degree')
        axes[2].set_ylabel('CV MSE')
        axes[2].set_title('5-Fold CV Model Selection')
        axes[2].legend()

        plt.tight_layout()
        fig


    _run()
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Summary

    | Method | What it does | Key formula | ML connection |
    |---|---|---|---|
    | MLE | Maximizes likelihood | $\hat{\theta} = \arg\max \prod p(x_i \mid \theta)$ | Training with a loss function |
    | MAP | Maximizes posterior | $\hat{\theta} = \arg\max \; p(\theta \mid \mathbf{x})$ | Training with regularization |
    | Full Bayes | Computes entire posterior | $p(\theta \mid \mathbf{x}) \propto p(\mathbf{x} \mid \theta)p(\theta)$ | Uncertainty quantification |
    | Bootstrap | Resamples data | Empirical sampling distribution | Bagging (Random Forests!) |
    | AIC/BIC | Balances fit + complexity | $-2\ell + \text{penalty}$ | Model/hyperparameter selection |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Code It: Implementation Exercises

    Work through these exercises to build estimation algorithms from scratch. Each one gives you a problem statement and skeleton code -- fill in the blanks.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Code It 1: MLE for the Exponential Distribution

    Given samples from $\text{Exp}(\lambda)$ with density $f(x) = \lambda e^{-\lambda x}$, the MLE is $\hat{\lambda} = 1/\bar{x}$. Implement this analytically, then verify by numerically minimizing the negative log-likelihood.
    """)
    return


@app.cell
def _(np):
    def _run():
        # Code It 1: MLE for Exponential distribution
        rng_exp = np.random.default_rng(42)
        true_lam = 2.5
        samples_exp = rng_exp.exponential(1 / true_lam, size=200)

        # TODO: Analytical MLE (hint: lambda_hat = 1 / x_bar)
        lam_analytical = ...  # fill in

        # TODO: Numerical MLE — minimize negative log-likelihood
        # NLL = -sum(log(lambda) - lambda * x_i) = -n*log(lambda) + lambda*sum(x_i)
        from scipy.optimize import minimize_scalar
        nll_exp = lambda lam: ...  # fill in
        # result_exp = minimize_scalar(nll_exp, bounds=(0.01, 20), method="bounded")
        # lam_numerical = result_exp.x

        # print(f"True lambda:      {true_lam}")
        # print(f"Analytical MLE:   {lam_analytical:.4f}")
        # print(f"Numerical MLE:    {lam_numerical:.4f}")


    _run()
    return

@app.cell
def _(mo):
    mo.md(r"""
    ### Code It 2: Simulate Bias-Variance Tradeoff

    Compare two estimators of the population mean from $\mathcal{N}(\mu, \sigma^2)$:
    - **Estimator A**: sample mean $\bar{x}$ (unbiased)
    - **Estimator B**: shrinkage estimator $0.8 \bar{x}$ (biased toward zero)

    Run 10,000 simulations and compute bias, variance, and MSE for each. Which one has lower MSE, and when?
    """)
    return


@app.cell
def _(np):
    def _run():
        # Code It 2: Bias-variance simulation
        rng_bvex = np.random.default_rng(0)
        true_mu_bv = 3.0
        sigma_bv = 5.0
        n_obs_bv = 10
        n_sims = 10_000

        estimates_A = np.zeros(n_sims)
        estimates_B = np.zeros(n_sims)

        for i in range(n_sims):
            sample = rng_bvex.normal(true_mu_bv, sigma_bv, size=n_obs_bv)
            # TODO: Estimator A — sample mean
            estimates_A[i] = ...  # fill in
            # TODO: Estimator B — shrinkage: 0.8 * sample mean
            estimates_B[i] = ...  # fill in

        # TODO: Compute bias, variance, MSE for each estimator
        # bias_A = estimates_A.mean() - true_mu_bv
        # var_A = estimates_A.var()
        # mse_A = bias_A**2 + var_A
        # (same for B)

        # print(f"Estimator A (x_bar):     Bias={bias_A:.4f}, Var={var_A:.4f}, MSE={mse_A:.4f}")
        # print(f"Estimator B (0.8*x_bar): Bias={bias_B:.4f}, Var={var_B:.4f}, MSE={mse_B:.4f}")


    _run()
    return

@app.cell
def _(mo):
    mo.md(r"""
    ### Code It 3: Bootstrap Confidence Interval for Correlation

    Given bivariate data $(x_i, y_i)$, construct a bootstrap 95% CI for the Pearson correlation coefficient. There is no simple analytical formula for the CI of a correlation -- the bootstrap shines here.
    """)
    return


@app.cell
def _(np):
    def _run():
        # Code It 3: Bootstrap CI for correlation
        rng_corr = np.random.default_rng(55)
        n_corr = 50
        # Generate correlated data
        mean_corr = [0, 0]
        cov_corr = [[1, 0.6], [0.6, 1]]
        xy = rng_corr.multivariate_normal(mean_corr, cov_corr, size=n_corr)
        x_corr, y_corr = xy[:, 0], xy[:, 1]

        sample_corr = np.corrcoef(x_corr, y_corr)[0, 1]
        print(f"Sample correlation: {sample_corr:.4f}")

        n_boot_corr = 10_000
        boot_corrs = np.zeros(n_boot_corr)
        for i in range(n_boot_corr):
            # TODO: resample indices with replacement
            # idx = rng_corr.choice(n_corr, size=n_corr, replace=True)
            # boot_corrs[i] = np.corrcoef(x_corr[idx], y_corr[idx])[0, 1]
            pass

        # TODO: Compute 95% CI from percentiles
        # ci_lo, ci_hi = np.percentile(boot_corrs, [2.5, 97.5])
        # print(f"Bootstrap 95% CI for correlation: [{ci_lo:.4f}, {ci_hi:.4f}]")


    _run()
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Code It 4: Bayesian Posterior Predictive

    Using a Beta-Bernoulli model: after observing coin flips, compute the posterior predictive probability of the next flip being heads. Compare the Bayesian prediction (which integrates over parameter uncertainty) with the MLE plug-in prediction.
    """)
    return


@app.cell
def _(np):
    def _run():
        # Code It 4: Posterior predictive for Beta-Bernoulli
        rng_pp = np.random.default_rng(8)
        true_p_pp = 0.7
        flips_pp = rng_pp.binomial(1, true_p_pp, size=15)
        k_pp = flips_pp.sum()
        n_pp = len(flips_pp)
        print(f"Observed: {k_pp} heads in {n_pp} flips")

        # Prior: Beta(a, b)
        a_prior, b_prior = 2, 2

        # TODO: MLE plug-in prediction for P(next flip = heads)
        # p_mle_pred = k_pp / n_pp

        # TODO: Bayesian posterior predictive
        # For Beta-Bernoulli, P(next=1 | data) = (a + k) / (a + b + n)
        # This integrates over the full posterior, not just the point estimate
        # p_bayes_pred = (a_prior + k_pp) / (a_prior + b_prior + n_pp)

        # print(f"MLE plug-in prediction:        {p_mle_pred:.4f}")
        # print(f"Bayesian posterior predictive:  {p_bayes_pred:.4f}")
        # print(f"True p:                         {true_p_pp}")


    _run()
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Code It 5: Compare AIC, BIC, and Cross-Validation

    Fit polynomial models of degrees 1-8 to noisy cubic data. Compute AIC, BIC, and 10-fold CV error. Print a table showing all three criteria and which degree each selects.
    """)
    return


@app.cell
def _(np):
    def _run():
        # Code It 5: Model selection comparison
        rng_ms = np.random.default_rng(17)
        n_ms = 120
        x_ms = np.sort(rng_ms.uniform(-2, 2, n_ms))
        y_ms = 0.8 * x_ms ** 3 - 1.5 * x_ms + 1 + rng_ms.normal(0, 2, n_ms)

        # TODO: For each degree d in 1..8:
        #   1. Fit polynomial with np.polyfit
        #   2. Compute residuals and MLE sigma^2
        #   3. Compute log-likelihood, AIC, BIC
        #   4. Compute 10-fold CV MSE (use sklearn cross_val_score)
        # Print a table of results

        # degrees_ms = range(1, 9)
        # print(f"{'Degree':>6} {'AIC':>10} {'BIC':>10} {'CV MSE':>10}")
        # print("-" * 40)
        # for d in degrees_ms:
        #     coeffs_ms = np.polyfit(x_ms, y_ms, d)
        #     y_pred_ms = np.polyval(coeffs_ms, x_ms)
        #     sigma2_ms = np.mean((y_ms - y_pred_ms) ** 2)
        #     k_ms = d + 2
        #     ll_ms = -0.5 * n_ms * np.log(2 * np.pi * sigma2_ms) - 0.5 * n_ms
        #     aic_ms = -2 * ll_ms + 2 * k_ms
        #     bic_ms = -2 * ll_ms + k_ms * np.log(n_ms)
        #     # CV: use sklearn
        #     from sklearn.preprocessing import PolynomialFeatures
        #     from sklearn.linear_model import LinearRegression
        #     from sklearn.model_selection import cross_val_score
        #     X_ms = PolynomialFeatures(d, include_bias=False).fit_transform(x_ms.reshape(-1,1))
        #     cv_ms = -cross_val_score(LinearRegression(), X_ms, y_ms, cv=10,
        #                              scoring='neg_mean_squared_error').mean()
        #     print(f"{d:>6} {aic_ms:>10.2f} {bic_ms:>10.2f} {cv_ms:>10.4f}")


    _run()
    return

@app.cell
def _(mo):
    mo.md(r"""
    ---

    Next up: **0F — Optimization**, where you'll learn the algorithms that actually solve these optimization problems — because in practice, you rarely get a nice closed-form solution like we did for Gaussian MLE. You'll need gradient descent.
    """)
    return


if __name__ == "__main__":
    app.run()
