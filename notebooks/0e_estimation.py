import marimo

app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


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


@app.cell
def _(mo):
    mo.md(r"""
    ### Python Implementation

    **Exercise 6: MLE by hand and by optimization.** Generate 100 samples from $\mathcal{N}(\mu=3, \sigma^2=4)$. (a) Compute the MLE analytically using your derived formulas. (b) Compute it numerically using `scipy.optimize.minimize` on the negative log-likelihood. Verify they match.
    """)
    return


@app.cell
def _():
    import numpy as np
    from scipy.optimize import minimize

    np.random.seed(42)
    data = np.random.normal(loc=3, scale=2, size=100)

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
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Exercise 7: MAP vs MLE with small samples.** Generate 5 coin flips from a coin with $p = 0.7$. Suppose all 5 come up heads. (a) Compute the MLE. (b) Compute the MAP with a $\text{Beta}(2, 2)$ prior. (c) Repeat with a $\text{Beta}(10, 10)$ prior. (d) Plot the prior, likelihood, and posterior for each case.
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
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
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Exercise 8: Bootstrap confidence intervals.** Using the Gaussian data from exercise 6: (a) Compute the bootstrap 95% CI for the mean using 10,000 bootstrap samples. (b) Compare with the analytical CI using the t-distribution. (c) Now compute a bootstrap CI for the median.
    """)
    return


@app.cell
def _():
    import numpy as np
    from scipy import stats

    np.random.seed(42)
    data = np.random.normal(loc=3, scale=2, size=100)

    # (a) Bootstrap CI for the mean
    n_bootstrap = 10000
    boot_means = np.array([
        np.random.choice(data, size=len(data), replace=True).mean()
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
        np.median(np.random.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstrap)
    ])
    ci_boot_median = np.percentile(boot_medians, [2.5, 97.5])
    print(f"Bootstrap 95% CI for median: [{ci_boot_median[0]:.4f}, {ci_boot_median[1]:.4f}]")
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Exercise 9: MLE for linear regression.** Generate data from $y = 2x_1 + 3x_2 + 1 + \epsilon$ where $\epsilon \sim \mathcal{N}(0, 1)$. (a) Compute OLS estimates using the normal equation. (b) Compute MLE by numerically maximizing the Gaussian log-likelihood. (c) Verify they give the same weights. (d) Add an L2 penalty and compute the MAP (Ridge) solution. Plot how the weights change as $\lambda$ varies from 0 to 100.
    """)
    return


@app.cell
def _():
    import numpy as np
    from scipy.optimize import minimize
    import matplotlib.pyplot as plt

    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 2)
    X_bias = np.column_stack([np.ones(n), X])  # add intercept
    true_w = np.array([1.0, 2.0, 3.0])
    y = X_bias @ true_w + np.random.randn(n)

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
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Exercise 10: AIC/BIC model selection.** Generate data from a degree-3 polynomial. Fit polynomials of degrees 1 through 10. For each, compute AIC and BIC. Plot both criteria against polynomial degree. Which degree does each criterion select? Compare with the result from 5-fold cross-validation.
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score

    np.random.seed(42)
    n = 100
    x = np.sort(np.random.uniform(-3, 3, n))
    # True model: degree 3
    y_true = 0.5 * x**3 - 2 * x**2 + x + 3
    y = y_true + np.random.randn(n) * 3

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
    return


@app.cell
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

    Next up: **0F — Optimization**, where you'll learn the algorithms that actually solve these optimization problems — because in practice, you rarely get a nice closed-form solution like we did for Gaussian MLE. You'll need gradient descent.
    """)
    return


if __name__ == "__main__":
    app.run()
