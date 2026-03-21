import marimo

app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Module 1A: The ML Landscape

    Everything you've built so far — the calculus, the linear algebra, the probability and statistics, the optimization — has been scaffolding. Important scaffolding, load-bearing scaffolding, but scaffolding nonetheless. This is the lecture where we start constructing the building.

    Welcome to machine learning.

    ---

    ## 1. What Is Machine Learning, Really?

    Let's start with the definition that has anchored this field for three decades. Tom Mitchell (1997) put it precisely:

    > A computer program is said to **learn** from experience $E$ with respect to some class of tasks $T$ and performance measure $P$, if its performance at tasks in $T$, as measured by $P$, improves with experience $E$.

    Read that carefully. It's deceptively simple and remarkably complete. There are three ingredients: a **task** you care about (classifying emails, predicting stock prices, generating images), a **performance measure** that tells you how well you're doing, and **experience** — data — from which the system improves. If all three are present and improvement happens, the system is learning.

    From a statistical perspective, machine learning is function estimation. You have some unknown function $f: \mathcal{X} \to \mathcal{Y}$ that maps inputs to outputs, and you're trying to approximate it using a finite sample of input-output pairs $\{(x_1, y_1), \ldots, (x_n, y_n)\}$. Your approximation $\hat{f}$ should be close to $f$ not just on the data you've seen, but on new, unseen inputs.

    That last phrase is the entire game. If all we cared about was fitting the data we already have, this would be curve fitting — a solved problem. Lagrange interpolation can pass a polynomial through any finite set of points exactly. But **memorizing is not learning**. The difference between curve fitting and machine learning is **generalization**: performing well on data you haven't seen yet.

    This is what makes ML genuinely hard, and genuinely interesting. We'll spend the rest of this module — and honestly, the rest of this course — unpacking what generalization means and how to achieve it.

    > **Reading:** [Bishop, Ch 1 (pp. 1-4)](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) gives a beautiful introduction through the lens of polynomial curve fitting. [Murphy, Ch 1](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf) provides a more modern and comprehensive overview.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2. Types of Learning

    Not all learning problems look the same. The taxonomy below isn't just organizational tidiness — the type of learning you're doing fundamentally constrains which algorithms apply, which theory governs your guarantees, and what kind of data you need.

    ### Supervised Learning

    This is the bread and butter. You have a dataset of labeled examples: inputs $x_i$ paired with known outputs $y_i$. The goal is to learn a mapping $\hat{f}$ so that $\hat{f}(x) \approx y$ for new, unseen $x$.

    Supervised learning splits into two flavors based on the nature of $y$:

    - **Regression:** $y \in \mathbb{R}$ (or $\mathbb{R}^k$). You're predicting a continuous quantity. How much will this house sell for? What will the temperature be tomorrow? What's the expected return on this portfolio?
    - **Classification:** $y \in \{1, 2, \ldots, C\}$. You're predicting a discrete category. Is this email spam or not? Which digit is in this image? Does this patient have the disease?

    The distinction matters because it changes your loss function, your evaluation metrics, and often your model architecture. But conceptually, both are instances of the same problem: estimate $f$ from labeled data.

    Examples worth internalizing: predicting housing prices from square footage, location, and features (regression). Detecting spam from email text (binary classification). Diagnosing disease from medical imaging (classification, often with severe class imbalance). Each of these illustrates different practical challenges that the theory must address.

    ### Unsupervised Learning

    Here you have inputs $x_i$ but no labels $y_i$. The goal is to discover **structure** in the data itself. This is fundamentally harder to evaluate — how do you measure performance when there's no ground truth? — but it's also where some of the most fascinating problems live.

    The main unsupervised tasks:

    - **Clustering:** Group similar data points together. Customer segmentation, gene expression analysis, topic discovery in text.
    - **Dimensionality reduction:** Find lower-dimensional representations that capture the essential structure. PCA is the classic example — we'll derive it properly in Module 1D.
    - **Density estimation:** Estimate the probability distribution $p(x)$ that generated the data. This is the most general unsupervised task, and in some sense all others are special cases of it.
    - **Anomaly detection:** Identify data points that don't fit the learned structure. Fraud detection, equipment failure prediction, network intrusion detection.

    ### Semi-Supervised Learning

    In the real world, labeled data is expensive and unlabeled data is cheap. A hospital might have millions of X-rays but only thousands with radiologist annotations. Semi-supervised learning exploits the structure in unlabeled data to improve supervised predictions. The unlabeled data doesn't tell you the answer, but it tells you something about the shape of the input space — and that matters.

    ### Reinforcement Learning

    An agent interacts with an environment, takes actions, receives rewards, and learns a policy that maximizes cumulative reward over time. This is a fundamentally different setup: there's no static dataset, the agent's actions affect future observations, and the feedback (reward) is often delayed and sparse.

    Reinforcement learning is behind AlphaGo, robotic control, and much of the fine-tuning that makes modern language models useful. We'll cover it properly in Module 3D — it deserves its own careful treatment. For now, just register that it exists and that it requires different mathematical machinery (Markov decision processes, value functions, policy gradients).

    > **Reading:** [Sutton & Barto, Ch 1](file:///C:/Users/landa/ml-course/textbooks/Sutton-RL.pdf) for RL motivation.

    ### Self-Supervised Learning

    This is the paradigm that has eaten much of modern ML. The idea is deceptively clever: create your own labels from the structure of the data itself. Mask out a word in a sentence and predict it. Corrupt an image and reconstruct it. Predict the next token in a sequence.

    Self-supervised learning is technically supervised (you have inputs and targets), but the labels are free — generated automatically from the data. This means you can train on internet-scale datasets without any human annotation. GPT, BERT, and the entire foundation model paradigm are built on this insight. We won't formalize it until Part 2, but it's worth knowing that this is where the field's center of gravity has shifted.

    > **Reading:** [ISLR, Ch 1](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf) provides excellent motivation for supervised vs. unsupervised. [ESL, Ch 1](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf) takes a broader view.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3. The Fundamental Problem: Generalization

    This section is the intellectual core of the entire lecture — possibly of the entire course. Everything in ML, every algorithm, every regularization trick, every architectural choice, is ultimately in service of one goal: **generalization**.

    ### Memorizing vs. Learning

    Suppose I give you a dataset of 100 points and ask you to fit a model. A degree-99 polynomial will pass through every single point with zero error. Perfect performance! But if I give you a new point not in the training set, that polynomial will almost certainly give a wildly wrong prediction, oscillating absurdly between the training points.

    This is **overfitting**: the model has memorized the training data, including its noise, and learned nothing about the underlying pattern. The training error is zero, but the test error — the error on new data — is catastrophic.

    The opposite failure mode is **underfitting**: the model is too simple to capture the real pattern. Fitting a straight line to clearly curved data. Training error is high, test error is high, everyone is unhappy.

    ### Training Error vs. Test Error

    Define the **training error** as the average loss on the data you trained on:

    $$\text{Err}_{\text{train}} = \frac{1}{n} \sum_{i=1}^{n} L(y_i, \hat{f}(x_i))$$

    and the **test error** (or generalization error) as the expected loss on a new, randomly drawn data point:

    $$\text{Err}_{\text{test}} = \mathbb{E}_{(x,y) \sim P}\left[L(y, \hat{f}(x))\right]$$

    Training error almost always decreases as model complexity increases — you can always fit the training data better with a more flexible model. But test error has a U-shape: it decreases at first (the model captures real patterns), then increases (the model starts fitting noise). The gap between training and test error is the **generalization gap**, and controlling it is the central challenge of ML.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The Bias-Variance Tradeoff

    This is THE fundamental tension. Let me derive it so you see exactly where it comes from.

    Assume the true data-generating process is $y = f(x) + \varepsilon$, where $\varepsilon$ is noise with $\mathbb{E}[\varepsilon] = 0$ and $\text{Var}(\varepsilon) = \sigma^2$. For a given input $x$, the expected prediction error of our learned model $\hat{f}$ (averaged over different training sets we could have drawn) decomposes as:

    $$\mathbb{E}\left[(y - \hat{f}(x))^2\right] = \underbrace{\left(f(x) - \mathbb{E}[\hat{f}(x)]\right)^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}\left[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2\right]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Irreducible Noise}}$$

    Let me make sure you understand each term:

    - **Bias** measures how far off your model is *on average* from the truth. A linear model fit to quadratic data will be biased no matter how much data you have. High bias means your model class isn't expressive enough — it **underfits**.
    - **Variance** measures how much your model's predictions fluctuate across different training sets. A degree-20 polynomial will give wildly different fits depending on which 50 points you sampled. High variance means your model is too sensitive to the particular training data — it **overfits**.
    - **Irreducible noise** is the floor. Even the true $f$ can't predict $y$ perfectly because of inherent randomness. No algorithm can beat this.

    The tradeoff: simpler models have high bias but low variance. Complex models have low bias but high variance. The optimal complexity is the one that minimizes their **sum** — and that optimum depends on how much data you have and how noisy it is. More data lets you safely use more complex models because variance decreases with sample size (roughly as $1/n$).

    This decomposition is not just a theoretical curiosity. It's the lens through which you should evaluate every modeling decision. When your model performs poorly, your first question should be: is this a bias problem or a variance problem? The answer dictates completely different remedies.

    > **Reading:** [ISLR, Section 2.2](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf) gives an exceptionally clear treatment of the bias-variance tradeoff with visual examples. [ESL, Section 2.9](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf) provides the full mathematical derivation. Both are essential reading.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4. The ML Pipeline

    Theory is necessary but not sufficient. Let me walk you through what doing ML actually looks like in practice, because the gap between textbook ML and real ML is vast.

    **Data collection and cleaning.** This is where 80% of the time goes and approximately 0% of the papers are written. Real data is messy: missing values, inconsistent formats, duplicate entries, label errors, distribution shifts between your training data and the real world. If your data is garbage, no algorithm will save you. This is unsexy, unglamorous work, and it is the single most important step.

    **Feature engineering.** Raw data is rarely in a form that algorithms can use directly. Feature engineering is the art of encoding domain knowledge as numerical representations. A date becomes day-of-week, month, is-holiday. An address becomes latitude, longitude, distance-to-city-center. Text becomes word counts or embeddings. Good features can make a simple model outperform a complex one with bad features. (Deep learning has partially automated this step, which is one of its major selling points — but even there, data representation choices matter enormously.)

    **Model selection.** You choose a hypothesis class — the family of functions you're searching over. Linear models? Decision trees? Neural networks? This choice embodies your assumptions about the problem and controls the bias-variance tradeoff. There is no universally best choice (more on this in Section 7).

    **Training.** Here's where your optimization knowledge pays off. Training a model means minimizing a loss function over the training data, subject to whatever constraints or regularization you've chosen. Gradient descent, the algorithm you studied in Module 0F, is the workhorse. The loss function encodes what you care about: squared error for regression, cross-entropy for classification, and many task-specific variants.

    **Evaluation.** You must evaluate on data the model has never seen during training. This is non-negotiable. The standard approach is to split your data into training, validation, and test sets. Train on the training set, tune hyperparameters on the validation set, report final performance on the test set (touched exactly once). Cross-validation is a more sophisticated version of this idea. We'll formalize evaluation in Module 1B.

    **Deployment and monitoring.** A trained model sitting on your laptop is useless. Deployment means integrating it into a system where it processes real inputs and produces real outputs. Monitoring means watching for distribution shift — the phenomenon where real-world data gradually (or suddenly) stops looking like your training data. A spam filter trained in 2020 will perform poorly on 2026 spam. Models need retraining, and knowing *when* is its own challenge.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 5. Parametric vs. Nonparametric Models

    This distinction runs deep and has consequences you'll feel throughout the course.

    **Parametric models** assume a specific functional form with a fixed, finite number of parameters. Linear regression assumes $\hat{f}(x) = w^T x + b$ — the parameters are $w$ and $b$, and their count is determined by the input dimensionality, not the amount of data. You could have 100 data points or 100 million; the model still has the same number of parameters. Parametric models are computationally efficient and interpretable, but they're only as good as their assumptions. If the true function isn't well-approximated by your parametric form, no amount of data will fix the bias.

    **Nonparametric models** don't assume a fixed functional form. Their effective complexity grows with the amount of data. $k$-nearest neighbors is the canonical example: to make a prediction for a new point, you find the $k$ closest training points and average their outputs (regression) or take a majority vote (classification). There are no "parameters" learned during training — the entire training set *is* the model. Decision trees, kernel density estimators, and Gaussian processes are other examples. Nonparametric models can approximate any function given enough data, but they tend to need a lot of data (especially in high dimensions), and prediction can be computationally expensive since you may need to reference the entire training set.

    The tradeoff maps directly onto bias-variance. Parametric models trade variance for bias (strong assumptions reduce variance but may introduce bias). Nonparametric models trade bias for variance (few assumptions, but highly sensitive to the particular data you have, especially with small samples).

    > **Reading:** [ISLR, Section 2.1](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf) distinguishes parametric and nonparametric approaches with clear examples. [ESL, Section 2.3](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf) goes deeper.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 6. The Curse of Dimensionality

    High-dimensional spaces are profoundly unintuitive, and this unintuition will bite you if you're not prepared for it.

    Consider a unit hypercube $[0, 1]^d$. To capture 10% of the volume (and thus roughly 10% of uniformly distributed data), you need a sub-cube with side length $0.1^{1/d}$. In $d = 1$, that side length is 0.1. In $d = 10$, it's $0.1^{0.1} \approx 0.79$. To capture just 10% of the data, you need to span 79% of each dimension's range. Your "local" neighborhood is nearly global. Locality — the foundation of many ML algorithms — breaks down.

    It gets stranger. In high dimensions, **most of the volume of a hypersphere concentrates near the surface**. The ratio of the volume of a sphere of radius $r - \epsilon$ to one of radius $r$ is $(1 - \epsilon/r)^d$, which goes to 0 exponentially fast in $d$. Points sampled uniformly within a high-dimensional sphere are almost all near the shell.

    Distances become problematic too. For points uniformly distributed in a high-dimensional cube, the ratio of the maximum to minimum pairwise distance converges to 1 as $d \to \infty$. Every point is roughly the same distance from every other point. If all distances are similar, distance-based algorithms (nearest neighbors, clustering by proximity) lose their discriminative power.

    The practical consequence: to maintain the same "density" of data coverage, you need exponentially more data as dimensionality increases. If 100 points give you reasonable coverage in 1D, you need $100^d$ for the same coverage in $d$ dimensions. This is why dimensionality reduction (Module 1D) and feature selection are not optional niceties — they're often survival strategies.

    > **Reading:** [ESL, Section 2.5](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf) has an excellent treatment of the curse of dimensionality. [MML, Ch 12](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) discusses it in the context of dimensionality reduction.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 7. The No Free Lunch Theorem

    Here's a humbling result. Wolpert and Macready (1997) proved that **no learning algorithm is universally better than any other** when averaged over all possible problems. For every problem where algorithm A beats algorithm B, there exists another problem where B beats A. Averaged over the uniform distribution over all possible functions, every algorithm has the same expected performance — including random guessing.

    The implication is not that ML is hopeless. The implication is that **assumptions matter**. The reason ML works in practice is that real-world problems are not uniformly distributed over all possible functions — they have structure (smoothness, symmetry, sparsity, low intrinsic dimensionality) that good algorithms exploit. The No Free Lunch theorem tells you that choosing an algorithm is choosing a set of assumptions, and those assumptions had better match your problem.

    This is why domain knowledge isn't a nice-to-have — it's a core ingredient. Knowing that your function is smooth, or sparse, or periodic, or invariant to certain transformations, lets you choose models and regularizers that encode those priors, giving you an edge that no algorithm-agnostic approach can match.

    > **Reading:** [Murphy, Section 1.4](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf) discusses the No Free Lunch theorem and its practical implications.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 8. A Taxonomy of What's Coming

    Let me give you the map before we start walking. Here's what the rest of Part 1 covers, and why each piece matters:

    **Linear models (Modules 1B and 1C).** We start here because linear models are the foundation everything else builds on. Linear regression, logistic regression, regularization (Ridge, Lasso). These are not toy models — they're powerful, interpretable, and often surprisingly competitive. More importantly, understanding their behavior deeply (when they work, when they fail, and why) gives you the intuition to understand everything else. Module 1C extends into kernel methods, which implicitly map data into high-dimensional feature spaces, making linear models act nonlinearly.

    **Dimensionality reduction (Module 1D).** PCA, and the idea that high-dimensional data often lives on low-dimensional manifolds. Direct defense against the curse of dimensionality.

    **Tree-based and ensemble methods (Module 1E).** Decision trees are intuitive but unstable. Random forests and gradient boosting fix the instability by combining many models. Gradient-boosted trees (XGBoost, LightGBM) are the workhorses of applied ML on tabular data — the algorithms that win Kaggle competitions and power real-world prediction systems.

    **Neural networks (Part 2).** The function approximators that scale. We'll build from single neurons to deep networks to convolutional and recurrent architectures to transformers. This is where the field's most dramatic recent progress has happened, but you need Part 1's foundations to understand *why* these architectures work and when they don't.

    **Kernel methods (Module 1C).** The "kernel trick" lets you work in infinite-dimensional feature spaces without ever computing the coordinates. Elegant, theoretically grounded, and the basis for support vector machines.

    **Ensemble methods (Module 1E).** The meta-insight: combining multiple weak models can produce a strong one. Bagging reduces variance. Boosting reduces bias. Stacking combines diverse models. These ideas are universal and apply across all model families.

    > **Reading:** [Bishop, Ch 1](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) provides an excellent overview that bridges many of these topics. [Murphy, Ch 1](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf) is the most comprehensive modern survey.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Looking Forward

    You now have the vocabulary, the conceptual framework, and the central questions. What is learning? (Generalization.) What makes it hard? (The bias-variance tradeoff, the curse of dimensionality, finite data.) What does it look like in practice? (A pipeline from messy data to deployed model.) What are the main paradigms? (Supervised, unsupervised, reinforcement, self-supervised.) What are the main model families? (Linear, tree-based, neural, kernel.)

    Starting in Module 1B, we'll get concrete. We'll derive our first learning algorithm — linear regression — from scratch, connect it to the optimization you already know, formalize the evaluation framework, and start building real intuition for how models behave. The abstractions in this lecture will become very tangible very quickly.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Practice Problems

    These are conceptual — no code required. The point is to test whether the ideas have actually landed.

    **P1.** A friend fits a degree-15 polynomial to 20 data points and reports a training $R^2$ of 0.99. They claim the model is excellent. What's wrong with this reasoning? What would you recommend they do to honestly evaluate their model?

    **P2.** You're building a model to predict apartment rental prices. You have features like square footage, number of bedrooms, neighborhood, distance to transit, and year built. For each feature, describe what feature engineering you might apply before feeding it to a model. Which features are already numeric? Which need transformation?

    **P3.** Consider two models for the same problem: (a) linear regression with 5 features, and (b) $k$-nearest neighbors with $k = 1$. Which has higher bias? Which has higher variance? How would you expect each to perform with 50 training points vs. 50,000 training points? Explain using the bias-variance tradeoff.

    **P4.** You have a dataset of 10,000 medical images, each with 1,000,000 pixels (features). You want to classify each image as healthy or diseased. Explain why the curse of dimensionality is relevant here and suggest at least two strategies to address it.

    **P5.** Explain the No Free Lunch theorem in your own words. Then argue: if no algorithm is universally best, why do we bother studying specific algorithms at all? (This is not a trick question — there's a real, satisfying answer.)

    **P6.** For each of the following problems, identify the type of learning (supervised/unsupervised/RL/self-supervised) and, if supervised, whether it's regression or classification:
    - (a) Grouping news articles by topic without predefined categories
    - (b) Predicting the selling price of a used car
    - (c) Teaching a robot to walk
    - (d) Training a language model by predicting masked words
    - (e) Diagnosing whether a tumor is malignant or benign from a biopsy image

    **P7.** The bias-variance decomposition gives $\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}^2 + \text{Variance} + \sigma^2$. Suppose you know the irreducible noise is $\sigma^2 = 4$. Your model achieves an expected test MSE of 12. If you estimate the bias to be 2, what is the variance? Now suppose you switch to a more complex model that reduces bias to 0 but increases variance to 15. What's the new test MSE? Was increasing complexity a good idea?

    **P8.** A parametric model assumes $f(x) = ax^2 + bx + c$. A nonparametric model uses $k$-nearest neighbors. If the true function is actually $f(x) = \sin(x)$, which approach do you think would perform better with (a) 20 data points, and (b) 10,000 data points? Why?
    """)
    return


if __name__ == "__main__":
    app.run()
