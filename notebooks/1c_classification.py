import marimo

app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import make_moons, make_circles
    from sklearn.linear_model import LogisticRegression
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, confusion_matrix,
                                 classification_report, roc_curve)
    return (mo, np, plt, SVC, StandardScaler, make_moons, make_circles,
            LogisticRegression, LinearDiscriminantAnalysis, GaussianNB,
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix, classification_report, roc_curve)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Module 1C: Classification

    You've spent the last module learning how to predict continuous values — house prices, temperatures, stock returns. Now we shift to a fundamentally different question: **what category does this thing belong to?** Is this email spam or not? Is this tumor malignant or benign? Which digit is written in this image?

    This is classification, and it's arguably the most practically important problem in all of machine learning. The good news: many of the ideas from regression carry over directly. The bad news: some of them carry over in ways that will surprise you.

    ---

    ## 1. From Regression to Classification

    Your first instinct might be: "I already know linear regression. Can't I just assign 0 to one class and 1 to the other, then fit a line?" You can try. Here's what goes wrong.

    Suppose you're classifying tumors as malignant (1) or benign (0) based on size. You fit a linear regression $\hat{y} = w_0 + w_1 x$. For a small dataset with well-separated classes, this actually looks reasonable — the line crosses 0.5 somewhere sensible, and you threshold there.

    Now add a single large benign tumor far to the right. The regression line tilts to accommodate it. Your threshold shifts. Points that were correctly classified as malignant are now misclassified. One outlier broke your classifier.

    The deeper problem: linear regression outputs can be anything — $-3.7$, $42$, $\pi$. But for classification, we need **probabilities** between 0 and 1, or at least something that respects the structure of categorical outputs. We need a model that says "I'm 87% confident this is malignant," not "the malignancy score is 1.3."

    See [ISLR Section 4.2](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf) for a nice visual demonstration of exactly this failure mode.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2. Logistic Regression

    ### The Sigmoid Function

    We want a function that takes any real number and squashes it into $(0, 1)$. The **sigmoid** (or logistic) function does exactly this:

    $$\sigma(z) = \frac{1}{1 + e^{-z}}$$

    Why this specific function? A few reasons:
    - It maps $\mathbb{R} \to (0, 1)$ — perfect for probabilities
    - It's symmetric: $\sigma(-z) = 1 - \sigma(z)$
    - Its derivative is remarkably clean: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$
    - It arises naturally from the exponential family (we'll see this in the multiclass section)
    - It's the canonical link function for Bernoulli distributions in generalized linear models

    The deeper reason: if you model the **log-odds** as linear in $x$, you get the sigmoid. That is, if $\log \frac{P(y=1|x)}{P(y=0|x)} = w^\top x$, then solving for $P(y=1|x)$ gives you $\sigma(w^\top x)$.

    ### The Model

    Logistic regression models the probability of class membership:

    $$P(y = 1 | x) = \sigma(w^\top x) = \frac{1}{1 + e^{-w^\top x}}$$

    This is still a **linear model** in an important sense: the decision boundary (where $P = 0.5$) is the hyperplane $w^\top x = 0$. Everything on one side gets classified as 1, everything on the other side as 0. What's nonlinear is the mapping from $w^\top x$ to probability.

    ### The Decision Boundary

    Setting $P(y=1|x) = 0.5$ gives $\sigma(w^\top x) = 0.5$, which means $w^\top x = 0$. This is a hyperplane in feature space. In 2D, it's a line. In 3D, a plane. The weight vector $w$ is **normal** to this hyperplane — it points toward the "class 1" side.

    The magnitude of $w^\top x$ tells you the **confidence**: points far from the boundary get probabilities close to 0 or 1; points near the boundary get probabilities near 0.5. This is much more informative than a hard classification.
    """)
    return


@app.cell
def _(np, plt):
    # Sigmoid function: sigma(z) = 1 / (1 + exp(-z))
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    # Plot it to see the squashing behavior
    z_vals = np.linspace(-8, 8, 200)
    fig_sig, ax_sig = plt.subplots(figsize=(7, 3))
    ax_sig.plot(z_vals, sigmoid(z_vals), linewidth=2)
    ax_sig.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax_sig.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax_sig.set_xlabel("z")
    ax_sig.set_ylabel(r"$\sigma(z)$")
    ax_sig.set_title("Sigmoid function")
    plt.tight_layout()

    # Verify the symmetry property: sigma(-z) = 1 - sigma(z)
    z_test = np.array([-2.0, 0.0, 1.5, 4.0])
    print("Symmetry check: sigma(-z) == 1 - sigma(z)?",
          np.allclose(sigmoid(-z_test), 1 - sigmoid(z_test)))

    # Verify the derivative: sigma'(z) = sigma(z) * (1 - sigma(z))
    s = sigmoid(z_test)
    print("Derivative at z =", z_test, ":", s * (1 - s))
    fig_sig
    return (sigmoid,)


@app.cell
def _(np, sigmoid):
    # Decision boundary demo: w^T x = 0 is where P(y=1) = 0.5
    w_demo = np.array([2.0, -1.0])  # weight vector
    b_demo = 0.5                     # bias

    # Points on either side of the boundary
    x_pos = np.array([1.0, 0.0])     # w^T x + b = 2.5 > 0
    x_neg = np.array([-1.0, 1.0])    # w^T x + b = -2.5 < 0

    # Probabilities: further from boundary -> more confident
    p_pos = sigmoid(w_demo @ x_pos + b_demo)
    p_neg = sigmoid(w_demo @ x_neg + b_demo)
    print(f"P(y=1 | x_pos) = {p_pos:.4f}  (confident positive)")
    print(f"P(y=1 | x_neg) = {p_neg:.4f}  (confident negative)")

    # A point near the boundary
    x_boundary = np.array([0.0, 0.5])  # w^T x + b = 0
    p_boundary = sigmoid(w_demo @ x_boundary + b_demo)
    print(f"P(y=1 | x_boundary) = {p_boundary:.4f}  (uncertain)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Deriving the Loss Function (Cross-Entropy from MLE)

    This is where your probabilistic background from Module 1B pays off. We have $n$ data points $(x_i, y_i)$ where $y_i \in \{0, 1\}$. The likelihood of observing the data is:

    $$L(w) = \prod_{i=1}^{n} P(y_i | x_i; w) = \prod_{i=1}^{n} \hat{p}_i^{y_i} (1 - \hat{p}_i)^{1-y_i}$$

    where $\hat{p}_i = \sigma(w^\top x_i)$. This is just the Bernoulli likelihood — each label is a coin flip with a data-dependent probability.

    Taking the negative log-likelihood:

    $$\text{NLL}(w) = -\sum_{i=1}^{n} \left[ y_i \log \hat{p}_i + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

    This is the **binary cross-entropy loss** (also called the log loss). It has a beautiful interpretation: it penalizes confident wrong predictions severely. If you predict $\hat{p} = 0.99$ and the true label is 0, the loss is $-\log(0.01) \approx 4.6$. If you predict $\hat{p} = 0.51$ and you're wrong, the loss is only $-\log(0.49) \approx 0.71$.

    ### Why Not Squared Error?

    You might wonder: why not minimize $\sum(y_i - \sigma(w^\top x_i))^2$? Two problems:

    1. **Non-convexity.** Squared error composed with the sigmoid creates a non-convex loss surface with local minima. Cross-entropy with the sigmoid is convex — guaranteed to find the global optimum.
    2. **Gradient saturation.** When the sigmoid saturates (outputs near 0 or 1), squared error gradients vanish. Cross-entropy gradients don't — they keep pushing in the right direction even when the model is confidently wrong.

    See [Bishop Section 4.3.2](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) for the full derivation.

    ### The Gradient

    Here's where things get satisfying. The gradient of the NLL with respect to $w$ is:

    $$\nabla_w \text{NLL} = \sum_{i=1}^{n} (\sigma(w^\top x_i) - y_i) \, x_i = X^\top (\sigma(Xw) - y)$$

    Look at that. It's the same form as the linear regression gradient: $X^\top(\hat{y} - y)$, except $\hat{y} = \sigma(Xw)$ instead of $\hat{y} = Xw$. The residual $(\hat{p}_i - y_i)$ gets multiplied by the feature vector $x_i$ and summed. This isn't a coincidence — it comes from both models being in the exponential family.

    **Derivation sketch:** Use the chain rule and $\sigma'(z) = \sigma(z)(1-\sigma(z))$:

    $$\frac{\partial}{\partial w_j} \text{NLL} = -\sum_{i=1}^n \left[\frac{y_i}{\hat{p}_i} - \frac{1-y_i}{1-\hat{p}_i}\right] \hat{p}_i(1-\hat{p}_i) \, x_{ij} = \sum_{i=1}^n (\hat{p}_i - y_i) \, x_{ij}$$

    The $\hat{p}(1-\hat{p})$ terms cancel beautifully. This clean gradient is one of the deep reasons the sigmoid + cross-entropy combination is so natural.

    ### No Closed-Form Solution

    Unlike linear regression, setting $\nabla_w \text{NLL} = 0$ doesn't give a closed-form solution — the sigmoid makes the equation transcendental. We use iterative methods:

    - **Gradient descent**: $w_{t+1} = w_t - \eta \, X^\top(\sigma(Xw_t) - y)$
    - **Newton's method** (IRLS): uses second-order information, converges faster. The Hessian is $X^\top S X$ where $S$ is diagonal with $S_{ii} = \hat{p}_i(1-\hat{p}_i)$. This is always positive semi-definite, confirming convexity.

    See [ISLR Section 4.3](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf) and [MML Section 12.2](file:///C:/Users/landa/ml-course/textbooks/MML.pdf).

    ### Regularized Logistic Regression

    Just like ridge and lasso in regression, we add penalty terms:

    $$\text{NLL}(w) + \lambda \|w\|_2^2 \quad \text{(L2 / ridge)}$$
    $$\text{NLL}(w) + \lambda \|w\|_1 \quad \text{(L1 / lasso)}$$

    L2 regularization shrinks coefficients toward zero, reducing overfitting. L1 produces sparse solutions — useful for feature selection. In `scikit-learn`, the `C` parameter is the inverse of $\lambda$: smaller `C` means more regularization.
    """)
    return


@app.cell
def _(np, sigmoid):
    # Binary cross-entropy loss: NLL = -sum[ y*log(p) + (1-y)*log(1-p) ]
    def cross_entropy_loss(y_true, p_hat):
        eps = 1e-12  # avoid log(0)
        p_hat = np.clip(p_hat, eps, 1 - eps)
        return -np.mean(y_true * np.log(p_hat) + (1 - y_true) * np.log(1 - p_hat))

    # See how the loss penalizes confident wrong predictions
    y_label = 1.0
    probs = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
    losses = [-np.log(p) for p in probs]  # loss when y=1
    print("True label = 1")
    for p, l in zip(probs, losses):
        print(f"  Predicted p={p:.2f} -> loss = {l:.3f}")

    # Verify: sigmoid + cross-entropy gradient = X^T (p_hat - y)
    X_demo = np.array([[1, 2], [1, -1], [1, 0.5]])  # 3 samples, 2 features
    y_demo = np.array([1, 0, 1])
    w_demo_ce = np.array([0.5, -0.3])
    p_hat_demo = sigmoid(X_demo @ w_demo_ce)
    grad = X_demo.T @ (p_hat_demo - y_demo) / len(y_demo)
    print(f"\nGradient of NLL: {grad}")
    return (cross_entropy_loss,)


@app.cell
def _(np, plt, sigmoid, cross_entropy_loss):
    # Logistic regression from scratch via gradient descent
    np.random.seed(0)
    # Simple 1D dataset with bias term
    X_gd = np.column_stack([np.ones(80), np.random.randn(80)])
    w_true_gd = np.array([0.5, 2.0])
    y_gd = (np.random.rand(80) < sigmoid(X_gd @ w_true_gd)).astype(float)

    # Gradient descent: w <- w - lr * X^T (sigma(Xw) - y) / n
    w_gd = np.zeros(2)
    lr = 0.5
    loss_history = []
    for epoch in range(100):
        p_hat_gd = sigmoid(X_gd @ w_gd)
        loss_history.append(cross_entropy_loss(y_gd, p_hat_gd))
        gradient = X_gd.T @ (p_hat_gd - y_gd) / len(y_gd)
        w_gd -= lr * gradient

    fig_loss, ax_loss = plt.subplots(figsize=(6, 3))
    ax_loss.plot(loss_history, linewidth=2)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.set_title("Logistic regression gradient descent convergence")
    plt.tight_layout()
    print(f"Learned weights: {w_gd}  (true: {w_true_gd})")
    fig_loss
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Code: Logistic Regression Example""")
    return


@app.cell
def _(np, LogisticRegression, make_moons):
    # Generate a 2D binary classification dataset
    np.random.seed(42)
    X_lr, y_lr = make_moons(n_samples=200, noise=0.2, random_state=42)

    model_lr = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs')
    model_lr.fit(X_lr, y_lr)

    # Predicted probabilities (not just labels)
    probs_lr = model_lr.predict_proba(X_lr)  # shape: (n_samples, 2)
    print(f"Accuracy: {model_lr.score(X_lr, y_lr):.3f}")

    # Inspect the decision boundary
    print(f"Weights: {model_lr.coef_}")
    print(f"Intercept: {model_lr.intercept_}")
    return (X_lr, y_lr)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3. Multiclass Classification

    ### One-vs-Rest (OvR)

    The simplest extension: train $K$ separate binary classifiers, one per class. Classifier $k$ treats class $k$ as positive and everything else as negative. At prediction time, pick the class whose classifier gives the highest score.

    This works, but it's inelegant — the classifiers are trained independently, so their probability estimates aren't calibrated against each other. You can get situations where all classifiers say "not me" or multiple say "me."

    ### Softmax Regression

    The principled multiclass extension uses the **softmax function**. For $K$ classes, we have $K$ weight vectors $w_1, \ldots, w_K$, and:

    $$P(y = k | x) = \frac{e^{w_k^\top x}}{\sum_{j=1}^{K} e^{w_j^\top x}}$$

    This is the softmax function applied to the vector of logits $z_k = w_k^\top x$. It guarantees that probabilities are positive and sum to 1. Note that for $K=2$, softmax reduces to logistic regression (one of the weight vectors is redundant).

    The loss function is the **categorical cross-entropy**:

    $$\text{NLL} = -\sum_{i=1}^{n} \sum_{k=1}^{K} \mathbb{1}[y_i = k] \log P(y_i = k | x_i)$$

    which simplifies to $-\sum_i \log P(y_i = k_i | x_i)$ where $k_i$ is the true class of example $i$. The gradient has the same clean form as the binary case.

    ### Connection to the Exponential Family

    The softmax arises naturally as the distribution that maximizes entropy subject to linear sufficient statistics. Both the Bernoulli (binary) and categorical (multiclass) are exponential family distributions, and logistic/softmax regression are their canonical generalized linear models. This isn't just mathematical trivia — it explains why the gradients are so clean and why these models have such nice optimization properties.

    See [Bishop Section 4.3.4](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) and [Murphy Section 10.3](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf).
    """)
    return


@app.cell
def _(np):
    # Softmax: P(y=k|x) = exp(z_k) / sum_j exp(z_j)
    def softmax(z):
        # Subtract max for numerical stability (doesn't change result)
        z_stable = z - np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(z_stable)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    # Example: 3-class logits for one sample
    logits = np.array([2.0, 1.0, 0.1])
    probs_sm = softmax(logits)
    print(f"Logits:        {logits}")
    print(f"Probabilities: {probs_sm}")
    print(f"Sum:           {probs_sm.sum():.6f}")

    # Categorical cross-entropy for one sample (true class = 0)
    true_class = 0
    cat_ce = -np.log(probs_sm[true_class])
    print(f"\nCross-entropy (true class {true_class}): {cat_ce:.4f}")

    # Verify: for K=2, softmax reduces to sigmoid
    logits_2 = np.array([1.5, 0.0])  # second logit is redundant
    p_softmax = softmax(logits_2)[0]
    p_sigmoid = 1 / (1 + np.exp(-(logits_2[0] - logits_2[1])))
    print(f"\nK=2: softmax P(class 0) = {p_softmax:.6f}")
    print(f"     sigmoid equivalent  = {p_sigmoid:.6f}")
    return (softmax,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Code: Multiclass Softmax""")
    return


@app.cell
def _(np, LogisticRegression):
    from sklearn.datasets import make_classification as _make_classification
    # Generate a multiclass dataset
    X_mc, y_mc = _make_classification(n_samples=300, n_features=5, n_informative=3,
                                       n_classes=3, n_clusters_per_class=1, random_state=42)

    model_mc = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    model_mc.fit(X_mc, y_mc)

    # Per-class probabilities
    probs_mc = model_mc.predict_proba(X_mc)  # shape: (n_samples, K)
    print(f"Multiclass accuracy: {model_mc.score(X_mc, y_mc):.3f}")
    print(f"Probability matrix shape: {probs_mc.shape}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4. Discriminative vs Generative Classifiers

    This is a fundamental conceptual divide that you'll see throughout machine learning.

    **Discriminative models** learn the decision boundary directly. They model $P(y|x)$ — given these features, what's the class? Logistic regression is discriminative. It doesn't care what the data "looks like," only where to draw the line.

    **Generative models** learn what data from each class "looks like." They model $P(x|y)$ (the class-conditional density) and $P(y)$ (the prior), then use Bayes' theorem:

    $$P(y|x) = \frac{P(x|y) P(y)}{P(x)}$$

    Think of it this way: a discriminative model learns to distinguish cats from dogs. A generative model learns what cats look like and what dogs look like, then classifies by asking "does this look more like a cat or a dog?"

    **Tradeoffs:**
    - Discriminative models are usually **more accurate** with enough data — they focus on the boundary, not on modeling the full data distribution
    - Generative models can work with **less data** because the modeling assumptions provide inductive bias
    - Generative models handle **missing features** more naturally
    - Generative models can **generate** synthetic data (hence the name)
    - The Ng & Jordan (2002) result: logistic regression is asymptotically more efficient, but Naive Bayes reaches its (lower) asymptotic error faster

    See [ISLR Section 4.4](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf) and [Bishop Section 4.2](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 5. Linear Discriminant Analysis (LDA)

    LDA is the classic generative classifier. The key assumption: the data from each class follows a multivariate Gaussian, and **all classes share the same covariance matrix**.

    $$P(x | y = k) = \mathcal{N}(x | \mu_k, \Sigma)$$

    Combined with Bayes' theorem:

    $$P(y = k | x) \propto P(x | y = k) \, P(y = k) = \mathcal{N}(x | \mu_k, \Sigma) \, \pi_k$$

    Taking the log and dropping terms that don't depend on $k$:

    $$\log P(y=k|x) \propto -\frac{1}{2}(x - \mu_k)^\top \Sigma^{-1} (x - \mu_k) + \log \pi_k$$

    Expanding the quadratic and dropping the $x^\top \Sigma^{-1} x$ term (it's the same for all classes because $\Sigma$ is shared):

    $$\delta_k(x) = x^\top \Sigma^{-1} \mu_k - \frac{1}{2} \mu_k^\top \Sigma^{-1} \mu_k + \log \pi_k$$

    This is **linear in $x$**. The decision boundary between classes $k$ and $l$ is $\delta_k(x) = \delta_l(x)$, which is a hyperplane. That's where the "linear" in LDA comes from.

    ### Fisher's Discriminant: The Dimensionality Reduction View

    There's a beautiful alternative derivation due to Fisher. Instead of thinking probabilistically, Fisher asked: what direction $w$ should I project the data onto to maximally separate the classes?

    Formally, maximize the **between-class variance** relative to the **within-class variance**:

    $$J(w) = \frac{w^\top S_B w}{w^\top S_W w}$$

    where $S_B$ is the between-class scatter matrix and $S_W$ is the within-class scatter. The solution is $w \propto S_W^{-1}(\mu_1 - \mu_2)$ for two classes. For $K$ classes, you get up to $K-1$ discriminant directions — this gives you a dimensionality reduction technique.

    The Fisher and Bayesian-Gaussian views give the same decision boundary. That's not obvious, and it's worth appreciating.

    ### When LDA Wins

    LDA beats logistic regression when:
    - **Small sample size**: the Gaussian assumption provides strong regularization
    - **Classes are well-separated**: logistic regression can have numerical issues when classes are perfectly separable (the weights diverge to infinity)
    - **The Gaussian assumption is roughly correct**: if your features are actually approximately Gaussian with shared covariance, LDA exploits this

    When the Gaussian assumption is badly violated (e.g., binary or highly skewed features), logistic regression typically wins.

    See [ISLR Section 4.4.1-4.4.4](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf), [ESL Section 4.3](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf), and [Bishop Section 4.2.1](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf).
    """)
    return


@app.cell
def _(np):
    # LDA from scratch: compute discriminant function delta_k(x)
    # Generate two Gaussian classes sharing one covariance
    np.random.seed(42)
    mu0 = np.array([0.0, 0.0])
    mu1 = np.array([2.0, 1.0])
    Sigma_shared = np.array([[1.0, 0.3], [0.3, 0.8]])

    X0_lda = np.random.multivariate_normal(mu0, Sigma_shared, 50)
    X1_lda = np.random.multivariate_normal(mu1, Sigma_shared, 50)

    # Estimate parameters from data
    mu0_hat = X0_lda.mean(axis=0)
    mu1_hat = X1_lda.mean(axis=0)
    Sigma_hat = 0.5 * (np.cov(X0_lda.T) + np.cov(X1_lda.T))
    Sigma_inv = np.linalg.inv(Sigma_hat)

    # Discriminant: delta_k(x) = x^T Sigma^{-1} mu_k - 0.5 mu_k^T Sigma^{-1} mu_k + log(pi_k)
    # For equal priors, log(pi_k) cancels
    # Fisher direction: w = Sigma^{-1} (mu1 - mu0)
    w_fisher = Sigma_inv @ (mu1_hat - mu0_hat)
    print(f"Fisher discriminant direction: {w_fisher}")
    print(f"Classify by sign of w^T x + threshold")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Code: LDA""")
    return


@app.cell
def _(X_lr, y_lr, LinearDiscriminantAnalysis):
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_lr, y_lr)

    # LDA can also do dimensionality reduction
    X_projected = lda.transform(X_lr)  # project onto discriminant directions

    print(f"Accuracy: {lda.score(X_lr, y_lr):.3f}")
    print(f"Class means:\n{lda.means_}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 6. Naive Bayes

    Naive Bayes is a generative classifier that makes a dramatic simplifying assumption: **all features are conditionally independent given the class**.

    $$P(x | y = k) = \prod_{j=1}^{d} P(x_j | y = k)$$

    This is almost certainly wrong. Features are almost never independent. Your height and weight are correlated. Words in a document are correlated. Pixel values in an image are correlated.

    And yet Naive Bayes works remarkably well, especially for:
    - **Text classification** (spam detection, sentiment analysis, topic classification)
    - **High-dimensional data** where estimating a full covariance would be hopeless
    - **Small datasets** where the independence assumption acts as strong regularization

    Why does it work despite the wrong assumption? The key insight is that **classification only requires getting the decision boundary right**, not the probability estimates. Even if $P(y|x)$ is poorly estimated, the *argmax* over classes can still be correct. The independence assumption biases probabilities toward 0 and 1 (overconfident), but the ranking of classes is often preserved.

    For text classification, each feature is a word count and we model $P(x_j | y=k)$ as Multinomial or Bernoulli. The posterior becomes:

    $$P(y=k|x) \propto \pi_k \prod_j P(x_j | y = k)$$

    In log space, this is just a sum of log-probabilities — extremely fast to compute.
    """)
    return


@app.cell
def _(np):
    # Naive Bayes from scratch (Gaussian) for 2 features, 2 classes
    # P(y=k|x) proportional to pi_k * prod_j N(x_j; mu_kj, sigma_kj^2)
    np.random.seed(7)
    # Class 0: feature means [0, 0], Class 1: feature means [1.5, 1]
    X0_nb = np.random.randn(40, 2) + np.array([0, 0])
    X1_nb = np.random.randn(40, 2) + np.array([1.5, 1])

    # Estimate per-class, per-feature mean and variance (the "naive" part)
    mu_nb = np.array([X0_nb.mean(axis=0), X1_nb.mean(axis=0)])
    var_nb = np.array([X0_nb.var(axis=0), X1_nb.var(axis=0)])
    prior_nb = np.array([0.5, 0.5])

    # Predict: log P(y=k|x) = log pi_k + sum_j log N(x_j; mu_kj, var_kj)
    x_new = np.array([0.8, 0.5])
    log_posts = []
    for k in range(2):
        log_lik = -0.5 * np.sum((x_new - mu_nb[k])**2 / var_nb[k] + np.log(var_nb[k]))
        log_posts.append(np.log(prior_nb[k]) + log_lik)
    log_posts = np.array(log_posts)
    pred_class = np.argmax(log_posts)
    print(f"Log-posteriors: {log_posts}")
    print(f"Predicted class: {pred_class}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Code: Naive Bayes""")
    return


@app.cell
def _(X_lr, y_lr, GaussianNB):
    # For continuous features
    gnb = GaussianNB()
    gnb.fit(X_lr, y_lr)
    print(f"Gaussian NB accuracy: {gnb.score(X_lr, y_lr):.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    See [ISLR Section 4.4.4](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf) and [ESL Section 6.6.3](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf).

    ---

    ## 7. Support Vector Machines (SVMs)

    SVMs take a completely different approach to classification. Instead of modeling probabilities, they look for the **decision boundary that maximizes the margin** — the distance between the boundary and the nearest data points.

    ### The Maximum Margin Idea

    Consider a linearly separable binary classification problem. There are infinitely many hyperplanes that separate the classes. Which one should we pick?

    The SVM answer: pick the one that is **as far as possible from the nearest data points on either side**. The intuition is robustness — a wider margin means small perturbations to new data points are less likely to cause misclassification.

    The distance from a point $x_i$ to the hyperplane $w^\top x + b = 0$ is $\frac{|w^\top x_i + b|}{\|w\|}$. For correctly classified points, $y_i(w^\top x_i + b) > 0$, so the distance is $\frac{y_i(w^\top x_i + b)}{\|w\|}$.

    ### Hard Margin SVM

    We want to maximize the minimum margin. With a rescaling trick (we can always rescale $w$ so the closest point has $y_i(w^\top x_i + b) = 1$), this becomes:

    $$\min_{w,b} \frac{1}{2}\|w\|^2 \quad \text{subject to} \quad y_i(w^\top x_i + b) \geq 1, \; \forall i$$

    This is a **convex quadratic program** — it has a unique global solution and can be solved efficiently. The factor $\frac{1}{2}\|w\|^2$ is there because minimizing $\|w\|$ is the same as maximizing the margin $\frac{2}{\|w\|}$.

    ### Support Vectors

    Here's what makes SVMs elegant: at the solution, **only a few data points matter**. These are the points sitting exactly on the margin ($y_i(w^\top x_i + b) = 1$). They're called **support vectors**, and the entire solution is determined by them alone. Remove any non-support-vector point and the solution doesn't change.

    This is very different from logistic regression, where every data point influences the solution. SVMs are determined by the hardest cases — the points closest to the boundary.

    ### Soft Margin SVM

    Real data is rarely perfectly separable. The **soft margin SVM** introduces slack variables $\xi_i \geq 0$ that allow points to violate the margin:

    $$\min_{w,b,\xi} \frac{1}{2}\|w\|^2 + C \sum_{i=1}^n \xi_i \quad \text{subject to} \quad y_i(w^\top x_i + b) \geq 1 - \xi_i, \; \xi_i \geq 0$$

    The parameter $C$ controls the tradeoff: large $C$ penalizes violations heavily (narrow margin, few violations); small $C$ allows more violations (wider margin, more robust). This is the SVM analogue of regularization.

    The loss function for the soft margin SVM is the **hinge loss**: $\max(0, 1 - y_i f(x_i))$. Compare this to logistic loss: both penalize wrong predictions, but hinge loss is exactly zero once the margin exceeds 1. This is why SVMs produce sparse solutions (few support vectors).

    See [ISLR Section 9.1-9.3](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf) and [ESL Section 12.2](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf).
    """)
    return


@app.cell
def _(np, plt):
    # Hinge loss vs logistic loss: visual comparison
    # Hinge: max(0, 1 - y*f(x)),  Logistic: log(1 + exp(-y*f(x)))
    margins = np.linspace(-3, 3, 200)  # y * f(x) = functional margin
    hinge = np.maximum(0, 1 - margins)
    logistic_loss = np.log(1 + np.exp(-margins))
    zero_one = (margins < 0).astype(float)  # ideal 0-1 loss

    fig_hl, ax_hl = plt.subplots(figsize=(7, 4))
    ax_hl.plot(margins, zero_one, "k--", label="0-1 loss", linewidth=1.5)
    ax_hl.plot(margins, hinge, label="Hinge loss (SVM)", linewidth=2)
    ax_hl.plot(margins, logistic_loss, label="Logistic loss", linewidth=2)
    ax_hl.set_xlabel(r"Functional margin $y \cdot f(x)$")
    ax_hl.set_ylabel("Loss")
    ax_hl.set_title("Hinge loss is zero beyond margin = 1 (sparse support vectors)")
    ax_hl.legend()
    ax_hl.set_ylim(-0.1, 4)
    plt.tight_layout()
    fig_hl
    return


@app.cell
def _(np):
    # RBF kernel: K(x, z) = exp(-gamma ||x - z||^2)
    # It computes inner products in infinite-dimensional space!
    def rbf_kernel(X, gamma=1.0):
        """Compute the RBF (Gaussian) kernel matrix."""
        sq_dists = np.sum(X**2, axis=1, keepdims=True) - 2 * X @ X.T + np.sum(X**2, axis=1)
        return np.exp(-gamma * sq_dists)

    # Small example: 4 points
    X_kern = np.array([[0, 0], [1, 0], [0, 1], [5, 5]])
    K = rbf_kernel(X_kern, gamma=0.5)
    print("RBF kernel matrix (nearby points -> values near 1):")
    print(np.round(K, 3))
    # Note: K[0,1] and K[0,2] are high (nearby), K[0,3] is near 0 (far away)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The Kernel Trick

    This is the big idea — arguably one of the most beautiful ideas in machine learning.

    **The problem:** linear decision boundaries are limited. Many real datasets aren't linearly separable.

    **Naive solution:** map data to a higher-dimensional space $\phi(x)$ where it becomes linearly separable. For example, $\phi(x) = (x_1, x_2, x_1^2, x_2^2, x_1 x_2)$ maps 2D data to 5D, where a linear boundary in 5D corresponds to a quadratic boundary in the original space.

    **The problem with the naive solution:** if you map to a very high-dimensional space (or infinite-dimensional), computing and storing $\phi(x)$ is expensive or impossible.

    **The kernel trick:** notice that the SVM optimization and prediction only depend on **inner products** $\phi(x_i)^\top \phi(x_j)$ between data points, never on $\phi(x)$ alone. A kernel function $K(x_i, x_j) = \phi(x_i)^\top \phi(x_j)$ computes this inner product **without ever computing $\phi$**. You get the power of a high-dimensional feature space at the cost of computing in the original space.

    **Common kernels:**
    - **Linear**: $K(x, z) = x^\top z$ — standard linear SVM
    - **Polynomial**: $K(x, z) = (x^\top z + c)^d$ — degree-$d$ polynomial boundaries
    - **RBF/Gaussian**: $K(x, z) = \exp(-\gamma \|x - z\|^2)$ — corresponds to an infinite-dimensional feature space. This is the most commonly used kernel.

    **Mercer's theorem** tells us that any function $K$ that produces a positive semi-definite Gram matrix can be used as a kernel — it's guaranteed to correspond to some (possibly implicit) feature map. This gives a clean mathematical criterion for valid kernels.

    See [ESL Section 12.3](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf) and [Boyd Section 4.4](file:///C:/Users/landa/ml-course/textbooks/Boyd-ConvexOptimization.pdf) for the convex optimization formulation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Code: SVM with Different Kernels""")
    return


@app.cell
def _(X_lr, y_lr, SVC, StandardScaler):
    # SVMs are sensitive to feature scaling!
    scaler_svm = StandardScaler()
    X_train_svm = scaler_svm.fit_transform(X_lr)

    # Linear SVM
    svm_linear = SVC(kernel='linear', C=1.0)
    svm_linear.fit(X_train_svm, y_lr)

    # RBF kernel SVM
    svm_rbf = SVC(kernel='rbf', C=10.0, gamma='scale')
    svm_rbf.fit(X_train_svm, y_lr)

    print(f"Linear SVM accuracy: {svm_linear.score(X_train_svm, y_lr):.3f}")
    print(f"RBF SVM accuracy: {svm_rbf.score(X_train_svm, y_lr):.3f}")
    print(f"Number of support vectors: {svm_rbf.n_support_}")
    return (scaler_svm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### SVMs vs Logistic Regression

    When to use which?

    | Criterion | Logistic Regression | SVM |
    |---|---|---|
    | Probability estimates | Native (well-calibrated) | Requires Platt scaling |
    | Large datasets | Scales well | Kernel SVMs scale poorly ($O(n^2)$ to $O(n^3)$) |
    | Nonlinear boundaries | Needs manual feature engineering | Kernel trick handles it |
    | Interpretability | Coefficients are log-odds ratios | Support vectors, less interpretable |
    | Sparse data (NLP) | Works well with L1 | Linear SVM works very well |
    | Outlier sensitivity | Moderate | Less sensitive (only support vectors matter) |

    In practice: for large datasets, logistic regression (or linear SVM, which is very similar). For small-to-medium datasets with complex boundaries, kernel SVMs.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ### Interactive: SVM Decision Boundary Explorer

    Use the sliders below to control the SVM's C parameter and kernel type, then observe how the decision boundary changes.
    """)
    return


@app.cell
def _(mo):
    mo.image(src="../animations/rendered/DecisionBoundaries.gif")
    return


@app.cell
def _(mo):
    c_slider = mo.ui.slider(start=-2, stop=3, step=0.1, value=0, label=r"log₁₀(C)")
    kernel_dropdown = mo.ui.dropdown(options=["linear", "rbf", "poly"], value="rbf", label="Kernel")
    mo.hstack([c_slider, kernel_dropdown])
    return (c_slider, kernel_dropdown)


@app.cell
def _(c_slider, kernel_dropdown, np, plt, SVC, StandardScaler, make_moons):
    _C = 10 ** c_slider.value
    _kernel = kernel_dropdown.value

    # Generate dataset
    np.random.seed(42)
    _X, _y = make_moons(n_samples=200, noise=0.2, random_state=42)
    _scaler = StandardScaler()
    _X_scaled = _scaler.fit_transform(_X)

    # Fit SVM
    _svm = SVC(kernel=_kernel, C=_C, gamma='scale')
    _svm.fit(_X_scaled, _y)
    _acc = _svm.score(_X_scaled, _y)
    _n_sv = _svm.n_support_.sum()

    # Create decision boundary mesh
    _x_min, _x_max = _X_scaled[:, 0].min() - 0.5, _X_scaled[:, 0].max() + 0.5
    _y_min, _y_max = _X_scaled[:, 1].min() - 0.5, _X_scaled[:, 1].max() + 0.5
    _xx, _yy = np.meshgrid(np.linspace(_x_min, _x_max, 200),
                            np.linspace(_y_min, _y_max, 200))
    _Z = _svm.predict(np.c_[_xx.ravel(), _yy.ravel()])
    _Z = _Z.reshape(_xx.shape)

    _fig, _ax = plt.subplots(1, 1, figsize=(9, 6))
    _ax.contourf(_xx, _yy, _Z, alpha=0.3, cmap="RdBu")
    _ax.contour(_xx, _yy, _Z, colors="k", linewidths=0.5)
    _ax.scatter(_X_scaled[_y == 0, 0], _X_scaled[_y == 0, 1], c="steelblue", s=30, edgecolors="k", linewidths=0.5, label="Class 0")
    _ax.scatter(_X_scaled[_y == 1, 0], _X_scaled[_y == 1, 1], c="tomato", s=30, edgecolors="k", linewidths=0.5, label="Class 1")

    # Highlight support vectors
    _sv = _svm.support_vectors_
    _ax.scatter(_sv[:, 0], _sv[:, 1], s=100, facecolors='none', edgecolors='gold', linewidths=2, label=f"Support vectors ({_n_sv})")

    _ax.set_xlabel("Feature 1 (scaled)")
    _ax.set_ylabel("Feature 2 (scaled)")
    _ax.set_title(f"SVM  |  Kernel: {_kernel}  |  C = {_C:.3f}  |  Accuracy = {_acc:.3f}")
    _ax.legend(loc="lower left")
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 8. Evaluating Classifiers

    A model is only as good as the metric you evaluate it on. Choosing the right metric is an underappreciated skill.

    ### Accuracy and Its Problems

    $$\text{Accuracy} = \frac{\text{correct predictions}}{\text{total predictions}}$$

    Simple and intuitive, but dangerously misleading with **class imbalance**. If 99% of emails are not spam, a classifier that always predicts "not spam" has 99% accuracy — and is completely useless.

    ### The Confusion Matrix

    For binary classification, the confusion matrix organizes predictions into four categories:

    |  | Predicted Positive | Predicted Negative |
    |---|---|---|
    | **Actual Positive** | True Positive (TP) | False Negative (FN) |
    | **Actual Negative** | False Positive (FP) | True Negative (TN) |

    Every evaluation metric can be derived from these four numbers.

    ### Precision, Recall, and the F1 Score

    $$\text{Precision} = \frac{TP}{TP + FP}$$

    "Of all things I called positive, how many actually were?" High precision means few false alarms.

    $$\text{Recall} = \frac{TP}{TP + FN}$$

    "Of all actual positives, how many did I catch?" High recall means few missed detections.

    **There's always a tradeoff.** You can get perfect recall by predicting everything as positive (but precision tanks). You can get perfect precision by only predicting positive when absolutely certain (but recall tanks).

    The **F1 score** balances both via the harmonic mean:

    $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

    The harmonic mean punishes imbalance: if either precision or recall is low, F1 is low. This is more informative than the arithmetic mean.

    ### ROC Curve and AUC

    The **Receiver Operating Characteristic (ROC)** curve plots True Positive Rate (recall) vs False Positive Rate ($\frac{FP}{FP + TN}$) as you sweep the classification threshold from 0 to 1.

    A perfect classifier hugs the top-left corner. A random classifier follows the diagonal. The **Area Under the Curve (AUC)** summarizes performance in a single number: 1.0 is perfect, 0.5 is random.

    AUC has a nice interpretation: it's the probability that a randomly chosen positive example gets a higher score than a randomly chosen negative example.

    ### When to Use Which Metric

    - **Balanced classes, equal costs:** accuracy is fine
    - **Imbalanced classes:** F1, precision, recall, or AUC
    - **Cost of false positives is high** (e.g., spam filter — don't lose real email): optimize precision
    - **Cost of false negatives is high** (e.g., cancer screening — don't miss a case): optimize recall
    - **Ranking quality matters** (e.g., search results): AUC
    """)
    return


@app.cell
def _(np):
    # Computing precision, recall, F1 from scratch using the confusion matrix
    # Simulated predictions
    y_true_eval = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    y_pred_eval = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 0])

    TP = np.sum((y_pred_eval == 1) & (y_true_eval == 1))
    FP = np.sum((y_pred_eval == 1) & (y_true_eval == 0))
    FN = np.sum((y_pred_eval == 0) & (y_true_eval == 1))
    TN = np.sum((y_pred_eval == 0) & (y_true_eval == 0))

    precision_manual = TP / (TP + FP)
    recall_manual = TP / (TP + FN)
    f1_manual = 2 * precision_manual * recall_manual / (precision_manual + recall_manual)
    accuracy_manual = (TP + TN) / len(y_true_eval)

    print(f"TP={TP}, FP={FP}, FN={FN}, TN={TN}")
    print(f"Precision: {precision_manual:.3f}")
    print(f"Recall:    {recall_manual:.3f}")
    print(f"F1 score:  {f1_manual:.3f}")
    print(f"Accuracy:  {accuracy_manual:.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Code: Evaluation Metrics and ROC Curve""")
    return


@app.cell
def _(X_lr, y_lr, np, plt, LogisticRegression, classification_report, roc_curve, roc_auc_score):
    from sklearn.model_selection import train_test_split as _tts
    np.random.seed(42)
    _X_train, _X_test, _y_train, _y_test = _tts(X_lr, y_lr, test_size=0.3, random_state=42)

    _model = LogisticRegression(C=1.0, solver='lbfgs')
    _model.fit(_X_train, _y_train)

    _y_pred = _model.predict(_X_test)
    _y_prob = _model.predict_proba(_X_test)[:, 1]  # probability of positive class

    # All metrics at once
    print(classification_report(_y_test, _y_pred))

    # ROC curve
    _fpr, _tpr, _thresholds = roc_curve(_y_test, _y_prob)
    _auc = roc_auc_score(_y_test, _y_prob)

    _fig, _ax = plt.subplots(figsize=(7, 5))
    _ax.plot(_fpr, _tpr, label=f'AUC = {_auc:.3f}')
    _ax.plot([0, 1], [0, 1], 'k--', label='Random')
    _ax.set_xlabel('False Positive Rate')
    _ax.set_ylabel('True Positive Rate')
    _ax.set_title('ROC Curve')
    _ax.legend()
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    See [ISLR Section 4.5](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf) for a thorough treatment of classification assessment.

    ---

    ## Summary: The Classification Landscape

    | Method | Type | Key Idea | When to Use |
    |---|---|---|---|
    | Logistic Regression | Discriminative, linear | Model log-odds as linear | Default starting point, large data |
    | LDA | Generative, linear | Gaussian class-conditionals, shared $\Sigma$ | Small data, roughly Gaussian features |
    | Naive Bayes | Generative, linear | Feature independence | Text, very high dimensions, tiny data |
    | SVM (linear) | Discriminative, linear | Maximum margin | Similar to logistic, sparse data |
    | SVM (kernel) | Discriminative, nonlinear | Kernel trick | Complex boundaries, moderate data |

    Notice that the first four all produce linear decision boundaries. The conceptual differences are big, but the practical differences can be small. The kernel SVM is the first method we've seen that can learn genuinely nonlinear boundaries — and in Module 2, neural networks will take this much further.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Key References

    - [ISLR Chapter 4: Classification](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf) — Clear introductions to logistic regression, LDA, and evaluation
    - [ISLR Chapter 9: Support Vector Machines](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf) — Excellent geometric treatment of SVMs
    - [Bishop Chapter 4: Linear Models for Classification](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) — Rigorous probabilistic treatment
    - [ESL Chapter 4: Linear Methods for Classification](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf) — More advanced, connects to optimal scoring
    - [ESL Chapter 12: Support Vector Machines](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf) — Thorough kernel methods treatment
    - [Murphy Chapter 10: Linear Discriminant Analysis](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf) — Modern perspective with connections to deep learning
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Practice Exercises

    **Conceptual:**

    1. Show that the sigmoid function satisfies $\sigma(-z) = 1 - \sigma(z)$. Why does this property make sense probabilistically?

    2. Derive the gradient of the binary cross-entropy loss. Start from $\text{NLL} = -\sum[y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)]$ and use $\hat{p}_i = \sigma(w^\top x_i)$. Verify you get $X^\top(\hat{p} - y)$.

    3. Explain geometrically why the shared covariance assumption in LDA produces linear decision boundaries, while unequal covariances (QDA) produce quadratic boundaries.

    4. Why do SVMs only depend on support vectors? Hint: think about the KKT complementary slackness conditions from constrained optimization.

    5. Construct a 2D dataset where Naive Bayes with the independence assumption gives the wrong decision boundary. Then construct one where it gives the right boundary despite the assumption being wrong.

    **Computational:**

    6. Load the `sklearn.datasets.load_breast_cancer` dataset. Compare logistic regression, LDA, Naive Bayes, linear SVM, and RBF SVM using 5-fold cross-validation. Report accuracy, F1, and AUC for each. Which method wins? Why?

    7. Implement logistic regression from scratch using gradient descent. Compare your convergence to `sklearn`'s implementation on a synthetic dataset. Plot the loss curve.

    8. Generate a 2D dataset with two concentric circles (`sklearn.datasets.make_circles`). Show that linear methods fail. Then use an RBF kernel SVM and plot the nonlinear decision boundary. Also try manually constructing a feature map $\phi(x) = (x_1, x_2, x_1^2 + x_2^2)$ and using a linear classifier in the transformed space.

    9. Build a spam classifier using Naive Bayes on the 20 Newsgroups dataset. Compute the confusion matrix. Which categories get confused most often? Does this make intuitive sense?

    10. For the breast cancer dataset, plot the ROC curves of all five classifiers on the same axes. Which has the highest AUC? At a false positive rate of 5%, which classifier has the best recall?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Code It: Implementation Exercises

    Now it is your turn. Implement the core algorithms from scratch. Each exercise gives you a problem statement and starter code -- fill in the missing parts.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 1: Implement the Sigmoid and its Gradient

    Write a numerically stable sigmoid function and verify that its derivative equals $\sigma(z)(1 - \sigma(z))$ by comparing to a finite-difference approximation.
    """)
    return


@app.cell
def _(np):
    def my_sigmoid(z):
        # TODO: implement sigmoid with numerical stability
        # Hint: for large negative z, exp(-z) overflows. Use np.clip or
        # the identity sigma(z) = exp(z)/(1+exp(z)) for z < 0
        pass

    def my_sigmoid_grad(z):
        # TODO: implement sigma'(z) = sigma(z) * (1 - sigma(z))
        pass

    # Test your implementation
    z_exercise = np.array([-100, -1, 0, 1, 100])
    # print("sigmoid:", my_sigmoid(z_exercise))
    # print("gradient:", my_sigmoid_grad(z_exercise))

    # Verify gradient with finite differences
    # eps_fd = 1e-7
    # numerical_grad = (my_sigmoid(z_exercise + eps_fd) - my_sigmoid(z_exercise - eps_fd)) / (2 * eps_fd)
    # print("numerical grad:", numerical_grad)
    # print("match:", np.allclose(my_sigmoid_grad(z_exercise), numerical_grad))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 2: Binary Cross-Entropy Loss from Scratch

    Implement the binary cross-entropy loss and show it penalizes confident wrong predictions more than uncertain ones.
    """)
    return


@app.cell
def _(np):
    def my_cross_entropy(y_true_ce, p_hat_ce):
        # TODO: implement NLL = -mean[ y*log(p) + (1-y)*log(1-p) ]
        # Hint: clip p_hat to avoid log(0)
        pass

    # Test: compare loss for confident-correct vs confident-wrong
    # y_test_ce = np.array([1, 1, 0, 0])
    # p_good = np.array([0.95, 0.9, 0.1, 0.05])   # good predictions
    # p_bad  = np.array([0.05, 0.1, 0.9, 0.95])    # terrible predictions
    # print(f"Loss (good predictions): {my_cross_entropy(y_test_ce, p_good):.4f}")
    # print(f"Loss (bad predictions):  {my_cross_entropy(y_test_ce, p_bad):.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 3: Logistic Regression with Gradient Descent

    Implement full logistic regression training from scratch. Use gradient descent with the update rule $w \leftarrow w - \eta \cdot X^T(\hat{p} - y) / n$.
    """)
    return


@app.cell
def _(np, plt, make_moons):
    def fit_logistic_regression_gd(X_fit, y_fit, lr_fit=0.1, n_epochs=200):
        """Train logistic regression via gradient descent. Return weights and loss history."""
        n, d = X_fit.shape
        w_fit = np.zeros(d)
        losses = []

        for _ in range(n_epochs):
            # TODO: 1. compute predictions p_hat = sigmoid(X @ w)
            # TODO: 2. compute cross-entropy loss, append to losses
            # TODO: 3. compute gradient = X^T (p_hat - y) / n
            # TODO: 4. update w = w - lr * gradient
            pass

        return w_fit, losses

    # Generate data and test your implementation
    # np.random.seed(42)
    # X_ex, y_ex = make_moons(n_samples=200, noise=0.2, random_state=42)
    # X_ex_bias = np.column_stack([np.ones(len(X_ex)), X_ex])  # add bias column
    # w_learned, loss_curve = fit_logistic_regression_gd(X_ex_bias, y_ex, lr_fit=0.5, n_epochs=300)

    # Plot convergence
    # fig_ex, ax_ex = plt.subplots(figsize=(6, 3))
    # ax_ex.plot(loss_curve)
    # ax_ex.set_xlabel("Epoch"); ax_ex.set_ylabel("Loss")
    # ax_ex.set_title("Training loss"); plt.tight_layout()
    # fig_ex
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 4: Softmax from Scratch

    Implement the softmax function and categorical cross-entropy loss for a 3-class problem. Verify that probabilities sum to 1 and that the gradient has the right shape.
    """)
    return


@app.cell
def _(np):
    def my_softmax(z_sm):
        # TODO: implement softmax with numerical stability (subtract max)
        # z_sm shape: (n_samples, K) or (K,)
        # Return: same shape, rows sum to 1
        pass

    def categorical_cross_entropy(y_classes, probs_cce):
        # TODO: NLL = -mean[ log(probs[i, y[i]]) ] for each sample i
        # y_classes: integer labels (n,), probs_cce: (n, K)
        pass

    # Test
    # logits_ex = np.array([[2.0, 1.0, 0.1], [-1.0, 2.0, 0.5]])
    # probs_ex = my_softmax(logits_ex)
    # print("Probabilities:\n", probs_ex)
    # print("Row sums:", probs_ex.sum(axis=1))
    # y_ex_mc = np.array([0, 1])
    # print("Cat. cross-entropy:", categorical_cross_entropy(y_ex_mc, probs_ex))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 5: The Kernel Trick -- Manual Feature Map vs RBF Kernel

    For `make_circles` data (not linearly separable), show that:
    1. A linear classifier fails
    2. Manually adding $x_1^2 + x_2^2$ as a feature makes it linearly separable
    3. An RBF kernel SVM solves it without manual feature engineering
    """)
    return


@app.cell
def _(np, plt, make_circles, SVC, LogisticRegression):
    # Generate concentric circles
    X_circ, y_circ = make_circles(n_samples=200, noise=0.1, factor=0.4, random_state=42)

    # TODO 1: Fit a linear logistic regression -- check accuracy
    # lin_model = LogisticRegression(...)
    # print(f"Linear accuracy: {lin_model.score(X_circ, y_circ):.3f}")

    # TODO 2: Add radial feature phi(x) = x1^2 + x2^2, fit linear model
    # r = (X_circ[:, 0]**2 + X_circ[:, 1]**2).reshape(-1, 1)
    # X_circ_aug = np.hstack([X_circ, r])
    # aug_model = LogisticRegression(...)
    # print(f"Augmented accuracy: {aug_model.score(X_circ_aug, y_circ):.3f}")

    # TODO 3: RBF kernel SVM (no manual features needed)
    # rbf_model = SVC(kernel='rbf', ...)
    # print(f"RBF SVM accuracy: {rbf_model.score(X_circ, y_circ):.3f}")

    # Scatter plot to visualize the data
    fig_circ, ax_circ = plt.subplots(figsize=(5, 5))
    ax_circ.scatter(X_circ[y_circ == 0, 0], X_circ[y_circ == 0, 1], s=20, label="Class 0")
    ax_circ.scatter(X_circ[y_circ == 1, 0], X_circ[y_circ == 1, 1], s=20, label="Class 1")
    ax_circ.set_title("Concentric circles -- not linearly separable")
    ax_circ.legend()
    ax_circ.set_aspect("equal")
    plt.tight_layout()
    fig_circ
    return


if __name__ == "__main__":
    app.run()
