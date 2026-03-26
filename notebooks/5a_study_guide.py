import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Complete Algorithm Study Guide

    Every algorithm in the course — grouped by how they relate to each other, with strengths, weaknesses, equations, and connections.

    Use the filters below to focus on what you need.
    """)
    return


@app.cell
def _(mo):
    section_filter = mo.ui.dropdown(
        options={
            "All Sections": "all",
            "1. Statistical Foundations": "foundations",
            "2. Linear Models": "linear",
            "3. Classifiers (Discriminative vs Generative)": "classifiers",
            "4. Evaluation & Model Selection": "evaluation",
            "5. Trees & Ensembles": "trees",
            "6. Clustering": "clustering",
            "7. Dimensionality Reduction": "dimreduce",
            "8. Neural Networks & Backprop": "nn",
            "9. DL Optimization": "dlopt",
            "10. DL Regularization": "dlreg",
            "11. CNNs": "cnn",
            "12. Sequence Models": "seq",
            "13. Transformers": "transformers",
            "14. Generative Models": "generative",
            "15. Self-Supervised Learning": "ssl",
            "16. Reinforcement Learning": "rl",
            "17. NLP": "nlp",
            "18. Computer Vision": "cv",
            "19. Bayesian ML": "bayesian",
            "20. When to Use What": "decision",
        },
        value="All Sections",
        label="Section",
    )

    type_filter = mo.ui.dropdown(
        options={
            "All Types": "all",
            "Supervised": "supervised",
            "Unsupervised": "unsupervised",
            "Statistical / Foundational": "statistical",
            "Deep Learning": "deep",
            "Reinforcement Learning": "rl",
            "Evaluation / Selection": "eval",
        },
        value="All Types",
        label="Algorithm Type",
    )

    mo.hstack([section_filter, type_filter], justify="start", gap=1)
    return (section_filter, type_filter)


@app.cell(hide_code=True)
def _(mo, section_filter, type_filter):
    sec = section_filter.value
    typ = type_filter.value

    sections = []

    # ===================== 1. FOUNDATIONS =====================
    if sec in ("all", "foundations") and typ in ("all", "statistical"):
        sections.append(mo.md(r"""
---

## 1. Statistical Foundations: Means, Variances, and Estimation

These concepts are the atoms everything else is built from.

### Sample Mean: $\bar{x} = \frac{1}{n}\sum x_i$

| | |
|---|---|
| **What** | Average of observed values. Simplest estimator of the true mean. |
| **Strengths** | Unbiased, consistent, intuitive. |
| **Weaknesses** | Sensitive to outliers. One extreme value can shift it heavily. |
| **Used by** | OLS (target prediction), K-Means (centroid update), Batch Norm (batch mean), PCA (centering), Gradient Boosting ($F_0$). |

### Sample Variance: $s^2 = \frac{1}{n-1}\sum(x_i - \bar{x})^2$

| | |
|---|---|
| **What** | Spread around the mean. Bessel's correction ($n{-}1$) makes it unbiased. |
| **Strengths** | Unbiased estimator of true variance. |
| **Weaknesses** | Sensitive to outliers (squared). Biased version (divide by $n$) is the MLE. |
| **Used by** | PCA (eigenvalues), Batch Norm (normalize by std), Bias-Variance Decomposition. |

### Covariance & Covariance Matrix

$$\text{Cov}(X,Y) = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)], \qquad S = \frac{1}{n-1}X^\top X$$

| | |
|---|---|
| **What** | How variables move together. Matrix form captures all pairwise relationships. |
| **Strengths** | Foundation for PCA, LDA, GPs, all Gaussian models. |
| **Weaknesses** | Only captures linear relationships. $O(p^2)$ storage. |

### Bias-Variance Decomposition

$$\text{MSE} = \text{Bias}^2 + \text{Variance} + \sigma^2$$

| | |
|---|---|
| **What** | Error = systematic error + sensitivity to data + irreducible noise. |
| **Key insight** | Complex model → low bias, high variance. Simple model → high bias, low variance. |
| **Connects to** | Regularization (↑bias, ↓variance), Ensembles (bagging ↓variance, boosting ↓bias). |

### Maximum Likelihood Estimation (MLE)

$$\hat{\theta} = \arg\max \sum_i \log p(x_i \mid \theta)$$

| | |
|---|---|
| **Strengths** | Consistent, asymptotically efficient, equivariant. |
| **Weaknesses** | Overfits with small samples. Degenerate solutions possible (GMM collapse). |
| **Used by** | OLS (MLE under Gaussian noise), Logistic Regression (Bernoulli MLE), GMMs (EM maximizes likelihood). |

### MAP Estimation

$$\hat{\theta} = \arg\max [\log p(D|\theta) + \log p(\theta)]$$

| | |
|---|---|
| **What** | MLE + prior. Point estimate at posterior mode. |
| **Key insight** | Gaussian prior → Ridge. Laplace prior → Lasso. |
| **Weaknesses** | Still a point estimate — no uncertainty quantification. |
        """))

    # ===================== 2. LINEAR MODELS =====================
    if sec in ("all", "linear") and typ in ("all", "supervised", "statistical"):
        sections.append(mo.md(r"""
---

## 2. Linear Prediction Models

All share the form $y = \mathbf{w}^\top \mathbf{x}$. Differ in loss function and output.

### Ordinary Least Squares (OLS)

$$\mathbf{w}^* = (X^\top X)^{-1} X^\top y$$

| | |
|---|---|
| **Loss** | $\|y - Xw\|^2$ |
| **Strengths** | Closed-form. Geometric (orthogonal projection). Convex. = MLE under Gaussian noise. |
| **Weaknesses** | $O(p^3)$ inversion. Overfits when $p$ large. Multicollinearity. Outlier-sensitive. |
| **Use when** | $p \ll n$, features not too correlated, interpretability matters. |

### Ridge Regression (L2)

$$\mathbf{w}^* = (X^\top X + \lambda I)^{-1} X^\top y$$

| | |
|---|---|
| **Loss** | $\|y - Xw\|^2 + \lambda \|w\|^2$ |
| **Strengths** | Always invertible. Handles multicollinearity. Bayesian: Gaussian prior. |
| **Weaknesses** | Keeps all features (no sparsity). Must tune $\lambda$. |
| **Use when** | Correlated features, want to keep all but control magnitude. |

### Lasso (L1)

| | |
|---|---|
| **Loss** | $\|y - Xw\|^2 + \lambda \|w\|_1$ |
| **Strengths** | Drives weights to **exactly zero** — automatic feature selection. Diamond constraint geometry. |
| **Weaknesses** | No closed form. Picks arbitrarily among correlated features. |
| **Use when** | Many irrelevant features, want automatic selection. |

### Elastic Net (L1 + L2)

| | |
|---|---|
| **Loss** | $\|y - Xw\|^2 + \lambda_1 \|w\|_1 + \lambda_2 \|w\|^2$ |
| **Strengths** | Sparsity from L1 + correlated feature handling from L2. |
| **Weaknesses** | Two hyperparameters. |

### Polynomial Regression

| | |
|---|---|
| **Model** | $y = w^\top \phi(x)$ where $\phi(x) = [1, x, x^2, \ldots, x^d]$ |
| **Key insight** | Nonlinear in $x$ but **still linear in $w$** — same OLS math applies. |
| **Weaknesses** | Overfits for high $d$. Oscillation at edges. |

### Logistic Regression

$$P(y{=}1 \mid x) = \sigma(w^\top x) = \frac{1}{1 + e^{-w^\top x}}$$

| | |
|---|---|
| **Loss** | Binary cross-entropy: $-\sum[y\log\hat{y} + (1{-}y)\log(1{-}\hat{y})]$ |
| **Gradient** | $X^\top(\sigma(Xw) - y)$ — same form as OLS! |
| **Strengths** | Calibrated probabilities. Convex. Interpretable (log-odds). |
| **Weaknesses** | Linear boundary only. No closed form. |
| **Use when** | Baseline for any classification task. |

### Softmax Regression

$$P(y{=}k \mid x) = \frac{e^{w_k^\top x}}{\sum_j e^{w_j^\top x}}$$

| | |
|---|---|
| **Loss** | Categorical cross-entropy. |
| **Key insight** | Generalizes sigmoid to $K$ classes. For $K{=}2$, reduces to logistic. |
        """))

    # ===================== 3. CLASSIFIERS =====================
    if sec in ("all", "classifiers") and typ in ("all", "supervised"):
        sections.append(mo.md(r"""
---

## 3. Discriminative vs. Generative Classifiers

**Discriminative** (learn $P(y|x)$ directly): Logistic Regression, SVM, Neural Nets
— More accurate with enough data. Can't generate.

**Generative** (learn $P(x|y)$ and $P(y)$, use Bayes): LDA, Naive Bayes, GMMs
— Better with small data. Can generate. Reach lower error faster with limited data.

### Linear Discriminant Analysis (LDA)

| | |
|---|---|
| **Assumes** | $P(x \mid y{=}k) \sim \mathcal{N}(\mu_k, \Sigma)$ — shared covariance, different means. |
| **Boundary** | $\delta_k(x) = x^\top \Sigma^{-1}\mu_k - \tfrac{1}{2}\mu_k^\top\Sigma^{-1}\mu_k + \log\pi_k$ |
| **Strengths** | Works with small $n$ (Gaussian assumption regularizes). Fisher's view: max between/within variance ratio. |
| **Weaknesses** | Assumes shared covariance and Gaussian classes. |

### Naive Bayes

| | |
|---|---|
| **Assumes** | $P(x \mid y{=}k) = \prod_j P(x_j \mid y{=}k)$ — conditional independence. |
| **Strengths** | $O(n)$ training. Fast. Works with tiny data. Surprisingly effective. |
| **Weaknesses** | Independence assumption almost always wrong. Probabilities poorly calibrated. |
| **Use when** | Text, high-dimensional sparse data, speed critical. |

### Support Vector Machines (SVM)

$$\min \tfrac{1}{2}\|w\|^2 + C\sum \xi_i \quad \text{s.t.} \quad y_i(w^\top x_i + b) \geq 1 - \xi_i$$

| | |
|---|---|
| **Loss** | Hinge: $\max(0, 1 - y \cdot f(x))$ |
| **Strengths** | Maximum margin. Kernel trick (linear, polynomial, RBF=infinite-dim). Only support vectors matter. |
| **Weaknesses** | No native probabilities. $O(n^2)$–$O(n^3)$ training. Kernel + $C$ tuning. |
| **Use when** | Medium $n$, clear margin, kernel captures structure. |
        """))

    # ===================== 4. EVALUATION =====================
    if sec in ("all", "evaluation") and typ in ("all", "eval"):
        sections.append(mo.md(r"""
---

## 4. Evaluation Metrics and Model Selection

### Regression Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| **MSE** | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Penalizes large errors heavily |
| **RMSE** | $\sqrt{\text{MSE}}$ | Same units as $y$ |
| **MAE** | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | Robust to outliers |
| **R²** | $1 - SS_{res}/SS_{tot}$ | Fraction of variance explained. Can be negative on test. |

### Classification Metrics

| Metric | Formula | Use when |
|--------|---------|----------|
| **Precision** | $\text{TP}/(\text{TP}+\text{FP})$ | False positives costly |
| **Recall** | $\text{TP}/(\text{TP}+\text{FN})$ | Missing positives costly |
| **F1** | $2PR/(P+R)$ | Imbalanced data |
| **AUC-ROC** | Area under TPR vs FPR | Overall ranking quality |

### Cross-Validation

| Method | How | Trade-off |
|--------|-----|-----------|
| **k-Fold** | Each fold is validation once, average $k$ scores | Standard: $k{=}5$ or $10$ |
| **Stratified** | Preserves class proportions | Critical for imbalanced data |
| **LOOCV** | $k{=}n$ | Lowest bias, highest variance, very slow |
| **Nested CV** | Outer evaluates, inner tunes | Only unbiased pipeline estimate |

### Hyperparameter Tuning

| Method | Strengths | Weaknesses |
|--------|-----------|------------|
| **Grid Search** | Exhaustive | Exponential cost |
| **Random Search** | More unique values per dim | Not exhaustive |
| **Bayesian Optimization** | Sample-efficient | Complex |

### Information Criteria

| | Formula | Behavior |
|---|---------|----------|
| **AIC** | $2k - 2\ln\hat{L}$ | Lighter penalty |
| **BIC** | $k\ln n - 2\ln\hat{L}$ | Heavier for large $n$, simpler models |

### Bootstrap & OOB

- Sample $n$ with replacement → ~63.2% unique. The ~36.8% out-of-bag = free validation.
        """))

    # ===================== 5. TREES =====================
    if sec in ("all", "trees") and typ in ("all", "supervised"):
        sections.append(mo.md(r"""
---

## 5. Decision Trees and Tree Ensembles

Each builds on the previous one's weakness.

### Decision Tree (CART)

| | |
|---|---|
| **How** | Greedily pick feature + threshold that most reduces impurity. Recurse. |
| **Gini** | $G = 1 - \sum p_k^2$ |
| **Entropy** | $H = -\sum p_k \log p_k$ |
| **Pruning** | $C_\alpha(T) = \text{loss} + \alpha \cdot |\text{leaves}|$ |
| **Strengths** | Interpretable. Mixed types. No scaling. Captures interactions. |
| **Weaknesses** | **High variance.** Axis-aligned only. Overfits without pruning. |

### Bagging

| | |
|---|---|
| **How** | $B$ bootstrap samples → full trees → average. |
| **Variance** | $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$ — correlation $\rho$ limits gains. |
| **Weakness** | Trees correlated (strong features dominate). |

### Random Forest

| | |
|---|---|
| **Innovation** | Random $m$ features at each split → decorrelation. |
| **Defaults** | $m = \sqrt{p}$ (classification), $p/3$ (regression). |
| **Strengths** | More trees never overfits. OOB error. Feature importance. Robust. |
| **Weaknesses** | Less interpretable. Doesn't reduce bias. |

### Gradient Boosting

| | |
|---|---|
| **How** | $F_m = F_{m-1} + \eta \cdot h_m(x)$ where $h_m$ fits pseudo-residuals. |
| **Key params** | $\eta \in [0.01, 0.1]$, depth 3–8, early stopping. |
| **Strengths** | Reduces bias. SOTA for tabular data. Any differentiable loss. |
| **Weaknesses** | **Can overfit** (unlike RF). Sequential. More tuning. |

### XGBoost / LightGBM / CatBoost

| Library | Key Feature |
|---------|-------------|
| **XGBoost** | $\gamma|T| + \lambda\sum w_j^2$ regularization |
| **LightGBM** | Histogram splits (~256 bins), leaf-wise growth, fastest |
| **CatBoost** | Native categorical handling |

### Stacking

| | |
|---|---|
| **How** | Diverse base models → out-of-fold predictions → meta-model. |
| **Gains** | 1–3% marginal. Common in competitions. |

### Bias-Variance Summary

| Method | Reduces | Mechanism |
|--------|---------|-----------|
| Bagging/RF | **Variance** | Average decorrelated models |
| Boosting | **Bias** | Sequential error correction |
| Stacking | **Both** | Meta-model learns combination |
        """))

    # ===================== 6. CLUSTERING =====================
    if sec in ("all", "clustering") and typ in ("all", "unsupervised"):
        sections.append(mo.md(r"""
---

## 6. Clustering

### K-Means

| | |
|---|---|
| **Objective** | $\min J = \sum_k \sum_{x \in C_k} \|x - \mu_k\|^2$ |
| **Algorithm** | Assign → update centroids → repeat. |
| **K-Means++ init** | Choose centroids with prob $\propto D(x)^2$. |
| **Strengths** | Fast $O(nK)$. Simple. Spherical clusters. |
| **Weaknesses** | Specify $K$. Local optima. Spherical assumption. Outlier-sensitive. |
| **Connection** | K-Means = EM on GMM with $\Sigma_k = \sigma^2 I$, $\sigma \to 0$. |

### GMM + EM

| | |
|---|---|
| **Model** | $p(x) = \sum_k \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)$ |
| **E-step** | Responsibilities $r_{ik}$ = posterior prob of cluster $k$ for point $i$ (Bayes). |
| **M-step** | Update $\pi_k, \mu_k, \Sigma_k$ using responsibilities as weights. |
| **Strengths** | Soft assignments. Different shapes/sizes. Probabilistic. BIC for model selection. |
| **Weaknesses** | Slower. Degenerate solutions. Needs $K$. |

### Hierarchical (Agglomerative)

| Linkage | Behavior |
|---------|----------|
| **Single** | Elongated, prone to chaining |
| **Complete** | Compact, outlier-sensitive |
| **Average** | Compromise |
| **Ward's** | Minimizes within-cluster variance (K-Means-like) |

| | |
|---|---|
| **Strengths** | Dendrogram. Don't need $K$. Any distance metric. |
| **Weaknesses** | $O(n^2)$–$O(n^3)$. Greedy. Sensitive to linkage. |

### Silhouette Score

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))} \in [-1, +1]$$

Near +1 = well clustered. Near 0 = boundary. Negative = misassigned.
        """))

    # ===================== 7. DIM REDUCTION =====================
    if sec in ("all", "dimreduce") and typ in ("all", "unsupervised"):
        sections.append(mo.md(r"""
---

## 7. Dimensionality Reduction

### PCA

| | |
|---|---|
| **How** | Eigendecomposition of $S$. Eigenvectors = PCs. Eigenvalues = variance. |
| **Equivalent** | SVD of centered data: $X = U\Sigma V^\top$, $V$ columns = PCs. |
| **Strengths** | Optimal linear DR. Fast. Denoising (discard low-variance). |
| **Weaknesses** | Linear only. Assumes variance = importance. Scale-sensitive. |

### t-SNE

| | |
|---|---|
| **How** | Gaussian similarities (high-D) vs Student-t (low-D). Minimize $KL(P \| Q)$. |
| **Strengths** | Excellent visualization. Preserves local neighborhoods. |
| **Weaknesses** | **Visualization only** — don't use as features. Non-deterministic. Between-cluster distances meaningless. $O(n^2)$. |

### UMAP

| | |
|---|---|
| **Strengths** | Faster. More global structure. Arbitrary output dims. General DR. |
| **Preferred over t-SNE** in most cases. |

### Isolation Forest

| | |
|---|---|
| **How** | Random splits. Anomalies isolated in fewer splits (shorter path). |
| **Use** | Outlier / anomaly detection in tabular data. |
        """))

    # ===================== 8. NEURAL NETS =====================
    if sec in ("all", "nn") and typ in ("all", "deep"):
        sections.append(mo.md(r"""
---

## 8. Neural Networks and Backpropagation

### Activation Functions

| Activation | Formula | Use |
|-----------|---------|-----|
| **ReLU** | $\max(0, z)$ | Default hidden layer |
| **Leaky ReLU** | $\max(0.01z, z)$ | Dead neuron fix |
| **GELU** | $z \cdot \Phi(z)$ | Transformers |
| **Sigmoid** | $1/(1+e^{-z})$ | Binary output |
| **Tanh** | $(e^z - e^{-z})/(e^z + e^{-z})$ | RNN hidden |
| **Softmax** | $e^{z_k} / \sum e^{z_j}$ | Multiclass output |

### Feedforward Network (MLP)

$$z^{(j)} = W^{(j)} h^{(j-1)} + b^{(j)}, \quad h^{(j)} = \sigma(z^{(j)})$$

- **Universal Approximation:** 1 hidden layer can approximate any continuous function (enough neurons). But depth is exponentially more efficient.

### Backpropagation

- Chain rule through computational graph. Reverse-mode autodiff: **one backward pass** computes all parameter gradients.
- Cost: ~2–3× forward pass.

### Loss Functions

| Loss | Use |
|------|-----|
| **MSE** | Regression |
| **Huber** | Regression (robust) |
| **Binary CE** | Binary classification |
| **Categorical CE** | Multiclass classification |
| **Hinge** | SVM-style |
        """))

    # ===================== 9. DL OPTIMIZATION =====================
    if sec in ("all", "dlopt") and typ in ("all", "deep"):
        sections.append(mo.md(r"""
---

## 9. Deep Learning Optimization

### Optimizers

| Optimizer | Update Rule | Key Property |
|-----------|------------|--------------|
| **SGD** | $\theta \leftarrow \theta - \eta g$ | Simple, noise helps generalization |
| **Momentum** | $v = \beta v + g; \; \theta \leftarrow \theta - \eta v$ | Accumulates velocity, dampens oscillation |
| **Adam** | 1st moment (momentum) + 2nd moment (RMSProp) + bias correction | Adaptive per-parameter LR, fast convergence |
| **AdamW** | Adam with **decoupled** weight decay | Proper L2 behavior. **Modern default.** |

**Adam vs AdamW:** In Adam, L2 penalty gets divided by $\sqrt{v_t}$ — heavily-updated params get *less* regularization (opposite of intent). AdamW fixes this.

### Learning Rate Schedules

| Schedule | How |
|----------|-----|
| **Warmup** | Linear ramp for first ~1000 steps |
| **Cosine annealing** | Half-cosine from max to min (modern default) |
| **One-cycle** | Low → high → very low (excellent empirical results) |
| **Step decay** | Multiply by 0.1 at fixed epochs |

### Batch Size

| Small (32–256) | Large (4K–64K) |
|----------------|----------------|
| More noise, better generalization | Less noise, better GPU utilization |
| Underutilize GPU | May find sharp minima |

**Linear scaling rule:** batch size × $k$ → LR × $k$.
        """))

    # ===================== 10. DL REGULARIZATION =====================
    if sec in ("all", "dlreg") and typ in ("all", "deep"):
        sections.append(mo.md(r"""
---

## 10. Deep Learning Regularization

### Weight Decay (L2)

$\theta \leftarrow (1 - \eta\lambda)\theta - \eta \nabla L$ — weights shrink by $(1 - \eta\lambda)$ each step. Smooths loss landscape. Biases typically not regularized.

### Dropout

| | |
|---|---|
| **Training** | Zero neurons with prob $p$. Scale remaining by $1/(1{-}p)$. |
| **Test** | Off (use all neurons). |
| **Why** | Ensemble of $2^n$ sub-networks. Forces redundant representations. Bayesian connection (MC Dropout → uncertainty). |
| **Typical** | Hidden: 0.5. Input: 0.1–0.2. Transformers: 0.1–0.3. |

### Early Stopping

Monitor validation loss → save best checkpoint → stop after $k$ epochs of no improvement. Return **best checkpoint**, not final model. Approximately equivalent to L2 regularization.

### Batch Normalization

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}}, \quad y_i = \gamma\hat{x}_i + \beta$$

| | |
|---|---|
| **Strengths** | Smooths landscape. Higher LR. Acts as regularizer (batch noise). |
| **Weaknesses** | Train ≠ test behavior. Bad for small batches, RNNs. |
| **Variants** | Layer Norm (Transformers), Group Norm (small batch), Instance Norm (style transfer). |

### Data Augmentation

- **Image:** Flips, rotations, crops, color jitter, Cutout, Mixup, CutMix.
- **Text:** Synonym replacement, back-translation.
- **Rule:** Transformation must preserve the label.

### Label Smoothing

Replace $[0, 1]$ with $[0.05, 0.95]$. Prevents overconfidence. Improves calibration.

### Double Descent

Test error decreases → increases (classical) → **decreases again** past interpolation threshold. Very large models generalize well. Challenges classical bias-variance.
        """))

    # ===================== 11. CNNs =====================
    if sec in ("all", "cnn") and typ in ("all", "deep"):
        sections.append(mo.md(r"""
---

## 11. Convolutional Neural Networks

### Core Ideas

- **Local connectivity** (not fully connected) + **weight sharing** (same kernel everywhere) = translation equivariance.
- Parameters: kernel size, stride, padding, dilation.

### Pooling

| Type | How | Use |
|------|-----|-----|
| **Max pool** | Max in window | Small translation invariance, downsample |
| **Avg pool** | Mean in window | Less common |
| **Strided conv** | Learned downsampling | Modern alternative |

### Landmark Architectures

| Architecture | Year | Innovation |
|-------------|------|------------|
| **AlexNet** | 2012 | ReLU, dropout, GPU training |
| **VGGNet** | 2014 | Uniform 3×3, simplicity |
| **GoogLeNet** | 2014 | Inception: parallel filter sizes |
| **ResNet** | 2015 | Skip connections: $y = F(x) + x$ |
| **EfficientNet** | 2019 | Compound scaling |

### Transfer Learning

| Strategy | How | When |
|----------|-----|------|
| **Feature extraction** | Freeze backbone, train new head | Small target dataset |
| **Fine-tuning** | Unfreeze layers, low LR | More target data |

Early layers learn universal features (edges, textures) — transfer across domains.
        """))

    # ===================== 12. SEQUENCE MODELS =====================
    if sec in ("all", "seq") and typ in ("all", "deep"):
        sections.append(mo.md(r"""
---

## 12. Sequence Models

### Vanilla RNN

$$h_t = \sigma(W_{hh}h_{t-1} + W_{xh}x_t + b)$$

Fatal weakness: **vanishing/exploding gradients** — $\partial h_t/\partial h_s$ involves $W_{hh}^{t-s}$.

### LSTM

| Gate | Controls |
|------|----------|
| **Forget** $f_t$ | What to discard from cell state |
| **Input** $i_t$ | What new info to store |
| **Output** $o_t$ | What to expose as hidden state |

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

**Key:** Additive cell update (not multiplicative) → gradients flow.

### GRU

2 gates (reset + update) instead of 3. Fewer params, similar performance, faster.

### Bidirectional RNNs

Forward + backward → concatenate. Full context, but can't do real-time generation.

### Seq2Seq Bottleneck → Attention

Encoder compresses to single vector → **attention** lets decoder look at all encoder states → leads directly to Transformers.
        """))

    # ===================== 13. TRANSFORMERS =====================
    if sec in ("all", "transformers") and typ in ("all", "deep"):
        sections.append(mo.md(r"""
---

## 13. Transformers

### Scaled Dot-Product Attention

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

$\sqrt{d_k}$ prevents softmax saturation.

### Multi-Head Attention

$h$ parallel heads with own $Q/K/V$ projections → concat → project. Different heads capture different relationships.

### Architecture

| Component | Role |
|-----------|------|
| **Self-attention** | Every position attends to all others |
| **Masked self-attention** | Decoder: prevent attending to future |
| **Cross-attention** | Decoder attends to encoder outputs |
| **Position-wise FFN** | Same 2-layer MLP at each position |
| **Positional encoding** | Inject sequence order (sinusoidal or learned) |
| **Residual + LayerNorm** | Gradient flow, stability |

### Why Transformers Won

- **Parallel:** All positions at once (RNNs are sequential).
- **Long-range:** Direct connections (no vanishing gradient over distance).
- **Scalable:** Performance improves predictably with compute/data/params.
        """))

    # ===================== 14. GENERATIVE =====================
    if sec in ("all", "generative") and typ in ("all", "deep"):
        sections.append(mo.md(r"""
---

## 14. Generative Models

### VAE (Variational Autoencoder)

| | |
|---|---|
| **Loss (ELBO)** | Reconstruction + $\text{KL}(q(z \mid x) \| \mathcal{N}(0,I))$ |
| **Reparameterization** | $z = \mu + \sigma \odot \varepsilon$ makes sampling differentiable |
| **Strengths** | Principled. Smooth latent space. Can sample. |
| **Weaknesses** | **Blurry** (MSE averages plausible outputs). Posterior collapse. |

### GAN (Generative Adversarial Network)

| | |
|---|---|
| **How** | Generator fools discriminator. Minimax game. |
| **Strengths** | Sharp outputs. No explicit density. |
| **Weaknesses** | Unstable training. Mode collapse. Hard to monitor. |

### Diffusion Models

| | |
|---|---|
| **How** | Forward: add noise over $T$ steps. Reverse: learn to denoise. |
| **Training** | $\|\varepsilon - \varepsilon_\theta(x_t, t)\|^2$ |
| **Strengths** | Stable training. SOTA image quality. Principled. |
| **Weaknesses** | Slow sampling. High compute. |
| **Status** | Current SOTA (DALL-E, Stable Diffusion, Imagen). |
        """))

    # ===================== 15. SSL =====================
    if sec in ("all", "ssl") and typ in ("all", "deep"):
        sections.append(mo.md(r"""
---

## 15. Self-Supervised Learning

| Method | Task | Context | Use |
|--------|------|---------|-----|
| **GPT-style** | Predict next token (autoregressive) | Unidirectional | Generation, in-context learning |
| **BERT-style** | Predict masked tokens | Bidirectional | Classification, understanding |
| **SimCLR/MoCo** | Pull augmented views together, push others apart | Contrastive | Vision representations |
| **T5** | Span corruption (text-to-text) | Bidirectional | General text tasks |

### Why Scale Matters

Emergent capabilities appear at sufficient scale — in-context learning, reasoning, instruction following. Scaling laws show predictable improvement.
        """))

    # ===================== 16. RL =====================
    if sec in ("all", "rl") and typ in ("all", "rl"):
        sections.append(mo.md(r"""
---

## 16. Reinforcement Learning

### Core: Bellman Equation

$$V(s) = \mathbb{E}[R + \gamma V(s')], \qquad Q(s,a) = \mathbb{E}[R + \gamma \max_{a'} Q(s', a')]$$

$\gamma$: 0 = greedy, 1 = far-sighted.

### Algorithms

| Algorithm | Type | Key Idea |
|-----------|------|----------|
| **Q-Learning** | Off-policy, model-free | $\max_{a'} Q(s',a')$ in update → learns optimal regardless of behavior |
| **SARSA** | On-policy | Uses actual next action (not max) |
| **DQN** | Deep Q-Learning | Neural net for $Q$. Experience replay + target network. |
| **REINFORCE** | Policy gradient | $\nabla J = \mathbb{E}[\nabla\log\pi(a \mid s) \cdot G_t]$. High variance. |
| **Actor-Critic** | Hybrid | Actor (policy) + Critic (value baseline). Lower variance. |
| **PPO** | Policy gradient | Clipped ratio $[1{-}\varepsilon, 1{+}\varepsilon]$. Stable. Widely used (RLHF). |

### Advanced

| | |
|---|---|
| **AlphaZero** | Self-play + MCTS, no human data |
| **Decision Transformer** | RL as sequence modeling — condition on desired return |
| **Offline RL** | Static dataset. Main challenge: distribution shift. |
| **Model-based** | Learn world model. More sample-efficient but compounding errors. |
        """))

    # ===================== 17. NLP =====================
    if sec in ("all", "nlp") and typ in ("all", "deep"):
        sections.append(mo.md(r"""
---

## 17. NLP

### Tokenization

| Method | How |
|--------|-----|
| **BPE** | Iteratively merge frequent character pairs. Handles unseen words. |
| **WordPiece** | Similar, used by BERT. Probabilistic. |

### Embeddings

| Method | Type |
|--------|------|
| **Word2Vec** | Static (one vector per word). Skip-gram: predict context from center. |
| **GloVe** | Static. Co-occurrence matrix factorization. |
| **BERT/GPT** | Contextual (different per context). |

### Sampling for Generation

| Method | How |
|--------|-----|
| **Temperature** | Divide logits by $T$. Low = sharp, high = diverse. |
| **Top-k** | Sample from $k$ most likely tokens. |
| **Top-p (nucleus)** | Smallest set exceeding cumulative prob $p$. |

### RLHF Pipeline

1. **SFT** — supervised fine-tuning on instructions.
2. **Reward model** — trained on human preference rankings.
3. **PPO** — optimize policy against reward model.
4. **DPO** — alternative: skip reward model, optimize preferences directly.
        """))

    # ===================== 18. CV =====================
    if sec in ("all", "cv") and typ in ("all", "deep"):
        sections.append(mo.md(r"""
---

## 18. Computer Vision

### Object Detection

| Method | Type | Trade-off |
|--------|------|-----------|
| **Faster R-CNN** | Two-stage (propose + classify) | More accurate, slower |
| **YOLO** | One-stage (single pass) | Real-time, slightly less accurate |
| **DETR** | Transformer-based | SOTA |

**IoU** = intersection / union. > 0.5 = correct detection. **NMS** removes duplicates.

### Segmentation

| Architecture | How |
|-------------|-----|
| **U-Net** | Encoder-decoder with skip connections for spatial detail |
| **DeepLab** | Dilated convolutions for large receptive field without losing resolution |
| **FCN** | Fully convolutional, no dense layers |

### Vision Transformer (ViT)

Split image into 16×16 patches → embed → position encodings → standard Transformer. Needs more data than CNNs (less inductive bias), but scales better.
        """))

    # ===================== 19. BAYESIAN =====================
    if sec in ("all", "bayesian") and typ in ("all", "statistical"):
        sections.append(mo.md(r"""
---

## 19. Bayesian Machine Learning

### Gaussian Processes

| | |
|---|---|
| **What** | Distribution over functions. Any finite points are jointly Gaussian. |
| **Kernels** | RBF, Matérn, periodic. Kernel = covariance function. |
| **Strengths** | Uncertainty estimates ($\sigma_*$). Nonparametric. Principled. Marginal likelihood for hyperparams. |
| **Weaknesses** | $O(n^3)$. Kernel design required. |
| **Use when** | Small $n$ + uncertainty matters. Bayesian optimization surrogate. |

### Approximate Inference

| Method | How | Trade-off |
|--------|-----|-----------|
| **Variational Inference** | Approximate posterior with tractable $q(\theta)$, maximize ELBO | Scalable (SGD). Underestimates uncertainty. |
| **MCMC (HMC/NUTS)** | Sample from posterior using gradients | Asymptotically exact. Slow. Scales poorly. |
| **MC Dropout** | Dropout on at test time → variance = uncertainty | Cheap. Crude approximation. |
        """))

    # ===================== 20. DECISION TABLE =====================
    if sec in ("all", "decision") and typ in ("all",):
        sections.append(mo.md(r"""
---

## 20. When to Use What

| Scenario | Recommendation |
|----------|----------------|
| Tabular, $n > 1000$ | Gradient Boosting (LightGBM / XGBoost) |
| Tabular, $n < 500$ | Regularized linear model or small RF |
| Images | CNN or Vision Transformer |
| Text | Transformer (BERT for understanding, GPT for generation) |
| Sequences / time series | LSTM/GRU or Transformer |
| Interpretability critical | Single decision tree or linear model |
| Quick baseline | Random Forest or Logistic Regression |
| Small labeled + large unlabeled | Self-supervised pretraining + fine-tuning |
| Uncertainty needed | GP (small $n$) or MC Dropout (neural nets) |
| Anomaly detection | Isolation Forest or GMM |
| Clustering | K-Means (spherical), GMM (flexible), Hierarchical (structure) |
| Dimensionality reduction | PCA (linear), UMAP (nonlinear), t-SNE (viz only) |
        """))

    if not sections:
        sections.append(mo.md("*No algorithms match the selected filters. Try a different combination.*"))

    mo.vstack(sections)
    return


if __name__ == "__main__":
    app.run()
