# ML Course — Complete Algorithm Study Guide

Grouped by how algorithms relate to and build on each other.

---

## 1. Statistical Foundations: Means, Variances, and Estimation

These concepts are the atoms everything else is built from.

### Sample Mean: x-bar = (1/n) sum(x_i)

- **What:** Average of observed values. Simplest estimator of the true mean.
- **Strengths:** Unbiased, consistent, intuitive.
- **Weaknesses:** Sensitive to outliers. One extreme value can shift it heavily.
- **Used by:** OLS (Ordinary Least Squares) (target prediction), K-Means (centroid update), Batch Normalization (batch mean), PCA (Principal Component Analysis) (centering data), Gradient Boosting (initial prediction F_0).

### Sample Variance: s^2 = (1/(n-1)) sum(x_i - x-bar)^2

- **What:** Measures spread of data around the mean. Bessel's correction (n-1) makes it unbiased.
- **Strengths:** Unbiased estimator of true variance.
- **Weaknesses:** Sensitive to outliers (squared deviations). Biased version (divide by n) is the MLE estimate.
- **Used by:** PCA (Principal Component Analysis) (eigenvalues = variance per component), Batch Normalization (normalize by batch std), Bias-Variance Decomposition, all hypothesis testing.

### Covariance: Cov(X,Y) = E[(X - mu_X)(Y - mu_Y)]

- **What:** Measures how two variables move together. Positive = same direction, negative = opposite.
- **Strengths:** Foundation for correlation, PCA (Principal Component Analysis), LDA (Linear Discriminant Analysis), Gaussian models.
- **Weaknesses:** Only captures linear relationships. Sensitive to scale.
- **Used by:** PCA (Principal Component Analysis) (covariance matrix S = X^T X / (n-1)), LDA (Linear Discriminant Analysis) (within-class and between-class scatter), GMMs (Gaussian Mixture Models) (each component has its own covariance), GPs (Gaussian Processes) (kernel = covariance function).

### Covariance Matrix: S = (1/(n-1)) X^T X

- **What:** p x p matrix where S_ij = covariance between features i and j, diagonal = variances.
- **Strengths:** Captures all pairwise linear relationships. Symmetric, positive semi-definite.
- **Weaknesses:** O(p^2) storage. Ill-conditioned when features are correlated.
- **Used by:** PCA (Principal Component Analysis) (eigendecomposition of S), LDA (Linear Discriminant Analysis) (shared covariance assumption), GMMs (Gaussian Mixture Models) (per-component covariance), Mahalanobis distance, GPs (Gaussian Processes).

### Bias-Variance Decomposition: MSE = Bias^2 + Variance + sigma^2

- **What:** Decomposes prediction error into three sources: systematic error (bias), sensitivity to training data (variance), and irreducible noise.
- **Strengths:** Explains the fundamental tradeoff in all of ML. Guides model selection.
- **Weaknesses:** Can't be computed exactly in practice (requires knowing the true function). Only holds for squared error loss.
- **Connects to:** Regularization (increases bias, decreases variance), Ensembles (bagging reduces variance, boosting reduces bias), Model complexity decisions.

### Maximum Likelihood Estimation (MLE): theta-hat = argmax sum log p(x_i | theta)

- **What:** Find parameters that make observed data most probable. Write likelihood, take log, differentiate, solve.
- **Strengths:** Consistent, asymptotically efficient (achieves Cramer-Rao bound), equivariant.
- **Weaknesses:** Overfits with small samples. Can find degenerate solutions (e.g., GMM (Gaussian Mixture Model) component collapsing to a single point).
- **Used by:** OLS (Ordinary Least Squares) (MLE under Gaussian noise), Logistic Regression (MLE of Bernoulli), GMMs (Gaussian Mixture Models) (EM (Expectation-Maximization) maximizes likelihood), Neural networks (cross-entropy = negative log-likelihood).

### MAP Estimation: theta-hat = argmax [log p(D|theta) + log p(theta)]

- **What:** MLE plus a prior. Point estimate at posterior mode.
- **Strengths:** Incorporates prior knowledge. Gaussian prior = Ridge, Laplace prior = Lasso.
- **Weaknesses:** Still a point estimate; no uncertainty quantification. Loses the full posterior.
- **Connects to:** Ridge (MAP with Gaussian prior), Lasso (MAP with Laplace prior), Bayesian ML (MAP (Maximum A Posteriori) is the simplest Bayesian approach).

---

## 2. Linear Prediction Models

All share the form y = w^T x (linear in parameters). Differ in loss function and what they predict.

### Ordinary Least Squares (OLS)

- **Equation:** w* = (X^T X)^(-1) X^T y
- **Loss:** ||y - Xw||^2 (sum of squared residuals)
- **Strengths:** Closed-form solution. Geometric interpretation (orthogonal projection). Equivalent to MLE under Gaussian noise. Convex — unique global minimum.
- **Weaknesses:** O(p^3) for matrix inversion. Overfits when p is large or features are correlated (multicollinearity). Sensitive to outliers.
- **Use when:** p << n, features aren't too correlated, you want interpretable coefficients.

### Ridge Regression (L2 Regularization)

- **Equation:** w* = (X^T X + lambda*I)^(-1) X^T y
- **Loss:** ||y - Xw||^2 + lambda * ||w||^2
- **Strengths:** Always invertible (lambda*I fixes ill-conditioning). Shrinks all coefficients toward zero. Bayesian interpretation: Gaussian prior on w. Handles multicollinearity.
- **Weaknesses:** Keeps all features — no sparsity. Must tune lambda (use CV).
- **Use when:** Many correlated features, you want to keep all features but control their magnitude.

### Lasso (L1 Regularization)

- **Equation:** No closed form — requires optimization (coordinate descent, proximal methods).
- **Loss:** ||y - Xw||^2 + lambda * ||w||_1
- **Strengths:** Drives some weights exactly to zero — automatic feature selection. Diamond-shaped constraint region has corners on axes (geometric reason for sparsity).
- **Weaknesses:** Picks one feature arbitrarily among correlated group. No closed form. Can underperform Ridge when all features are relevant.
- **Use when:** You suspect many features are irrelevant and want automatic selection.

### Elastic Net (L1 + L2)

- **Loss:** ||y - Xw||^2 + lambda_1 * ||w||_1 + lambda_2 * ||w||^2
- **Strengths:** Gets sparsity from L1 and handles correlated features from L2. More stable than pure Lasso.
- **Weaknesses:** Two hyperparameters to tune.
- **Use when:** Correlated features + desire for sparsity.

### Polynomial Regression

- **Model:** y = w^T phi(x) where phi(x) = [1, x, x^2, ..., x^d]
- **Strengths:** Captures nonlinear patterns while remaining linear in parameters (same OLS math). Easy to implement via feature expansion.
- **Weaknesses:** Overfits badly for high degree d. Runge phenomenon (oscillation at edges). Curse of dimensionality with multiple features.
- **Use when:** You see clear nonlinear patterns and have enough data relative to degree.

### Logistic Regression

- **Model:** P(y=1|x) = sigma(w^T x) = 1 / (1 + e^(-w^T x))
- **Loss:** Binary cross-entropy: -sum[y log(y-hat) + (1-y) log(1-y-hat)]
- **Gradient:** X^T (sigma(Xw) - y) — same form as OLS!
- **Strengths:** Outputs calibrated probabilities. Convex loss — unique global minimum. Interpretable (log-odds are linear). Well-understood theoretically.
- **Weaknesses:** Linear decision boundary only. No closed-form (need gradient descent or Newton's method). Sigmoid saturates for large |z|.
- **Use when:** Binary or multiclass classification with roughly linear boundaries. Baseline for any classification task.

### Softmax Regression (Multinomial Logistic)

- **Model:** P(y=k|x) = e^(w_k^T x) / sum_j e^(w_j^T x)
- **Loss:** Categorical cross-entropy: -sum log P(y_i = true class)
- **Strengths:** Natural generalization of logistic regression to K classes. Outputs valid probability distribution.
- **Weaknesses:** Linear boundaries between all class pairs. K weight vectors to learn.
- **Use when:** Multiclass problems. Output layer of neural networks.

---

## 3. Discriminative vs. Generative Classifiers

Two philosophies for classification. All the models below draw a decision boundary — they differ in *how*.

### Discriminative: Learn P(y|x) directly

**Logistic Regression, SVM, Neural Networks**

- **Strengths:** More accurate with enough data. Fewer assumptions about data distribution. Focus resources on the boundary.
- **Weaknesses:** Can't generate new data. Need more data to work well. No insight into data distribution.

### Generative: Learn P(x|y) and P(y), use Bayes' rule

**LDA, Naive Bayes, GMMs**

- **Strengths:** Work better with small data (strong inductive bias). Can generate synthetic data. Handle missing features naturally. Reach lower error faster with limited data (Ng & Jordan, 2002).
- **Weaknesses:** Assumptions about data distribution often wrong. Asymptotically less efficient.

### Linear Discriminant Analysis (LDA)

- **Assumes:** P(x|y=k) ~ N(mu_k, Sigma) — shared covariance, different means.
- **Boundary:** delta_k(x) = x^T Sigma^(-1) mu_k - (1/2) mu_k^T Sigma^(-1) mu_k + log pi_k
- **Strengths:** Works well with small samples (Gaussian assumption acts as regularizer). No iterative optimization needed. Fisher's view: maximizes between-class / within-class variance ratio.
- **Weaknesses:** Assumes shared covariance and Gaussian classes. Linear boundary only.
- **Use when:** Small n, roughly Gaussian features, shared covariance plausible.

### Naive Bayes

- **Assumes:** P(x|y=k) = product_j P(x_j|y=k) — features conditionally independent.
- **Strengths:** O(n) training. Works with tiny datasets. Fast inference. Surprisingly effective despite wrong assumption.
- **Weaknesses:** Independence assumption almost always wrong. Probability estimates are poorly calibrated.
- **Use when:** Text classification, high-dimensional sparse data, very small datasets, when you need speed.

### Support Vector Machines (SVM)

- **Objective:** min (1/2)||w||^2 + C * sum(xi_i) subject to y_i(w^T x_i + b) >= 1 - xi_i
- **Loss:** Hinge loss: max(0, 1 - y * f(x))
- **Strengths:** Maximum margin = good generalization. Kernel trick enables nonlinear boundaries without explicit feature computation. Sparse solution (only support vectors matter). Strong theoretical guarantees.
- **Weaknesses:** Doesn't output probabilities natively (Platt scaling needed). Slow for large n: O(n^2) to O(n^3). Kernel choice and C require tuning.
- **Kernels:** Linear (x^T z), Polynomial ((x^T z + c)^d), RBF (exp(-gamma||x-z||^2) — infinite-dimensional).
- **Use when:** Medium-sized datasets with clear margin. When kernels capture domain structure.

---

## 4. Evaluation Metrics and Model Selection

How you measure and choose models. These aren't algorithms that make predictions — they judge the ones that do.

### Regression Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| **MSE (Mean Squared Error)** | (1/n) sum(y_i - y-hat_i)^2 | Penalizes large errors heavily (squared) |
| **RMSE (Root Mean Squared Error)** | sqrt(MSE) | Same units as y |
| **MAE (Mean Absolute Error)** | (1/n) sum \|y_i - y-hat_i\| | Robust to outliers |
| **R^2** | 1 - SS_res/SS_tot | Fraction of variance explained. 0 = mean predictor, 1 = perfect. Can be negative on test set. |

### Classification Metrics

| Metric | Formula | Use when |
|--------|---------|----------|
| **Accuracy** | (TP+TN) / total | Balanced classes only |
| **Precision** | TP / (TP+FP) | False positives are costly (spam filter) |
| **Recall** | TP / (TP+FN) | Missing positives is costly (cancer screening) |
| **F1** | 2*P*R / (P+R) | Imbalanced data, need balance of P and R |
| **AUC-ROC (Area Under the Receiver Operating Characteristic Curve)** | Area under TPR vs FPR curve | Overall ranking quality across all thresholds |
| **PR-AUC (Precision-Recall Area Under Curve)** | Area under Precision-Recall curve | Better for highly imbalanced data |

### Cross-Validation

- **k-Fold:** Split into k folds, train k times, each fold is validation once. Average scores.
- **Stratified k-Fold:** Preserves class proportions in each fold.
- **LOOCV (Leave-One-Out Cross-Validation):** k = n. Lowest bias, highest variance, very slow.
- **Nested CV:** Outer loop evaluates, inner loop tunes. Only way to get unbiased estimate of full pipeline.
- **Strengths:** More data-efficient than single split. Reduces variance of estimate.
- **Weaknesses:** k times more expensive. Results still have variance.

### Hyperparameter Tuning

| Method | How | Strengths | Weaknesses |
|--------|-----|-----------|------------|
| **Grid Search** | Try all combos | Exhaustive | Exponential cost, wastes evals on unimportant dims |
| **Random Search** | Sample randomly | Explores more unique values per dim | Not exhaustive |
| **Bayesian Optimization** | GP (Gaussian Process) surrogate + acquisition function | Sample-efficient | Complex to implement |

### Information Criteria (no CV needed)

| Criterion | Formula | Behavior |
|-----------|---------|----------|
| **AIC (Akaike Information Criterion)** | 2k - 2 ln(L-hat) | Lighter penalty, favors complex models |
| **BIC (Bayesian Information Criterion)** | k*ln(n) - 2 ln(L-hat) | Heavier penalty for large n, favors simpler models |

### Bootstrap

- Sample n points with replacement -> ~63.2% unique.
- **OOB (Out-of-Bag):** The ~36.8% not sampled = free validation set.
- **.632 Bootstrap:** Err = 0.368 * Err_train + 0.632 * Err_oob (bias-corrected).

---

## 5. Decision Trees and Tree Ensembles

A family of related algorithms built on recursive binary splitting. Each builds on the previous one's weakness.

### Decision Tree (CART — Classification and Regression Trees)

- **How:** At each node, greedily pick the feature + threshold that most reduces impurity. Recurse.
- **Splitting criteria:** Gini = 1 - sum(p_k^2). Entropy = -sum(p_k log p_k). Both work similarly.
- **Regression:** Leaf predicts mean of its training points. Minimize RSS (Residual Sum of Squares).
- **Pruning:** Cost-complexity: C_alpha(T) = loss + alpha * |leaves|. Select alpha via CV.
- **Strengths:** Highly interpretable. Handles mixed types. No scaling needed. Captures interactions. Robust to outliers.
- **Weaknesses:** **High variance** — small data changes produce completely different trees. Axis-aligned splits only. Overfit without pruning.
- **Use when:** Interpretability is critical. Quick baseline.

### Bagging (Bootstrap Aggregating)

- **How:** Draw B bootstrap samples. Train full unpruned tree on each. Average predictions (regression) or majority vote (classification).
- **Variance formula:** Var(avg) = rho*sigma^2 + (1-rho)/B * sigma^2. The rho*sigma^2 term doesn't vanish — correlation between trees limits gains.
- **Strengths:** Reduces variance by averaging. OOB (Out-of-Bag) error is free validation.
- **Weaknesses:** Trees are correlated (all from same data, strong features dominate). Doesn't reduce bias.
- **Connects to:** Random Forest improves on this by decorrelating trees.

### Random Forest

- **Innovation:** At each split, only consider m random features (not all p).
- **Defaults:** m = sqrt(p) for classification, p/3 for regression.
- **Strengths:** Decorrelated trees -> lower rho -> better variance reduction. More trees never overfits. Feature importance (permutation or impurity-based). Remarkably robust.
- **Weaknesses:** Less interpretable than single tree. Can be slow for very large datasets. Doesn't reduce bias.
- **Use when:** Tabular data baseline. Quick, robust, hard to mess up.

### Gradient Boosting

- **How:** Start with F_0 = mean(y). Sequentially fit shallow trees to negative gradient (pseudo-residuals). F_m = F_{m-1} + eta * h_m(x).
- **Key hyperparameters:** Learning rate eta (0.01-0.1), number of trees M (use early stopping), tree depth (3-8, weak learners), subsampling (50-80%).
- **Strengths:** Reduces bias (sequential error correction). Works with any differentiable loss. State-of-the-art for tabular data.
- **Weaknesses:** **Can overfit** (unlike RF) — needs early stopping. Sequential (harder to parallelize). More hyperparameters to tune.
- **Use when:** Maximum predictive accuracy on tabular data.

### XGBoost / LightGBM / CatBoost

| Library | Key Feature |
|---------|-------------|
| **XGBoost** | Explicit regularization: gamma*\|T\| + lambda * sum(w_j^2). Widest ecosystem. |
| **LightGBM** | Histogram-based splits (~256 bins). Leaf-wise growth. Fastest. |
| **CatBoost** | Native categorical feature handling. Good defaults. |

### Stacking

- **How:** Train diverse base models. Use their cross-validated predictions as features for a meta-model.
- **Strengths:** Can combine strengths of different model types. Marginal gains (1-3%).
- **Weaknesses:** Complex. Must use out-of-fold predictions to prevent meta-model overfitting.
- **Use when:** Competitions. Marginal accuracy matters more than simplicity.

### Bias-Variance Summary for Ensembles

| Method | Reduces | Mechanism |
|--------|---------|-----------|
| Bagging/RF | **Variance** | Average many noisy, decorrelated models |
| Boosting | **Bias** | Sequentially correct errors |
| Stacking | **Both** | Meta-model learns optimal combination |

---

## 6. Unsupervised Learning: Clustering

No labels. Find structure in data.

### K-Means

- **Objective:** min J = sum_k sum_{x in C_k} ||x - mu_k||^2 (within-cluster variance)
- **Algorithm:** Assign to nearest centroid -> update centroid to cluster mean -> repeat.
- **K-Means++ init:** Choose new centroids with probability proportional to D(x)^2.
- **Strengths:** Fast O(nK per iteration). Simple. Works well for spherical clusters.
- **Weaknesses:** Must specify K. Local optima only. Assumes spherical, equal-size clusters. Sensitive to outliers and scale.
- **Choosing K:** Elbow method, silhouette score, gap statistic, domain knowledge.
- **Connection:** K-Means = EM (Expectation-Maximization) on GMM (Gaussian Mixture Model) with isotropic covariance and sigma -> 0 (hard assignments).

### Gaussian Mixture Models (GMM) + EM (Expectation-Maximization)

- **Model:** p(x) = sum_k pi_k * N(x | mu_k, Sigma_k)
- **EM E-step:** Compute responsibilities r_ik = posterior probability point i belongs to cluster k (Bayes' theorem).
- **EM M-step:** Update pi_k, mu_k, Sigma_k using responsibilities as weights.
- **Strengths:** Soft assignments (probabilities). Handles different cluster shapes and sizes. Principled probabilistic framework. Can compute likelihood, use BIC (Bayesian Information Criterion) for model selection.
- **Weaknesses:** Slower than K-Means. Can have degenerate solutions (component collapses to single point). Sensitive to initialization. Requires specifying K.
- **Use when:** Clusters have different shapes/sizes. You need probabilistic assignments.

### Hierarchical Clustering (Agglomerative)

- **How:** Start with n clusters, iteratively merge the two closest.
- **Linkage:** Single (chaining), Complete (compact, outlier-sensitive), Average (compromise), Ward's (minimize variance increase — K-Means-like).
- **Strengths:** Dendrogram shows structure at all granularities. Don't need to specify K. Any distance metric.
- **Weaknesses:** O(n^2) to O(n^3). Greedy, no backtracking. Sensitive to linkage choice.
- **Use when:** You want to see hierarchical structure. Small to medium datasets.

### Silhouette Score

- **Formula:** s(i) = (b(i) - a(i)) / max(a(i), b(i)) where a = avg within-cluster distance, b = avg nearest-other-cluster distance.
- **Range:** [-1, +1]. Near +1 = well clustered. Near 0 = boundary point. Negative = likely misassigned.

---

## 7. Unsupervised Learning: Dimensionality Reduction

Reduce features while preserving information.

### PCA (Principal Component Analysis)

- **How:** Eigendecomposition of covariance matrix S. Eigenvectors = principal components. Eigenvalues = variance per component.
- **Equivalent:** SVD (Singular Value Decomposition) of centered data: X = U Sigma V^T. V columns = PCs.
- **Choosing components:** Scree plot (elbow). Retain 90-95% explained variance.
- **Strengths:** Optimal linear DR (Dimensionality Reduction) (maximizes variance preserved). Fast. Well-understood. Useful for denoising (discard low-variance components).
- **Weaknesses:** Linear only. Assumes variance = importance. Sensitive to scaling (standardize first).
- **Use when:** Preprocessing, visualization, denoising, reducing multicollinearity.

### Kernel PCA

- **How:** Apply kernel trick to PCA. Compute inner products in high-D feature space without explicit transformation.
- **Strengths:** Captures nonlinear structure.
- **Weaknesses:** Must choose kernel. More expensive. No simple inverse transform.

### t-SNE (t-distributed Stochastic Neighbor Embedding)

- **How:** In high-D, compute Gaussian similarities. In low-D (2-3), compute Student-t similarities. Minimize KL (Kullback-Leibler) divergence KL(P||Q).
- **Hyperparameter:** Perplexity (5-50, effective number of neighbors).
- **Strengths:** Excellent for visualization. Preserves local neighborhoods.
- **Weaknesses:** **Visualization only** — don't use embeddings as features. Non-deterministic. Between-cluster distances and cluster sizes are meaningless. Slow O(n^2).
- **Use when:** 2D/3D visualization of high-dimensional data.

### UMAP (Uniform Manifold Approximation and Projection)

- **How:** Based on fuzzy topological structures. Optimizes cross-entropy between high-D and low-D fuzzy sets.
- **Strengths:** Faster than t-SNE. Preserves more global structure. Can embed to arbitrary dimensions (not just 2-3). Can be used for general DR (Dimensionality Reduction), not just visualization.
- **Weaknesses:** Hyperparameter-sensitive. Less theoretical grounding than PCA.
- **Use when:** Visualization or general nonlinear DR. Preferred over t-SNE in most cases.

### Isolation Forest (Anomaly Detection)

- **How:** Random axis-aligned splits. Anomalies are isolated in fewer splits (shorter path length).
- **Strengths:** Fast. Scales well. No density estimation needed.
- **Weaknesses:** Axis-aligned only. Anomaly score threshold must be chosen.
- **Use when:** Outlier/anomaly detection in tabular data.

---

## 8. Neural Networks and Backpropagation

The foundation of deep learning.

### The Neuron

- **Equation:** z = w^T x + b, output = activation(z)
- **Activation functions and when to use each:**

| Activation | Formula | Strengths | Weaknesses | Use |
|-----------|---------|-----------|------------|-----|
| **ReLU** | max(0, z) | No saturation (z>0), fast | Dead neurons (z<0 always) | Default hidden layer |
| **Leaky ReLU** | max(0.01z, z) | No dead neurons | Slight added complexity | When dead neurons are a problem |
| **GELU** | z * Phi(z) | Smooth, performs well | Slower to compute | Transformers |
| **Sigmoid** | 1/(1+e^-z) | Output in (0,1) | Saturates, vanishing gradient | Binary output layer |
| **Tanh** | (e^z - e^-z)/(e^z + e^-z) | Zero-centered | Still saturates | RNN hidden layers |
| **Softmax** | e^z_k / sum e^z_j | Valid probability distribution | — | Multiclass output layer |

### Feedforward Network (MLP — Multi-Layer Perceptron)

- **How:** Stack layers: z^(j) = W^(j) h^(j-1) + b^(j), h^(j) = activation(z^(j)).
- **Universal Approximation:** Single hidden layer with enough neurons can approximate any continuous function. But depth is exponentially more efficient.
- **Strengths:** Can learn any function (given enough capacity). Flexible architecture.
- **Weaknesses:** Need lots of data. Many hyperparameters. Black box.

### Backpropagation

- **How:** Chain rule through computational graph. Forward pass computes activations. Backward pass propagates gradients.
- **Cost:** ~2-3x forward pass.
- **Key insight:** Reverse-mode autodiff computes all parameter gradients in one backward pass (vs. forward mode needing one pass per parameter). This is why it works for neural nets (scalar loss, millions of parameters).

### Loss Functions for Neural Networks

| Loss | Formula | Use |
|------|---------|-----|
| **MSE** | (1/n) sum(y - y-hat)^2 | Regression |
| **MAE** | (1/n) sum \|y - y-hat\| | Regression (robust to outliers) |
| **Huber** | MSE near 0, MAE far away | Regression (compromise) |
| **Binary CE** | -sum[y log y-hat + (1-y) log(1-y-hat)] | Binary classification |
| **Categorical CE** | -sum log P(true class) | Multiclass classification |
| **Hinge** | sum max(0, 1 - y*f(x)) | SVM-style classification |

---

## 9. Deep Learning Optimization

How to navigate the loss landscape.

### SGD (Stochastic Gradient Descent)

- **Update:** theta <- theta - eta * gradient_batch
- **Strengths:** Simple. Noise helps escape sharp minima (better generalization). Sometimes beats Adam on final test accuracy.
- **Weaknesses:** Slow convergence. Oscillates in narrow valleys. Sensitive to learning rate.

### SGD + Momentum

- **Update:** v = beta*v + gradient; theta <- theta - eta*v
- **Strengths:** Accumulates velocity in consistent direction. Dampens oscillations. Standard beta = 0.9.
- **Weaknesses:** Extra hyperparameter. Can overshoot.

### Adam (Adaptive Moment Estimation)

- **How:** Tracks running mean of gradient (1st moment, like momentum) AND running mean of squared gradient (2nd moment, like RMSProp). Bias correction for early steps.
- **Defaults:** beta_1=0.9, beta_2=0.999, epsilon=1e-8.
- **Strengths:** Adaptive per-parameter learning rates. Converges fast. Works well with minimal tuning.
- **Weaknesses:** Can converge to sharper minima than SGD (worse generalization on some vision tasks). L2 regularization interacts badly with adaptive rates.

### AdamW (Decoupled Weight Decay)

- **Key insight:** In Adam, L2 regularization gets divided by sqrt(v_t), making heavily-updated parameters get LESS regularization (opposite of intent). AdamW applies weight decay outside the adaptive scaling.
- **Strengths:** Proper regularization behavior. Better generalization than Adam + L2.
- **Status:** **Default optimizer in modern deep learning.**

### Learning Rate Schedules

| Schedule | How | Use |
|----------|-----|-----|
| **Warmup** | Linear ramp from small LR for first ~1000 steps | Large batch training, Adam |
| **Step decay** | Multiply by 0.1 at fixed epochs | Classic CV training |
| **Cosine annealing** | Half-cosine from max to min | Modern default, minimal tuning |
| **One-cycle** | Low -> high -> very low over training | Excellent empirical results |
| **LR range test** | Exponentially increase LR, plot loss | Find good LR in minutes |

### Batch Size Tradeoffs

- **Small (32-256):** More noise, better generalization, underutilize GPU.
- **Large (4K-64K):** Less noise, better GPU utilization, may find sharp minima.
- **Linear scaling rule:** Multiply batch size by k -> multiply LR by k.

---

## 10. Deep Learning Regularization

Preventing neural nets from memorizing training data.

### Weight Decay (L2)

- **How:** Loss + (lambda/2)||theta||^2. Update: theta <- (1 - eta*lambda)*theta - eta*gradient.
- **Effect:** Shrinks weights. Smooths loss landscape (adds lambda to Hessian eigenvalues). Preferentially shrinks low-curvature directions.
- **Note:** Biases typically not regularized.

### Dropout

- **How:** Randomly zero neurons with probability p during training. Scale remaining by 1/(1-p) (inverted dropout). Off at test time.
- **Strengths:** Ensemble of 2^n sub-networks. Forces redundant representations. Bayesian connection (MC Dropout for uncertainty).
- **Weaknesses:** Slows training. Less effective for CNNs. Adds noise to gradients.
- **Typical rates:** Hidden layers: 0.5. Input: 0.1-0.2. Transformers: 0.1-0.3.

### Early Stopping

- **How:** Monitor validation loss. Save best checkpoint. Stop after k epochs of no improvement.
- **Key:** Return the best checkpoint, NOT the final model.
- **Equivalence:** Approximately equivalent to L2 regularization. Training steps ~ inverse regularization strength.

### Batch Normalization

- **How:** Per layer: normalize to zero mean, unit variance across batch. Then apply learned scale (gamma) and shift (beta).
- **Training vs inference:** Training uses batch stats. Inference uses running averages.
- **Strengths:** Smooths loss landscape. Allows higher learning rates. Acts as regularizer (batch noise).
- **Weaknesses:** Train/test behavior differs. Small batches = noisy estimates. Bad for RNNs.
- **Variants:** Layer Norm (across features — good for Transformers), Group Norm (groups of channels — good for small batches), Instance Norm (per sample per channel — style transfer).

### Data Augmentation

- **Image:** Flips, rotations, crops, color jitter, Cutout, Mixup, CutMix.
- **Text:** Synonym replacement, back-translation, random insertion/deletion.
- **Key principle:** Transformation must preserve the label.

### Label Smoothing

- **How:** Replace hard [0, 1] targets with soft [0.05, 0.95].
- **Strengths:** Prevents overconfident predictions. Improves calibration and generalization.

### Double Descent

- **What:** Test error decreases, then increases (classical), then **decreases again** past the interpolation threshold.
- **Why it matters:** Challenges classical bias-variance tradeoff. Very large models can generalize well.

---

## 11. Convolutional Neural Networks (CNNs)

Specialized for spatial data (images, audio).

### Convolution Layer

- **How:** Slide learned kernel over input. Element-wise multiply and sum at each position.
- **Key ideas:** Local connectivity (not fully connected). Weight sharing (same kernel everywhere = translation equivariance). Multiple filters = multiple feature maps.
- **Parameters:** Kernel size, stride, padding, dilation.

### Pooling

- **Max pooling:** Take max in window. Provides small translation invariance. Reduces spatial size.
- **Average pooling:** Take mean. Less common.
- **Modern alternative:** Strided convolution (learnable downsampling).

### Key Architectures

| Architecture | Year | Innovation | Depth |
|-------------|------|------------|-------|
| **AlexNet** | 2012 | ReLU, dropout, GPU training | 8 |
| **VGGNet** | 2014 | Uniform 3x3 filters, simplicity | 16-19 |
| **GoogLeNet** | 2014 | Inception modules (parallel filter sizes) | 22 |
| **ResNet** | 2015 | Skip connections: y = F(x) + x | 50-152+ |
| **EfficientNet** | 2019 | Compound scaling (width, depth, resolution) | varies |

### Transfer Learning

- **Feature extraction:** Freeze pretrained backbone, train new head only.
- **Fine-tuning:** Unfreeze some/all layers, train with low learning rate.
- **Why it works:** Early layers learn universal features (edges, textures) useful across domains.

---

## 12. Sequence Models

For data with temporal/sequential structure.

### Vanilla RNN (Recurrent Neural Network)

- **Equation:** h_t = activation(W_hh * h_{t-1} + W_xh * x_t + b)
- **Strengths:** Variable-length sequences. Parameter sharing across time.
- **Fatal weakness:** Vanishing/exploding gradients. gradient involves W_hh^(t-s) — exponential decay or growth.

### LSTM (Long Short-Term Memory)

- **Key innovation:** Cell state c_t as a gradient highway. Three gates (forget, input, output) control information flow.
- **Cell update:** c_t = f_t * c_{t-1} + i_t * c-tilde_t (additive, not multiplicative — gradients flow).
- **Strengths:** Captures long-range dependencies. Gates are learned. Mostly solves vanishing gradient.
- **Weaknesses:** Sequential (can't parallelize). More parameters than vanilla RNN. Still struggles with very long sequences.

### GRU (Gated Recurrent Unit)

- **How:** Two gates (reset, update) instead of three. Combines forget and input gates.
- **Strengths:** Fewer parameters than LSTM. Similar performance. Faster training.
- **Use when:** Speed matters more than marginal accuracy.

### Bidirectional RNNs

- **How:** Run forward and backward RNNs, concatenate hidden states.
- **Strengths:** Access to both past and future context.
- **Weakness:** Can't be used for real-time generation (needs full sequence).

### Seq2Seq + Attention

- **Bottleneck problem:** Encoder compresses entire input into single fixed-length vector.
- **Attention solution:** Decoder attends to all encoder hidden states with learned weights.
- **This leads directly to Transformers.**

---

## 13. Transformers and Attention

The architecture that dominates modern ML.

### Scaled Dot-Product Attention

- **Equation:** Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
- **Why sqrt(d_k):** Prevents softmax saturation for large d_k.
- **Q, K, V:** Linear projections of input. Query asks "what am I looking for?", Key says "what do I contain?", Value says "here's my content."

### Multi-Head Attention

- **How:** h parallel attention heads, each with own Q/K/V projections. Concatenate and project.
- **Why:** Different heads capture different relationship types (syntax, semantics, position, etc.).

### Transformer Architecture

- **Encoder:** Multi-head self-attention + FFN + residual connections + layer norm. N=6 blocks.
- **Decoder:** Masked self-attention (prevent future) + cross-attention (attend to encoder) + FFN. N=6 blocks.
- **Positional encoding:** Sinusoidal or learned. Needed because attention is permutation-invariant.
- **FFN:** Same 2-layer MLP at every position (no cross-position interaction).

### Why Transformers Won

- **Parallelizable:** All positions processed simultaneously (RNNs are sequential).
- **Long-range:** Direct connection between all positions (no vanishing gradient over distance).
- **Scalable:** Performance improves predictably with more compute, data, parameters.

---

## 14. Generative Models

Learning to create new data.

### Autoencoder

- **How:** Encoder: x -> z (compress). Decoder: z -> x-hat (reconstruct). Loss: ||x - x-hat||^2.
- **Not truly generative:** Latent space has holes — random z doesn't produce valid outputs.
- **Use for:** Dimensionality reduction, feature learning, anomaly detection.

### VAE (Variational Autoencoder)

- **How:** Encoder outputs distribution q(z|x) = N(mu, sigma^2), not a point. Sample z, decode.
- **Loss (ELBO):** Reconstruction + KL(q(z|x) || N(0,I)). Reconstruction = faithful output. KL = smooth latent space.
- **Reparameterization trick:** z = mu + sigma * epsilon (epsilon ~ N(0,I)) — makes sampling differentiable.
- **Strengths:** Principled probabilistic framework. Smooth latent space. Can generate by sampling z ~ N(0,I).
- **Weaknesses:** **Blurry outputs** (MSE averages over plausible images). KL term can dominate (posterior collapse).

### GAN (Generative Adversarial Network)

- **How:** Generator creates fake samples. Discriminator distinguishes real from fake. Minimax game.
- **Loss:** L_D = -E[log D(x)] - E[log(1-D(G(z)))]. Generator maximizes probability of fooling D.
- **Strengths:** Sharp, realistic outputs. No explicit density estimation needed.
- **Weaknesses:** Training instability. Mode collapse (generator produces limited variety). Hard to monitor convergence. No likelihood.

### Diffusion Models

- **How:** Forward: gradually add noise over T steps until pure noise. Reverse: learn neural net to denoise step by step.
- **Training:** Minimize ||epsilon - epsilon_theta(x_t, t)||^2 (predict the noise that was added).
- **Strengths:** Stable training (no adversarial dynamics). State-of-the-art image quality. Principled probabilistic framework.
- **Weaknesses:** Slow sampling (many denoising steps). High compute cost.
- **Status:** Current SOTA for image generation (DALL-E, Stable Diffusion, Imagen).

---

## 15. Self-Supervised Learning

Learn representations from unlabeled data.

### Autoregressive (GPT-style)

- **Task:** Predict next token given all previous. Causal mask prevents looking ahead.
- **Strengths:** Simple objective. Scales remarkably well. Enables text generation. Emergent abilities at scale (in-context learning, reasoning).
- **Weaknesses:** Unidirectional context only.

### Masked (BERT-style)

- **Task:** Mask 15% of tokens (80% [MASK], 10% random, 10% unchanged). Predict masked tokens.
- **Strengths:** Bidirectional context. Excellent for classification/understanding tasks.
- **Weaknesses:** Can't generate text autoregressively. [MASK] token doesn't exist at fine-tuning time (train-test mismatch).

### Contrastive Learning (SimCLR, MoCo)

- **Task:** Two augmentations of same image = positive pair. Different images = negatives. Pull positives together, push negatives apart.
- **Loss:** NT-Xent: -log(exp(sim(z_i, z_j)/tau) / sum_k exp(sim(z_i, z_k)/tau))
- **Strengths:** No labels needed. Learned representations transfer well.
- **Weaknesses:** Needs large batch sizes (more negatives). Sensitive to augmentation choice.

---

## 16. Reinforcement Learning

Learning from rewards through interaction.

### Core Concepts

- **MDP:** (States, Actions, Transitions, Rewards, gamma).
- **V(s):** Expected cumulative discounted reward from state s.
- **Q(s,a):** Expected return after taking action a in state s.
- **Bellman:** V(s) = E[R + gamma * V(s')]. Recursive definition.
- **Gamma:** 0 = greedy (immediate only). 1 = far-sighted (all future equal).

### Q-Learning (Off-Policy, Model-Free)

- **Update:** Q(s,a) <- Q(s,a) + alpha * [R + gamma * max_a' Q(s',a') - Q(s,a)]
- **Strengths:** Off-policy (learns optimal policy regardless of behavior). Simple.
- **Weaknesses:** Tabular only (need function approximation for large state spaces). Can overestimate Q-values.
- **DQN:** Neural net approximates Q. Experience replay + target network for stability.

### SARSA (On-Policy)

- **Update:** Same as Q-Learning but uses actual next action a' instead of max.
- **Difference:** Learns value of the policy it's actually following, not the optimal.

### Policy Gradient (REINFORCE)

- **How:** Directly optimize policy parameters. Gradient = E[grad log pi(a|s) * G_t].
- **Strengths:** Works with continuous actions. Directly optimizes what you care about.
- **Weaknesses:** High variance. Needs complete episodes (Monte Carlo returns).

### Actor-Critic

- **How:** Actor = policy network. Critic = value network estimates advantage.
- **Gradient:** E[grad log pi(a|s) * (r + gamma*V(s') - V(s))].
- **Strengths:** Lower variance than REINFORCE (critic provides baseline). Can use TD learning (no complete episodes needed).

### PPO (Proximal Policy Optimization)

- **How:** Clips probability ratio to [1-epsilon, 1+epsilon] to prevent destructively large updates.
- **Strengths:** Simple. Stable. Very widely used (RLHF for LLMs, robotics).
- **Weaknesses:** On-policy (sample-inefficient). Hyperparameter-sensitive.

### Advanced RL (Reinforcement Learning)

| Algorithm | Key Idea |
|-----------|----------|
| **AlphaZero** | Self-play + MCTS guided by neural net. No human data. |
| **PPO** | Clipped objective for stable policy updates. |
| **SAC** | Maximum entropy RL — explore more. |
| **Decision Transformer** | RL as sequence modeling — condition on desired return. |
| **Offline RL** | Learn from static dataset. Main challenge: distribution shift. |
| **Model-based RL** | Learn world model. More sample-efficient but compounding errors. |

---

## 17. NLP (Natural Language Processing)-Specific Methods

### Tokenization

| Method | How | Strengths |
|--------|-----|-----------|
| **Word-level** | Split on whitespace | Simple |
| **Character-level** | Each character is a token | Small vocabulary, handles anything |
| **BPE** | Iteratively merge frequent character pairs | Handles unseen words via subwords |
| **WordPiece** | Similar to BPE, used by BERT | Probabilistic selection |

### Word Embeddings

| Method | How | Static/Contextual |
|--------|-----|-------------------|
| **Word2Vec** | Skip-gram: predict context from center word | Static (one vector per word) |
| **GloVe** | Factorize co-occurrence matrix | Static |
| **BERT embeddings** | Transformer encoder output | Contextual (different per context) |
| **GPT embeddings** | Transformer decoder output | Contextual |

### Sampling Strategies for Generation

| Method | How |
|--------|-----|
| **Greedy** | Always pick highest-probability token |
| **Temperature** | Divide logits by T before softmax (T<1 = sharper, T>1 = flatter) |
| **Top-k** | Sample from k most likely tokens |
| **Top-p (nucleus)** | Sample from smallest set exceeding cumulative probability p |

### RLHF Pipeline

1. **SFT:** Supervised fine-tuning on instruction-following data.
2. **Reward model:** Train on human preference rankings (A > B).
3. **PPO:** Optimize policy against reward model.
4. **Alternative:** DPO (Direct Preference Optimization) — skip reward model.

---

## 18. Computer Vision Tasks

### Object Detection

| Method | Type | Speed | Accuracy |
|--------|------|-------|----------|
| **R-CNN -> Faster R-CNN** | Two-stage (propose + classify) | Slower | Higher |
| **YOLO** | One-stage (single pass) | Real-time | Good |
| **DETR** | Transformer-based | Medium | State-of-the-art |

- **IoU:** Intersection / Union of predicted and true boxes. > 0.5 = correct detection.
- **NMS:** Non-Maximum Suppression removes duplicate detections.

### Semantic Segmentation

- **FCN:** Fully convolutional (no dense layers).
- **U-Net:** Encoder-decoder with skip connections for fine spatial detail.
- **DeepLab:** Dilated (atrous) convolutions for large receptive fields without losing resolution.
- **Loss:** Pixel-wise cross-entropy.

### Vision Transformer (ViT)

- **How:** Split image into 16x16 patches. Embed patches. Add position embeddings. Standard Transformer.
- **Strengths:** Scalable. Competitive with CNNs at large scale.
- **Weaknesses:** Needs more data than CNNs (less inductive bias).

---

## 19. Bayesian Machine Learning

Quantifying uncertainty.

### Gaussian Processes

- **What:** Distribution over functions. Any finite set of points is jointly Gaussian.
- **Specified by:** Mean function + kernel function (RBF, Matern, periodic, etc.).
- **Prediction:** Closed-form posterior: mu_* and sigma_* for any new x_*.
- **Strengths:** Principled uncertainty estimates. Nonparametric (adapts complexity to data). Bayesian: marginal likelihood for hyperparameter selection.
- **Weaknesses:** O(n^3) — doesn't scale. Kernel design required.
- **Use when:** Small datasets where uncertainty matters. Bayesian optimization (surrogate model).

### Variational Inference (VI)

- **How:** Approximate intractable posterior p(theta|D) with tractable q(theta). Maximize ELBO.
- **Strengths:** Scalable (can use SGD). Works with neural nets.
- **Weaknesses:** Approximation quality depends on family choice. Underestimates uncertainty.

### MCMC (Markov Chain Monte Carlo)

- **Metropolis-Hastings:** Propose candidate, accept/reject based on posterior ratio.
- **HMC (Hamiltonian Monte Carlo):** Use gradients to propose distant, high-probability samples. Much more efficient in high dimensions.
- **NUTS:** Auto-tunes HMC step size. Used by Stan/PyMC.
- **Strengths:** Asymptotically exact (unlike VI).
- **Weaknesses:** Slow. Hard to diagnose convergence. Scales poorly.

### MC Dropout

- **How:** Keep dropout on at test time. Run multiple forward passes. Variance of predictions ~ uncertainty.
- **Strengths:** Cheap approximate Bayes — no architecture changes.
- **Weaknesses:** Crude approximation. Dropout rate implicitly sets prior.

---

## 20. When to Use What

| Scenario | Recommendation |
|----------|----------------|
| Tabular, n > 1000 | Gradient Boosting (LightGBM/XGBoost) |
| Tabular, n < 500 | Regularized linear model or small RF |
| Images | CNN or Vision Transformer |
| Text | Transformer (BERT for understanding, GPT for generation) |
| Sequences (time series) | LSTM/GRU or Transformer |
| Interpretability critical | Single decision tree or linear model |
| Quick baseline | Random Forest or Logistic Regression |
| Small labeled + large unlabeled | Self-supervised pretraining + fine-tuning |
| Uncertainty needed | GP (small n) or MC Dropout (neural nets) |
| Anomaly detection | Isolation Forest or GMM |
| Clustering | K-Means (spherical), GMM (flexible), Hierarchical (structure) |
| Dimensionality reduction | PCA (linear), UMAP (nonlinear), t-SNE (visualization only) |
