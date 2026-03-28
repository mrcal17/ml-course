#flashcards/part1

## Decision Trees and Ensembles

What is the fundamental weakness of a single decision tree?
?
High variance. Small changes in the training data can produce a completely different tree structure. This makes individual trees unreliable, which is why ensembles (Bagging, RF, Boosting) exist.

How does a decision tree choose splits?
?
Greedy search: at each node, evaluate every feature and every threshold, pick the one that most reduces impurity. Gini impurity = 1 - sum(p_k^2). Entropy = -sum(p_k log p_k). For regression, minimize RSS within each resulting partition.

What is cost-complexity pruning?
?
After growing a full tree, prune by minimizing C_alpha(T) = loss + alpha * |leaves|. Alpha controls the tradeoff between fit and complexity. Select the best alpha via cross-validation. This prevents overfitting while keeping useful splits.

How does Bagging reduce variance, and what limits its effectiveness?
?
Bagging averages B bootstrap-trained models. For independent models, Var(avg) = sigma^2/B. But bootstrap samples from the same data produce correlated trees (strong features always dominate splits). The variance formula is Var = rho*sigma^2 + (1-rho)/B * sigma^2, where rho is tree correlation. The rho*sigma^2 term never vanishes.

What is Random Forest's key innovation over Bagging?
?
At each split, only consider m randomly selected features (not all p). Defaults: m = sqrt(p) for classification, p/3 for regression. This decorrelates the trees, reducing rho in the variance formula. More trees never hurts because averaging decorrelated models only reduces variance.

Why can't Random Forest overfit by adding more trees?
?
Each additional tree is an independent sample from the bagging distribution. Averaging more independent estimates can only reduce variance, never increase it. This is unlike boosting, where each additional model can overfit by chasing noise in the residuals.

Explain the Gradient Boosting algorithm in three steps.
?
1. Initialize F_0 = mean(y). 2. For each round m, compute pseudo-residuals r_i = negative gradient of the loss at current predictions. Fit a shallow tree h_m to these residuals. 3. Update: F_m = F_{m-1} + eta * h_m(x), where eta is the learning rate. Sequential error correction that reduces bias.

What are the four key hyperparameters for Gradient Boosting?
?
1. Learning rate eta (0.01-0.1): smaller = more trees needed but better generalization. 2. Number of trees M: use early stopping on validation loss. 3. Tree depth (3-8): controls interaction order; shallow = weak learners. 4. Subsampling rate (50-80%): adds randomness, reduces overfitting.

How do Bagging/RF and Boosting reduce error differently?
?
Bagging/RF reduces variance by averaging many noisy, decorrelated models. Boosting reduces bias by sequentially correcting errors. This is why boosting can overfit (chasing noise in residuals) while RF cannot (just averaging).

What differentiates XGBoost, LightGBM, and CatBoost?
?
XGBoost: explicit regularization with gamma*|T| + lambda*sum(w_j^2); widest ecosystem. LightGBM: histogram-based splits (~256 bins) with leaf-wise growth; fastest. CatBoost: native categorical feature handling with ordered target encoding; best defaults out of the box.

How does Stacking work, and how do you prevent meta-model overfitting?
?
Train diverse base models. Use their cross-validated (out-of-fold) predictions as features for a meta-model. The meta-model learns the optimal combination. You must use out-of-fold predictions; using in-sample predictions would leak information and cause the meta-model to overfit.

What does each ensemble method reduce?

| Method | Reduces |
|--------|---------|
?
| Bagging/RF | Variance (averaging decorrelated models) |
| Boosting | Bias (sequential error correction) |
| Stacking | Both (meta-model learns optimal blend) |
