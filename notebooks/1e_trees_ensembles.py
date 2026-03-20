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
    from sklearn.datasets import make_classification, make_moons
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import (
        RandomForestClassifier, RandomForestRegressor,
        GradientBoostingClassifier, GradientBoostingRegressor,
        StackingClassifier
    )
    from sklearn.model_selection import train_test_split
    return (
        DecisionTreeClassifier,
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
        StackingClassifier,
        make_classification,
        make_moons,
        np,
        plt,
        train_test_split,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Trees & Ensembles

    You now have a solid toolkit — linear and logistic regression, regularization, cross-validation, the bias-variance tradeoff. Those are the foundations. But here is a blunt truth about applied machine learning: **on tabular data, tree-based ensembles beat linear models almost every time.** If someone hands you a spreadsheet and asks you to predict something, your first instinct should be to reach for a random forest or gradient boosting, not linear regression. This module explains why.

    We will build up from a single decision tree — a model so simple you can draw it on a whiteboard — to random forests and gradient boosting machines, which are among the most powerful predictive models ever devised for structured data. The progression is logical: each method fixes a specific flaw in the previous one.

    Reading: [ISLR Ch 8: Tree-Based Methods](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf), [ESL Ch 9.2: Tree-Based Methods](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf), [ESL Ch 15: Random Forests](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf), [ESL Ch 10: Boosting and Additive Trees](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 1. Decision Trees

    A decision tree partitions the feature space into rectangles and fits a simple model (usually a constant) in each one. For regression, it predicts the mean of the training targets in each region. For classification, it predicts the majority class.

    The structure is a binary tree. At each internal node, a single feature $x_j$ is compared to a threshold $t$: go left if $x_j \le t$, go right otherwise. The leaves contain predictions. This is what makes trees so interpretable — you can literally trace the path from root to leaf and explain *why* a prediction was made.
    """)
    return


@app.cell
def _(mo):
    mo.image(src="../animations/rendered/DecisionTreeGrowth.gif")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 1.1 Recursive Binary Splitting

    How do we build the tree? The algorithm is greedy. At each node, we consider every feature and every possible split point, and we pick the one that most reduces our loss. Then we recurse on the two child nodes.

    For **regression trees**, we minimize the residual sum of squares. If a split divides the data into regions $R_1$ and $R_2$, the loss is:

    $$\sum_{i \in R_1}(y_i - \hat{y}_{R_1})^2 + \sum_{i \in R_2}(y_i - \hat{y}_{R_2})^2$$

    where $\hat{y}_{R_k}$ is the mean of the targets in region $R_k$. We search over all features $j$ and all thresholds $t$ to find the split that minimizes this.

    For **classification trees**, we need a measure of node "impurity" — how mixed the classes are.

    ### 1.2 Splitting Criteria: Gini Impurity and Entropy

    Let $\hat{p}_{mk}$ be the proportion of training observations in node $m$ that belong to class $k$. Two standard impurity measures:

    **Gini impurity:**

    $$G_m = \sum_{k=1}^{K} \hat{p}_{mk}(1 - \hat{p}_{mk}) = 1 - \sum_{k=1}^{K} \hat{p}_{mk}^2$$

    **Entropy (information gain):**

    $$H_m = -\sum_{k=1}^{K} \hat{p}_{mk} \log_2 \hat{p}_{mk}$$

    Why do these work? Consider the binary case ($K=2$). If a node is pure — all class 0 or all class 1 — then $\hat{p} = 0$ or $\hat{p} = 1$, and both Gini and entropy equal zero. If the node is maximally mixed ($\hat{p} = 0.5$), Gini reaches its maximum of $0.5$ and entropy reaches $1$ bit. Both are concave functions that peak at the uniform distribution and hit zero at the extremes. This means the algorithm *prefers splits that push child nodes toward purity*.

    The derivation of Gini is elegant. Think of it as the expected misclassification rate if you randomly labeled each observation according to the class distribution in the node. The probability of picking class $k$ is $\hat{p}_{mk}$, and the probability of misclassifying it is $(1 - \hat{p}_{mk})$. Summing over classes gives Gini. Entropy, on the other hand, comes from information theory — it measures the expected number of bits needed to encode the class label.

    In practice, Gini and entropy produce nearly identical trees. The real choice is between these and misclassification error, which is *not* a good splitting criterion because it is piecewise linear and less sensitive to changes in class probabilities.

    See [ISLR Section 8.1.2](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf) and [ESL Section 9.2.3](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf) for the full treatment.

    ### 1.3 Tree Pruning

    A fully grown tree will overfit — it will keep splitting until every leaf contains one observation. We need to prune.

    **Cost-complexity pruning** (also called weakest-link pruning) adds a penalty for tree size. For a subtree $T$ with $|T|$ terminal nodes:

    $$C_\alpha(T) = \sum_{m=1}^{|T|} \sum_{i \in R_m} (y_i - \hat{y}_{R_m})^2 + \alpha |T|$$

    The parameter $\alpha \ge 0$ controls the trade-off between fit and complexity. When $\alpha = 0$, the full tree is optimal. As $\alpha$ increases, smaller trees become optimal. We select $\alpha$ via cross-validation.

    This is exactly the same idea as regularization in ridge/lasso — penalize complexity to control overfitting. The difference is that here "complexity" means number of leaves, not magnitude of coefficients. See [ISLR Section 8.1.1: Pruning a Tree](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf).

    ### 1.4 Strengths and Weaknesses

    **Pros:**
    - Highly interpretable — you can visualize and explain the decision rules
    - Handle mixed feature types (numeric and categorical) naturally
    - No feature scaling needed — splits depend only on ordering, not magnitude
    - Capture interactions automatically (a split on $x_2$ in the left child of a split on $x_1$ is an interaction)
    - Robust to outliers in features

    **Cons:**
    - **High variance** — this is the fatal flaw, and the entire rest of this module is about fixing it
    - Axis-aligned splits only — decision boundaries are always perpendicular to feature axes
    - Overfit easily without pruning
    - Generally low predictive accuracy compared to other methods
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ### Interactive: Decision Tree Depth

    Use the slider below to control the maximum depth of a decision tree trained on synthetic 2D data. Watch how the decision boundary becomes more complex as depth increases — this is the overfitting problem in action.
    """)
    return


@app.cell
def _(mo):
    depth_slider = mo.ui.slider(start=1, stop=20, step=1, value=3, label="Max depth")
    depth_slider
    return (depth_slider,)


@app.cell
def _(DecisionTreeClassifier, depth_slider, make_moons, np, plt):
    # Generate synthetic 2D data
    X_tree_demo, y_tree_demo = make_moons(n_samples=300, noise=0.3, random_state=42)

    # Train decision tree with selected depth
    dt = DecisionTreeClassifier(max_depth=depth_slider.value, random_state=42)
    dt.fit(X_tree_demo, y_tree_demo)

    # Create meshgrid for decision boundary
    x_min, x_max = X_tree_demo[:, 0].min() - 0.5, X_tree_demo[:, 0].max() + 0.5
    y_min, y_max = X_tree_demo[:, 1].min() - 0.5, X_tree_demo[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    Z = dt.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
    ax.scatter(X_tree_demo[y_tree_demo == 0, 0], X_tree_demo[y_tree_demo == 0, 1],
               c='blue', edgecolors='k', s=20, label='Class 0')
    ax.scatter(X_tree_demo[y_tree_demo == 1, 0], X_tree_demo[y_tree_demo == 1, 1],
               c='red', edgecolors='k', s=20, label='Class 1')
    ax.set_title(f"Decision Tree (max_depth={depth_slider.value}) — "
                 f"Train accuracy: {dt.score(X_tree_demo, y_tree_demo):.3f}, "
                 f"Leaves: {dt.get_n_leaves()}")
    ax.legend()
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    plt.tight_layout()
    fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 2. Why Single Trees Fail

    Here is the core problem. Grow a decision tree on your training data. Now remove one data point and grow it again. The tree can look *completely different*. A small change near the root propagates down to every subsequent split, reshaping the entire structure.

    This is high variance in the bias-variance sense. A single decision tree is an **unstable learner** — its predictions change drastically with small perturbations in the training data. The tree itself has low bias (it can approximate complex functions if grown deep), but the variance is enormous.

    This observation is the motivation for everything that follows. If we could somehow reduce the variance of trees while keeping their low bias, we would have a powerful model. That is exactly what ensemble methods do.

    ---

    ## 3. Bagging (Bootstrap Aggregating)

    Leo Breiman's insight (1996) was simple: if variance is the problem, *average multiple trees*.

    **The algorithm:**
    1. Draw $B$ bootstrap samples from the training data (sample $n$ observations with replacement)
    2. Grow a full, unpruned decision tree on each bootstrap sample
    3. For regression: average the $B$ predictions. For classification: take a majority vote.

    Why does this work? Consider the variance of an average. If we have $B$ independent random variables, each with variance $\sigma^2$, the variance of their mean is $\sigma^2 / B$. More trees, less variance.

    But here is the catch: the trees are **not independent**. They are all trained on bootstrap samples from the same dataset, so they are correlated. If the pairwise correlation between trees is $\rho$, the variance of the average is:

    $$\text{Var}\left(\frac{1}{B}\sum_{b=1}^{B} T_b(x)\right) = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$$

    The second term vanishes as $B \to \infty$, but the first term — $\rho\sigma^2$ — does not. As long as the trees are correlated, averaging can only reduce variance so far. This formula is critical: it tells us that **decorrelating the trees matters more than adding more trees.** Remember this — it is the key idea behind random forests.

    ### 3.1 Out-of-Bag Error

    Each bootstrap sample includes roughly $1 - 1/e \approx 63.2\%$ of the training observations. The remaining ~36.8% are **out-of-bag (OOB)** for that tree. For each observation $x_i$, we can average predictions only from trees where $x_i$ was OOB — giving us an estimate of test error without needing a separate validation set.

    OOB error is essentially free cross-validation. It is a remarkably good estimate of the true generalization error, and it means you can train a bagged ensemble without worrying about a held-out set.

    See [ISLR Section 8.2.1: Bagging](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf) and [ESL Section 15.2](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 4. Random Forests

    Random forests (Breiman, 2001) add one twist to bagging: **at each split, only consider a random subset of $m$ features** (out of the total $p$ features). This is the decorrelation trick.

    Why does this help? In a standard bagged ensemble, if one feature is very strong, every tree will split on it first, and the trees will be highly correlated. The $\rho\sigma^2$ term in our variance formula stays large. By forcing each split to consider only a subset of features, we prevent the dominant feature from always being chosen. Different trees explore different parts of the feature space, and the correlation between trees drops.

    ### 4.1 The `mtry` Hyperparameter

    The number of features considered at each split, commonly called `mtry` (or `max_features`), is the key hyperparameter:

    - **Classification:** $m = \sqrt{p}$ (the default in most implementations)
    - **Regression:** $m = p/3$

    These defaults work surprisingly well in practice. Decreasing $m$ reduces correlation between trees (good) but increases the bias of each individual tree (bad). The sweet spot is usually close to the defaults, but it is worth tuning via cross-validation.

    ### 4.2 Feature Importance

    One of the most useful byproducts of random forests is feature importance ranking.

    **Permutation importance:** For each feature $x_j$, randomly shuffle its values in the OOB data and measure the increase in OOB error. If the feature matters, shuffling it destroys its relationship with the target and error increases dramatically. If it does not matter, the error barely changes. This is the more reliable method.

    **Impurity-based importance:** Sum the total decrease in Gini impurity (or MSE) from all splits on feature $x_j$ across all trees. Faster to compute but biased toward high-cardinality features — a feature with many unique values gets more opportunities to split and can appear artificially important.

    Always prefer permutation importance for inference. Use impurity-based importance only as a quick screening tool.

    ### 4.3 Can Random Forests Overfit?

    A common claim is that random forests "cannot overfit." This is misleading. Increasing the number of trees $B$ does not cause overfitting — the ensemble converges as $B$ grows. But each individual tree can overfit. If you set `min_samples_leaf=1` (grow fully), each tree memorizes noise, and even though averaging helps, it does not eliminate the problem entirely. In practice, random forests are remarkably robust to overfitting, but not immune.

    See [ISLR Section 8.2.2: Random Forests](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf), [ESL Chapter 15: Random Forests](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf), and [Murphy PML1 Section 18.5](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf).
    """)
    return


@app.cell
def _():
    from sklearn.ensemble import RandomForestClassifier as RFC_demo
    from sklearn.inspection import permutation_importance
    from sklearn.datasets import load_breast_cancer

    data_rf = load_breast_cancer()
    X_rf, y_rf = data_rf.data, data_rf.target
    feature_names_rf = data_rf.feature_names

    # Classification
    rf_clf = RFC_demo(
        n_estimators=500,        # number of trees
        max_features='sqrt',     # mtry = sqrt(p) for classification
        min_samples_leaf=5,      # mild regularization
        oob_score=True,          # compute OOB error
        n_jobs=-1,               # parallelize across all cores
        random_state=42
    )
    rf_clf.fit(X_rf, y_rf)
    print(f"OOB accuracy: {rf_clf.oob_score_:.4f}")

    # Permutation importance (use OOB)
    perm_imp = permutation_importance(rf_clf, X_rf, y_rf, n_repeats=10, random_state=42)
    print("\nTop 10 features by permutation importance:")
    for i in perm_imp.importances_mean.argsort()[::-1][:10]:
        print(f"  {feature_names_rf[i]}: {perm_imp.importances_mean[i]:.4f}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Interactive: Random Forest — OOB Error vs Number of Trees

    Use the slider to control the number of trees in a random forest. Watch the OOB error stabilize as more trees are added — this demonstrates that adding trees does not cause overfitting.
    """)
    return


@app.cell
def _(mo):
    n_trees_slider = mo.ui.slider(start=1, stop=200, step=5, value=10, label="Number of trees")
    n_trees_slider
    return (n_trees_slider,)


@app.cell
def _(make_classification, n_trees_slider, np, plt):
    from sklearn.ensemble import RandomForestClassifier as RFC_oob

    # Generate synthetic data
    X_oob, y_oob = make_classification(
        n_samples=500, n_features=20, n_informative=10,
        n_redundant=5, random_state=42
    )

    # Compute OOB error for each number of trees from 1 to slider value
    n_range = list(range(1, n_trees_slider.value + 1))
    oob_errors = []

    for n in n_range:
        rf_temp = RFC_oob(
            n_estimators=n, max_features='sqrt', oob_score=True,
            random_state=42, n_jobs=-1, warm_start=False
        )
        rf_temp.fit(X_oob, y_oob)
        oob_errors.append(1 - rf_temp.oob_score_)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(n_range, oob_errors, 'b-', linewidth=1.5)
    ax.set_xlabel("Number of Trees")
    ax.set_ylabel("OOB Error Rate")
    ax.set_title(f"Random Forest: OOB Error vs Number of Trees (current: {n_trees_slider.value})")
    ax.axhline(y=oob_errors[-1], color='r', linestyle='--', alpha=0.5,
               label=f"Current OOB error: {oob_errors[-1]:.4f}")
    ax.legend()
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 5. Boosting — Learning from Mistakes

    Bagging and random forests tackle the variance problem. Boosting attacks from the other side: **it reduces bias.** Instead of building independent trees and averaging them, boosting builds trees *sequentially*, where each new tree focuses on the errors of the previous ensemble.

    ### 5.1 AdaBoost

    AdaBoost (Freund & Schapire, 1997) was the first major boosting algorithm. The idea:

    1. Start with equal weights on all training observations
    2. Fit a weak learner (a shallow tree, often just a stump)
    3. Increase the weights of misclassified observations
    4. Fit the next weak learner on the reweighted data
    5. Repeat. The final prediction is a weighted vote of all learners.

    Observations that are hard to classify accumulate weight over iterations, forcing subsequent learners to focus on them. The "boosting" name comes from the theoretical result that combining many weak learners (barely better than random) produces a strong learner.

    ### 5.2 Gradient Boosting

    Gradient boosting (Friedman, 2001) generalizes AdaBoost to arbitrary differentiable loss functions. This is the framework that dominates modern ML competitions.

    **The algorithm:**

    1. Initialize the model with a constant prediction: $F_0(x) = \arg\min_\gamma \sum_{i=1}^n L(y_i, \gamma)$
    2. For $m = 1, 2, \ldots, M$:
       a. Compute the **negative gradient** (pseudo-residuals): $r_{im} = -\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\bigg|_{F=F_{m-1}}$
       b. Fit a regression tree $h_m(x)$ to the pseudo-residuals $r_{im}$
       c. Update the model: $F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$

    For squared error loss, the negative gradient is simply the residual $y_i - F_{m-1}(x_i)$, so gradient boosting literally fits trees to residuals. For other losses (logistic, Huber, quantile), the pseudo-residuals are different, but the principle is the same: each tree corrects the direction of steepest descent in function space.

    This is **functional gradient descent** — we are doing gradient descent, but instead of updating parameters, we are adding functions to our model. Each tree is a step in function space. See [ESL Section 10.10: Gradient Boosting](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf) and [ESL Section 10.1-10.4](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf).

    ### 5.3 Key Hyperparameters

    **Learning rate (shrinkage) $\eta$:** Each tree's contribution is scaled by $\eta \in (0, 1]$. Smaller values require more trees but generalize better. The intuition is the same as in gradient descent — a small step size prevents overshooting. Typical values: 0.01 to 0.1.

    **Number of trees $M$:** Unlike random forests, more trees *can* cause overfitting in boosting. There is an optimal $M$ that balances underfitting and overfitting. Select via early stopping on a validation set — stop training when validation error stops improving.

    **Tree depth:** Boosted trees should be *shallow*. Depth 4-8 is typical. Each tree is a "weak learner" — it does not need to be accurate on its own, just slightly better than random. Interaction depth $d$ controls the order of interactions the model can capture (depth $d$ allows $d$-way interactions).

    **Subsampling (stochastic gradient boosting):** Sample a fraction of the training data for each tree (e.g., 50-80%). This adds a bagging-like variance reduction and speeds up training.

    ### 5.4 The Bias-Variance Perspective

    This is worth being explicit about:

    | Method | Primary effect |
    |--------|----------------|
    | Bagging / Random Forests | Reduces **variance** (averaging decorrelated trees) |
    | Boosting | Reduces **bias** (sequentially correcting errors) |

    Bagging takes high-variance, low-bias models (deep trees) and averages them. Boosting takes high-bias, low-variance models (shallow trees) and combines them additively. Both produce strong learners, but from opposite directions.
    """)
    return


@app.cell
def _():
    from sklearn.ensemble import GradientBoostingClassifier as GBC_demo
    from sklearn.datasets import load_breast_cancer as lbc_gb
    from sklearn.model_selection import train_test_split as tts_gb

    X_gb, y_gb = lbc_gb(return_X_y=True)
    X_train_gb, X_test_gb, y_train_gb, y_test_gb = tts_gb(
        X_gb, y_gb, test_size=0.2, random_state=42, stratify=y_gb
    )

    gb_clf = GBC_demo(
        n_estimators=1000,       # max number of trees
        learning_rate=0.05,      # shrinkage
        max_depth=5,             # shallow trees
        subsample=0.8,           # stochastic gradient boosting
        validation_fraction=0.1, # hold out 10% for early stopping
        n_iter_no_change=20,     # stop if no improvement for 20 rounds
        random_state=42
    )
    gb_clf.fit(X_train_gb, y_train_gb)
    print(f"Trees used: {gb_clf.n_estimators_}")
    print(f"Test accuracy: {gb_clf.score(X_test_gb, y_test_gb):.4f}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 6. XGBoost, LightGBM, CatBoost — Modern Gradient Boosting

    Scikit-learn's `GradientBoostingClassifier` implements gradient boosting correctly but slowly. The libraries that dominate practice are **XGBoost**, **LightGBM**, and **CatBoost**. These are engineered for speed and accuracy on large datasets.

    ### 6.1 What Makes Them Fast

    **Histogram-based splitting:** Instead of evaluating every unique feature value as a potential split point, these libraries bin continuous features into a fixed number of buckets (e.g., 256). This reduces the complexity of finding the best split from $O(n)$ to $O(\text{bins})$ and dramatically speeds up training.

    **Leaf-wise (best-first) growth:** Traditional trees grow level-by-level (all nodes at depth $d$ before any at depth $d+1$). LightGBM and modern XGBoost grow leaf-wise — they expand the leaf with the highest potential gain, regardless of depth. This produces deeper, more asymmetric trees that can be more accurate with fewer leaves.

    **Parallel and distributed training:** Feature-parallel and data-parallel computation across CPU cores and machines.

    ### 6.2 Built-in Regularization

    XGBoost's objective function includes explicit regularization on the tree structure:

    $$\text{Obj} = \sum_{i=1}^n L(y_i, \hat{y}_i) + \sum_{m=1}^M \left[\gamma |T_m| + \frac{1}{2}\lambda \sum_{j=1}^{|T_m|} w_j^2 + \alpha \sum_{j=1}^{|T_m|} |w_j|\right]$$

    where $w_j$ are leaf weights, $\gamma$ penalizes the number of leaves, $\lambda$ is L2 regularization, and $\alpha$ is L1 regularization on leaf outputs. This is a principled way to control model complexity beyond just tree depth and learning rate.

    ### 6.3 Practical Guidance

    **XGBoost:** The original modern implementation (Chen & Guestrin, 2016). Mature, well-documented, wide language support. Good default choice.

    **LightGBM:** Faster training, especially on large datasets. Uses exclusive feature bundling and histogram differencing. Often the best choice when speed matters. Handles categorical features natively.

    **CatBoost:** Handles categorical features without preprocessing (ordered target encoding). Good out-of-the-box performance with less tuning. Uses ordered boosting to reduce target leakage.

    **When to use which:**
    - Default starting point: LightGBM (fast, accurate, good defaults)
    - Many categorical features: CatBoost
    - Need maximum ecosystem support / deployment flexibility: XGBoost
    - All three are excellent — differences are usually small
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ```python
    import lightgbm as lgb

    lgb_clf = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=31,            # controls tree complexity (leaf-wise)
        max_depth=-1,             # no limit (leaf-wise growth handles this)
        subsample=0.8,
        colsample_bytree=0.8,    # random feature subsets (like RF)
        reg_alpha=0.1,            # L1 regularization
        reg_lambda=1.0,           # L2 regularization
        n_jobs=-1,
    )
    lgb_clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    ```

    ```python
    import xgboost as xgb

    xgb_clf = xgb.XGBClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method='hist',       # histogram-based (fast)
        early_stopping_rounds=50,
        eval_metric='logloss',
        n_jobs=-1,
    )
    xgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
    ```

    See [ESL Section 10.11-10.14: Boosting implementation details](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf) and [Murphy PML1 Section 18.6](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 7. Stacking

    Stacking (Wolpert, 1992) is a different ensemble philosophy. Instead of combining many copies of the same model type, it combines *different* model types.

    **The idea:**
    1. Train several diverse base models (e.g., random forest, gradient boosting, logistic regression, SVM)
    2. Use their predictions as features for a **meta-model** (often logistic regression or a simple linear model)
    3. The meta-model learns the optimal way to combine the base models

    Critically, the base model predictions used to train the meta-model must be generated via cross-validation (out-of-fold predictions), not on the training set. Otherwise the meta-model will simply learn to trust whichever base model overfits the most.

    Stacking typically gives a small improvement (1-3%) over the best single model. It is common in competitions but rare in production due to the complexity of maintaining multiple model types. When you see a Kaggle winning solution, it almost always involves stacking.
    """)
    return


@app.cell
def _():
    from sklearn.ensemble import StackingClassifier as SC_demo, RandomForestClassifier as RFC_stack
    from sklearn.ensemble import GradientBoostingClassifier as GBC_stack
    from sklearn.linear_model import LogisticRegression as LR_stack
    from sklearn.datasets import load_breast_cancer as lbc_stack
    from sklearn.model_selection import train_test_split as tts_stack

    X_st, y_st = lbc_stack(return_X_y=True)
    X_train_st, X_test_st, y_train_st, y_test_st = tts_stack(
        X_st, y_st, test_size=0.2, random_state=42, stratify=y_st
    )

    stacking_clf = SC_demo(
        estimators=[
            ('rf', RFC_stack(n_estimators=200, random_state=42)),
            ('gb', GBC_stack(n_estimators=200, random_state=42)),
        ],
        final_estimator=LR_stack(),
        cv=5  # out-of-fold predictions for training meta-model
    )
    stacking_clf.fit(X_train_st, y_train_st)
    print(f"Stacking test accuracy: {stacking_clf.score(X_test_st, y_test_st):.4f}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 8. Trees vs Linear Models vs Neural Nets — When to Use What

    This is the practical question. After learning all these methods, which should you actually use?

    **Tabular / structured data (spreadsheets, databases):**
    Tree ensembles (gradient boosting) usually win. They handle nonlinearities, interactions, mixed feature types, and missing values naturally. Multiple benchmark studies (Grinsztajn et al., 2022) confirm that XGBoost/LightGBM outperform deep learning on tabular data in most cases.

    **Images, text, audio, video:**
    Neural networks (CNNs, Transformers) win decisively. Tree-based models cannot exploit spatial or sequential structure in raw pixel/token data. Do not use random forests on images.

    **Small datasets ($n < 500$):**
    Simpler models — logistic/linear regression, or random forests with few trees. Gradient boosting and neural nets need more data to shine. Regularized linear models can outperform here because they have strong inductive bias.

    **Interpretability is paramount:**
    A single (pruned) decision tree or a linear model. Random forests and gradient boosting are powerful but opaque. Feature importance helps, but it does not explain individual predictions the way a decision tree path does. (SHAP values can help bridge this gap — a topic for a later module.)

    **Time series:**
    Specialized methods often dominate (ARIMA, exponential smoothing, temporal CNNs/Transformers), but tree ensembles with engineered lag features are a strong baseline.

    | Scenario | Recommended approach |
    |----------|---------------------|
    | Tabular data, $n > 1000$ | Gradient boosting (LightGBM / XGBoost) |
    | Tabular data, $n < 500$ | Regularized linear model or small random forest |
    | Image classification | CNNs or Vision Transformers |
    | Text classification | Transformer-based models (BERT, etc.) |
    | Need full interpretability | Single decision tree or linear model |
    | Quick baseline | Random forest with defaults |

    See [ISLR Section 8.2.3-8.2.4](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf) and [Bishop PRML Section 14.2-14.3: Ensemble methods](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Summary

    The progression in this module follows a clear logic:

    1. **Decision trees** are interpretable but unstable (high variance)
    2. **Bagging** reduces variance by averaging many trees, but trees remain correlated
    3. **Random forests** decorrelate the trees by randomizing feature subsets — this is the key insight
    4. **Boosting** attacks bias instead of variance, building trees sequentially on residuals
    5. **Modern implementations** (XGBoost, LightGBM, CatBoost) make gradient boosting fast, regularized, and practical
    6. **Stacking** combines heterogeneous models for marginal gains

    The conceptual thread is bias-variance: bagging and random forests reduce variance through averaging; boosting reduces bias through sequential correction. Understanding *why* each method works, not just *how*, is what separates ML practitioners from API callers.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Practice

    ### Conceptual

    1. **Gini vs entropy:** Compute Gini impurity and entropy for a node with class distribution $[0.1, 0.9]$ and for $[0.5, 0.5]$. Verify that both measures are maximized at the uniform distribution and zero at purity.

    2. **Variance reduction in bagging:** Suppose you have $B = 100$ trees, each with prediction variance $\sigma^2 = 1$, and pairwise correlation $\rho = 0.4$. Compute the variance of the bagged ensemble using $\rho\sigma^2 + (1-\rho)\sigma^2/B$. Now compute it for $\rho = 0.1$ (i.e., after random forest decorrelation). How much does reducing correlation help compared to simply adding more trees?

    3. **Boosting overfitting:** Explain why increasing the number of trees can cause overfitting in gradient boosting but not in random forests. What role does the learning rate play?

    4. **Depth and interactions:** A gradient boosted ensemble uses trees of depth $d = 3$. What is the maximum order of feature interactions it can capture? Why do we prefer shallow trees in boosting but deep trees in bagging?

    ### Applied

    5. **Random forest from scratch:** Using the Iris or Wine dataset, train a random forest with `n_estimators` in $\{10, 50, 100, 500, 1000\}$. Plot OOB error vs number of trees. Verify that error stabilizes and does not increase.

    6. **Gradient boosting tuning:** On a regression dataset (e.g., California Housing), train a LightGBM model. Use 5-fold CV to tune `learning_rate` in $\{0.01, 0.05, 0.1\}$, `num_leaves` in $\{15, 31, 63, 127\}$, and `n_estimators` with early stopping. Report the best configuration and test RMSE.

    7. **Bagging vs boosting:** On the same dataset, compare the test error of a random forest and a gradient boosting model (both tuned). Generate learning curves — plot test error against training set size. Which method benefits more from additional data?

    8. **Feature importance comparison:** Train a random forest and compute both permutation importance and impurity-based importance. Find a case where the two rankings disagree. Which do you trust more, and why?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    **Key references:**
    - [ISLR Chapter 8: Tree-Based Methods](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf) — Clear introduction to trees, bagging, RF, boosting
    - [ESL Chapter 10: Boosting and Additive Trees](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf) — Theoretical depth on gradient boosting
    - [ESL Chapter 15: Random Forests](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf) — Variance analysis, feature importance
    - [Murphy PML1 Sections 18.5-18.6](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf) — Modern treatment of tree ensembles
    - [Bishop PRML Chapter 14](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) — Ensemble methods and boosting theory
    """)
    return


if __name__ == "__main__":
    app.run()
