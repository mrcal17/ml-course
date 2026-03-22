import marimo

app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Model Selection & Evaluation

    ## The Evaluation Problem

    You've now built two kinds of models: regression (1A-1B) and classification (1C). You've fit parameters, minimized loss functions, and made predictions. Here's the uncomfortable question you should be asking: **how do you know your model is any good?**

    The naive answer is "look at the training error." The training error tells you how well your model fits the data it was trained on. This is almost completely useless information. A polynomial of degree $n-1$ will perfectly interpolate $n$ training points — training error zero — and it will make catastrophically bad predictions on new data. You already know this intuitively; it's called overfitting.

    What you actually care about is **generalization error**: how well the model performs on data it has never seen. The entire field of model selection and evaluation is about estimating this quantity honestly, and then using that estimate to choose between competing models. It's harder than it sounds, because every shortcut you take here will give you an inflated, dishonest estimate of your model's true performance.

    This lecture is about the tools for doing it right: how to split your data, how to estimate generalization error, how to tune hyperparameters without fooling yourself, and how to compare models fairly.

    For a thorough treatment of all these topics, see [ISLR Chapter 5: Resampling Methods](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf) and [ESL Chapter 7: Model Assessment and Selection](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf). The ESL chapter is more mathematical; the ISLR chapter is gentler. Read both eventually.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Overfitting in Action

    Before we get into evaluation methods, let's see why training error is misleading. We'll fit polynomials of increasing degree to noisy data and watch training error go to zero while true error explodes.
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    # Generate noisy data from a simple quadratic
    rng = np.random.default_rng(42)
    n_pts = 20
    X_demo = rng.uniform(0, 1, n_pts)
    y_demo = 2 * X_demo - 3 * X_demo**2 + rng.normal(0, 0.3, n_pts)  # true function + noise

    # Fit polynomials of degree 1 through 15, track train error
    degrees = range(1, 16)
    train_errors = []
    for d in degrees:
        coeffs = np.polyfit(X_demo, y_demo, d)
        y_pred = np.polyval(coeffs, X_demo)
        mse = np.mean((y_demo - y_pred) ** 2)  # MSE = (1/n) * sum((y - y_hat)^2)
        train_errors.append(mse)

    plt.figure(figsize=(7, 3.5))
    plt.plot(list(degrees), train_errors, "o-")
    plt.xlabel("Polynomial degree")
    plt.ylabel("Training MSE")
    plt.title("Training error drops to zero — but the model is getting WORSE")
    plt.yscale("log")
    plt.tight_layout()
    plt.gca()
    return (np, plt, rng)


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 1. Train / Validation / Test Split

    ### The Basic Idea

    The simplest approach to estimating generalization error: take your dataset, hide some portion of it, train your model on the rest, and then evaluate on the hidden portion. The hidden portion simulates "data the model has never seen."

    This is the **train/test split**, and you've seen it before. But there's a subtlety that trips people up.

    ### Why You Need Three Sets, Not Two

    Suppose you train a model on the training set and evaluate on the test set. You get 85% accuracy. You try a different model — maybe you add polynomial features, or change the regularization strength. You evaluate on the test set again. 87%. You keep going, trying dozens of configurations, always evaluating on the test set.

    Here's the problem: **you are now optimizing against the test set.** Every time you make a decision based on test set performance, you're leaking information from the test set into your modeling choices. After 50 rounds of this, your reported "test accuracy" is optimistic — it reflects how well you tuned to this particular test set, not how well your model generalizes to truly unseen data.

    The solution is to split your data three ways:

    | Set | Purpose | When you look at it |
    |-----|---------|-------------------|
    | **Training set** | Fit model parameters | Every training run |
    | **Validation set** | Tune hyperparameters, select models | During development |
    | **Test set** | Final, unbiased performance estimate | Once, at the very end |

    The **validation set** is your "disposable test set." You use it to make decisions — which model architecture, which hyperparameters, which features. You accept that your estimate on the validation set is slightly optimistic (because you're selecting based on it), but that's fine. The test set remains untouched, pristine, ready to give you an honest final number.

    **The cardinal sin**: using the test set to guide any modeling decision. If you evaluate on the test set and then go back and change your model, the test set is no longer a test set — it's a validation set, and you no longer have a clean estimate of generalization error.

    ### Typical Split Ratios

    The conventional wisdom:

    - **Small datasets** (hundreds to low thousands): 60/20/20 train/val/test
    - **Medium datasets** (tens of thousands): 70/15/15 or 80/10/10
    - **Large datasets** (millions+): 98/1/1 — when you have a million examples, 10,000 is plenty for validation

    These are guidelines, not laws. The key constraint is that each set must be large enough to give reliable estimates.

    See [ISLR Section 5.1: The Validation Set Approach](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf) for discussion of the limitations of a single validation split.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Train/Val/Test Split in Code

    Here is the three-way split implemented from scratch with numpy, then with sklearn.
    """)
    return


@app.cell
def _(np):
    from sklearn.datasets import load_diabetes

    # Load data
    X_diabetes, y_diabetes = load_diabetes(return_X_y=True)
    n_samples = len(y_diabetes)

    # --- Pure numpy three-way split ---
    rng_split = np.random.default_rng(42)
    indices = rng_split.permutation(n_samples)  # shuffle indices
    n_train = int(0.6 * n_samples)
    n_val = int(0.2 * n_samples)
    # train = first 60%, val = next 20%, test = last 20%
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    print(f"Numpy split — train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")

    # --- sklearn shortcut: two calls to train_test_split ---
    from sklearn.model_selection import train_test_split
    X_temp, X_test_split, y_temp, y_test_split = train_test_split(
        X_diabetes, y_diabetes, test_size=0.2, random_state=42)
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2
    print(f"Sklearn split — train: {len(X_train_split)}, val: {len(X_val_split)}, test: {len(X_test_split)}")
    return (X_diabetes, y_diabetes)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2. Cross-Validation

    ### The Problem with a Single Split

    A single train/validation split has a flaw: your performance estimate depends on which particular data points ended up in which set. If you happened to put all the "easy" examples in the validation set, you'll think your model is better than it is. If you got unlucky and put the hardest examples there, you'll underestimate.

    With small datasets, this variance is severe. Cross-validation solves it.

    ### k-Fold Cross-Validation

    The procedure, step by step:

    1. Shuffle your dataset randomly
    2. Split it into $k$ roughly equal-sized groups (folds)
    3. For $i = 1, 2, \ldots, k$:
       - Use fold $i$ as validation
       - Train on the remaining $k-1$ folds combined
       - Record the validation error on fold $i$
    4. Average the $k$ validation errors

    That's it. Every data point gets to be in the validation set exactly once, and in the training set $k-1$ times. The average gives you a more stable estimate than any single split.

    **$k = 5$ and $k = 10$ are the standard choices.** Empirically, 10-fold CV tends to give a good balance of bias and variance in the error estimate. 5-fold is faster and often nearly as good.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### k-Fold CV from Scratch

    Let's implement the k-fold loop manually so you see exactly what's happening — no sklearn magic.
    """)
    return


@app.cell
def _(X_diabetes, np, y_diabetes):
    from sklearn.linear_model import Ridge

    # Manual k-fold cross-validation
    k = 5
    n = len(y_diabetes)
    fold_size = n // k
    rng_cv = np.random.default_rng(0)
    perm = rng_cv.permutation(n)

    mse_folds_manual = []
    for i in range(k):
        # Carve out fold i as validation
        val_mask = perm[i * fold_size : (i + 1) * fold_size]
        train_mask = np.concatenate([perm[:i * fold_size], perm[(i + 1) * fold_size:]])

        model = Ridge(alpha=1.0)
        model.fit(X_diabetes[train_mask], y_diabetes[train_mask])
        y_pred = model.predict(X_diabetes[val_mask])
        mse = np.mean((y_diabetes[val_mask] - y_pred) ** 2)  # MSE for this fold
        mse_folds_manual.append(mse)

    print(f"Manual {k}-fold CV  — Mean MSE: {np.mean(mse_folds_manual):.2f} ± {np.std(mse_folds_manual):.2f}")
    return


@app.cell
def _(X_diabetes, y_diabetes):
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import Ridge as Ridge_cv

    # 10-fold CV with Ridge regression
    scores = cross_val_score(Ridge_cv(alpha=1.0), X_diabetes, y_diabetes, cv=10, scoring='neg_mean_squared_error')
    mse_per_fold = -scores  # sklearn uses negative MSE (convention: higher = better)

    print(f"Mean MSE: {mse_per_fold.mean():.4f}")
    print(f"Std MSE:  {mse_per_fold.std():.4f}")
    print(f"\nPer-fold MSEs: {[f'{m:.1f}' for m in mse_per_fold]}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The standard deviation across folds tells you how variable the estimate is. If one fold gives you MSE of 2.3 and another gives 8.7, your model's performance is unstable — probably because your dataset is small or heterogeneous.

    ### Leave-One-Out Cross-Validation (LOOCV)

    The extreme case: $k = n$. Each fold contains a single data point. You train $n$ models, each on $n-1$ points, and test on the held-out one.

    **Pros:** Almost no bias (training on nearly the full dataset each time). Deterministic (no randomness in the split).
    **Cons:** Extremely expensive — you train $n$ separate models. High variance in the error estimate, because the $n$ training sets overlap almost completely, making the individual errors highly correlated.

    LOOCV makes sense when $n$ is very small (say, under 100) and training is cheap. For linear models there's a computational shortcut that makes LOOCV essentially free — see [ISLR Section 5.1.3](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf) and [ESL Section 7.10](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf).

    ### Stratified k-Fold

    When doing classification, you want each fold to have roughly the same class distribution as the full dataset. If your dataset is 90% class 0 and 10% class 1, a naive random split might produce a fold with zero class 1 examples.

    **Stratified k-fold** handles this by sampling from each class proportionally. In scikit-learn:
    """)
    return


@app.cell
def _():
    from sklearn.model_selection import StratifiedKFold
    from sklearn.datasets import load_breast_cancer

    X_bc, y_bc = load_breast_cancer(return_X_y=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_bc, y_bc)):
        X_train_fold, X_val_fold = X_bc[train_idx], X_bc[val_idx]
        y_train_fold, y_val_fold = y_bc[train_idx], y_bc[val_idx]
        print(f"Fold {fold_idx+1}: train={len(train_idx)}, val={len(val_idx)}, "
              f"class 1 ratio in val={y_val_fold.mean():.3f}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    Always use stratified folds for classification. There's almost never a reason not to.

    ### Nested Cross-Validation

    Here's a subtle problem. Suppose you use 5-fold CV to select the best hyperparameters (say, the best regularization strength $\alpha$). Now you want to report the model's performance. Can you use that same 5-fold CV score?

    No. That score is optimistic, because you selected the $\alpha$ that looked best on those folds. You've used the validation data for model selection, so it can't also give you an unbiased performance estimate. This is the same logic as the train/val/test split — using the same data for both selection and evaluation produces optimistic estimates.

    **Nested CV** solves this with two layers of cross-validation:

    - **Outer loop** (e.g., 5-fold): each fold is a train/test split for evaluation
    - **Inner loop** (e.g., 5-fold): within each outer training set, run CV to select the best hyperparameters

    ```
    For each outer fold:
        outer_train, outer_test = split data
        For each hyperparameter setting:
            inner_cv_score = cross_val on outer_train  # inner loop
        Pick best hyperparameters based on inner CV
        Train final model on outer_train with best hyperparameters
        Evaluate on outer_test  →  this gives unbiased estimate
    Average the outer test scores
    ```

    This gives you an honest estimate of the performance of "the entire pipeline, including hyperparameter tuning." It's computationally expensive ($k_{\text{outer}} \times k_{\text{inner}} \times$ number of hyperparameter settings) but it's correct.
    """)
    return


@app.cell
def _(X_diabetes, y_diabetes):
    from sklearn.model_selection import cross_val_score as cvs_nested, GridSearchCV as GSV_nested
    from sklearn.linear_model import Ridge as Ridge_nested

    # Inner CV is handled by GridSearchCV
    inner_cv = GSV_nested(Ridge_nested(), {'alpha': [0.01, 0.1, 1, 10, 100]}, cv=5)

    # Outer CV estimates the performance of the entire tuning procedure
    outer_scores = cvs_nested(inner_cv, X_diabetes, y_diabetes, cv=5, scoring='neg_mean_squared_error')
    print(f"Nested CV MSE: {(-outer_scores).mean():.4f} ± {(-outer_scores).std():.4f}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    For more on the bias-variance tradeoff in cross-validation, see [ESL Section 7.10: Cross-Validation](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf).

    ---

    ## 3. The Bootstrap

    ### Sampling with Replacement

    The bootstrap is a different resampling idea. Instead of partitioning the data into folds, you create new datasets by **sampling with replacement** from the original data.

    Given $n$ data points, draw $n$ samples with replacement. Some points will appear multiple times; others won't appear at all. The probability that any particular point is not selected in $n$ draws is $(1 - 1/n)^n \approx e^{-1} \approx 0.368$. So roughly 63.2% of the original data points appear in each bootstrap sample, and 36.8% are left out.

    Repeat this process $B$ times (say, $B = 200$), training your model on each bootstrap sample and evaluating on the left-out points.

    ### Bootstrap Estimates of Standard Error

    The bootstrap's most common use is estimating the **standard error** of a statistic. Want to know how variable your accuracy estimate is? Compute accuracy on 200 bootstrap samples and take the standard deviation. That's your bootstrap standard error.

    This is more general than cross-validation: you can bootstrap any statistic (median, correlation, regression coefficient — anything), whereas cross-validation is specifically about prediction error.

    ### The .632 Bootstrap

    The naive bootstrap error estimate (train on bootstrap sample, test on out-of-bag points) is biased. The training set is smaller than the original dataset (only ~63.2% unique points), so the model slightly underfits, and the error estimate is slightly pessimistic.

    The **.632 bootstrap** corrects for this:

    $$\hat{Err}_{.632} = 0.368 \cdot \overline{err}_{\text{train}} + 0.632 \cdot \overline{err}_{\text{oob}}$$

    where $\overline{err}_{\text{train}}$ is the average training error and $\overline{err}_{\text{oob}}$ is the average out-of-bag error. This blends the optimistic training error with the pessimistic out-of-bag error.

    See [ESL Section 7.11: Bootstrap Methods](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf) for the full derivation and the improved .632+ estimator.

    ### When to Use Bootstrap vs Cross-Validation

    - **For prediction error estimation**: Cross-validation is generally preferred. It's simpler, well-understood, and has less bias.
    - **For standard errors of statistics**: Bootstrap. It's the universal tool for uncertainty quantification.
    - **For very small datasets**: Bootstrap can be useful because it doesn't require partitioning your already-limited data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Bootstrap in Code

    Let's implement a bootstrap to estimate the standard error of a regression coefficient and verify the ~63.2% coverage.
    """)
    return


@app.cell
def _(X_diabetes, np, y_diabetes):
    from sklearn.linear_model import LinearRegression as LR_boot

    # Bootstrap: estimate std error of the first regression coefficient
    B = 200  # number of bootstrap samples
    n_boot = len(y_diabetes)
    rng_boot = np.random.default_rng(42)
    coef_samples = []

    for _ in range(B):
        # Sample n points WITH replacement — P(point not chosen) = (1-1/n)^n ≈ 0.368
        idx = rng_boot.integers(0, n_boot, size=n_boot)
        model_b = LR_boot().fit(X_diabetes[idx], y_diabetes[idx])
        coef_samples.append(model_b.coef_[0])  # first feature's coefficient

    coef_samples = np.array(coef_samples)
    print(f"Bootstrap estimate of coef[0]: {coef_samples.mean():.2f}")
    print(f"Bootstrap standard error:      {coef_samples.std():.2f}")

    # Verify ~63.2% unique points per sample
    sample_idx = rng_boot.integers(0, n_boot, size=n_boot)
    pct_unique = len(np.unique(sample_idx)) / n_boot
    print(f"\nUnique points in one sample:   {pct_unique:.1%} (theory: ~63.2%)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4. Hyperparameter Tuning

    ### Parameters vs Hyperparameters

    A distinction you need to have crystal clear:

    - **Parameters** are learned from the data during training. The weights $w$ in linear regression. The coefficients in logistic regression. These are found by optimization (gradient descent, closed-form solutions, etc.).
    - **Hyperparameters** are set before training and control the learning process. The regularization strength $\lambda$. The number of folds $k$ in CV. The learning rate. The polynomial degree. These are not learned — you choose them.

    The model can't tune its own hyperparameters (not without nesting — which is what this section is about). You, the practitioner, must choose them. The question is how.

    ### Grid Search
    """)
    return


@app.cell
def _():
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_breast_cancer as load_bc_gs

    X_gs, y_gs = load_bc_gs(return_X_y=True)
    X_train_gs, X_test_gs, y_train_gs, y_test_gs = train_test_split(
        X_gs, y_gs, test_size=0.2, random_state=42, stratify=y_gs
    )

    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],       # inverse regularization strength
        'penalty': ['l1', 'l2'],                      # regularization type
    }

    grid_search = GridSearchCV(
        LogisticRegression(solver='saga', max_iter=5000),
        param_grid,
        cv=5,                    # 5-fold CV for each combo
        scoring='accuracy',
        n_jobs=-1,               # parallelize across CPU cores
        return_train_score=True  # track overfitting
    )
    grid_search.fit(X_train_gs, y_train_gs)

    print(f"Best params: {grid_search.best_params_}")
    print(f"Best CV accuracy: {grid_search.best_score_:.4f}")

    # Final evaluation on test set
    test_acc = grid_search.score(X_test_gs, y_test_gs)
    print(f"Test accuracy: {test_acc:.4f}")
    return (X_train_gs, y_train_gs,)


@app.cell
def _(mo):
    mo.md(r"""
    Grid search is exhaustive and reliable, but it scales exponentially with the number of hyperparameters. Two hyperparameters with 10 values each = 100 combinations. Five hyperparameters with 10 values each = 100,000 combinations. Each combination requires a full cross-validation loop. This gets expensive fast.

    ### Random Search

    Here's a result that surprises people: **random search is almost always more efficient than grid search.** Bergstra and Bengio (2012) showed this convincingly. The intuition is that for most problems, only one or two hyperparameters actually matter much. Grid search wastes budget exploring combinations that differ only in the unimportant hyperparameters. Random search, by contrast, explores a wider range of each important hyperparameter.
    """)
    return


@app.cell
def _(X_train_gs, y_train_gs):
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.linear_model import LogisticRegression as LR_rand
    from scipy.stats import loguniform

    param_distributions = {
        'C': loguniform(1e-3, 1e3),    # sample C on a log scale
        'penalty': ['l1', 'l2'],
    }

    random_search = RandomizedSearchCV(
        LR_rand(solver='saga', max_iter=5000),
        param_distributions,
        n_iter=50,               # try 50 random combinations
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42
    )
    random_search.fit(X_train_gs, y_train_gs)

    print(f"Best params (random search): {random_search.best_params_}")
    print(f"Best CV accuracy (random search): {random_search.best_score_:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visualizing the Hyperparameter Landscape

    Let's plot how CV accuracy changes with the regularization strength C. This makes clear why log-scale sampling matters.
    """)
    return


@app.cell
def _(X_train_gs, np, plt, y_train_gs):
    from sklearn.model_selection import cross_val_score as cvs_landscape
    from sklearn.linear_model import LogisticRegression as LR_landscape

    # Sweep C on a log scale and record CV accuracy
    C_values = np.logspace(-3, 3, 15)  # 15 values from 0.001 to 1000
    mean_accs = []
    std_accs = []
    for C in C_values:
        scores_c = cvs_landscape(
            LR_landscape(C=C, solver='saga', max_iter=5000, penalty='l2'),
            X_train_gs, y_train_gs, cv=5, scoring='accuracy')
        mean_accs.append(scores_c.mean())
        std_accs.append(scores_c.std())

    mean_accs = np.array(mean_accs)
    std_accs = np.array(std_accs)

    plt.figure(figsize=(7, 3.5))
    plt.semilogx(C_values, mean_accs, "o-")
    plt.fill_between(C_values, mean_accs - std_accs, mean_accs + std_accs, alpha=0.2)
    plt.xlabel("C (inverse regularization)")
    plt.ylabel("CV Accuracy")
    plt.title("Hyperparameter landscape — accuracy vs C")
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    Notice `loguniform` for the regularization parameter — hyperparameters that span orders of magnitude should be sampled on a log scale, not a linear one. Sampling $C$ uniformly from $[0.001, 1000]$ would almost never try values near 0.001; sampling on a log scale gives equal probability to each order of magnitude.

    ### Bayesian Optimization (Conceptual)

    Grid and random search treat each trial independently — they don't learn from previous evaluations. Bayesian optimization is smarter: it builds a surrogate model (usually a Gaussian process) of the function "hyperparameters → validation score" and uses it to decide which hyperparameters to try next.

    The idea: after a few evaluations, the surrogate model has an estimate of where the validation score is likely high (exploit) and where it's uncertain (explore). An acquisition function balances exploration and exploitation to pick the next point. This typically finds good hyperparameters in far fewer evaluations than random search.

    Libraries like `optuna` and `scikit-optimize` implement this. We won't go deep here — the point is to know that smarter methods exist when tuning is expensive (e.g., training a neural network takes hours per configuration).

    ### Early Stopping as Implicit Regularization

    For iterative algorithms (gradient descent, boosting), there's a hyperparameter hiding in plain sight: **how many iterations to run.** If you train too long, the model overfits. If you stop early, you get a simpler model.

    Early stopping monitors validation error during training and stops when it starts increasing. This acts as a regularizer — it constrains the effective complexity of the model without explicitly adding a penalty term. We'll see this repeatedly in deep learning.

    See [ISLR Chapter 5](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf) and [ESL Chapter 7](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf) for the formal treatment of these topics.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 5. Model Comparison

    ### Comparing Models Fairly

    You've trained three models on the same data using cross-validation. Model A gets 87% accuracy, Model B gets 85%, Model C gets 86%. Is Model A genuinely better, or is the difference within the noise?

    This matters. A 2% difference in CV accuracy could easily be due to which points ended up in which folds. If you always pick the model with the highest CV score without asking "is this difference meaningful?", you'll sometimes pick a worse model that got lucky.

    ### Paired Tests

    Because the models are evaluated on the same folds, you can use **paired comparisons**. For each fold $i$, compute the difference $d_i = \text{error}_A^{(i)} - \text{error}_B^{(i)}$. Then test whether the average difference is significantly different from zero using a paired $t$-test (or a Wilcoxon signed-rank test if you're uncomfortable with normality assumptions).

    ### Practical vs Statistical Significance

    A paired $t$-test might tell you that the 2% difference is statistically significant ($p < 0.05$). But is 2% accuracy worth the added complexity of Model A? If Model A is a complex ensemble that takes 10 minutes to train and Model B is a logistic regression that trains in milliseconds, you might prefer Model B.

    Statistical significance tells you whether a difference is real. **Practical significance** tells you whether it matters. Always consider both.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Paired Model Comparison in Code

    Let's compare Ridge and Lasso on the same CV folds and run a paired t-test.
    """)
    return


@app.cell
def _(X_diabetes, np, y_diabetes):
    from sklearn.model_selection import KFold, cross_val_score as cvs_compare
    from sklearn.linear_model import Ridge as Ridge_cmp, Lasso as Lasso_cmp
    from scipy.stats import ttest_rel

    # Use the SAME folds for both models — this is what makes the test paired
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    ridge_scores = cvs_compare(Ridge_cmp(alpha=1.0), X_diabetes, y_diabetes,
                               cv=kf, scoring='neg_mean_squared_error')
    lasso_scores = cvs_compare(Lasso_cmp(alpha=1.0), X_diabetes, y_diabetes,
                               cv=kf, scoring='neg_mean_squared_error')

    # Paired differences: d_i = error_ridge(fold i) - error_lasso(fold i)
    diff = (-ridge_scores) - (-lasso_scores)  # positive means Ridge is worse
    t_stat, p_value = ttest_rel(-ridge_scores, -lasso_scores)

    print(f"Ridge mean MSE: {(-ridge_scores).mean():.2f}")
    print(f"Lasso mean MSE: {(-lasso_scores).mean():.2f}")
    print(f"Mean difference: {diff.mean():.2f} ± {diff.std():.2f}")
    print(f"Paired t-test:  t={t_stat:.3f}, p={p_value:.4f}")
    print(f"Significant at 0.05? {'Yes' if p_value < 0.05 else 'No'}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 6. Information Criteria

    ### AIC and BIC

    When your model is fit by maximum likelihood, there are analytic alternatives to cross-validation for model selection. The idea: penalize the log-likelihood by a term that grows with model complexity.

    **Akaike Information Criterion (AIC):**

    $$\text{AIC} = 2k - 2\ln(\hat{L})$$

    **Bayesian Information Criterion (BIC):**

    $$\text{BIC} = k \ln(n) - 2\ln(\hat{L})$$

    where $k$ is the number of parameters, $n$ is the number of observations, and $\hat{L}$ is the maximized likelihood. Lower is better for both.

    The key difference: BIC penalizes complexity more heavily (especially for large $n$, since $\ln(n) > 2$ once $n > 7$). BIC tends to select simpler models than AIC.

    ### When to Use Information Criteria vs Cross-Validation

    Information criteria are fast — they require fitting the model only once, not $k$ times. But they rely on assumptions:

    - The model must be fit by maximum likelihood (or an approximation)
    - They assume the true model is in your candidate set (AIC) or is approximated by it
    - They don't account for overfitting due to model search

    Cross-validation is more general: it works for any model, any loss function, and directly estimates what you care about (prediction error). Use information criteria when you need speed and your models satisfy the assumptions. Use cross-validation otherwise.

    For a deeper treatment, see [ESL Section 7.5-7.7](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf) and [Bishop PRML Section 1.3](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) on model selection.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Computing AIC and BIC from Scratch

    For linear regression with Gaussian errors, the log-likelihood has a closed form. Let's compute AIC/BIC and compare models of different complexity.
    """)
    return


@app.cell
def _(X_diabetes, np, plt, y_diabetes):
    from sklearn.linear_model import LinearRegression as LR_ic
    from sklearn.preprocessing import PolynomialFeatures

    n_ic = len(y_diabetes)
    # Use just one feature for simplicity
    X_1d = X_diabetes[:, 2:3]  # BMI feature

    aic_vals, bic_vals = [], []
    degrees_ic = range(1, 10)
    for deg in degrees_ic:
        # Create polynomial features of degree `deg`
        X_poly = PolynomialFeatures(deg, include_bias=False).fit_transform(X_1d)
        model_ic = LR_ic().fit(X_poly, y_diabetes)
        y_hat = model_ic.predict(X_poly)
        rss = np.sum((y_diabetes - y_hat) ** 2)

        k = deg + 1  # number of parameters (coefficients + intercept)
        # Log-likelihood for Gaussian: -n/2 * ln(RSS/n) + const
        log_lik = -n_ic / 2 * np.log(rss / n_ic)
        aic = 2 * k - 2 * log_lik          # AIC = 2k - 2ln(L)
        bic = k * np.log(n_ic) - 2 * log_lik  # BIC = k*ln(n) - 2ln(L)
        aic_vals.append(aic)
        bic_vals.append(bic)

    plt.figure(figsize=(7, 3.5))
    plt.plot(list(degrees_ic), aic_vals, "o-", label="AIC")
    plt.plot(list(degrees_ic), bic_vals, "s-", label="BIC")
    plt.xlabel("Polynomial degree")
    plt.ylabel("Information criterion (lower is better)")
    plt.legend()
    plt.title("AIC vs BIC — BIC penalizes complexity more")
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 7. Feature Selection

    ### Why Select Features?

    More features isn't always better. Irrelevant features add noise, increase overfitting, slow down training, and make models harder to interpret. Feature selection is about finding the subset of features that gives you the best predictive performance.

    ### Subset Selection Methods

    **Forward selection:** Start with no features. Add the one that improves the model the most. Repeat until adding more features doesn't help (or until you reach a budget).

    **Backward elimination:** Start with all features. Remove the one whose removal hurts the model the least. Repeat until all remaining features are "important."

    **Stepwise:** Combine forward and backward — at each step, consider both adding and removing features.

    These methods are greedy and can miss the globally optimal subset, but they're practical. See [ISLR Section 6.1: Subset Selection](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf) for a full treatment.
    """)
    return


@app.cell
def _():
    from sklearn.feature_selection import SequentialFeatureSelector
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split as tts_fs
    from sklearn.datasets import load_diabetes as ld_fs

    data_fs = ld_fs()
    X_fs, y_fs = data_fs.data, data_fs.target
    feature_names_fs = data_fs.feature_names
    X_train_fs, X_test_fs, y_train_fs, y_test_fs = tts_fs(X_fs, y_fs, test_size=0.2, random_state=42)

    # Forward selection with 5-fold CV
    sfs = SequentialFeatureSelector(
        LinearRegression(),
        n_features_to_select=5,
        direction='forward',
        cv=5,
        scoring='neg_mean_squared_error'
    )
    sfs.fit(X_train_fs, y_train_fs)
    selected_mask = sfs.get_support()
    print(f"Selected features: {[feature_names_fs[i] for i, s in enumerate(selected_mask) if s]}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### L1 Regularization as Automatic Feature Selection

    Recall from Module 1B that L1 regularization (Lasso) drives some coefficients exactly to zero. This is feature selection built into the optimization — the model automatically decides which features to keep and which to discard. No explicit search required.

    This is one of the most elegant results in all of machine learning. The L1 penalty acts as a continuous relaxation of the discrete "select a subset" problem. See [ISLR Section 6.2.2: The Lasso](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf) for a refresher.

    ### Feature Importance from Models

    Some models naturally produce a ranking of feature importance. Decision trees (which you'll see next) rank features by how much they reduce impurity. Random forests average these importances across many trees, giving robust rankings. We'll explore this in Module 1E.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Lasso Feature Selection in Code

    Let's see Lasso zeroing out irrelevant features on the diabetes dataset.
    """)
    return


@app.cell
def _(X_diabetes, np, plt, y_diabetes):
    from sklearn.linear_model import Lasso as Lasso_fs
    from sklearn.datasets import load_diabetes as ld_fs2

    feature_names_lasso = ld_fs2().feature_names

    # Sweep alpha: large alpha → more zeros → fewer features
    alphas_lasso = [0.01, 0.1, 1.0, 10.0]
    for alpha in alphas_lasso:
        lasso = Lasso_fs(alpha=alpha).fit(X_diabetes, y_diabetes)
        n_nonzero = np.sum(lasso.coef_ != 0)  # L1 drives some coefs to exactly 0
        print(f"alpha={alpha:>5.2f} → {n_nonzero} non-zero coefficients: "
              f"{[feature_names_lasso[i] for i in np.where(lasso.coef_ != 0)[0]]}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 8. Common Pitfalls

    ### Data Leakage: The Silent Killer

    Data leakage is when information from the test (or validation) set contaminates the training process. It produces models that look great in development and fail in production. Examples:

    - **Preprocessing on the full dataset before splitting.** If you standardize features (subtract mean, divide by std) using the entire dataset, then split, the training set's standardization includes information from the test set. The correct approach: fit the scaler on the training set only, then transform both training and test.
    """)
    return


@app.cell
def _():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression as LR_leak
    from sklearn.model_selection import cross_val_score as cvs_leak
    from sklearn.datasets import load_breast_cancer as lbc_leak

    X_leak, y_leak = lbc_leak(return_X_y=True)

    # CORRECT: scaler is fit only on training data within each CV fold
    pipe_correct = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LR_leak(max_iter=5000))
    ])
    scores_correct = cvs_leak(pipe_correct, X_leak, y_leak, cv=5)
    print(f"CORRECT (pipeline) - Mean accuracy: {scores_correct.mean():.4f}")

    # WRONG: scaler fit on all data before CV
    scaler_wrong = StandardScaler()
    X_scaled_leak = scaler_wrong.fit_transform(X_leak)  # leakage!
    scores_wrong = cvs_leak(LR_leak(max_iter=5000), X_scaled_leak, y_leak, cv=5)
    print(f"WRONG (leakage)    - Mean accuracy: {scores_wrong.mean():.4f}")
    print(f"\nDifference: {(scores_wrong.mean() - scores_correct.mean())*100:.2f} percentage points")
    return


@app.cell
def _(mo):
    mo.md(r"""
    - **Temporal leakage.** In time series, randomly shuffling data before splitting lets the model "see the future." Always split temporally: train on the past, test on the future.
    - **Target leakage.** Features derived from the target variable, even indirectly. If you're predicting loan defaults and one of your features is "was sent to collections," that feature is a consequence of the default, not a predictor. The model will look amazing and be useless.

    ### Overfitting to the Validation Set

    Every time you evaluate on the validation set and make a decision based on the result, you consume a little bit of the validation set's "freshness." After hundreds of experiments, your validation score becomes optimistic. This is why the test set exists — but even the test set isn't immune if you keep going back to "check one more thing."

    The defense: commit to a limited number of final evaluations on the test set. Ideally, one.

    ### Selection Bias in Reported Results

    If you try 100 models and report the best one's validation score, you're reporting a biased estimate. The best of 100 random variables is systematically higher than the average. This is why nested CV exists — it accounts for the selection process in the error estimate.

    This is also why published ML papers should be read with skepticism when they report large benchmarks. If a paper tried 50 architectures and reports the best one, that result is optimistically biased.

    ### The Multiple Comparisons Problem

    Related to the above: the more comparisons you make, the more likely you are to find a "significant" result by chance. If you test 20 models and declare any $p < 0.05$ difference significant, you expect one false positive even if all models are identical. Corrections like Bonferroni ($\alpha / m$ for $m$ comparisons) exist but are conservative. The best defense is honest reporting and pre-registration of your evaluation plan.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Putting It All Together

    Here's the workflow for a real ML project:

    1. **Split** your data into train and test (hold out the test set)
    2. **Explore and preprocess** using only the training data
    3. **Define candidate models** with their hyperparameter spaces
    4. **Tune** each model using cross-validation on the training set (or nested CV)
    5. **Select** the best model/hyperparameter combination based on CV scores
    6. **Retrain** on the full training set with the chosen hyperparameters
    7. **Evaluate** once on the test set — this is your final reported number
    """)
    return


@app.cell
def _():
    import numpy as np
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report

    # Load data
    X, y = load_breast_cancer(return_X_y=True)

    # Step 1: Hold out test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Steps 3-5: Define pipeline and tune with CV
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=5000))
    ])

    param_grid = {
        'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'clf__penalty': ['l1', 'l2'],
        'clf__solver': ['saga']
    }

    grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)

    print(f"Best CV accuracy: {grid.best_score_:.4f}")
    print(f"Best params: {grid.best_params_}")

    # Step 7: Final evaluation (only do this once!)
    print("\n--- Test Set Performance (final, unbiased) ---")
    print(classification_report(y_test, grid.predict(X_test)))
    return


@app.cell
def _(mo):
    mo.md(r"""
    Notice how `StandardScaler` is inside the pipeline. This is critical — it ensures the scaler is fit only on the training fold during each CV iteration, preventing data leakage.

    ---

    ## Key Takeaways

    1. **Training error is meaningless for evaluation.** Always estimate generalization error on held-out data.
    2. **Three-way splits** (train/val/test) separate model fitting, model selection, and final evaluation. Don't conflate these roles.
    3. **Cross-validation** reduces the variance of your error estimate by averaging over multiple splits. Use stratified $k$-fold for classification.
    4. **Nested CV** gives unbiased performance estimates when hyperparameter tuning is part of the pipeline.
    5. **Random search usually beats grid search** in efficiency. Sample hyperparameters on appropriate scales (log scale for parameters spanning orders of magnitude).
    6. **Data leakage** is the most dangerous pitfall. All preprocessing must be fit on training data only. Use pipelines to enforce this.
    7. **Information criteria** (AIC, BIC) are fast alternatives to CV for likelihood-based models, but less general.
    8. **Statistical significance ≠ practical significance.** A tiny improvement might not be worth the added complexity.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Practice

    ### Conceptual

    1. You standardize your entire dataset, then run 5-fold cross-validation. Explain why your CV score is an optimistic estimate of true generalization error.

    2. You run 10-fold CV and get per-fold accuracies of [0.91, 0.88, 0.85, 0.93, 0.79, 0.90, 0.87, 0.92, 0.88, 0.86]. What is the mean and standard error? What concerns does the standard deviation raise?

    3. Explain the difference between LOOCV and 10-fold CV in terms of bias and variance of the error estimate. When would you choose one over the other?

    4. You train 50 different models with different hyperparameters and report the best validation accuracy. Your friend trains one model and reports its validation accuracy. Whose reported number is more likely to be close to the true generalization error? Why?

    5. Why does BIC tend to select simpler models than AIC, especially for large $n$?

    ### Applied

    6. Load `sklearn.datasets.load_diabetes`. Compare Ridge regression with $\alpha \in$ {0.01, 0.1, 1, 10, 100, 1000} using 10-fold CV. Plot mean CV MSE vs $\alpha$ with error bars ($\pm 1$ std). Which $\alpha$ do you choose?

    7. Repeat the above using `RandomizedSearchCV` with $\alpha$ sampled from `loguniform(1e-3, 1e4)` with 30 iterations. Compare the best $\alpha$ found with grid search.

    8. Implement nested cross-validation for Lasso regression on the diabetes dataset. Compare the nested CV score with the non-nested CV score (where you just pick the best $\alpha$ from a single CV run). Which gives a higher (more optimistic) score?

    9. Demonstrate data leakage. On any classification dataset, compare two approaches: (a) scale the data first, then cross-validate; (b) put the scaler inside a pipeline and cross-validate. Are the scores different? By how much? (On some datasets the difference is small, on others it's dramatic — can you find a case where it matters?)

    10. Using the breast cancer dataset, perform forward feature selection (selecting 5 features) with `SequentialFeatureSelector`. Compare the 5-feature model's CV accuracy against the full-feature model. Is the simpler model competitive?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Code It

    Time to implement these concepts yourself. Each exercise gives you a skeleton — fill in the `TODO` placeholders.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 1: Manual k-Fold Cross-Validation

    Implement k-fold CV from scratch without using `cross_val_score`. This forces you to understand the fold-splitting logic.
    """)
    return


@app.cell
def _():
    import numpy as np_ex1
    from sklearn.linear_model import Ridge as Ridge_ex1
    from sklearn.datasets import load_diabetes as ld_ex1

    X_ex1, y_ex1 = ld_ex1(return_X_y=True)
    n_ex1 = len(y_ex1)
    k_ex1 = 5

    # TODO: Shuffle indices
    rng_ex1 = np_ex1.random.default_rng(0)
    indices_ex1 = ...  # rng_ex1.permutation(?)

    fold_size_ex1 = n_ex1 // k_ex1
    mse_list_ex1 = []

    for i in range(k_ex1):
        # TODO: Create val_idx and train_idx for fold i
        val_idx_ex1 = ...  # indices_ex1[? : ?]
        train_idx_ex1 = ...  # np_ex1.concatenate the parts before and after

        # TODO: Fit Ridge(alpha=1.0) on training indices, predict on val indices
        model_ex1 = ...
        y_pred_ex1 = ...

        # TODO: Compute MSE = mean((y_true - y_pred)^2)
        mse_ex1 = ...
        mse_list_ex1.append(mse_ex1)

    # Uncomment when done:
    # print(f"Manual {k_ex1}-fold CV MSE: {np_ex1.mean(mse_list_ex1):.2f} ± {np_ex1.std(mse_list_ex1):.2f}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Exercise 2: Bootstrap Confidence Interval

    Use the bootstrap to compute a 95% confidence interval for the mean squared error of a Ridge model. The key idea: resample, fit, evaluate, repeat — then use percentiles.
    """)
    return


@app.cell
def _():
    import numpy as np_ex2
    from sklearn.linear_model import Ridge as Ridge_ex2
    from sklearn.datasets import load_diabetes as ld_ex2
    from sklearn.model_selection import train_test_split as tts_ex2

    X_ex2, y_ex2 = ld_ex2(return_X_y=True)
    X_tr_ex2, X_te_ex2, y_tr_ex2, y_te_ex2 = tts_ex2(X_ex2, y_ex2, test_size=0.2, random_state=42)

    B_ex2 = 200
    rng_ex2 = np_ex2.random.default_rng(42)
    mse_bootstrap_ex2 = []

    for _ in range(B_ex2):
        # TODO: Sample training indices WITH replacement
        idx_ex2 = ...  # rng_ex2.integers(0, len(X_tr_ex2), size=len(X_tr_ex2))

        # TODO: Fit Ridge on bootstrap sample
        model_ex2 = ...

        # TODO: Evaluate on the held-out TEST set (not out-of-bag)
        y_pred_ex2 = ...
        mse_ex2 = ...
        mse_bootstrap_ex2.append(mse_ex2)

    # Uncomment when done:
    # ci_low = np_ex2.percentile(mse_bootstrap_ex2, 2.5)
    # ci_high = np_ex2.percentile(mse_bootstrap_ex2, 97.5)
    # print(f"Bootstrap 95% CI for test MSE: [{ci_low:.2f}, {ci_high:.2f}]")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 3: Detect Data Leakage

    Build two pipelines: one with leakage (scale first, then CV) and one without (scale inside CV). Measure the difference on synthetic data where leakage is amplified.
    """)
    return


@app.cell
def _():
    import numpy as np_ex3
    from sklearn.pipeline import Pipeline as Pipeline_ex3
    from sklearn.preprocessing import StandardScaler as Scaler_ex3
    from sklearn.linear_model import LogisticRegression as LR_ex3
    from sklearn.model_selection import cross_val_score as cvs_ex3

    # Synthetic data: 50 samples, 500 features (p >> n makes leakage worse)
    rng_ex3 = np_ex3.random.default_rng(42)
    X_ex3 = rng_ex3.normal(size=(50, 500))
    y_ex3 = rng_ex3.integers(0, 2, size=50)  # random labels — no real signal

    # TODO: WRONG way — scale all data, then cross-validate
    # scaler_ex3 = Scaler_ex3()
    # X_scaled_ex3 = ...  # fit_transform on ALL of X_ex3
    # scores_leak_ex3 = cvs_ex3(LR_ex3(max_iter=1000), X_scaled_ex3, y_ex3, cv=5)

    # TODO: RIGHT way — put scaler inside a pipeline
    # pipe_ex3 = Pipeline_ex3([('scaler', Scaler_ex3()), ('clf', LR_ex3(max_iter=1000))])
    # scores_clean_ex3 = cvs_ex3(pipe_ex3, X_ex3, y_ex3, cv=5)

    # print(f"With leakage:    {scores_leak_ex3.mean():.3f}")
    # print(f"Without leakage: {scores_clean_ex3.mean():.3f}")
    # print(f"Difference:      {(scores_leak_ex3.mean() - scores_clean_ex3.mean())*100:.1f} pp")
    # Note: with random labels, accuracy should be ~50%. Leakage inflates it.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 4: Full Model Selection Pipeline

    Put it all together: load data, hold out a test set, define two candidate models with hyperparameter grids, tune each with CV, pick the winner, and report final test performance. Use pipelines to avoid leakage.
    """)
    return


@app.cell
def _():
    import numpy as np_ex4
    from sklearn.datasets import load_breast_cancer as lbc_ex4
    from sklearn.model_selection import train_test_split as tts_ex4, GridSearchCV as GS_ex4
    from sklearn.pipeline import Pipeline as Pipe_ex4
    from sklearn.preprocessing import StandardScaler as SS_ex4
    from sklearn.linear_model import LogisticRegression as LR_ex4, RidgeClassifier as RC_ex4

    X_ex4, y_ex4 = lbc_ex4(return_X_y=True)

    # TODO: Step 1 — hold out 20% test set (stratified)
    # X_train_ex4, X_test_ex4, y_train_ex4, y_test_ex4 = tts_ex4(...)

    # TODO: Step 2 — define two pipelines
    # pipe_lr = Pipe_ex4([('scaler', SS_ex4()), ('clf', LR_ex4(max_iter=5000, solver='saga'))])
    # pipe_rc = Pipe_ex4([('scaler', SS_ex4()), ('clf', RC_ex4())])

    # TODO: Step 3 — define hyperparameter grids
    # grid_lr = {'clf__C': [0.01, 0.1, 1, 10], 'clf__penalty': ['l1', 'l2']}
    # grid_rc = {'clf__alpha': [0.01, 0.1, 1, 10, 100]}

    # TODO: Step 4 — run GridSearchCV for each, print best CV score
    # gs_lr = GS_ex4(pipe_lr, grid_lr, cv=5, scoring='accuracy').fit(X_train_ex4, y_train_ex4)
    # gs_rc = GS_ex4(pipe_rc, grid_rc, cv=5, scoring='accuracy').fit(X_train_ex4, y_train_ex4)
    # print(f"LogReg best CV: {gs_lr.best_score_:.4f}")
    # print(f"Ridge  best CV: {gs_rc.best_score_:.4f}")

    # TODO: Step 5 — pick winner based on CV, evaluate ONCE on test
    # winner = gs_lr if gs_lr.best_score_ > gs_rc.best_score_ else gs_rc
    # print(f"Test accuracy: {winner.score(X_test_ex4, y_test_ex4):.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Further Reading

    - [ISLR Chapter 5: Resampling Methods](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf) — Cross-validation and bootstrap, gently explained
    - [ISLR Chapter 6: Linear Model Selection and Regularization](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf) — Subset selection, shrinkage, and dimension reduction
    - [ESL Chapter 7: Model Assessment and Selection](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf) — The definitive technical treatment
    - [Bishop PRML Section 1.3: Model Selection](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) — Bayesian perspective on model complexity
    - [Murphy PML1 Chapter 5.3: Model Selection](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf) — Modern treatment with practical advice
    """)
    return


if __name__ == "__main__":
    app.run()
