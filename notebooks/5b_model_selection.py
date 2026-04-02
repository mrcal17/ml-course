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
    # Model Selection in Practice

    In Module 1D you learned the *theory* of model selection: the bias-variance tradeoff, cross-validation, the bootstrap, information criteria, why training error is a lie. That was the conceptual foundation. This module is about *doing it* — running real experiments, comparing real models on real datasets, tuning hyperparameters with scikit-learn's actual tools, and reading the diagnostic plots that tell you what to fix next.

    The gap between theory and practice is wider than you might expect. Knowing that cross-validation reduces variance in your error estimate is one thing. Knowing how to set up a `cross_validate` call with the right scoring metric, interpret the `cv_results_` dictionary from `GridSearchCV`, and read a validation curve to decide whether your model is underfitting or overfitting — that is something else entirely. This module bridges that gap.

    We will work through two complete pipelines — one for regression, one for classification — and then move to hyperparameter tuning and diagnostic curves. By the end you will have a repeatable experimental workflow that you can apply to any supervised learning problem.

    References: [Geron Chapter 2: End-to-End Machine Learning Project](file:///C:/Users/landa/ml-course/textbooks/Geron.pdf), [ISLR Chapter 5: Resampling Methods](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf), [ESL Chapter 7: Model Assessment and Selection](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 1. The Experimental Mindset

    Here is the single most important thing to internalize about applied machine learning: **you do not know which model will win until you test it.** You can have strong priors — tree ensembles tend to dominate tabular data, linear models work well in high dimensions with few samples — but priors are not knowledge. The data has the final word.

    This means ML is fundamentally an empirical science. You form hypotheses ("Ridge will outperform OLS on this noisy dataset"), you design experiments (cross-validated comparison with a specific metric), you run them, and you interpret the results. The quality of your conclusions depends entirely on the quality of your experimental protocol.

    ### The Experimental Protocol

    Every model comparison you run should follow this structure:

    1. **Choose your metric before looking at any results.** This is not optional. If you pick your metric after seeing the data, you are giving yourself a free parameter to game. This is the "researcher degrees of freedom" problem — the more choices you make after seeing results, the more likely you are to find something that looks good but doesn't generalize. Decide upfront: are you optimizing for MSE? MAE? Accuracy? F1? ROC-AUC? The answer depends on your problem, not on which metric makes your model look best.

       Consider a concrete example. You train five models and compute accuracy, F1, and AUC for each. Model A wins on accuracy, Model B wins on F1, Model C wins on AUC. If you now "choose" the metric that happens to favor the model you like, you have effectively run a hidden model selection step without accounting for it. This inflates your reported performance. The antidote is to commit to a single primary metric in advance, recorded in your experiment log before any code runs.

    2. **Hold out a test set that you do not touch until the very end.** You already know this from Module 1D. The test set is your final, unbiased estimate of generalization performance. Every decision you make — which models to try, which hyperparameters to tune, which features to include — is based on cross-validation scores on the *training* set. The test set is opened once, like a sealed envelope.

    3. **Use cross-validation, not a single validation split.** A single split is noisy. Five-fold or ten-fold CV gives you a mean *and* a standard deviation, which tells you both how good the model is and how stable that estimate is.

    4. **Compare models on the same folds.** If you compare Model A on one random split and Model B on a different split, the comparison is confounded by which data points ended up where. Use the same `cv` object (same `KFold` or `StratifiedKFold` instance with the same `random_state`) for all models.

    ### Stratified Splits for Classification

    When your target is categorical, always use stratified splits. If your dataset is 95% class 0 and 5% class 1, a naive random split could produce a fold with zero positive examples. Stratification ensures each fold mirrors the overall class distribution. In scikit-learn, `train_test_split` accepts `stratify=y`, and `StratifiedKFold` handles cross-validation. There is almost never a reason not to stratify for classification.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2. Regression -- An End-to-End Pipeline

    Let's work through a complete regression experiment on the California Housing dataset. This dataset contains 20,640 samples with 8 features (median income, house age, average rooms, etc.) and the target is median house value in hundreds of thousands of dollars.

    ### Load and Explore
    """)
    return


@app.cell
def _(np, plt):
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split

    housing = fetch_california_housing()
    X_housing, y_housing = housing.data, housing.target

    print(f"Features:  {housing.feature_names}")
    print(f"X shape:   {X_housing.shape}")
    print(f"y shape:   {y_housing.shape}")
    print(f"y range:   [{y_housing.min():.2f}, {y_housing.max():.2f}]")
    print(f"y mean:    {y_housing.mean():.2f}")
    print(f"y median:  {np.median(y_housing):.2f}")

    # Target distribution
    fig_hist, ax_hist = plt.subplots(figsize=(7, 3))
    ax_hist.hist(y_housing, bins=50, edgecolor="k", alpha=0.7)
    ax_hist.set_xlabel("Median house value ($100k)")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Target distribution — California Housing")
    plt.tight_layout()
    fig_hist
    return (X_housing, housing, train_test_split, y_housing)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The target is right-skewed and capped at 5.0 (representing $500k+). That cap will affect our metrics — some "errors" are actually censored data. In a real project you would investigate this further. For now, we proceed.

    ### Train/Test Split and Candidate Models

    We hold out 20% for testing, then compare four linear models — all wrapped in `Pipeline` with `StandardScaler` to avoid data leakage. This is the pattern you should use every time: preprocessing steps go inside the pipeline so that they are fit only on training data during each CV fold.
    """)
    return


@app.cell
def _(X_housing, train_test_split, y_housing):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.model_selection import cross_validate

    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
        X_housing, y_housing, test_size=0.2, random_state=42
    )
    print(f"Train: {X_train_h.shape[0]} samples, Test: {X_test_h.shape[0]} samples")

    # Define candidate models — all with StandardScaler in a Pipeline
    models = {
        "OLS":        Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())]),
        "Ridge":      Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))]),
        "Lasso":      Pipeline([("scaler", StandardScaler()), ("reg", Lasso(alpha=0.01))]),
        "ElasticNet": Pipeline([("scaler", StandardScaler()), ("reg", ElasticNet(alpha=0.01, l1_ratio=0.5))]),
    }

    # 5-fold CV with multiple metrics
    # cross_validate returns a dict with test scores, train scores, and fit/score times.
    # cross_val_score returns only test scores for a single metric — simpler but less informative.
    # Note: sklearn uses "neg_" prefix for metrics where lower is better (MSE, MAE), because
    # its convention is that higher scores are always better. You negate them when reporting.
    scoring = ["neg_mean_squared_error", "neg_mean_absolute_error", "r2"]
    results = {}
    for name, pipe in models.items():
        cv_result = cross_validate(pipe, X_train_h, y_train_h, cv=5,
                                   scoring=scoring, return_train_score=True)
        results[name] = cv_result
        mse = -cv_result["test_neg_mean_squared_error"].mean()
        mae = -cv_result["test_neg_mean_absolute_error"].mean()
        r2 = cv_result["test_r2"].mean()
        print(f"{name:12s} | MSE={mse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")
    return (
        ElasticNet,
        Lasso,
        LinearRegression,
        Pipeline,
        Ridge,
        StandardScaler,
        X_test_h,
        X_train_h,
        cross_validate,
        models,
        results,
        y_test_h,
        y_train_h,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Understanding the Metrics

    Three metrics, three different stories:

    **Mean Squared Error (MSE)** is the average of $(y_i - \hat{y}_i)^2$. Because errors are squared, large errors are penalized much more than small ones. A prediction that is off by 10 contributes 100 to the MSE; a prediction off by 1 contributes only 1. This makes MSE sensitive to outliers. Use MSE when large errors are genuinely more costly than small ones — for example, when predicting structural loads where a big miss could be catastrophic.

    **Mean Absolute Error (MAE)** is the average of $|y_i - \hat{y}_i|$. All errors are weighted equally regardless of magnitude. MAE is more robust to outliers than MSE. Use MAE when you care about the typical error magnitude and don't want a few extreme cases to dominate the metric.

    **$R^2$ (coefficient of determination)** measures how much better your model is than a naive mean-predictor. $R^2 = 1 - \text{MSE}/\text{Var}(y)$. An $R^2$ of 0.6 means your model explains 60% of the variance in the target. An $R^2$ of 0 means your model is no better than predicting the mean every time. $R^2$ can be negative — that means your model is *worse* than the mean. The advantage of $R^2$ is that it is scale-free: you can compare it across datasets with different target ranges.

    For this dataset, all four models perform similarly. That tells us the linear relationship is strong and regularization is not doing much work — the dataset has many samples relative to features, so overfitting is not a major concern.

    A bar chart with error bars makes the comparison visual. The error bars show $\pm 1$ standard deviation across folds — if the bars overlap, the difference between models is likely not meaningful.
    """)
    return


@app.cell
def _(np, plt, results):
    # Visual comparison of regression models
    model_names_bar = list(results.keys())
    mse_means = [-results[n]["test_neg_mean_squared_error"].mean() for n in model_names_bar]
    mse_stds  = [results[n]["test_neg_mean_squared_error"].std() for n in model_names_bar]

    fig_bar, ax_bar = plt.subplots(figsize=(7, 4))
    x_pos = np.arange(len(model_names_bar))
    ax_bar.bar(x_pos, mse_means, yerr=mse_stds, capsize=6, alpha=0.7,
               color=["tab:blue", "tab:orange", "tab:green", "tab:red"])
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(model_names_bar)
    ax_bar.set_ylabel("Cross-validated MSE")
    ax_bar.set_title("Regression model comparison — California Housing (5-fold CV)")
    plt.tight_layout()
    fig_bar
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The error bars overlap substantially — none of these linear models is clearly better than the others on this dataset. In practice, when models perform this similarly, pick the simplest one (OLS if you want interpretability, Ridge if you want a safety margin against multicollinearity).

    ### Prediction vs Actual Plot
    """)
    return


@app.cell
def _(X_train_h, np, plt, y_train_h):
    from sklearn.model_selection import cross_val_predict

    # Get cross-validated predictions for OLS (predicted on held-out folds)
    pipe_ols = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", LinearRegression())
    ])
    y_pred_cv = cross_val_predict(pipe_ols, X_train_h, y_train_h, cv=5)

    fig_pred, ax_pred = plt.subplots(figsize=(6, 5))
    ax_pred.scatter(y_train_h, y_pred_cv, alpha=0.15, s=8, edgecolors="none")
    ax_pred.plot([0, 5.5], [0, 5.5], "r--", lw=1.5, label="Perfect prediction")
    ax_pred.set_xlabel("Actual median value ($100k)")
    ax_pred.set_ylabel("Predicted median value ($100k)")
    ax_pred.set_title("Cross-validated predictions vs actual — OLS")
    ax_pred.legend()
    ax_pred.set_xlim(0, 5.5)
    ax_pred.set_ylim(-0.5, 6)
    plt.tight_layout()
    fig_pred
    return (cross_val_predict, y_pred_cv)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The diagonal line is perfect prediction. Points above the line are overpredictions; points below are underpredictions. Notice the cluster at the top — those are the capped values at $500k+. The model cannot predict them well because the true values are censored. You can also see that predictions for expensive houses tend to be too low. This systematic pattern tells us the linear model is *misspecified* — the true relationship is nonlinear.

    ### Residual Analysis

    Residuals are the errors: $e_i = y_i - \hat{y}_i$. For a well-specified model, residuals should be roughly random — no patterns, no trends, no heteroscedasticity (changing variance). If you see structure in the residuals, your model is missing something.
    """)
    return


@app.cell
def _(np, plt, y_pred_cv, y_train_h):
    residuals = y_train_h - y_pred_cv

    fig_resid, axes_resid = plt.subplots(1, 2, figsize=(12, 4))

    # Residuals vs predicted
    axes_resid[0].scatter(y_pred_cv, residuals, alpha=0.15, s=8, edgecolors="none")
    axes_resid[0].axhline(0, color="r", ls="--", lw=1)
    axes_resid[0].set_xlabel("Predicted value")
    axes_resid[0].set_ylabel("Residual (actual - predicted)")
    axes_resid[0].set_title("Residuals vs predicted")

    # Residual histogram
    axes_resid[1].hist(residuals, bins=60, edgecolor="k", alpha=0.7)
    axes_resid[1].axvline(0, color="r", ls="--", lw=1)
    axes_resid[1].set_xlabel("Residual")
    axes_resid[1].set_ylabel("Count")
    axes_resid[1].set_title(f"Residual distribution (mean={np.mean(residuals):.3f}, std={np.std(residuals):.3f})")

    plt.tight_layout()
    fig_resid
    return (residuals,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The residual plot reveals two problems. First, the variance of residuals increases with predicted value — that is heteroscedasticity, meaning the model is less reliable for expensive houses. Second, there is a visible curve at the high end — the model systematically underpredicts expensive houses. Both patterns suggest that a nonlinear model (trees, ensembles, or polynomial features) would do better here. We will not pursue that now — the point is that *you looked*, and the residuals told you something actionable.

    **What good residuals look like:** Random scatter around zero, constant variance (homoscedasticity), no trends, no curves. The histogram should be roughly bell-shaped and centered at zero. If you see any of the following, your model has a problem:

    - **Funnel shape** (variance increasing or decreasing): heteroscedasticity. Consider a log transform on the target, or use a model that handles non-constant variance.
    - **Curved pattern**: the model is misspecified. The true relationship is nonlinear and the linear model cannot capture it. Add polynomial features or switch to a flexible model.
    - **Clusters or gaps**: possible subgroups in the data that the model treats identically. Investigate whether an interaction term or separate models per subgroup would help.
    - **Outliers** (isolated points far from zero): investigate these individually. They may be data errors, or they may be genuinely unusual cases that your model cannot handle.

    ---

    ## 3. Classification -- The Metrics Zoo

    Classification metrics are more subtle than regression metrics because there are multiple ways to be wrong, and different kinds of mistakes have different costs. You must understand each metric, know when to use it, and be able to explain *why* you chose it.

    We will use the Breast Cancer Wisconsin dataset: 569 samples, 30 features, binary target (malignant vs benign).
    """)
    return


@app.cell
def _(np):
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import StratifiedKFold

    bc = load_breast_cancer()
    X_bc, y_bc = bc.data, bc.target
    print(f"Features:  {X_bc.shape[1]}")
    print(f"Samples:   {X_bc.shape[0]}")
    print(f"Classes:   {bc.target_names}")
    print(f"Class distribution: benign={np.sum(y_bc == 1)}, malignant={np.sum(y_bc == 0)}")
    print(f"Positive rate:      {np.mean(y_bc):.2%}")
    return (StratifiedKFold, X_bc, bc, load_breast_cancer, y_bc)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The dataset is moderately imbalanced — about 63% benign, 37% malignant. This is not extreme, but it is enough to make accuracy misleading in some scenarios.

    ### The Metrics, One by One

    **Accuracy** is the fraction of correct predictions. Simple, intuitive, and dangerous. Consider a spam detector where 99% of emails are legitimate. A model that *always* predicts "not spam" gets 99% accuracy and catches zero spam. Accuracy rewards laziness when classes are imbalanced. Use it only when classes are roughly balanced and all mistakes are equally costly.

    **Precision** answers: "Of the samples I flagged as positive, how many actually are?" If you build a cancer screening tool that flags 100 patients for biopsy, and 80 of them actually have cancer, your precision is 80%. High precision means few false positives — few healthy patients get unnecessary biopsies. Precision matters when false positives are expensive (spam filter that deletes legitimate emails, fraud detector that freezes real transactions).

    **Recall** (sensitivity) answers: "Of all the actual positives, how many did I catch?" If there are 50 cancer patients in the dataset and your model catches 45 of them, your recall is 90%. High recall means few false negatives — few sick patients slip through. Recall matters when false negatives are dangerous (missing a malignant tumor, missing a fraudulent transaction, missing a structural defect).

    **F1 score** is the harmonic mean of precision and recall: $F_1 = 2 \cdot \frac{P \cdot R}{P + R}$. The harmonic mean punishes extreme imbalances — a model with 100% precision and 1% recall gets $F_1 = 0.02$, not 50.5%. F1 is a good default for imbalanced classification when you don't have a clear reason to favor precision or recall.

    **ROC-AUC** (Receiver Operating Characteristic — Area Under the Curve) measures ranking quality independent of the classification threshold. A model that assigns higher scores to positives than to negatives will have a high AUC, even if the raw scores are not well-calibrated probabilities. AUC = 1.0 means perfect ranking; AUC = 0.5 means random. Use ROC-AUC when you want to evaluate the model's discriminative ability without committing to a specific threshold.

    ### The Precision-Recall Tradeoff

    Precision and recall are fundamentally in tension. You can always increase recall by lowering your classification threshold — flag more things as positive and you will catch more true positives. But this comes at the cost of precision, because you will also flag more false positives. Conversely, raising the threshold increases precision but misses more true positives.

    A concrete example: suppose you have 1000 patients, 100 of whom have cancer. A model assigns each a probability score.

    - At threshold 0.5: flags 120 patients, catches 80 of the 100 cancers. Precision = 80/120 = 67%, Recall = 80/100 = 80%.
    - At threshold 0.3: flags 200 patients, catches 95 of the 100 cancers. Precision = 95/200 = 48%, Recall = 95/100 = 95%.
    - At threshold 0.8: flags 60 patients, catches 50 of the 100 cancers. Precision = 50/60 = 83%, Recall = 50/100 = 50%.

    Which threshold is best? It depends entirely on the *costs* of false positives vs false negatives. In cancer screening, missing a tumor (false negative) is far worse than an unnecessary biopsy (false positive), so you would lean toward lower thresholds with high recall. In email spam filtering, deleting a legitimate email (false positive) might be worse than letting some spam through (false negative), so you would lean toward higher precision. There is no universally correct threshold — it is a decision that must be informed by the problem domain.

    ### Comparing Classifiers
    """)
    return


@app.cell
def _(Pipeline, StandardScaler, StratifiedKFold, X_bc, cross_validate, y_bc):
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(
        X_bc, y_bc, test_size=0.2, random_state=42, stratify=y_bc
    )

    clf_models = {
        "LogReg":  Pipeline([("scaler", StandardScaler()),
                             ("clf", LogisticRegression(max_iter=5000))]),
        "SVC":     Pipeline([("scaler", StandardScaler()),
                             ("clf", SVC(probability=True))]),
        "RF":      Pipeline([("scaler", StandardScaler()),
                             ("clf", RandomForestClassifier(n_estimators=100, random_state=42))]),
        "GBM":     Pipeline([("scaler", StandardScaler()),
                             ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42))]),
    }

    # Same folds for all models — critical for fair comparison
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf_scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    clf_results = {}
    print(f"{'Model':8s} | {'Acc':>6s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'AUC':>6s}")
    print("-" * 50)
    for name, pipe in clf_models.items():
        cv_res = cross_validate(pipe, X_train_bc, y_train_bc, cv=skf,
                                scoring=clf_scoring, return_train_score=False)
        clf_results[name] = cv_res
        acc  = cv_res["test_accuracy"].mean()
        prec = cv_res["test_precision"].mean()
        rec  = cv_res["test_recall"].mean()
        f1   = cv_res["test_f1"].mean()
        auc  = cv_res["test_roc_auc"].mean()
        print(f"{name:8s} | {acc:.4f} {prec:.4f} {rec:.4f} {f1:.4f} {auc:.4f}")
    return (
        GradientBoostingClassifier,
        LogisticRegression,
        RandomForestClassifier,
        SVC,
        X_test_bc,
        X_train_bc,
        clf_models,
        clf_results,
        skf,
        y_test_bc,
        y_train_bc,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    All four models perform well on this dataset. That is typical for clean, well-separated data with many features. The differences are small — probably within the noise of 5-fold CV. In production you would pick the simplest model (LogReg) unless there is a compelling reason for added complexity.

    ### Confusion Matrices

    The confusion matrix is the single most informative visualization in classification. It shows you exactly where your model is succeeding and where it is failing — how many true positives, true negatives, false positives, and false negatives. Every scalar metric (accuracy, precision, recall, F1) can be computed directly from the confusion matrix. When in doubt, look at the confusion matrix.

    Use the dropdown below to switch between classifiers and see how their error patterns differ.
    """)
    return


@app.cell
def _(mo):
    clf_dropdown = mo.ui.dropdown(
        options=["LogReg", "SVC", "RF", "GBM"],
        value="LogReg",
        label="Classifier"
    )
    clf_dropdown
    return (clf_dropdown,)


@app.cell
def _(Pipeline, StandardScaler, X_train_bc, clf_dropdown, clf_models, np, plt, y_train_bc):
    from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                                 accuracy_score, precision_score,
                                 recall_score, f1_score, roc_auc_score)
    from sklearn.model_selection import cross_val_predict as cvp_clf

    # Get cross-validated predictions for the selected classifier
    selected_clf = clf_models[clf_dropdown.value]
    y_pred_clf = cvp_clf(selected_clf, X_train_bc, y_train_bc, cv=5)

    # If model supports predict_proba, also get probabilities for AUC
    try:
        y_proba_clf = cvp_clf(selected_clf, X_train_bc, y_train_bc, cv=5, method="predict_proba")[:, 1]
        auc_val = roc_auc_score(y_train_bc, y_proba_clf)
    except Exception:
        auc_val = None

    cm = confusion_matrix(y_train_bc, y_pred_clf)
    acc_val  = accuracy_score(y_train_bc, y_pred_clf)
    prec_val = precision_score(y_train_bc, y_pred_clf)
    rec_val  = recall_score(y_train_bc, y_pred_clf)
    f1_val   = f1_score(y_train_bc, y_pred_clf)

    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Malignant", "Benign"])
    disp.plot(ax=ax_cm, cmap="Blues", colorbar=False)
    metrics_str = f"Acc={acc_val:.3f}  Prec={prec_val:.3f}  Rec={rec_val:.3f}  F1={f1_val:.3f}"
    if auc_val is not None:
        metrics_str += f"  AUC={auc_val:.3f}"
    ax_cm.set_title(f"{clf_dropdown.value} — Confusion Matrix\n{metrics_str}", fontsize=10)
    plt.tight_layout()
    fig_cm
    return (cm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read the confusion matrix carefully. The top-left cell is true negatives (correctly identified malignant tumors); bottom-right is true positives (correctly identified benign tumors). Off-diagonal cells are errors. In medical screening, false negatives (bottom-left: malignant tumors called benign) are far more dangerous than false positives (top-right: benign tumors called malignant). This is why recall matters more than precision for cancer detection — missing a tumor can be fatal.

    ### ROC Curves

    The ROC curve plots the true positive rate (recall) against the false positive rate at every possible classification threshold. A perfect model hugs the top-left corner (TPR = 1, FPR = 0). A random model follows the diagonal. The area under this curve (AUC) summarizes the model's discriminative quality in a single number.

    Why is the ROC curve useful? Because it decouples model evaluation from the threshold choice. Two models might achieve the same F1 at different thresholds, but the one with higher AUC is fundamentally better at separating the classes. You choose the threshold *after* selecting the model.
    """)
    return


@app.cell
def _(Pipeline, StandardScaler, X_train_bc, clf_models, np, plt, y_train_bc):
    from sklearn.metrics import RocCurveDisplay
    from sklearn.model_selection import cross_val_predict as cvp_roc

    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    for name, pipe in clf_models.items():
        y_proba_roc = cvp_roc(pipe, X_train_bc, y_train_bc, cv=5, method="predict_proba")[:, 1]
        RocCurveDisplay.from_predictions(y_train_bc, y_proba_roc, ax=ax_roc, name=name)
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.5)")
    ax_roc.set_title("ROC Curves — Cross-Validated Predictions")
    ax_roc.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    fig_roc
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    All four models achieve high AUC on this dataset — the classes are well-separated. On harder problems with more class overlap, the ROC curves spread apart and the differences between models become more visible.

    ---

    ## 4. Hyperparameter Tuning

    You have candidate models. Now you need to find the best hyperparameters for each. This is where `GridSearchCV` and `RandomizedSearchCV` come in.

    ### GridSearchCV: Exhaustive Search

    `GridSearchCV` tries every combination of hyperparameters you specify and evaluates each with cross-validation. It is simple, reliable, and exhaustive — which is both its strength and its weakness. If you have 3 hyperparameters with 10 values each, that is $10^3 = 1000$ combinations, each requiring $k$ model fits for $k$-fold CV. It scales exponentially with the number of hyperparameters.

    We will tune a Random Forest on the Wine dataset to keep things fast. Note the `__` (double underscore) notation for accessing parameters inside a pipeline: `clf__max_depth` means "the `max_depth` parameter of the step named `clf`."
    """)
    return


@app.cell
def _(Pipeline, StandardScaler):
    from sklearn.datasets import load_wine
    from sklearn.model_selection import GridSearchCV

    wine = load_wine()
    X_wine, y_wine = wine.data, wine.target

    X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(
        X_wine, y_wine, test_size=0.2, random_state=42, stratify=y_wine
    )

    pipe_rf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(random_state=42))
    ])

    param_grid_rf = {
        "clf__n_estimators": [50, 100, 200],
        "clf__max_depth": [3, 5, 10, None],
        "clf__min_samples_leaf": [1, 2, 5],
    }

    grid_search = GridSearchCV(
        pipe_rf, param_grid_rf, cv=5,
        scoring="accuracy", n_jobs=-1,
        return_train_score=True
    )
    grid_search.fit(X_train_w, y_train_w)

    print(f"Total combinations: {len(grid_search.cv_results_['mean_test_score'])}")
    print(f"Best CV accuracy:   {grid_search.best_score_:.4f}")
    print(f"Best params:        {grid_search.best_params_}")
    print(f"Test accuracy:      {grid_search.score(X_test_w, y_test_w):.4f}")
    return (
        GridSearchCV,
        X_test_w,
        X_train_w,
        X_wine,
        grid_search,
        load_wine,
        param_grid_rf,
        pipe_rf,
        y_test_w,
        y_train_w,
        y_wine,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpreting cv_results_

    The `cv_results_` dictionary from `GridSearchCV` contains everything: mean and std of test scores, training scores, fit times, and the parameter combinations. Converting it to a DataFrame makes it much easier to inspect.

    A few things to look for when inspecting the results table:

    - **Are the top configurations clustered?** If the top 10 configs all have similar scores, the hyperparameter surface is flat and tuning does not matter much. Pick the simplest one.
    - **Does one hyperparameter dominate?** Sort by each parameter and see if there is a clear trend. If `max_depth` matters but `min_samples_leaf` does not, you can fix `min_samples_leaf` at a reasonable value and spend your budget on finer-grained search over `max_depth`.
    - **Are training and test scores far apart?** A gap of more than 5-10 percentage points is a sign of overfitting. Consider more regularization or simpler models.
    """)
    return


@app.cell
def _(grid_search):
    import pandas as pd

    cv_df = pd.DataFrame(grid_search.cv_results_)
    # Show the top 10 configurations sorted by mean test score
    cols = ["param_clf__n_estimators", "param_clf__max_depth",
            "param_clf__min_samples_leaf", "mean_test_score", "std_test_score",
            "mean_train_score", "mean_fit_time"]
    top10 = cv_df[cols].sort_values("mean_test_score", ascending=False).head(10)
    top10
    return (cv_df, pd, top10)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Look at the gap between `mean_train_score` and `mean_test_score`. A large gap means overfitting — the model is memorizing the training data. If both are low, the model is underfitting. The ideal is when both are high and close together.

    ### RandomizedSearchCV: Why Random Beats Grid

    Bergstra and Bengio (2012) showed a surprising result: random search is almost always more efficient than grid search. The intuition is straightforward. In most ML problems, only one or two hyperparameters truly matter. Grid search wastes evaluations on combinations that differ only in the unimportant parameters. Random search, by contrast, explores a wider range of each individual parameter.

    Consider a 2D grid with 9 points (3 values per parameter). If only parameter 1 matters, you are really only testing 3 distinct values of it — the same 3 values repeated across the 3 irrelevant values of parameter 2. With 9 random samples, you get 9 distinct values of parameter 1. That is three times the effective resolution for the same computational budget.

    The practical rule: use grid search when you have 2-3 hyperparameters with small, known-good ranges. Use random search for everything else.
    """)
    return


@app.cell
def _(Pipeline, RandomForestClassifier, StandardScaler, X_train_w, y_train_w):
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint, uniform

    pipe_rf_rand = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(random_state=42))
    ])

    param_dist_rf = {
        "clf__n_estimators": randint(50, 300),
        "clf__max_depth": [3, 5, 7, 10, 15, None],
        "clf__min_samples_leaf": randint(1, 10),
        "clf__max_features": ["sqrt", "log2", None],
    }

    random_search = RandomizedSearchCV(
        pipe_rf_rand, param_dist_rf,
        n_iter=36,   # same budget as grid (36 combos)
        cv=5, scoring="accuracy", n_jobs=-1,
        random_state=42
    )
    random_search.fit(X_train_w, y_train_w)

    print(f"Random search — best CV accuracy: {random_search.best_score_:.4f}")
    print(f"Best params: {random_search.best_params_}")
    return (RandomizedSearchCV, random_search)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What to Tune First

    Not all hyperparameters are created equal. Some have a large effect on model performance; others barely matter. Here is a rough priority ordering for common models:

    **Random Forest:** `n_estimators` (more is almost always better, until diminishing returns), `max_depth` (controls overfitting), `max_features` (controls decorrelation of trees).

    **Gradient Boosting (GBM/XGBoost/LightGBM):** `learning_rate` (the most important parameter — smaller values need more trees), `n_estimators` (coupled with learning rate), `max_depth` (usually 3-8 is sufficient).

    **SVM:** `C` (regularization — large C means less regularization, small C means more), `gamma` (RBF kernel bandwidth — controls how far each training example's influence reaches).

    **Logistic Regression / Ridge / Lasso:** `C` or `alpha` (the single most important hyperparameter — controls the bias-variance tradeoff directly).

    **k-Nearest Neighbors:** `n_neighbors` (the only parameter that really matters — small k means flexible and noisy, large k means smooth and biased).

    Start with the parameters that matter most. Tune them on a coarse grid first, then refine around the best region. Don't waste time tuning parameters that barely affect performance. A good heuristic: if changing a hyperparameter by a factor of 2 does not change the validation score by more than 0.1%, that hyperparameter does not matter for your problem.

    Also note: for parameters that span orders of magnitude (like `C`, `alpha`, `learning_rate`, `gamma`), always search on a log scale. Use `np.logspace` for grid search or `scipy.stats.loguniform` for random search. Searching `alpha` from 0.001 to 1000 on a linear scale wastes almost all your budget testing values near 1000, because the scale of variation is multiplicative, not additive.

    ---

    ## 5. Diagnostic Curves

    Two plots that every ML practitioner should know how to make and read: the **validation curve** and the **learning curve**. They answer different questions, and using the wrong one wastes your time.

    ### Validation Curves: Score vs Hyperparameter Value

    A validation curve plots the training score and cross-validation score as a function of a single hyperparameter. It answers the question: "Is my model underfitting or overfitting for this hyperparameter setting?"

    The three regimes are:

    - **Underfitting (both scores low):** The model is too simple. Increase complexity — deeper tree, less regularization, more features.
    - **Overfitting (training score high, validation score low):** The model is too complex. Decrease complexity — shallower tree, more regularization, fewer features.
    - **Sweet spot (both scores high, close together):** You are in the Goldilocks zone. The gap between training and validation score should be small.
    """)
    return


@app.cell
def _(Pipeline, RandomForestClassifier, StandardScaler, X_train_w, np, plt, y_train_w):
    from sklearn.model_selection import validation_curve

    # Validation curve: max_depth for RandomForest
    param_range_depth = np.array([1, 2, 3, 5, 7, 10, 15, 20])

    train_scores_vc, val_scores_vc = validation_curve(
        Pipeline([("scaler", StandardScaler()),
                  ("clf", RandomForestClassifier(n_estimators=100, random_state=42))]),
        X_train_w, y_train_w,
        param_name="clf__max_depth",
        param_range=param_range_depth,
        cv=5, scoring="accuracy", n_jobs=-1
    )

    train_mean = train_scores_vc.mean(axis=1)
    train_std  = train_scores_vc.std(axis=1)
    val_mean   = val_scores_vc.mean(axis=1)
    val_std    = val_scores_vc.std(axis=1)

    fig_vc, ax_vc = plt.subplots(figsize=(7, 4))
    ax_vc.plot(param_range_depth, train_mean, "o-", label="Training score", color="tab:blue")
    ax_vc.fill_between(param_range_depth, train_mean - train_std, train_mean + train_std,
                       alpha=0.15, color="tab:blue")
    ax_vc.plot(param_range_depth, val_mean, "o-", label="Validation score", color="tab:orange")
    ax_vc.fill_between(param_range_depth, val_mean - val_std, val_mean + val_std,
                       alpha=0.15, color="tab:orange")
    ax_vc.set_xlabel("max_depth")
    ax_vc.set_ylabel("Accuracy")
    ax_vc.set_title("Validation curve — RandomForest max_depth")
    ax_vc.legend(loc="lower right")
    ax_vc.set_ylim(0.8, 1.02)
    plt.tight_layout()
    fig_vc
    return (validation_curve,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read this plot carefully. The training score stays near 1.0 (the forest can always memorize training data with enough depth). The validation score rises quickly, peaks around depth 5-7, and then levels off or slightly declines. The gap between training and validation widens as depth increases — that is the overfitting signature. The shaded bands show the standard deviation across folds; narrower bands mean a more stable estimate.

    For this dataset, `max_depth` around 5-10 is the sweet spot. Deeper trees waste capacity without improving generalization.

    ### Learning Curves: Score vs Training Set Size

    A learning curve plots training and validation scores as a function of the number of training samples. It answers a different question: **will collecting more data help?**

    The reading:

    - **Curves still converging (gap is closing, validation score still rising):** More data will help. Go collect more.
    - **Curves have plateaued (both flat, gap is stable):** More data will not help. You need a different model or better features.
    - **Large gap, both plateaued:** The model is overfitting even with all available data. Simplify the model or add regularization.
    """)
    return


@app.cell
def _(Pipeline, RandomForestClassifier, StandardScaler, X_train_w, np, plt, y_train_w):
    from sklearn.model_selection import learning_curve

    train_sizes_lc, train_scores_lc, val_scores_lc = learning_curve(
        Pipeline([("scaler", StandardScaler()),
                  ("clf", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))]),
        X_train_w, y_train_w,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, scoring="accuracy", n_jobs=-1
    )

    train_mean_lc = train_scores_lc.mean(axis=1)
    train_std_lc  = train_scores_lc.std(axis=1)
    val_mean_lc   = val_scores_lc.mean(axis=1)
    val_std_lc    = val_scores_lc.std(axis=1)

    fig_lc, ax_lc = plt.subplots(figsize=(7, 4))
    ax_lc.plot(train_sizes_lc, train_mean_lc, "o-", label="Training score", color="tab:blue")
    ax_lc.fill_between(train_sizes_lc, train_mean_lc - train_std_lc,
                       train_mean_lc + train_std_lc, alpha=0.15, color="tab:blue")
    ax_lc.plot(train_sizes_lc, val_mean_lc, "o-", label="Validation score", color="tab:orange")
    ax_lc.fill_between(train_sizes_lc, val_mean_lc - val_std_lc,
                       val_mean_lc + val_std_lc, alpha=0.15, color="tab:orange")
    ax_lc.set_xlabel("Training set size")
    ax_lc.set_ylabel("Accuracy")
    ax_lc.set_title("Learning curve — RandomForest (max_depth=5)")
    ax_lc.legend(loc="lower right")
    ax_lc.set_ylim(0.7, 1.02)
    plt.tight_layout()
    fig_lc
    return (learning_curve,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### When to Use Which

    **Validation curve:** Use when you are tuning a specific hyperparameter and want to visualize the bias-variance tradeoff for that parameter. It tells you *where* to set the knob.

    **Learning curve:** Use when you are deciding whether to collect more data or change your modeling approach. It tells you whether the *data* or the *model* is the bottleneck.

    These two plots together give you a remarkably complete diagnostic picture. Here are the common scenarios and what to do in each:

    | Validation curve | Learning curve | Diagnosis | Action |
    |---|---|---|---|
    | Clear sweet spot, gap is small | Both scores high, close together | Model is well-tuned | Ship it |
    | All configs underfit | Curves converging, val still rising | Need more data or capacity | Collect data or increase model complexity |
    | High train, low val at all values | Large gap, both plateaued | Overfitting, data-limited | More data, regularization, or fewer features |
    | Sweet spot exists | Both scores plateau at low level | Good tuning, weak features | Feature engineering, different model family |

    If the validation curve shows a good sweet spot but the learning curve shows convergence at a low level, you need better features, not more tuning. If the learning curve shows the curves are still converging, stop tuning and go get more data.

    ---

    ## 6. When to Use What

    With all these tools in hand, how do you decide which model to try first? Here is a practical decision framework, not derived from theory but from decades of collective experience in the ML community.

    **Tabular data (spreadsheets, databases, CSV files):** Start with gradient boosting (XGBoost, LightGBM, or scikit-learn's `GradientBoostingClassifier/Regressor`). Tree ensembles dominate tabular data in nearly every benchmark and Kaggle competition. Use a linear model (Ridge, LogReg) as a baseline — if the linear model is competitive, the problem is easy and you should use the simpler model.

    **Small $n$, many features ($p \gg n$):** Regularized linear models (Ridge, Lasso, ElasticNet). Trees overfit when data is scarce. L1 regularization (Lasso) also performs automatic feature selection, which is valuable when you have hundreds or thousands of features.

    **Need calibrated probabilities:** Logistic Regression gives well-calibrated probabilities out of the box. Tree ensembles give decent rankings (high AUC) but the raw probabilities are often poorly calibrated. You can fix this with `CalibratedClassifierCV`, but LogReg is simpler if you need probabilities.

    **Need interpretability:** Linear models, small decision trees, or models with post-hoc explanations (SHAP values for tree ensembles). In regulated industries (healthcare, finance, insurance), being able to explain *why* a prediction was made is often a hard requirement.

    **Images:** Convolutional Neural Networks. We will cover these in Module 5C-5D.

    **Sequences (text, time series):** Recurrent networks or Transformers. Also covered later.

    ### The No Free Lunch Theorem in Practice

    The No Free Lunch theorem states that no single algorithm is universally best across all possible problems. Averaged over *all* possible data distributions, every algorithm performs identically. This is a theoretical result with a practical punchline: **you must test on your specific data.**

    In practice, the No Free Lunch theorem is less nihilistic than it sounds. Real-world problems are not uniformly distributed over all possible problems. They have structure — smoothness, sparsity, locality, symmetry — and some algorithms exploit that structure better than others. The practical truth: for most tabular problems, try 3-4 models, tune the best one, and move on. Diminishing returns set in quickly.

    ### A Practical Workflow

    Here is the workflow that experienced practitioners converge on:

    1. **Baseline first.** Fit the simplest reasonable model (linear regression for regression, logistic regression for classification). This takes seconds and gives you a floor to beat. If the baseline is already excellent, stop — you don't need a complex model.
    2. **Try 3-4 candidate models** — typically a linear model, a tree ensemble (Random Forest or GBM), and maybe an SVM or a nearest-neighbor method. Don't try 20 models; it wastes time and inflates your risk of overfitting to the validation set.
    3. **Tune the best 1-2 models** using `RandomizedSearchCV`. Start coarse (wide parameter ranges, fewer iterations), then refine around the best region.
    4. **Inspect diagnostic plots** — validation curves and learning curves — to understand whether your model is data-limited or capacity-limited.
    5. **Report final performance** on the held-out test set, once. Resist the temptation to "peek" and then go back for more tuning.

    This workflow is boring. That is a feature, not a bug. Boring, systematic workflows produce reliable results. Flashy, ad hoc experimentation produces models that look good on your laptop and fail in production.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interactive: Model Comparison Dashboard

    Use the slider below to control the regularization strength and see how it affects model performance across different model families. This illustrates the central tradeoff: too little regularization and you overfit, too much and you underfit.
    """)
    return


@app.cell
def _(mo):
    alpha_slider = mo.ui.slider(
        start=-3, stop=3, step=0.5, value=0,
        label="log10(alpha) for Ridge / Lasso"
    )
    alpha_slider
    return (alpha_slider,)


@app.cell
def _(Pipeline, StandardScaler, X_train_h, alpha_slider, np, plt, y_train_h):
    from sklearn.model_selection import cross_val_score

    alpha_val = 10 ** alpha_slider.value
    models_compare = {
        "Ridge":      Pipeline([("s", StandardScaler()), ("r", Ridge(alpha=alpha_val))]),
        "Lasso":      Pipeline([("s", StandardScaler()), ("r", Lasso(alpha=alpha_val, max_iter=5000))]),
        "ElasticNet": Pipeline([("s", StandardScaler()), ("r", ElasticNet(alpha=alpha_val, l1_ratio=0.5, max_iter=5000))]),
    }

    fig_comp, ax_comp = plt.subplots(figsize=(7, 4))
    for name, pipe in models_compare.items():
        scores = cross_val_score(pipe, X_train_h, y_train_h, cv=5,
                                 scoring="neg_mean_squared_error")
        mse_vals = -scores
        ax_comp.bar(name, mse_vals.mean(), yerr=mse_vals.std(), capsize=5, alpha=0.7)

    ax_comp.set_ylabel("Cross-validated MSE")
    ax_comp.set_title(f"Regression models at alpha = {alpha_val:.4f}")
    plt.tight_layout()
    fig_comp
    return (cross_val_score,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Try sliding alpha to the extremes. At very small alpha (log10 = -3), regularization is negligible and all three models behave like OLS. At very large alpha (log10 = 3), the penalty dominates and the models underfit — Lasso may drive all coefficients to zero, giving terrible MSE. The sweet spot is somewhere in between, and finding it is exactly what `GridSearchCV` automates for you.

    ### Common Gotchas

    Before moving to the exercises, here are the mistakes that trip up students (and professionals) most often:

    **Forgetting to put preprocessing inside the pipeline.** If you scale your features before calling `cross_val_score`, the scaler has seen the validation fold's statistics. This is data leakage. Always put `StandardScaler` (or any preprocessing) as a step in the `Pipeline`.

    **Using accuracy on imbalanced data.** A model that always predicts the majority class gets high accuracy on imbalanced datasets. Always check the confusion matrix and use appropriate metrics (F1, precision, recall, AUC).

    **Not setting `random_state`.** Without a fixed random state, your results will change every time you run the notebook. This makes debugging impossible and comparisons unreliable. Set `random_state` on your models, your splitters, and your random search.

    **Evaluating on the test set more than once.** Every time you look at test performance and then go back to tune, you are leaking information from the test set into your model selection. If you find yourself wanting to "check just one more thing" on the test set, stop — use cross-validation on the training set instead.

    **Ignoring the standard deviation.** A mean CV score of 0.92 with std of 0.08 is much less reliable than a mean of 0.90 with std of 0.01. Always report both. If two models differ by less than one standard deviation, the difference is probably not meaningful.

    ---

    ## 7. Code It + Exercises

    Time to implement these ideas yourself. Each "Code It" exercise gives you a skeleton with `TODO` placeholders. Fill them in. The point is not just to make the code run — it is to build the muscle memory of the experimental workflow: split, define models, cross-validate, inspect diagnostics, tune, and evaluate.

    ---

    ### Code It: Exercise 1 -- Full Regression Pipeline

    Build a complete regression pipeline with cross-validation on the California Housing dataset. Compare at least three models and report MSE, MAE, and R2 for each.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np_ex1
        from sklearn.datasets import fetch_california_housing as fch_ex1
        from sklearn.model_selection import train_test_split as tts_ex1, cross_validate as cv_ex1
        from sklearn.pipeline import Pipeline as Pipe_ex1
        from sklearn.preprocessing import StandardScaler as SS_ex1
        from sklearn.linear_model import Ridge as Ridge_ex1, Lasso as Lasso_ex1
        from sklearn.ensemble import RandomForestRegressor as RFR_ex1

        X_ex1, y_ex1 = fch_ex1(return_X_y=True)

        # TODO: Split into 80% train, 20% test
        # X_tr, X_te, y_tr, y_te = tts_ex1(...)

        # TODO: Define at least 3 pipelines (with StandardScaler!)
        # models_ex1 = {
        #     "Ridge":  Pipe_ex1([("scaler", SS_ex1()), ("reg", Ridge_ex1(alpha=1.0))]),
        #     "Lasso":  Pipe_ex1([("scaler", SS_ex1()), ("reg", Lasso_ex1(alpha=0.01))]),
        #     "RF":     Pipe_ex1([("scaler", SS_ex1()), ("reg", RFR_ex1(n_estimators=100, random_state=42))]),
        # }

        # TODO: Run 5-fold CV with scoring=["neg_mean_squared_error", "neg_mean_absolute_error", "r2"]
        # for name, pipe in models_ex1.items():
        #     result = cv_ex1(pipe, X_tr, y_tr, cv=5,
        #                     scoring=["neg_mean_squared_error", "neg_mean_absolute_error", "r2"])
        #     mse = -result["test_neg_mean_squared_error"].mean()
        #     mae = -result["test_neg_mean_absolute_error"].mean()
        #     r2  = result["test_r2"].mean()
        #     print(f"{name:8s} | MSE={mse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")

        # TODO: Pick the best model, retrain on full training set, evaluate on test set ONCE
        # best_pipe = ...
        # best_pipe.fit(X_tr, y_tr)
        # print(f"Test R2: {best_pipe.score(X_te, y_te):.4f}")
        pass


    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Code It: Exercise 2 -- Classification Metrics

    Train a classifier on the Breast Cancer dataset with stratified 5-fold CV. Compute all five metrics (accuracy, precision, recall, F1, ROC-AUC) and plot the confusion matrix.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np_ex2
        import matplotlib.pyplot as plt_ex2
        from sklearn.datasets import load_breast_cancer as lbc_ex2
        from sklearn.model_selection import (train_test_split as tts_ex2,
                                             StratifiedKFold as SKF_ex2,
                                             cross_validate as cv_ex2,
                                             cross_val_predict as cvp_ex2)
        from sklearn.pipeline import Pipeline as Pipe_ex2
        from sklearn.preprocessing import StandardScaler as SS_ex2
        from sklearn.linear_model import LogisticRegression as LR_ex2
        from sklearn.metrics import confusion_matrix as cm_ex2, ConfusionMatrixDisplay as CMD_ex2

        X_ex2, y_ex2 = lbc_ex2(return_X_y=True)

        # TODO: Stratified train/test split (20% test)
        # X_tr, X_te, y_tr, y_te = tts_ex2(X_ex2, y_ex2, test_size=0.2,
        #                                    random_state=42, stratify=y_ex2)

        # TODO: Define a pipeline with StandardScaler + LogisticRegression
        # pipe = Pipe_ex2([("scaler", SS_ex2()), ("clf", LR_ex2(max_iter=5000))])

        # TODO: Run stratified 5-fold CV with all 5 metrics
        # skf = SKF_ex2(n_splits=5, shuffle=True, random_state=42)
        # scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        # result = cv_ex2(pipe, X_tr, y_tr, cv=skf, scoring=scoring)
        # for metric in scoring:
        #     vals = result[f"test_{metric}"]
        #     print(f"{metric:12s}: {vals.mean():.4f} +/- {vals.std():.4f}")

        # TODO: Get cross-validated predictions and plot confusion matrix
        # y_pred = cvp_ex2(pipe, X_tr, y_tr, cv=skf)
        # cm = cm_ex2(y_tr, y_pred)
        # fig, ax = plt_ex2.subplots(figsize=(5, 4))
        # CMD_ex2(cm, display_labels=["Malignant", "Benign"]).plot(ax=ax, cmap="Blues")
        # plt_ex2.tight_layout()
        # plt_ex2.show()
        pass


    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Code It: Exercise 3 -- GridSearchCV on Gradient Boosting

    Tune a `GradientBoostingClassifier` on the Wine dataset. Search over `learning_rate`, `max_depth`, and `n_estimators`. Report the best parameters and test accuracy.
    """)
    return


@app.cell
def _():
    def _run():
        from sklearn.datasets import load_wine as lw_ex3
        from sklearn.model_selection import (train_test_split as tts_ex3,
                                             GridSearchCV as GS_ex3)
        from sklearn.pipeline import Pipeline as Pipe_ex3
        from sklearn.preprocessing import StandardScaler as SS_ex3
        from sklearn.ensemble import GradientBoostingClassifier as GBC_ex3

        X_ex3, y_ex3 = lw_ex3(return_X_y=True)

        # TODO: Stratified train/test split
        # X_tr, X_te, y_tr, y_te = tts_ex3(X_ex3, y_ex3, test_size=0.2,
        #                                    random_state=42, stratify=y_ex3)

        # TODO: Define pipeline
        # pipe = Pipe_ex3([("scaler", SS_ex3()),
        #                  ("clf", GBC_ex3(random_state=42))])

        # TODO: Define parameter grid
        # param_grid = {
        #     "clf__learning_rate": [0.01, 0.1, 0.2],
        #     "clf__max_depth": [2, 3, 5],
        #     "clf__n_estimators": [50, 100, 200],
        # }

        # TODO: Run GridSearchCV with 5-fold CV
        # gs = GS_ex3(pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
        # gs.fit(X_tr, y_tr)
        # print(f"Best CV accuracy: {gs.best_score_:.4f}")
        # print(f"Best params:      {gs.best_params_}")
        # print(f"Test accuracy:    {gs.score(X_te, y_te):.4f}")
        pass


    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Code It: Exercise 4 -- Validation Curve and Learning Curve

    For a `GradientBoostingClassifier` on the Wine dataset, plot a validation curve for `max_depth` and a learning curve. Interpret both.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np_ex4
        import matplotlib.pyplot as plt_ex4
        from sklearn.datasets import load_wine as lw_ex4
        from sklearn.model_selection import (train_test_split as tts_ex4,
                                             validation_curve as vc_ex4,
                                             learning_curve as lc_ex4)
        from sklearn.pipeline import Pipeline as Pipe_ex4
        from sklearn.preprocessing import StandardScaler as SS_ex4
        from sklearn.ensemble import GradientBoostingClassifier as GBC_ex4

        X_ex4, y_ex4 = lw_ex4(return_X_y=True)
        X_tr, X_te, y_tr, y_te = tts_ex4(X_ex4, y_ex4, test_size=0.2,
                                           random_state=42, stratify=y_ex4)

        pipe = Pipe_ex4([("scaler", SS_ex4()),
                         ("clf", GBC_ex4(n_estimators=100, random_state=42))])

        # TODO: Validation curve for clf__max_depth
        # param_range = np_ex4.array([1, 2, 3, 5, 7, 10])
        # train_sc, val_sc = vc_ex4(pipe, X_tr, y_tr,
        #                           param_name="clf__max_depth",
        #                           param_range=param_range,
        #                           cv=5, scoring="accuracy", n_jobs=-1)
        # fig, axes = plt_ex4.subplots(1, 2, figsize=(14, 5))
        # axes[0].plot(param_range, train_sc.mean(axis=1), "o-", label="Train")
        # axes[0].fill_between(param_range, train_sc.mean(axis=1) - train_sc.std(axis=1),
        #                      train_sc.mean(axis=1) + train_sc.std(axis=1), alpha=0.15)
        # axes[0].plot(param_range, val_sc.mean(axis=1), "o-", label="Validation")
        # axes[0].fill_between(param_range, val_sc.mean(axis=1) - val_sc.std(axis=1),
        #                      val_sc.mean(axis=1) + val_sc.std(axis=1), alpha=0.15)
        # axes[0].set_xlabel("max_depth")
        # axes[0].set_ylabel("Accuracy")
        # axes[0].set_title("Validation Curve")
        # axes[0].legend()

        # TODO: Learning curve
        # train_sizes, train_sc_lc, val_sc_lc = lc_ex4(
        #     pipe, X_tr, y_tr,
        #     train_sizes=np_ex4.linspace(0.1, 1.0, 8),
        #     cv=5, scoring="accuracy", n_jobs=-1)
        # axes[1].plot(train_sizes, train_sc_lc.mean(axis=1), "o-", label="Train")
        # axes[1].fill_between(train_sizes,
        #                      train_sc_lc.mean(axis=1) - train_sc_lc.std(axis=1),
        #                      train_sc_lc.mean(axis=1) + train_sc_lc.std(axis=1), alpha=0.15)
        # axes[1].plot(train_sizes, val_sc_lc.mean(axis=1), "o-", label="Validation")
        # axes[1].fill_between(train_sizes,
        #                      val_sc_lc.mean(axis=1) - val_sc_lc.std(axis=1),
        #                      val_sc_lc.mean(axis=1) + val_sc_lc.std(axis=1), alpha=0.15)
        # axes[1].set_xlabel("Training set size")
        # axes[1].set_ylabel("Accuracy")
        # axes[1].set_title("Learning Curve")
        # axes[1].legend()
        # plt_ex4.tight_layout()
        # plt_ex4.show()
        pass


    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Practice

    ### Pencil and Paper

    1. You plot a learning curve and observe: training score = 0.99, validation score = 0.72, and both curves have plateaued after 1000 training samples. The gap is large and stable. What is the diagnosis? What should you do next? Would collecting 10,000 more samples help?

    2. Your `GridSearchCV` on a Random Forest with 4 hyperparameters (each with 5 values) and 5-fold CV requires how many total model fits? Show the calculation. Is this practical on a dataset with 100,000 samples? What alternative would you use, and why?

    3. Explain why the harmonic mean (F1) is more informative than the arithmetic mean for combining precision and recall. Compute both for a model with precision = 0.99 and recall = 0.01. What does the arithmetic mean tell you? What does F1 tell you? Which is more honest?

    4. You build a cancer screening model. The cost of a false negative (missed cancer) is roughly 100x the cost of a false positive (unnecessary biopsy). Should you optimize for precision, recall, F1, or something else? Justify your answer. What classification threshold would you set, and why?

    5. Your model achieves ROC-AUC of 0.95 but accuracy of 0.60. How is this possible? What does it tell you about the model and the data? Hint: consider what happens when classes are heavily imbalanced and the default threshold of 0.5 is suboptimal.

    ### Coding

    6. Load `sklearn.datasets.fetch_california_housing`. Build pipelines for OLS, Ridge, Lasso, ElasticNet, and `RandomForestRegressor`. Compare all five using 5-fold CV with MSE, MAE, and R2. Plot a bar chart of MSE with error bars ($\pm 1$ std). Which model wins? Is the difference statistically meaningful (hint: compare stds)?

    7. Repeat the classification comparison from Section 3 on `sklearn.datasets.load_digits` (10-class classification). Which metrics from the binary case generalize cleanly to multiclass? Which require modification (hint: look at the `average` parameter for precision, recall, and F1)? Try both `average='macro'` and `average='weighted'` — when do they differ?

    8. Use `RandomizedSearchCV` to tune a `GradientBoostingClassifier` on the Breast Cancer dataset. Search over `learning_rate` (loguniform from 0.001 to 1.0), `max_depth` (randint from 2 to 10), `n_estimators` (randint from 50 to 500), and `subsample` (uniform from 0.5 to 1.0). Use 60 iterations and 5-fold CV. Compare the best result with an untuned default GBM.

    9. Plot both a validation curve (for `max_depth` ranging from 1 to 15) and a learning curve for a `GradientBoostingRegressor` on the California Housing dataset. Based on your plots, answer: (a) What is the optimal `max_depth`? (b) Would more training data improve performance? (c) Is the model underfitting, overfitting, or well-tuned?

    10. Build a nested cross-validation loop: outer 5-fold for evaluation, inner `GridSearchCV` for tuning. Apply it to `LogisticRegression` on the Breast Cancer dataset with `C` in [0.01, 0.1, 1, 10, 100]. Compare the nested CV score with the non-nested (biased) CV score — the non-nested score is what you get from `GridSearchCV.best_score_`. How large is the optimistic bias? Is it large enough to change your conclusions?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Key Takeaways

    1. **ML is experimental.** You do not know which model wins until you test it. Design your experiments before running them: fix the metric, hold out a test set, use cross-validation.
    2. **Use pipelines.** Put all preprocessing inside `Pipeline` to prevent data leakage. This is non-negotiable.
    3. **Understand your metrics.** MSE penalizes large errors; MAE is robust; R2 is scale-free. For classification, accuracy is misleading on imbalanced data — use F1, precision/recall, or AUC as appropriate.
    4. **The confusion matrix is your best friend.** Every scalar metric is a lossy summary. The confusion matrix shows you exactly where mistakes happen.
    5. **Random search beats grid search** for most tuning tasks. Tune the parameters that matter most on log scales.
    6. **Read your diagnostic plots.** Validation curves tell you where to set a hyperparameter. Learning curves tell you whether to collect more data. Together they give a complete diagnostic picture.
    7. **Report standard deviations.** A model comparison without uncertainty estimates is meaningless. Always report mean $\pm$ std from cross-validation.

    **Key references:**
    - [Geron Chapter 2: End-to-End Machine Learning Project](file:///C:/Users/landa/ml-course/textbooks/Geron.pdf) -- Practical walkthrough of a complete pipeline
    - [ISLR Chapter 5: Resampling Methods](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf) -- Cross-validation and bootstrap theory
    - [ESL Chapter 7: Model Assessment and Selection](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf) -- The definitive technical treatment of all topics in this module
    - Bergstra & Bengio (2012), "Random Search for Hyper-Parameter Optimization" -- Why random search beats grid search

    > **Next**: 5C -- Deep Learning with PyTorch
    """)
    return


if __name__ == "__main__":
    app.run()
