import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    return (mo, np, plt, warnings)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Practical ML Lab: scikit-learn & PyTorch

    You have built linear regression from scratch, coded backprop by hand, and implemented gradient descent with nothing but NumPy. That was the point — you needed to understand what these algorithms *are* before trusting a library to run them for you.

    Now it is time to learn the tools that practitioners actually use. This module is almost entirely exercises. The goal is fluency: given a dataset and a modeling task, you should be able to go from raw data to evaluated model in minutes, not hours.

    **What this module covers:**
    - **scikit-learn**: pipelines, preprocessing, supervised models, unsupervised models, hyperparameter tuning, evaluation
    - **PyTorch**: tensors, autograd, `nn.Module`, training loops, CNNs, transfer learning, autoencoders

    **Prerequisites**: Parts 1–2 (classical ML + deep learning fundamentals). You do not need Part 3 or 4.

    **How to use this module**: Each exercise gives you a task, context connecting it to the theory you already know, and skeleton code. Fill in the TODOs, run the cell, and check your output against the expected results. Solutions are provided inline — try before you peek.

    ---
    """)
    return


# ---------------------------------------------------------------------------
# SECTION 1: scikit-learn — Supervised Pipelines
# ---------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Section 1: scikit-learn — Supervised Pipelines

    scikit-learn's API is built on three verbs: **fit**, **transform**, **predict**. Every estimator follows the same pattern:

    ```python
    model = SomeEstimator(hyperparams)
    model.fit(X_train, y_train)          # learn from data
    predictions = model.predict(X_test)  # apply what was learned
    ```

    Transformers (scalers, encoders, PCA) add `.transform()` and `.fit_transform()`. Pipelines chain transformers and estimators so the entire workflow is a single object that can be cross-validated without data leakage.

    This is the API you will use for the rest of your career. Learn it well.
    """)
    return


# ---- Exercise 1: End-to-End Regression ----

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Exercise 1: End-to-End Regression Pipeline

    **Task**: Build a complete regression pipeline on the California Housing dataset. Fit three models (OLS, Ridge, Lasso), evaluate with MSE and R², and compare via 5-fold cross-validation.

    **Connects to**: Module 1B (linear regression, regularization), Module 1D (cross-validation, model selection)

    The California Housing dataset has 8 features (median income, house age, average rooms, etc.) predicting median house value. It is a clean, medium-sized regression benchmark — 20,640 samples, no missing values.
    """)
    return


@app.cell
def _(np, plt):
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_squared_error, r2_score

    # Load data
    housing = fetch_california_housing()
    X_h, y_h = housing.data, housing.target
    feature_names_h = housing.feature_names

    print(f"Dataset: {X_h.shape[0]} samples, {X_h.shape[1]} features")
    print(f"Features: {feature_names_h}")
    print(f"Target range: [{y_h.min():.2f}, {y_h.max():.2f}] (median house value in $100k)")

    # Train/test split
    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
        X_h, y_h, test_size=0.2, random_state=42
    )

    # Build pipelines: StandardScaler → Model
    pipelines_h = {
        "OLS": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "Ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "Lasso": Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=0.01))]),
    }

    # Fit, predict, evaluate
    results_h = {}
    for name, pipe in pipelines_h.items():
        pipe.fit(X_train_h, y_train_h)
        y_pred = pipe.predict(X_test_h)
        cv_scores = cross_val_score(pipe, X_train_h, y_train_h, cv=5, scoring="r2")
        results_h[name] = {
            "test_mse": mean_squared_error(y_test_h, y_pred),
            "test_r2": r2_score(y_test_h, y_pred),
            "cv_r2_mean": cv_scores.mean(),
            "cv_r2_std": cv_scores.std(),
            "y_pred": y_pred,
        }
        print(f"{name:6s} | Test MSE: {results_h[name]['test_mse']:.4f} | "
              f"Test R²: {results_h[name]['test_r2']:.4f} | "
              f"CV R²: {results_h[name]['cv_r2_mean']:.4f} ± {results_h[name]['cv_r2_std']:.4f}")

    # Plot predictions vs actuals for each model
    fig_h, axes_h = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (name, res) in zip(axes_h, results_h.items()):
        ax.scatter(y_test_h, res["y_pred"], alpha=0.15, s=8, color="steelblue")
        ax.plot([0, 5], [0, 5], "r--", lw=1.5)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{name} (R²={res['test_r2']:.3f})")
        ax.set_xlim(0, 5.2)
        ax.set_ylim(0, 5.2)
        ax.set_aspect("equal")
    fig_h.suptitle("Predictions vs Actuals — California Housing", y=1.02)
    plt.tight_layout()
    plt.show()
    return (
        Pipeline, StandardScaler, cross_val_score, fetch_california_housing,
        mean_squared_error, r2_score, train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Key takeaways from Exercise 1:**
    - All three models perform similarly here — this dataset is well-behaved and not overparameterized, so regularization helps only slightly.
    - The `Pipeline` ensures that the scaler is fit *only* on training data during cross-validation — no data leakage. If you had scaled before splitting, test fold statistics would leak into training.
    - `cross_val_score` handles the train/val splitting internally — you pass the *training* set, not the full dataset.

    ---
    """)
    return


# ---- Exercise 2: Classification Showdown ----

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 2: Classification Showdown

    **Task**: Compare four classifiers on the Breast Cancer dataset: Logistic Regression, SVM, Random Forest, and Gradient Boosting. Use pipelines with scaling, compute a full metrics suite, and plot confusion matrices.

    **Connects to**: Module 1C (classification, metrics), Module 1E (ensembles)
    """)
    return


@app.cell
def _(np, plt, Pipeline, StandardScaler, cross_val_score, train_test_split):
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression as LR_clf
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    )

    # Load data
    cancer = load_breast_cancer()
    X_c, y_c = cancer.data, cancer.target  # 569 samples, 30 features, binary
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_c, y_c, test_size=0.2, random_state=42, stratify=y_c
    )

    # Four classifiers in pipelines
    classifiers = {
        "LogReg": Pipeline([("scaler", StandardScaler()), ("clf", LR_clf(max_iter=2000))]),
        "SVM": Pipeline([("scaler", StandardScaler()), ("clf", SVC(probability=True))]),
        "RF": Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))]),
        "GBM": Pipeline([("scaler", StandardScaler()), ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42))]),
    }

    # Fit and evaluate
    print(f"{'Model':8s} | {'Acc':>6s} | {'Prec':>6s} | {'Rec':>6s} | {'F1':>6s} | {'AUC':>6s}")
    print("-" * 55)
    clf_results = {}
    for name, pipe in classifiers.items():
        pipe.fit(X_train_c, y_train_c)
        y_pred_c = pipe.predict(X_test_c)
        y_prob_c = pipe.predict_proba(X_test_c)[:, 1]
        metrics = {
            "acc": accuracy_score(y_test_c, y_pred_c),
            "prec": precision_score(y_test_c, y_pred_c),
            "rec": recall_score(y_test_c, y_pred_c),
            "f1": f1_score(y_test_c, y_pred_c),
            "auc": roc_auc_score(y_test_c, y_prob_c),
            "y_pred": y_pred_c,
        }
        clf_results[name] = metrics
        print(f"{name:8s} | {metrics['acc']:6.4f} | {metrics['prec']:6.4f} | "
              f"{metrics['rec']:6.4f} | {metrics['f1']:6.4f} | {metrics['auc']:6.4f}")

    # Confusion matrices
    fig_c, axes_c = plt.subplots(1, 4, figsize=(16, 3.5))
    for ax, (name, res) in zip(axes_c, clf_results.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test_c, res["y_pred"], ax=ax, cmap="Blues", colorbar=False,
            display_labels=["Malignant", "Benign"],
        )
        ax.set_title(f"{name} (F1={res['f1']:.3f})")
    fig_c.suptitle("Confusion Matrices — Breast Cancer Classification", y=1.02)
    plt.tight_layout()
    plt.show()
    return (
        GradientBoostingClassifier, RandomForestClassifier, SVC,
        accuracy_score, f1_score, precision_score, recall_score,
        roc_auc_score, confusion_matrix,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Key takeaways from Exercise 2:**
    - `stratify=y_c` in `train_test_split` preserves the class balance in both splits — critical for imbalanced datasets.
    - `SVC(probability=True)` enables `predict_proba`, which SVM does not provide by default (it uses Platt scaling internally).
    - `StandardScaler` matters most for SVM and Logistic Regression (distance/gradient-based). Tree models are invariant to feature scaling, but including the scaler in the pipeline doesn't hurt them and keeps the code uniform.
    - All four models score high here because the Breast Cancer dataset is relatively separable. On harder problems, you would see larger gaps.

    ---
    """)
    return


# ---- Exercise 3: Hyperparameter Tuning ----

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 3: Hyperparameter Tuning

    **Task**: Use `GridSearchCV` and `RandomizedSearchCV` to tune a Random Forest on the Wine dataset. Plot validation curves and learning curves.

    **Connects to**: Module 1D (cross-validation, hyperparameter tuning, learning curves)

    The Wine dataset has 13 chemical features predicting one of 3 wine cultivars (178 samples). Small enough to grid-search exhaustively.
    """)
    return


@app.cell
def _(np, plt, Pipeline, StandardScaler, RandomForestClassifier):
    from sklearn.datasets import load_wine
    from sklearn.model_selection import (
        GridSearchCV, RandomizedSearchCV,
        validation_curve, learning_curve,
    )
    from scipy.stats import randint

    wine = load_wine()
    X_w, y_w = wine.data, wine.target

    # --- GridSearchCV on RandomForest ---
    pipe_w = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(random_state=42)),
    ])

    param_grid = {
        "clf__n_estimators": [50, 100, 200],
        "clf__max_depth": [3, 5, 10, None],
        "clf__min_samples_split": [2, 5, 10],
    }

    grid = GridSearchCV(pipe_w, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_w, y_w)
    print(f"GridSearchCV best params: {grid.best_params_}")
    print(f"GridSearchCV best CV accuracy: {grid.best_score_:.4f}")

    # --- RandomizedSearchCV on same pipeline ---
    param_dist = {
        "clf__n_estimators": randint(50, 300),
        "clf__max_depth": [3, 5, 10, 15, None],
        "clf__min_samples_split": randint(2, 20),
        "clf__min_samples_leaf": randint(1, 10),
    }

    random_search = RandomizedSearchCV(
        pipe_w, param_dist, n_iter=30, cv=5, scoring="accuracy",
        random_state=42, n_jobs=-1,
    )
    random_search.fit(X_w, y_w)
    print(f"\nRandomizedSearchCV best params: {random_search.best_params_}")
    print(f"RandomizedSearchCV best CV accuracy: {random_search.best_score_:.4f}")
    return (GridSearchCV, RandomizedSearchCV, learning_curve, validation_curve, X_w, y_w, grid)


@app.cell
def _(np, plt, Pipeline, StandardScaler, RandomForestClassifier, validation_curve, learning_curve, X_w, y_w):
    # --- Validation Curve: accuracy vs max_depth ---
    pipe_vc = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])

    depth_range = [2, 3, 5, 7, 10, 15, 20, None]
    # validation_curve doesn't handle None well in the range, so we use integers only
    depth_range_int = [2, 3, 5, 7, 10, 15, 20, 30]
    train_scores_vc, test_scores_vc = validation_curve(
        pipe_vc, X_w, y_w, param_name="clf__max_depth",
        param_range=depth_range_int, cv=5, scoring="accuracy",
    )

    fig_vc, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # Validation curve
    ax1.plot(depth_range_int, train_scores_vc.mean(axis=1), "o-", label="Train", color="steelblue")
    ax1.fill_between(depth_range_int,
                     train_scores_vc.mean(axis=1) - train_scores_vc.std(axis=1),
                     train_scores_vc.mean(axis=1) + train_scores_vc.std(axis=1),
                     alpha=0.15, color="steelblue")
    ax1.plot(depth_range_int, test_scores_vc.mean(axis=1), "o-", label="Validation", color="coral")
    ax1.fill_between(depth_range_int,
                     test_scores_vc.mean(axis=1) - test_scores_vc.std(axis=1),
                     test_scores_vc.mean(axis=1) + test_scores_vc.std(axis=1),
                     alpha=0.15, color="coral")
    ax1.set_xlabel("max_depth")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Validation Curve: RF max_depth")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Learning curve
    train_sizes_lc, train_scores_lc, test_scores_lc = learning_curve(
        pipe_vc, X_w, y_w, train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, scoring="accuracy",
    )

    ax2.plot(train_sizes_lc, train_scores_lc.mean(axis=1), "o-", label="Train", color="steelblue")
    ax2.fill_between(train_sizes_lc,
                     train_scores_lc.mean(axis=1) - train_scores_lc.std(axis=1),
                     train_scores_lc.mean(axis=1) + train_scores_lc.std(axis=1),
                     alpha=0.15, color="steelblue")
    ax2.plot(train_sizes_lc, test_scores_lc.mean(axis=1), "o-", label="Validation", color="coral")
    ax2.fill_between(train_sizes_lc,
                     test_scores_lc.mean(axis=1) - test_scores_lc.std(axis=1),
                     test_scores_lc.mean(axis=1) + test_scores_lc.std(axis=1),
                     alpha=0.15, color="coral")
    ax2.set_xlabel("Training Set Size")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Learning Curve: RF (max_depth=10)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Key takeaways from Exercise 3:**
    - `GridSearchCV` is exhaustive — it tries every combination. With 3 x 4 x 3 = 36 combos x 5 folds = 180 model fits, it is manageable here but explodes combinatorially with more hyperparameters.
    - `RandomizedSearchCV` samples from distributions instead. For the same compute budget, it typically finds better hyperparameters because it explores more of the space. The `n_iter` parameter controls how many random samples to try.
    - The prefix `clf__` in parameter names is how sklearn addresses parameters inside a `Pipeline`. The format is `<step_name>__<param_name>`.
    - The **validation curve** shows overfitting: when train accuracy is perfect but validation drops, the model is too complex. The **learning curve** shows whether more data would help: if train and validation curves haven't converged, more data will likely improve performance.

    ---
    """)
    return


# ---- Exercise 4: Feature Engineering Pipeline ----

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 4: Feature Engineering Pipeline

    **Task**: Build a preprocessing pipeline that handles mixed feature types — numerical columns get scaled, categorical columns get one-hot encoded — then feeds into a classifier. Cross-validate the whole thing as a single unit.

    **Connects to**: Module 1D (evaluation), real-world data preparation

    In practice, real datasets have mixed types. The `ColumnTransformer` is how sklearn handles this: you specify which transformers apply to which columns, and it stitches the outputs together.
    """)
    return


@app.cell
def _(np, plt, Pipeline, StandardScaler, GradientBoostingClassifier, cross_val_score):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    import pandas as pd

    # Create a synthetic mixed-type dataset (simulating real-world data)
    rng_fe = np.random.default_rng(42)
    n_fe = 500
    df_fe = pd.DataFrame({
        "age": rng_fe.normal(45, 15, n_fe).clip(18, 90),
        "income": rng_fe.lognormal(10.5, 0.8, n_fe),
        "credit_score": rng_fe.normal(680, 80, n_fe).clip(300, 850),
        "employment": rng_fe.choice(["salaried", "self-employed", "freelance", "retired"], n_fe),
        "education": rng_fe.choice(["high_school", "bachelors", "masters", "phd"], n_fe),
        "region": rng_fe.choice(["north", "south", "east", "west"], n_fe),
    })
    # Target: loan approval (influenced by income, credit_score, employment)
    logit_fe = (
        0.002 * df_fe["income"].values / 1000
        + 0.01 * df_fe["credit_score"].values
        - 5.0
        + 0.5 * (df_fe["employment"] == "salaried").astype(float).values
    )
    y_fe = (rng_fe.random(n_fe) < 1 / (1 + np.exp(-logit_fe))).astype(int)

    print(f"Dataset: {df_fe.shape}, target balance: {y_fe.mean():.2f}")
    print(f"\nColumn types:\n{df_fe.dtypes}")

    # Define column groups
    num_cols = ["age", "income", "credit_score"]
    cat_cols = ["employment", "education", "region"]

    # Build ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(drop="first", sparse_output=False)),
            ]), cat_cols),
        ]
    )

    # Full pipeline: preprocess → classify
    full_pipe = Pipeline([
        ("preprocess", preprocessor),
        ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ])

    # Cross-validate the ENTIRE pipeline — no data leakage
    cv_scores_fe = cross_val_score(full_pipe, df_fe, y_fe, cv=5, scoring="accuracy")
    print(f"\n5-Fold CV Accuracy: {cv_scores_fe.mean():.4f} ± {cv_scores_fe.std():.4f}")

    # Fit on full data to inspect feature importances
    full_pipe.fit(df_fe, y_fe)
    feature_names_out = full_pipe.named_steps["preprocess"].get_feature_names_out()
    importances = full_pipe.named_steps["clf"].feature_importances_

    # Plot feature importances
    idx_sorted = np.argsort(importances)
    fig_fe, ax_fe = plt.subplots(figsize=(8, 5))
    ax_fe.barh(range(len(importances)), importances[idx_sorted], color="steelblue")
    ax_fe.set_yticks(range(len(importances)))
    ax_fe.set_yticklabels(feature_names_out[idx_sorted], fontsize=9)
    ax_fe.set_xlabel("Feature Importance")
    ax_fe.set_title("GBM Feature Importances (Mixed-Type Pipeline)")
    plt.tight_layout()
    plt.show()
    return (ColumnTransformer, OneHotEncoder, pd)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Key takeaways from Exercise 4:**
    - `ColumnTransformer` applies different transformations to different column subsets, then concatenates the results. This is essential for real-world data where you have numerical, categorical, and sometimes text features.
    - `OneHotEncoder(drop="first")` avoids the dummy variable trap (multicollinearity from redundant binary columns).
    - The entire pipeline — imputation, scaling, encoding, and classification — is cross-validated as a single unit. This means the imputer learns fill values and the scaler learns means/stds *only from the training fold each time*. If you had done these steps outside the pipeline before cross-validation, you would have data leakage.
    - `get_feature_names_out()` tells you what the transformed feature matrix looks like after the `ColumnTransformer` — useful for interpreting feature importances.

    ---
    """)
    return


# ---------------------------------------------------------------------------
# SECTION 2: scikit-learn — Unsupervised Learning
# ---------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Section 2: scikit-learn — Unsupervised Learning

    Unsupervised models follow the same API pattern, but without `y`:

    ```python
    model = KMeans(n_clusters=3)
    model.fit(X)                      # learn structure from X alone
    labels = model.predict(X)         # assign clusters
    # or equivalently:
    labels = model.fit_predict(X)
    ```

    Transformers like PCA use `fit_transform`:
    ```python
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)  # learn + project in one step
    ```
    """)
    return


# ---- Exercise 5: Clustering Comparison ----

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 5: Clustering Comparison

    **Task**: Apply K-Means, DBSCAN, and Agglomerative Clustering to the Digits dataset. Use the elbow method and silhouette scores to pick K for K-Means. Visualize all three clusterings in 2D via PCA.

    **Connects to**: Module 1F (K-means, clustering evaluation, PCA)
    """)
    return


@app.cell
def _(np, plt, StandardScaler):
    from sklearn.datasets import load_digits
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA

    digits = load_digits()
    X_d, y_d = digits.data, digits.target  # 1797 samples, 64 features (8x8 images), 10 classes

    # Scale the data
    X_d_scaled = StandardScaler().fit_transform(X_d)

    # --- Elbow method + silhouette scores for K-Means ---
    K_range = range(2, 16)
    inertias = []
    silhouettes = []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_d_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_d_scaled, km.labels_))

    fig_cl, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(list(K_range), inertias, "o-", color="steelblue")
    ax1.set_xlabel("K")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Method")
    ax1.axvline(x=10, color="red", linestyle="--", alpha=0.5, label="K=10 (true)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(list(K_range), silhouettes, "o-", color="coral")
    ax2.set_xlabel("K")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Analysis")
    ax2.axvline(x=10, color="red", linestyle="--", alpha=0.5, label="K=10 (true)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    best_k = list(K_range)[np.argmax(silhouettes)]
    print(f"Best K by silhouette: {best_k} (score={max(silhouettes):.4f})")
    print(f"True number of classes: 10")
    return (PCA, KMeans, DBSCAN, AgglomerativeClustering, silhouette_score,
            load_digits, X_d_scaled, y_d)


@app.cell
def _(np, plt, PCA, KMeans, DBSCAN, AgglomerativeClustering, silhouette_score, X_d_scaled, y_d):
    # --- Compare three clustering algorithms ---
    # Project to 2D for visualization
    pca_2d = PCA(n_components=2)
    X_2d = pca_2d.fit_transform(X_d_scaled)

    # K-Means with K=10
    km_10 = KMeans(n_clusters=10, random_state=42, n_init=10)
    labels_km = km_10.fit_predict(X_d_scaled)

    # DBSCAN — needs tuning of eps
    dbscan = DBSCAN(eps=6.5, min_samples=8)
    labels_db = dbscan.fit_predict(X_d_scaled)

    # Agglomerative
    agg = AgglomerativeClustering(n_clusters=10)
    labels_agg = agg.fit_predict(X_d_scaled)

    # Plot
    fig_cl2, axes_cl2 = plt.subplots(1, 4, figsize=(18, 4))

    scatter_kwargs = dict(s=8, alpha=0.6)
    axes_cl2[0].scatter(X_2d[:, 0], X_2d[:, 1], c=y_d, cmap="tab10", **scatter_kwargs)
    axes_cl2[0].set_title("True Labels")

    axes_cl2[1].scatter(X_2d[:, 0], X_2d[:, 1], c=labels_km, cmap="tab10", **scatter_kwargs)
    sil_km = silhouette_score(X_d_scaled, labels_km)
    axes_cl2[1].set_title(f"K-Means (sil={sil_km:.3f})")

    # DBSCAN may produce noise points (label=-1)
    n_noise = (labels_db == -1).sum()
    mask_db = labels_db >= 0
    axes_cl2[2].scatter(X_2d[mask_db, 0], X_2d[mask_db, 1], c=labels_db[mask_db], cmap="tab10", **scatter_kwargs)
    axes_cl2[2].scatter(X_2d[~mask_db, 0], X_2d[~mask_db, 1], c="gray", s=5, alpha=0.3, label="noise")
    n_clusters_db = len(set(labels_db) - {-1})
    axes_cl2[2].set_title(f"DBSCAN ({n_clusters_db} clusters, {n_noise} noise)")
    axes_cl2[2].legend(fontsize=7)

    axes_cl2[3].scatter(X_2d[:, 0], X_2d[:, 1], c=labels_agg, cmap="tab10", **scatter_kwargs)
    sil_agg = silhouette_score(X_d_scaled, labels_agg)
    axes_cl2[3].set_title(f"Agglomerative (sil={sil_agg:.3f})")

    for ax in axes_cl2:
        ax.set_xticks([])
        ax.set_yticks([])

    fig_cl2.suptitle("Clustering Comparison on Digits Dataset (PCA 2D projection)", y=1.02)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Key takeaways from Exercise 5:**
    - The **elbow method** looks for a "kink" in the inertia curve — the point where adding more clusters gives diminishing returns. For digits, the elbow is not sharp because the clusters overlap in high dimensions.
    - The **silhouette score** measures how well each point fits its own cluster vs. the nearest other cluster. It ranges from -1 to 1; higher is better. It may not pick K=10 because some digits (like 1 and 7, or 3 and 8) genuinely overlap.
    - **DBSCAN** does not require you to specify K — it discovers clusters based on density. But it requires tuning `eps` (neighborhood radius) and `min_samples`. It also labels sparse points as noise (-1), which can be useful or problematic depending on the task.
    - **Agglomerative clustering** builds a hierarchy of merges. With `n_clusters=10` it cuts the dendrogram at the right level.
    - Clustering labels are arbitrary — cluster 3 in K-Means has no relation to digit "3". Evaluating clustering quality without ground truth is fundamentally harder than evaluating classification.

    ---
    """)
    return


# ---- Exercise 6: Dimensionality Reduction ----

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 6: Dimensionality Reduction

    **Task**: Apply PCA and t-SNE to the Digits dataset. For PCA: plot the scree plot and cumulative explained variance, reconstruct digits from varying numbers of components. For t-SNE: compare perplexity settings.

    **Connects to**: Module 1F (PCA, dimensionality reduction), Module 0C (SVD, eigendecomposition)
    """)
    return


@app.cell
def _(np, plt, PCA, load_digits, StandardScaler):
    from sklearn.manifold import TSNE

    digits_dr = load_digits()
    X_dr = StandardScaler().fit_transform(digits_dr.data)
    y_dr = digits_dr.target

    # --- PCA: Scree plot and cumulative variance ---
    pca_full = PCA().fit(X_dr)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)

    fig_dr, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.bar(range(1, len(pca_full.explained_variance_ratio_) + 1),
            pca_full.explained_variance_ratio_, color="steelblue", alpha=0.7)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_title("Scree Plot")
    ax1.set_xlim(0, 30)
    ax1.grid(True, alpha=0.3)

    ax2.plot(range(1, len(cumvar) + 1), cumvar, "o-", color="coral", markersize=3)
    ax2.axhline(y=0.90, color="gray", linestyle="--", alpha=0.5, label="90%")
    ax2.axhline(y=0.95, color="gray", linestyle=":", alpha=0.5, label="95%")
    n_90 = np.searchsorted(cumvar, 0.90) + 1
    n_95 = np.searchsorted(cumvar, 0.95) + 1
    ax2.axvline(x=n_90, color="steelblue", linestyle="--", alpha=0.5)
    ax2.axvline(x=n_95, color="steelblue", linestyle=":", alpha=0.5)
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Explained Variance")
    ax2.set_title("Cumulative Variance")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    print(f"Components for 90% variance: {n_90}")
    print(f"Components for 95% variance: {n_95}")
    return (TSNE, X_dr, y_dr, digits_dr)


@app.cell
def _(np, plt, PCA, X_dr, digits_dr):
    # --- PCA Reconstruction at different component counts ---
    n_components_list = [2, 5, 10, 20, 40, 64]
    sample_indices = [0, 1, 4, 7, 9]  # pick a few digits to reconstruct

    fig_rc, axes_rc = plt.subplots(len(sample_indices), len(n_components_list) + 1,
                                    figsize=(14, 8))

    for row, idx in enumerate(sample_indices):
        # Original (on the unscaled data for visualization)
        original = digits_dr.data[idx].reshape(8, 8)
        axes_rc[row, 0].imshow(original, cmap="gray_r")
        axes_rc[row, 0].set_title("Original" if row == 0 else "")
        axes_rc[row, 0].set_ylabel(f"digit={digits_dr.target[idx]}")

        for col, n_comp in enumerate(n_components_list):
            pca_rc = PCA(n_components=n_comp)
            X_proj = pca_rc.fit_transform(X_dr)
            X_recon = pca_rc.inverse_transform(X_proj)
            # inverse_transform gives us back in the scaled space; for viz, reshape
            recon = X_recon[idx].reshape(8, 8)
            axes_rc[row, col + 1].imshow(recon, cmap="gray_r")
            axes_rc[row, col + 1].set_title(f"n={n_comp}" if row == 0 else "")

    for ax in axes_rc.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig_rc.suptitle("PCA Reconstruction Quality vs Number of Components", y=1.01)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(plt, TSNE, X_dr, y_dr):
    # --- t-SNE: perplexity comparison ---
    perplexities = [5, 30, 100]
    fig_tsne, axes_tsne = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, perp in zip(axes_tsne, perplexities):
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42, n_iter=1000)
        X_tsne = tsne.fit_transform(X_dr)
        scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_dr, cmap="tab10", s=8, alpha=0.6)
        ax.set_title(f"t-SNE (perplexity={perp})")
        ax.set_xticks([])
        ax.set_yticks([])

    fig_tsne.colorbar(scatter, ax=axes_tsne, label="Digit", shrink=0.8)
    fig_tsne.suptitle("t-SNE Perplexity Comparison on Digits", y=1.02)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Key takeaways from Exercise 6:**
    - PCA is a **linear** dimensionality reduction. It finds the directions of maximum variance and projects onto them. With just 20 of 64 components, you capture ~90% of the variance — the remaining 44 dimensions are mostly noise.
    - The reconstruction visualization makes this concrete: at n=10, digits are recognizable but blurry. At n=20, they are nearly perfect. This is the core PCA tradeoff: compression vs. information loss.
    - t-SNE is a **nonlinear** method designed for visualization (2D/3D only). It preserves local neighborhoods: nearby points in high-D stay nearby in 2D. But distances between distant clusters are *not* meaningful — don't interpret the gaps.
    - **Perplexity** controls the effective number of neighbors t-SNE considers. Low perplexity (5) shows fine-grained local structure but may fragment real clusters. High perplexity (100) shows global structure but can merge distinct clusters. Perplexity ~30 is a common default.
    - t-SNE is stochastic and slow — different random seeds give different layouts. PCA is deterministic and fast. Use PCA for preprocessing and t-SNE for visualization, not as a preprocessing step for downstream models.

    ---
    """)
    return


# ---------------------------------------------------------------------------
# SECTION 3: PyTorch — Neural Network Fundamentals
# ---------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Section 3: PyTorch — Neural Network Fundamentals

    PyTorch's core abstraction is the **tensor** — a multi-dimensional array that can live on CPU or GPU and tracks gradients automatically. The training loop is explicit:

    ```python
    for epoch in range(n_epochs):
        y_hat = model(X)           # forward pass
        loss = loss_fn(y_hat, y)   # compute loss
        loss.backward()            # backward pass (compute gradients)
        optimizer.step()           # update parameters
        optimizer.zero_grad()      # reset gradients for next iteration
    ```

    Unlike sklearn's `.fit()`, PyTorch gives you full control over every step. This is more verbose but lets you customize anything: the loss function, the optimizer, the data loading, the training schedule. The exercises below build up from raw tensors to a complete CNN.
    """)
    return


# ---- Exercise 7: Tensors, Autograd, and Linear Regression ----

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 7: Tensors, Autograd, and Linear Regression in PyTorch

    **Task**: Implement linear regression using (1) raw tensors with manual autograd, then (2) `nn.Linear` with an optimizer. Compare both to the NumPy OLS solution.

    **Connects to**: Module 2A (backpropagation, computational graphs), Module 0F (gradient descent), Module 1B (linear regression)
    """)
    return


@app.cell
def _(np, plt):
    import torch
    import torch.nn as nn
    import torch.optim as optim

    # Generate synthetic data: y = 3x1 - 2x2 + 1 + noise
    torch.manual_seed(42)
    n_t = 200
    X_t = torch.randn(n_t, 2)
    w_true_t = torch.tensor([3.0, -2.0])
    b_true_t = 1.0
    y_t = X_t @ w_true_t + b_true_t + 0.3 * torch.randn(n_t)

    print(f"True weights: {w_true_t.numpy()}, bias: {b_true_t}")
    print(f"X shape: {X_t.shape}, y shape: {y_t.shape}")

    # --- Method 1: Raw tensors with autograd ---
    w_manual = torch.randn(2, requires_grad=True)
    b_manual = torch.zeros(1, requires_grad=True)
    lr_manual = 0.01
    losses_manual = []

    for i in range(500):
        # Forward
        y_pred = X_t @ w_manual + b_manual
        loss = ((y_pred - y_t) ** 2).mean()
        losses_manual.append(loss.item())

        # Backward
        loss.backward()

        # Update (no_grad because we don't want to track this operation)
        with torch.no_grad():
            w_manual -= lr_manual * w_manual.grad
            b_manual -= lr_manual * b_manual.grad
            w_manual.grad.zero_()
            b_manual.grad.zero_()

    print(f"\nMethod 1 (raw autograd):")
    print(f"  Learned w: {w_manual.detach().numpy()}, b: {b_manual.item():.4f}")

    # --- Method 2: nn.Linear + optimizer ---
    model_lr = nn.Linear(2, 1)
    optimizer_lr = optim.SGD(model_lr.parameters(), lr=0.01)
    loss_fn_lr = nn.MSELoss()
    losses_nn = []

    for i in range(500):
        y_pred = model_lr(X_t).squeeze()
        loss = loss_fn_lr(y_pred, y_t)
        losses_nn.append(loss.item())

        loss.backward()
        optimizer_lr.step()
        optimizer_lr.zero_grad()

    w_nn = model_lr.weight.detach().numpy().flatten()
    b_nn = model_lr.bias.item()
    print(f"\nMethod 2 (nn.Linear + SGD):")
    print(f"  Learned w: {w_nn}, b: {b_nn:.4f}")

    # --- Method 3: NumPy OLS for comparison ---
    X_np_aug = np.c_[X_t.numpy(), np.ones(n_t)]
    w_ols = np.linalg.solve(X_np_aug.T @ X_np_aug, X_np_aug.T @ y_t.numpy())
    print(f"\nMethod 3 (NumPy OLS):")
    print(f"  Learned w: {w_ols[:2]}, b: {w_ols[2]:.4f}")

    # Plot loss curves
    fig_lr, ax_lr = plt.subplots(figsize=(8, 4))
    ax_lr.plot(losses_manual, label="Raw autograd", alpha=0.8)
    ax_lr.plot(losses_nn, label="nn.Linear + SGD", alpha=0.8)
    ax_lr.set_xlabel("Iteration")
    ax_lr.set_ylabel("MSE Loss")
    ax_lr.set_title("Linear Regression: Autograd vs nn.Module")
    ax_lr.legend()
    ax_lr.grid(True, alpha=0.3)
    ax_lr.set_yscale("log")
    plt.tight_layout()
    plt.show()
    return (torch, nn, optim)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Key takeaways from Exercise 7:**
    - `requires_grad=True` tells PyTorch to track all operations on that tensor. When you call `.backward()`, it walks the computational graph in reverse and fills in `.grad` for every leaf tensor.
    - `torch.no_grad()` is essential during the parameter update — without it, PyTorch would try to build a graph for the update step itself, wasting memory and potentially causing errors.
    - `optimizer.zero_grad()` is required because PyTorch *accumulates* gradients by default (useful for gradient accumulation with large models). If you forget this, gradients grow each iteration and training diverges.
    - The `nn.Linear` + optimizer approach is equivalent to the manual version but cleaner. As models get larger, the manual approach becomes impractical — `nn.Module` handles parameter registration, and `optim` handles update rules.
    - All three methods converge to the same solution — they are all minimizing the same MSE objective.

    ---
    """)
    return


# ---- Exercise 8: MNIST Feedforward Classifier ----

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 8: MNIST Feedforward Classifier

    **Task**: Build a 2-hidden-layer feedforward network to classify MNIST digits. Implement the full training loop with DataLoader, track loss and accuracy, and evaluate on the test set.

    **Connects to**: Module 2A (MLPs, activation functions), Module 2C (dropout)
    """)
    return


@app.cell
def _(np, plt, torch, nn, optim):
    from torch.utils.data import DataLoader, TensorDataset
    from torchvision import datasets, transforms

    # Load MNIST
    transform_mnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean/std
    ])

    train_data_mnist = datasets.MNIST("./data", train=True, download=True, transform=transform_mnist)
    test_data_mnist = datasets.MNIST("./data", train=False, download=True, transform=transform_mnist)

    train_loader_mnist = DataLoader(train_data_mnist, batch_size=128, shuffle=True)
    test_loader_mnist = DataLoader(test_data_mnist, batch_size=256, shuffle=False)

    print(f"Training set: {len(train_data_mnist)} images")
    print(f"Test set: {len(test_data_mnist)} images")
    print(f"Image shape: {train_data_mnist[0][0].shape}")  # (1, 28, 28)

    # Define feedforward network
    class FeedforwardNet(nn.Module):
        def __init__(self, hidden1=256, hidden2=128, dropout=0.2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),                # (batch, 1, 28, 28) → (batch, 784)
                nn.Linear(784, hidden1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden1, hidden2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden2, 10),
            )

        def forward(self, x):
            return self.net(x)

    device_ff = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_ff = FeedforwardNet().to(device_ff)
    optimizer_ff = optim.Adam(model_ff.parameters(), lr=1e-3)
    loss_fn_ff = nn.CrossEntropyLoss()

    print(f"\nDevice: {device_ff}")
    print(f"Parameters: {sum(p.numel() for p in model_ff.parameters()):,}")
    return (DataLoader, TensorDataset, datasets, transforms,
            FeedforwardNet, device_ff, model_ff, optimizer_ff, loss_fn_ff,
            train_loader_mnist, test_loader_mnist)


@app.cell
def _(torch, model_ff, optimizer_ff, loss_fn_ff, device_ff,
      train_loader_mnist, test_loader_mnist, plt):
    # Training loop
    n_epochs_ff = 5
    history_ff = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(n_epochs_ff):
        # --- Train ---
        model_ff.train()
        running_loss, correct, total = 0.0, 0, 0
        for X_batch, y_batch in train_loader_mnist:
            X_batch, y_batch = X_batch.to(device_ff), y_batch.to(device_ff)

            logits = model_ff(X_batch)
            loss = loss_fn_ff(logits, y_batch)

            loss.backward()
            optimizer_ff.step()
            optimizer_ff.zero_grad()

            running_loss += loss.item() * X_batch.size(0)
            correct += (logits.argmax(1) == y_batch).sum().item()
            total += X_batch.size(0)

        history_ff["train_loss"].append(running_loss / total)
        history_ff["train_acc"].append(correct / total)

        # --- Evaluate ---
        model_ff.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader_mnist:
                X_batch, y_batch = X_batch.to(device_ff), y_batch.to(device_ff)
                logits = model_ff(X_batch)
                loss = loss_fn_ff(logits, y_batch)
                running_loss += loss.item() * X_batch.size(0)
                correct += (logits.argmax(1) == y_batch).sum().item()
                total += X_batch.size(0)

        history_ff["test_loss"].append(running_loss / total)
        history_ff["test_acc"].append(correct / total)

        print(f"Epoch {epoch+1}/{n_epochs_ff} | "
              f"Train Loss: {history_ff['train_loss'][-1]:.4f} | "
              f"Test Loss: {history_ff['test_loss'][-1]:.4f} | "
              f"Test Acc: {history_ff['test_acc'][-1]:.4f}")

    # Plot training curves
    fig_ff, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs_range = range(1, n_epochs_ff + 1)

    ax1.plot(epochs_range, history_ff["train_loss"], "o-", label="Train", color="steelblue")
    ax1.plot(epochs_range, history_ff["test_loss"], "o-", label="Test", color="coral")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Cross-Entropy Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, history_ff["train_acc"], "o-", label="Train", color="steelblue")
    ax2.plot(epochs_range, history_ff["test_acc"], "o-", label="Test", color="coral")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Classification Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return (history_ff,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Key takeaways from Exercise 8:**
    - `model.train()` enables dropout and batch norm training behavior. `model.eval()` disables them. Forgetting `model.eval()` during testing is a common bug — dropout randomly zeroes activations, artificially lowering test accuracy.
    - `torch.no_grad()` during evaluation saves memory by not building the computation graph. You do not need gradients for inference.
    - `CrossEntropyLoss` in PyTorch expects raw logits (not softmax outputs). It applies `log_softmax` + `NLLLoss` internally. Passing softmax outputs is a common mistake that makes training unstable.
    - `logits.argmax(1)` picks the class with the highest score for each sample in the batch. The `1` means "along dimension 1" (the class dimension).
    - A simple 2-layer feedforward net reaches ~97-98% on MNIST in just 5 epochs. This is a sanity check — if you get much lower, something is wrong with your training loop.

    ---
    """)
    return


# ---- Exercise 9: CNN for Fashion-MNIST ----

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 9: CNN for Fashion-MNIST

    **Task**: Build a CNN for Fashion-MNIST and compare its performance to the feedforward network. Fashion-MNIST is harder than MNIST — the classes (T-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot) have more visual overlap.

    **Connects to**: Module 2D (convolution, pooling, CNN architectures)
    """)
    return


@app.cell
def _(torch, nn, optim, plt, DataLoader, datasets, transforms):
    # Load Fashion-MNIST
    transform_fmnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    train_fmnist = datasets.FashionMNIST("./data", train=True, download=True, transform=transform_fmnist)
    test_fmnist = datasets.FashionMNIST("./data", train=False, download=True, transform=transform_fmnist)
    train_loader_f = DataLoader(train_fmnist, batch_size=128, shuffle=True)
    test_loader_f = DataLoader(test_fmnist, batch_size=256, shuffle=False)

    class_names_f = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                     "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    # --- CNN architecture ---
    class ConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                # Block 1: 1 → 32 channels
                nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (28,28)
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),                              # (14,14)
                # Block 2: 32 → 64 channels
                nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (14,14)
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),                              # (7,7)
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),                                  # 64*7*7 = 3136
                nn.Linear(64 * 7 * 7, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 10),
            )

        def forward(self, x):
            x = self.features(x)
            return self.classifier(x)

    # --- Feedforward baseline for comparison ---
    class FFBaseline(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 10),
            )
        def forward(self, x):
            return self.net(x)

    device_cnn = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train both models
    def train_model(model, train_loader, test_loader, n_epochs=8, lr=1e-3):
        model = model.to(device_cnn)
        opt = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        hist = {"train_loss": [], "test_acc": []}

        for epoch in range(n_epochs):
            model.train()
            total_loss, n = 0.0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device_cnn), yb.to(device_cnn)
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
                opt.zero_grad()
                total_loss += loss.item() * xb.size(0)
                n += xb.size(0)
            hist["train_loss"].append(total_loss / n)

            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(device_cnn), yb.to(device_cnn)
                    correct += (model(xb).argmax(1) == yb).sum().item()
                    total += yb.size(0)
            hist["test_acc"].append(correct / total)
            print(f"  Epoch {epoch+1}/{n_epochs} | Loss: {hist['train_loss'][-1]:.4f} | "
                  f"Test Acc: {hist['test_acc'][-1]:.4f}")
        return hist

    print("Training Feedforward baseline:")
    ff_model = FFBaseline()
    hist_ff_f = train_model(ff_model, train_loader_f, test_loader_f, n_epochs=8)

    print("\nTraining CNN:")
    cnn_model = ConvNet()
    hist_cnn_f = train_model(cnn_model, train_loader_f, test_loader_f, n_epochs=8)

    n_params_ff = sum(p.numel() for p in ff_model.parameters())
    n_params_cnn = sum(p.numel() for p in cnn_model.parameters())

    # Compare
    fig_cnn, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs_r = range(1, 9)

    ax1.plot(epochs_r, hist_ff_f["train_loss"], "o-", label=f"FF ({n_params_ff:,} params)", color="steelblue")
    ax1.plot(epochs_r, hist_cnn_f["train_loss"], "o-", label=f"CNN ({n_params_cnn:,} params)", color="coral")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_r, hist_ff_f["test_acc"], "o-", label="Feedforward", color="steelblue")
    ax2.plot(epochs_r, hist_cnn_f["test_acc"], "o-", label="CNN", color="coral")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Test Accuracy")
    ax2.set_title("Test Accuracy: FF vs CNN on Fashion-MNIST")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\nFinal test accuracy — FF: {hist_ff_f['test_acc'][-1]:.4f}, CNN: {hist_cnn_f['test_acc'][-1]:.4f}")
    print(f"Parameters — FF: {n_params_ff:,}, CNN: {n_params_cnn:,}")
    return (ConvNet, device_cnn, cnn_model, train_loader_f, test_loader_f, class_names_f)


@app.cell
def _(torch, np, plt, cnn_model, test_loader_f, class_names_f, device_cnn):
    # Visualize first-layer filters
    filters_1 = cnn_model.features[0].weight.detach().cpu().numpy()  # (32, 1, 3, 3)

    fig_filt, axes_filt = plt.subplots(4, 8, figsize=(12, 6))
    for i, ax in enumerate(axes_filt.flat):
        ax.imshow(filters_1[i, 0], cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
    fig_filt.suptitle("Learned First-Layer CNN Filters (3x3)", y=1.01)
    plt.tight_layout()
    plt.show()

    # Show some predictions
    cnn_model.eval()
    test_iter = iter(test_loader_f)
    xb_show, yb_show = next(test_iter)
    xb_show = xb_show.to(device_cnn)
    with torch.no_grad():
        preds_show = cnn_model(xb_show).argmax(1).cpu()

    fig_pred, axes_pred = plt.subplots(2, 8, figsize=(14, 4))
    for i, ax in enumerate(axes_pred.flat):
        img = xb_show[i, 0].cpu().numpy()
        true_label = class_names_f[yb_show[i]]
        pred_label = class_names_f[preds_show[i]]
        color = "green" if yb_show[i] == preds_show[i] else "red"
        ax.imshow(img, cmap="gray_r")
        ax.set_title(f"{pred_label}", fontsize=8, color=color)
        ax.set_xticks([])
        ax.set_yticks([])
    fig_pred.suptitle("CNN Predictions (green=correct, red=wrong)", y=1.02)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Key takeaways from Exercise 9:**
    - The CNN outperforms the feedforward network by ~2-4% accuracy on Fashion-MNIST, often with fewer total parameters. This is the power of **parameter sharing** and **translation invariance** — convolutional layers exploit the spatial structure of images that fully-connected layers ignore.
    - `BatchNorm2d` normalizes activations per channel across the spatial dimensions and batch. It stabilizes training and often allows higher learning rates.
    - The first-layer filters learn simple edge and texture detectors — horizontal, vertical, diagonal edges, blobs. This is consistent with what Module 2D predicts: early layers learn local features, later layers compose them into higher-level patterns.
    - `padding=1` with a 3x3 kernel preserves the spatial dimensions ("same" padding). `MaxPool2d(2)` halves each spatial dimension.
    - Fashion-MNIST is a better benchmark than MNIST for comparing architectures — MNIST is "too easy" (even a linear classifier gets ~92%).

    ---
    """)
    return


# ---------------------------------------------------------------------------
# SECTION 4: PyTorch — Going Deeper
# ---------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Section 4: PyTorch — Going Deeper

    The final three exercises tackle problems where naive approaches fail and more sophisticated techniques are required: transfer learning for small datasets, recurrent networks for sequences, and autoencoders for unsupervised representation learning.
    """)
    return


# ---- Exercise 10: Transfer Learning ----

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 10: Transfer Learning

    **Task**: Compare three strategies on a small CIFAR-10 subset (1000 training images): (a) training a small CNN from scratch, (b) using a frozen pretrained ResNet18 as a feature extractor, (c) fine-tuning the full ResNet18.

    **Connects to**: Module 2D (transfer learning, ResNet), Module 3C (pretrain-finetune paradigm)

    Transfer learning is one of the most practically important techniques in deep learning. When you have limited data, features learned on a large dataset (ImageNet) transfer remarkably well to new tasks.
    """)
    return


@app.cell
def _(torch, nn, optim, plt, DataLoader, datasets, transforms):
    from torchvision.models import resnet18, ResNet18_Weights

    # CIFAR-10 with ResNet-compatible preprocessing
    transform_cifar = transforms.Compose([
        transforms.Resize(224),  # ResNet expects 224x224
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_cifar_full = datasets.CIFAR10("./data", train=True, download=True, transform=transform_cifar)
    test_cifar = datasets.CIFAR10("./data", train=False, download=True, transform=transform_cifar)

    # Subsample: only 1000 training images (100 per class)
    rng_tl = torch.Generator().manual_seed(42)
    indices_tl = []
    targets_arr = torch.tensor(train_cifar_full.targets)
    for c in range(10):
        class_idx = (targets_arr == c).nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(class_idx), generator=rng_tl)[:100]
        indices_tl.extend(class_idx[perm].tolist())

    train_subset = torch.utils.data.Subset(train_cifar_full, indices_tl)
    train_loader_tl = DataLoader(train_subset, batch_size=32, shuffle=True)
    test_loader_tl = DataLoader(test_cifar, batch_size=64, shuffle=False)

    cifar_classes = ["airplane", "auto", "bird", "cat", "deer",
                     "dog", "frog", "horse", "ship", "truck"]

    print(f"Training subset: {len(train_subset)} images (100 per class)")
    print(f"Test set: {len(test_cifar)} images")

    # --- Strategy A: Small CNN from scratch ---
    class SmallCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(4),  # 56
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(4), # 14
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64, 10))
        def forward(self, x):
            return self.classifier(self.features(x))

    # --- Strategy B: Frozen ResNet18 feature extractor ---
    def make_frozen_resnet():
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False  # freeze everything
        model.fc = nn.Linear(model.fc.in_features, 10)  # new head (trainable)
        return model

    # --- Strategy C: Fine-tune full ResNet18 ---
    def make_finetune_resnet():
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 10)
        return model

    device_tl = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_and_eval(model, train_loader, test_loader, n_epochs=5, lr=1e-3):
        model = model.to(device_tl)
        opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        hist = {"test_acc": []}

        for epoch in range(n_epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device_tl), yb.to(device_tl)
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
                opt.zero_grad()

            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(device_tl), yb.to(device_tl)
                    correct += (model(xb).argmax(1) == yb).sum().item()
                    total += yb.size(0)
            acc = correct / total
            hist["test_acc"].append(acc)
            print(f"  Epoch {epoch+1}/{n_epochs} | Test Acc: {acc:.4f}")
        return hist

    print("\n--- Strategy A: Small CNN from scratch ---")
    hist_a = train_and_eval(SmallCNN(), train_loader_tl, test_loader_tl, n_epochs=5)

    print("\n--- Strategy B: Frozen ResNet18 (feature extractor) ---")
    hist_b = train_and_eval(make_frozen_resnet(), train_loader_tl, test_loader_tl, n_epochs=5, lr=1e-3)

    print("\n--- Strategy C: Fine-tune full ResNet18 ---")
    hist_c = train_and_eval(make_finetune_resnet(), train_loader_tl, test_loader_tl, n_epochs=5, lr=1e-4)

    # Plot comparison
    fig_tl, ax_tl = plt.subplots(figsize=(8, 5))
    ep = range(1, 6)
    ax_tl.plot(ep, hist_a["test_acc"], "o-", label="A: Scratch CNN", color="steelblue")
    ax_tl.plot(ep, hist_b["test_acc"], "s-", label="B: Frozen ResNet", color="coral")
    ax_tl.plot(ep, hist_c["test_acc"], "^-", label="C: Fine-tuned ResNet", color="seagreen")
    ax_tl.set_xlabel("Epoch")
    ax_tl.set_ylabel("Test Accuracy")
    ax_tl.set_title("Transfer Learning: 1000 CIFAR-10 Training Images")
    ax_tl.legend()
    ax_tl.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return (device_tl,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Key takeaways from Exercise 10:**
    - With only 1000 training images, the **from-scratch CNN struggles** — not enough data to learn good features. Transfer learning dramatically closes this gap.
    - **Frozen features** (Strategy B) work surprisingly well because the early/middle layers of a network trained on ImageNet learn universal features (edges, textures, shapes) that transfer to CIFAR-10. Only the final classification head is trained.
    - **Fine-tuning** (Strategy C) typically performs best because it can adapt the feature layers to the new task. But it requires a lower learning rate (1e-4 vs 1e-3) to avoid destroying the pretrained features — this is a key practical consideration.
    - `filter(lambda p: p.requires_grad, model.parameters())` ensures the optimizer only updates unfrozen parameters. Without this filter, the optimizer would track frozen parameters and waste memory.
    - The `Resize(224)` transform is necessary because ResNet18 was designed for 224x224 ImageNet images. CIFAR-10 images are 32x32 — upscaling them loses nothing (the information is still there) and matches the architecture's expected input size.

    ---
    """)
    return


# ---- Exercise 11: Sequence Modeling with LSTM ----

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 11: Sequence Modeling with LSTM

    **Task**: Train an LSTM to predict the next value in a noisy sine wave. Build a sliding-window DataLoader, train the model, and visualize its predictions on unseen future timesteps.

    **Connects to**: Module 2E (RNNs, LSTMs, hidden state, sequence-to-one)
    """)
    return


@app.cell
def _(np, torch, nn, optim, plt, DataLoader, TensorDataset):
    # Generate noisy sine wave
    t_seq = np.linspace(0, 20 * np.pi, 2000)
    signal = np.sin(t_seq) + 0.1 * np.random.randn(len(t_seq))
    signal = signal.astype(np.float32)

    # Sliding window dataset
    seq_len = 50
    X_seq_list, y_seq_list = [], []
    for i in range(len(signal) - seq_len):
        X_seq_list.append(signal[i:i + seq_len])
        y_seq_list.append(signal[i + seq_len])

    X_seq = torch.tensor(np.array(X_seq_list)).unsqueeze(-1)  # (N, seq_len, 1)
    y_seq = torch.tensor(np.array(y_seq_list))                # (N,)

    # Train/test split (temporal — no shuffling across time)
    split_seq = int(0.8 * len(X_seq))
    train_ds_seq = TensorDataset(X_seq[:split_seq], y_seq[:split_seq])
    test_ds_seq = TensorDataset(X_seq[split_seq:], y_seq[split_seq:])
    train_loader_seq = DataLoader(train_ds_seq, batch_size=64, shuffle=True)
    test_loader_seq = DataLoader(test_ds_seq, batch_size=64, shuffle=False)

    print(f"Sequence length: {seq_len}")
    print(f"Train sequences: {len(train_ds_seq)}, Test sequences: {len(test_ds_seq)}")
    print(f"Input shape per batch: (batch, {seq_len}, 1)")

    # LSTM model
    class LSTMPredictor(nn.Module):
        def __init__(self, input_size=1, hidden_size=64, num_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            # x: (batch, seq_len, 1)
            out, (h_n, c_n) = self.lstm(x)
            # Use the last hidden state
            last_hidden = out[:, -1, :]  # (batch, hidden_size)
            return self.fc(last_hidden).squeeze(-1)

    device_seq = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_seq = LSTMPredictor().to(device_seq)
    opt_seq = optim.Adam(model_seq.parameters(), lr=1e-3)
    loss_fn_seq = nn.MSELoss()

    print(f"Parameters: {sum(p.numel() for p in model_seq.parameters()):,}")
    return (model_seq, opt_seq, loss_fn_seq, device_seq,
            train_loader_seq, test_loader_seq,
            X_seq, y_seq, split_seq, signal, seq_len, t_seq)


@app.cell
def _(torch, plt, model_seq, opt_seq, loss_fn_seq, device_seq,
      train_loader_seq, test_loader_seq, X_seq, y_seq, split_seq, signal, seq_len, t_seq):
    # Training loop
    n_epochs_seq = 20
    losses_seq = []

    for epoch in range(n_epochs_seq):
        model_seq.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader_seq:
            xb, yb = xb.to(device_seq), yb.to(device_seq)
            pred = model_seq(xb)
            loss = loss_fn_seq(pred, yb)
            loss.backward()
            opt_seq.step()
            opt_seq.zero_grad()
            epoch_loss += loss.item()
            n_batches += 1
        losses_seq.append(epoch_loss / n_batches)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{n_epochs_seq} | Train Loss: {losses_seq[-1]:.6f}")

    # Predict on test set
    model_seq.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for xb, yb in test_loader_seq:
            xb = xb.to(device_seq)
            pred = model_seq(xb)
            all_preds.extend(pred.cpu().numpy())
            all_true.extend(yb.numpy())

    # Plot results
    fig_seq, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7))

    # Loss curve
    ax1.plot(range(1, n_epochs_seq + 1), losses_seq, "o-", color="steelblue")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("LSTM Training Loss")
    ax1.grid(True, alpha=0.3)

    # Prediction vs actual on test region
    test_t = t_seq[split_seq + seq_len:][:len(all_preds)]
    ax2.plot(test_t, all_true, label="Actual", alpha=0.7, color="steelblue")
    ax2.plot(test_t, all_preds, label="LSTM Predicted", alpha=0.7, color="coral", linestyle="--")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Value")
    ax2.set_title("LSTM Sine Wave Prediction (Test Set)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Key takeaways from Exercise 11:**
    - The LSTM processes the input sequence one timestep at a time, maintaining a hidden state that accumulates information. We use `out[:, -1, :]` — the hidden state after processing the entire sequence — to make the prediction. This is a **many-to-one** architecture.
    - `batch_first=True` means input shape is (batch, seq_len, features) rather than (seq_len, batch, features). The default in PyTorch is batch_first=False (a historical convention), so always set this explicitly.
    - The temporal train/test split is critical — shuffling across time would leak future information into training. In practice, you would also want a gap between train and test to prevent sequence overlap.
    - The LSTM learns the sinusoidal pattern quickly because it is periodic and low-dimensional. On real time series (stock prices, weather), the task is much harder because the underlying dynamics are nonstationary and high-dimensional.
    - Two LSTM layers (`num_layers=2`) let the second layer operate on the first layer's hidden states — a form of hierarchical temporal abstraction.

    ---
    """)
    return


# ---- Exercise 12: Autoencoder ----

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 12: Autoencoder for Unsupervised Representation Learning

    **Task**: Build an autoencoder that compresses MNIST digits to a 2D latent space, reconstructs them, and lets you visualize the latent manifold. This is *unsupervised* — we never use the digit labels during training.

    **Connects to**: Module 3B (autoencoders, VAEs, latent spaces)
    """)
    return


@app.cell
def _(torch, nn, optim, plt, np, DataLoader, datasets, transforms):
    # Load MNIST (reuse transform)
    transform_ae = transforms.Compose([transforms.ToTensor()])
    train_ae = datasets.MNIST("./data", train=True, download=True, transform=transform_ae)
    test_ae = datasets.MNIST("./data", train=False, download=True, transform=transform_ae)
    train_loader_ae = DataLoader(train_ae, batch_size=256, shuffle=True)
    test_loader_ae = DataLoader(test_ae, batch_size=256, shuffle=False)

    class Autoencoder(nn.Module):
        def __init__(self, latent_dim=2):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 256),
                nn.ReLU(),
                nn.Linear(256, 784),
                nn.Sigmoid(),  # output in [0, 1] to match pixel range
            )

        def forward(self, x):
            z = self.encoder(x)
            x_recon = self.decoder(z).view(-1, 1, 28, 28)
            return x_recon, z

    device_ae = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae = Autoencoder(latent_dim=2).to(device_ae)
    opt_ae = optim.Adam(ae.parameters(), lr=1e-3)

    print(f"Parameters: {sum(p.numel() for p in ae.parameters()):,}")
    print(f"Latent dimension: 2")
    return (ae, opt_ae, device_ae, train_loader_ae, test_loader_ae, train_ae, test_ae)


@app.cell
def _(torch, nn, plt, np, ae, opt_ae, device_ae, train_loader_ae, test_loader_ae):
    # Training loop — MSE reconstruction loss, no labels used
    n_epochs_ae = 20
    losses_ae = []

    for epoch in range(n_epochs_ae):
        ae.train()
        epoch_loss = 0.0
        n = 0
        for xb, _ in train_loader_ae:  # _ = labels, unused
            xb = xb.to(device_ae)
            x_recon, z = ae(xb)
            loss = nn.functional.mse_loss(x_recon, xb)
            loss.backward()
            opt_ae.step()
            opt_ae.zero_grad()
            epoch_loss += loss.item() * xb.size(0)
            n += xb.size(0)
        losses_ae.append(epoch_loss / n)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{n_epochs_ae} | Recon Loss: {losses_ae[-1]:.6f}")

    # Evaluate: collect latent codes and reconstructions
    ae.eval()
    all_z, all_labels, all_recon = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader_ae:
            xb = xb.to(device_ae)
            x_recon, z = ae(xb)
            all_z.append(z.cpu().numpy())
            all_labels.append(yb.numpy())
            all_recon.append(x_recon.cpu().numpy())
    all_z = np.concatenate(all_z)
    all_labels = np.concatenate(all_labels)
    all_recon = np.concatenate(all_recon)

    # --- Plot 1: Latent space ---
    fig_ae, axes_ae = plt.subplots(1, 3, figsize=(18, 5))

    scatter_ae = axes_ae[0].scatter(all_z[:, 0], all_z[:, 1], c=all_labels, cmap="tab10",
                                     s=4, alpha=0.5)
    axes_ae[0].set_xlabel("z₁")
    axes_ae[0].set_ylabel("z₂")
    axes_ae[0].set_title("2D Latent Space (colored by digit)")
    fig_ae.colorbar(scatter_ae, ax=axes_ae[0], label="Digit")

    # --- Plot 2: Reconstruction examples ---
    test_iter_ae = iter(test_loader_ae)
    xb_show_ae, yb_show_ae = next(test_iter_ae)
    xb_show_ae = xb_show_ae.to(device_ae)
    with torch.no_grad():
        recon_show, _ = ae(xb_show_ae)
    recon_show = recon_show.cpu().numpy()

    # Show 8 originals on top, reconstructions on bottom
    for i in range(8):
        # Original in row area of axes_ae[1]
        pass  # handled below

    # Use a sub-grid for originals vs reconstructions
    axes_ae[1].set_visible(False)
    axes_ae[2].set_visible(False)

    # Create inset axes for reconstruction comparison
    gs_recon = fig_ae.add_gridspec(2, 8, left=0.4, right=0.98, bottom=0.15, top=0.85, wspace=0.05, hspace=0.3)
    for i in range(8):
        ax_orig = fig_ae.add_subplot(gs_recon[0, i])
        ax_orig.imshow(xb_show_ae[i, 0].cpu().numpy(), cmap="gray_r")
        ax_orig.set_xticks([])
        ax_orig.set_yticks([])
        if i == 0:
            ax_orig.set_ylabel("Original", fontsize=9)

        ax_rec = fig_ae.add_subplot(gs_recon[1, i])
        ax_rec.imshow(recon_show[i, 0], cmap="gray_r")
        ax_rec.set_xticks([])
        ax_rec.set_yticks([])
        if i == 0:
            ax_rec.set_ylabel("Recon", fontsize=9)

    plt.show()
    return (all_z, all_labels)


@app.cell
def _(torch, np, plt, ae, device_ae):
    # --- Latent space interpolation ---
    # Pick two points in latent space and interpolate between them
    ae.eval()

    # Interpolate between two points
    z_start = torch.tensor([[-3.0, -2.0]], dtype=torch.float32).to(device_ae)
    z_end = torch.tensor([[3.0, 2.0]], dtype=torch.float32).to(device_ae)

    n_interp = 12
    alphas = np.linspace(0, 1, n_interp)
    z_interp = torch.stack([
        (1 - a) * z_start + a * z_end for a in alphas
    ]).squeeze(1)

    with torch.no_grad():
        decoded = ae.decoder(z_interp).view(-1, 28, 28).cpu().numpy()

    fig_interp, axes_interp = plt.subplots(1, n_interp, figsize=(16, 2))
    for i, ax in enumerate(axes_interp):
        ax.imshow(decoded[i], cmap="gray_r")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"α={alphas[i]:.1f}", fontsize=8)
    fig_interp.suptitle("Latent Space Interpolation: Walking Between Two Points", y=1.05)
    plt.tight_layout()
    plt.show()

    # --- Grid sample of the latent space ---
    grid_x = np.linspace(-4, 4, 15)
    grid_y = np.linspace(-4, 4, 15)
    grid_img = np.zeros((15 * 28, 15 * 28))

    with torch.no_grad():
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_pt = torch.tensor([[xi, yi]], dtype=torch.float32).to(device_ae)
                digit = ae.decoder(z_pt).view(28, 28).cpu().numpy()
                grid_img[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = digit

    fig_grid, ax_grid = plt.subplots(figsize=(8, 8))
    ax_grid.imshow(grid_img, cmap="gray_r")
    ax_grid.set_xlabel("z₁")
    ax_grid.set_ylabel("z₂")
    ax_grid.set_xticks(np.arange(0, 15 * 28, 28) + 14)
    ax_grid.set_xticklabels([f"{x:.1f}" for x in grid_x], fontsize=6, rotation=45)
    ax_grid.set_yticks(np.arange(0, 15 * 28, 28) + 14)
    ax_grid.set_yticklabels([f"{y:.1f}" for y in grid_y], fontsize=6)
    ax_grid.set_title("Decoded Grid: Sampling the 2D Latent Space")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Key takeaways from Exercise 12:**
    - The autoencoder learns to compress 784 pixels into **2 numbers** and reconstruct from them. The 2D latent space is a learned representation — nearby points decode to visually similar digits. This is unsupervised: the network never sees the labels.
    - `nn.Sigmoid()` in the decoder output ensures pixel values are in [0, 1], matching the input range. Using MSE loss between input and reconstruction trains the network to minimize pixel-wise error.
    - The **latent space visualization** shows that different digit classes naturally cluster — the network has discovered semantic structure without supervision. However, the clusters may overlap or be irregular, which is why Variational Autoencoders (VAEs, covered in Module 3B) add a regularization term to enforce a smooth, continuous latent space.
    - **Interpolation** between two latent points produces a smooth morphing between digit styles. This is the hallmark of a good latent representation — it captures the underlying factors of variation (stroke style, digit identity, angle) in a continuous way.
    - The **grid visualization** shows what the decoder "imagines" for each point in latent space. You can see different digit types emerge in different regions, with smooth transitions between them.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Summary: What You Have Practiced

    | Exercise | Library | Task | Key Concepts |
    |----------|---------|------|-------------|
    | 1 | sklearn | Regression pipeline | `Pipeline`, `StandardScaler`, CV, MSE/R² |
    | 2 | sklearn | Classification comparison | 4 classifiers, confusion matrix, ROC-AUC |
    | 3 | sklearn | Hyperparameter tuning | `GridSearchCV`, `RandomizedSearchCV`, validation/learning curves |
    | 4 | sklearn | Mixed-type preprocessing | `ColumnTransformer`, `OneHotEncoder`, no data leakage |
    | 5 | sklearn | Clustering comparison | K-Means, DBSCAN, Agglomerative, silhouette |
    | 6 | sklearn | Dimensionality reduction | PCA scree plot, t-SNE perplexity, reconstruction |
    | 7 | PyTorch | Tensors and autograd | `requires_grad`, `.backward()`, `no_grad()` |
    | 8 | PyTorch | MNIST feedforward | `nn.Module`, training loop, `DataLoader` |
    | 9 | PyTorch | Fashion-MNIST CNN | Conv layers, BatchNorm, FF vs CNN comparison |
    | 10 | PyTorch | Transfer learning | Frozen vs fine-tuned ResNet18, small data regime |
    | 11 | PyTorch | Sequence prediction | LSTM, sliding window, temporal splits |
    | 12 | PyTorch | Autoencoder | Latent space, reconstruction, interpolation |

    You now have working experience with the two most important ML libraries. The sklearn exercises cover the standard workflow for tabular data and classical models. The PyTorch exercises cover everything from manual gradient computation to transfer learning and generative models.

    The next step is to apply these patterns to your own datasets and problems. The APIs are the same — only the data and the model architecture change.

    > **Back to**: [Course Home](./home/) | [Algorithm Study Guide](./5a_study_guide/) | [Quiz & Flashcards](./quiz.html)
    """)
    return


if __name__ == "__main__":
    app.run()
