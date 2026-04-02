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
    from sklearn.datasets import (
        make_moons, make_classification, load_iris, load_wine
    )
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import (
        StandardScaler, MinMaxScaler, RobustScaler,
        OrdinalEncoder, OneHotEncoder
    )
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import accuracy_score
    import pandas as pd
    return (
        np, plt,
        make_moons, make_classification, load_iris, load_wine,
        train_test_split, cross_val_score,
        LinearRegression, LogisticRegression, Ridge,
        RandomForestClassifier, GradientBoostingClassifier,
        DecisionTreeClassifier, KMeans, PCA, SVC, KNeighborsClassifier,
        StandardScaler, MinMaxScaler, RobustScaler,
        OrdinalEncoder, OneHotEncoder,
        SimpleImputer,
        Pipeline, make_pipeline,
        ColumnTransformer,
        accuracy_score,
        pd,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Module 5A: The scikit-learn Ecosystem

    You have spent the first four parts of this course building things from scratch. You implemented OLS by solving the normal equations. You wrote logistic regression with gradient descent. You constructed decision trees node by node, backpropagation layer by layer. That was the right thing to do: you cannot effectively use a tool you do not understand, and you cannot debug a system whose internals are a black box.

    But now it is time to confront a fact that separates coursework from production machine learning: **the gap between "I can implement this algorithm" and "I can ship this model" is enormous.** In practice, you will not write your own StandardScaler. You will not hand-roll cross-validation. You will not manually split data, transform features, fit a model, and inverse-transform predictions in separate, disconnected steps. You will use a framework. And in the world of classical machine learning, that framework is scikit-learn.

    This module is not a tutorial that walks you through API docs. It is an explanation of the *design principles* behind sklearn — why the API looks the way it does, what problems it solves, and how to compose its pieces into systems that are correct by construction. You already know the algorithms. Now you learn the engineering that makes them usable.

    Reading: [Geron Ch 2-3](file:///C:/Users/landa/ml-course/textbooks/Geron.pdf) covers the sklearn workflow end-to-end. The original API design paper is Buitinck et al. (2013), "API design for machine learning software: experiences from the scikit-learn project" — it is short and worth reading for the philosophy.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 1. Why Frameworks Matter

    ### The Problem

    Imagine you are a data scientist at a company. You have built a churn prediction model in a Jupyter notebook. It works on your laptop. Now your manager asks: can you retrain it monthly on new data? Can you swap in a different algorithm without rewriting the data pipeline? Can someone else on the team understand your preprocessing steps without reading 200 lines of ad-hoc NumPy?

    If your code is a loose collection of NumPy operations — scale this array, encode that column, split here, fit there — the answer to all three questions is probably no. The logic is scattered. The ordering is implicit. The coupling between preprocessing and modeling is invisible. You will, at some point, fit a scaler on your test data and not notice. You will, at some point, forget to apply the same encoding at inference time. These are not hypothetical dangers. They are the default outcome of working without a framework.

    ### What Frameworks Buy You

    A well-designed ML framework provides four things:

    **Consistency.** Every model, every transformer, every preprocessor uses the same interface. You learn the pattern once and it works everywhere. You do not need to remember that Model A uses `.train()` while Model B uses `.fit()` while Model C takes the data in the constructor.

    **Reproducibility.** A Pipeline object captures the exact sequence of transformations applied to data. You can serialize it, version it, and apply it to new data months later with the guarantee that it does exactly the same thing. No implicit state, no forgotten steps.

    **Composability.** You can chain arbitrary transformers and estimators into pipelines, apply different transformations to different columns, nest pipelines inside other pipelines. The pieces snap together because they all follow the same interface contract.

    **Community.** When everyone uses the same API, code becomes portable. A preprocessing recipe from a Kaggle kernel works in your codebase. A custom transformer you write can be dropped into anyone else's pipeline. Bugs are found and fixed by thousands of users.

    ### The sklearn Design Philosophy

    scikit-learn was not the first ML library, but it won because of its API design. The key insight, articulated by Buitinck et al., is that machine learning workflows decompose into a small number of **object types**, each with a minimal, predictable interface:

    - **Estimators**: anything that learns from data. They have `.fit(X, y)` (supervised) or `.fit(X)` (unsupervised).
    - **Transformers**: estimators that also transform data. They add `.transform(X)` and the convenience `.fit_transform(X)`.
    - **Predictors**: estimators that also make predictions. They add `.predict(X)` and often `.predict_proba(X)` or `.score(X, y)`.

    That is the entire taxonomy. A `StandardScaler` is an Estimator + Transformer (it learns means and variances, then transforms data). A `RandomForestClassifier` is an Estimator + Predictor (it learns trees, then predicts classes). A `PCA` is an Estimator + Transformer (it learns principal components, then projects data). Every object in the library fits one of these categories, and the methods always have the same signatures.

    This is not just aesthetic cleanliness. It is what makes Pipelines possible — and Pipelines, as you will see, are the single most important abstraction in applied ML.
    """)
    return


@app.cell
def _(np, plt, make_classification, train_test_split, StandardScaler, LogisticRegression, RandomForestClassifier, KMeans, PCA):
    def _run():
        # The same 3-line pattern works for wildly different algorithms.
        # This is the payoff of a consistent API.
        X_demo, y_demo = make_classification(
            n_samples=200, n_features=4, n_informative=2,
            n_redundant=1, random_state=42
        )
        X_tr, X_te, y_tr, y_te = train_test_split(X_demo, y_demo, random_state=42)

        # Supervised: LogisticRegression
        lr = LogisticRegression(max_iter=200)
        lr.fit(X_tr, y_tr)
        print(f"LogisticRegression  — test accuracy: {lr.score(X_te, y_te):.3f}")

        # Supervised: RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X_tr, y_tr)
        print(f"RandomForest        — test accuracy: {rf.score(X_te, y_te):.3f}")

        # Unsupervised transformer: PCA
        pca = PCA(n_components=2)
        pca.fit(X_tr)
        X_tr_2d = pca.transform(X_tr)
        print(f"PCA                 — explained variance: {pca.explained_variance_ratio_.sum():.3f}")

        # Unsupervised predictor: KMeans
        km = KMeans(n_clusters=2, random_state=42, n_init=10)
        km.fit(X_tr)
        print(f"KMeans              — inertia: {km.inertia_:.1f}")

        print("\nFour completely different algorithms. Same .fit() pattern.")


    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice what just happened. We fitted a linear classifier, a tree ensemble, a dimensionality reduction method, and a clustering algorithm. The interface was identical every time: construct the object, call `.fit()`, then call `.predict()`, `.transform()`, or `.score()`. If you wanted to swap logistic regression for a gradient boosting classifier, you change exactly one line — the constructor. Nothing else in your code needs to change.

    This is not a minor convenience. It means you can write generic code that operates on *any* estimator. You can write a function that takes an estimator, fits it, evaluates it, and returns scores — and it works whether you pass it a `LinearRegression` or a `NeuralNetworkClassifier` from a third-party library that follows the sklearn interface. The API is a contract, and the contract enables composition.

    ---

    ## 2. The Estimator API — fit / transform / predict

    Let us look more carefully at the three core methods and what happens inside each one.

    ### fit(X, y=None)

    This is where learning happens. The estimator examines the training data and stores whatever it learns as **attributes with trailing underscores**. This is a sklearn convention: anything ending in `_` was set during `fit()`, not in the constructor.

    For a `StandardScaler`, `.fit()` computes and stores `mean_` and `scale_`. For a `DecisionTreeClassifier`, it builds and stores `tree_`. For a `PCA`, it computes and stores `components_` and `explained_variance_`. The constructor sets hyperparameters (things you choose); `.fit()` computes learned parameters (things estimated from data).

    ### transform(X)

    Available on Transformers. This applies the learned transformation to new data. Critically, it does **not** re-learn anything — it uses whatever was stored during `fit()`. When you call `scaler.transform(X_test)`, it subtracts the **training** mean and divides by the **training** standard deviation. This is essential for avoiding data leakage.

    ### predict(X)

    Available on Predictors. This applies the learned model to produce predictions. Like `transform()`, it uses the parameters learned during `fit()` without modifying them.

    ### fit_transform(X, y=None)

    A convenience method on Transformers that calls `fit(X, y)` followed by `transform(X)`. For most transformers, this is functionally identical to calling the two methods separately. For some (like `PCA` with randomized SVD), there is a computational shortcut that makes `fit_transform` faster than the two-step version. But semantically, it is always equivalent.

    ### The Underscore Convention

    This is worth emphasizing because it makes sklearn code self-documenting:
    """)
    return


@app.cell
def _(StandardScaler, np):
    def _run():
        # The underscore convention: constructor params vs learned attributes
        scaler = StandardScaler(with_mean=True, with_std=True)

        # Before fit: no learned attributes
        print("Before fit:")
        print(f"  with_mean (constructor param): {scaler.with_mean}")
        try:
            _ = scaler.mean_
        except AttributeError:
            print("  mean_ (learned attr): does not exist yet")

        # After fit: learned attributes appear
        X_example = np.array([[1, 10], [2, 20], [3, 30], [4, 40]], dtype=float)
        scaler.fit(X_example)
        print("\nAfter fit:")
        print(f"  mean_ = {scaler.mean_}")
        print(f"  scale_ = {scaler.scale_}")
        print(f"  n_features_in_ = {scaler.n_features_in_}")

        # Transform uses learned params — does NOT recompute
        X_new = np.array([[5, 50]])
        X_scaled = scaler.transform(X_new)
        print(f"\nTransform [5, 50] using TRAINING stats: {X_scaled.round(3)}")
        print(f"  (5 - {scaler.mean_[0]:.1f}) / {scaler.scale_[0]:.3f} = {X_scaled[0, 0]:.3f}")


    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interactive: One API, Many Models

    The dropdown below lets you pick any classifier. The data and evaluation code are identical — only the model object changes. This is the practical consequence of a uniform API.
    """)
    return


@app.cell
def _(mo):
    clf_dropdown = mo.ui.dropdown(
        options=["LogisticRegression", "DecisionTree", "RandomForest",
                 "GradientBoosting", "SVM (RBF)", "KNeighbors"],
        value="LogisticRegression",
        label="Classifier"
    )
    clf_dropdown
    return (clf_dropdown,)


@app.cell
def _(clf_dropdown, np, plt, make_moons, train_test_split,
      LogisticRegression, DecisionTreeClassifier, RandomForestClassifier,
      GradientBoostingClassifier, SVC, KNeighborsClassifier, StandardScaler):
    def _run():
        X_api, y_api = make_moons(n_samples=300, noise=0.25, random_state=42)
        X_tr, X_te, y_tr, y_te = train_test_split(X_api, y_api, test_size=0.3, random_state=42)

        # Scale for models that need it
        sc = StandardScaler().fit(X_tr)
        X_tr_s, X_te_s = sc.transform(X_tr), sc.transform(X_te)

        model_map = {
            "LogisticRegression": LogisticRegression(max_iter=300),
            "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "SVM (RBF)": SVC(kernel="rbf", gamma="scale"),
            "KNeighbors": KNeighborsClassifier(n_neighbors=5),
        }
        needs_scaling = {"LogisticRegression", "SVM (RBF)", "KNeighbors"}

        name = clf_dropdown.value
        model = model_map[name]
        Xtr = X_tr_s if name in needs_scaling else X_tr
        Xte = X_te_s if name in needs_scaling else X_te

        # Same API regardless of model
        model.fit(Xtr, y_tr)
        train_acc = model.score(Xtr, y_tr)
        test_acc = model.score(Xte, y_te)

        # Decision boundary
        x_min, x_max = Xtr[:, 0].min() - 0.5, Xtr[:, 0].max() + 0.5
        y_min, y_max = Xtr[:, 1].min() - 0.5, Xtr[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                              np.linspace(y_min, y_max, 200))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
        ax.scatter(Xtr[y_tr == 0, 0], Xtr[y_tr == 0, 1], c="steelblue",
                   edgecolors="k", s=20, label="Class 0")
        ax.scatter(Xtr[y_tr == 1, 0], Xtr[y_tr == 1, 1], c="tomato",
                   edgecolors="k", s=20, label="Class 1")
        ax.set_title(f"{name}  |  Train: {train_acc:.3f}  |  Test: {test_acc:.3f}")
        ax.legend(loc="lower left")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        plt.tight_layout()
        fig


    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3. Preprocessing — The Unglamorous Foundation

    The most common source of bugs in machine learning is not the model. It is the preprocessing. Features arrive in incompatible scales, with missing values, in mixed types. Getting this right is unglamorous work, but it is the difference between a model that generalizes and one that memorizes artifacts of your data pipeline.

    ### 3.1 Feature Scaling

    **Why it matters.** Many algorithms are sensitive to the scale of input features. Gradient-based methods (logistic regression, SVMs, neural networks) and distance-based methods (k-NN, k-Means) are the main culprits. Consider a dataset with two features: income (range: 20,000 to 200,000) and age (range: 18 to 80). Without scaling, the income dimension dominates every distance calculation and every gradient computation. The algorithm effectively ignores age.

    **Why trees do not care.** Decision trees split on thresholds: "is income > 50,000?" A monotone transformation of income (like standardization) does not change the ordering of values, so it does not change which split is optimal. The tree finds the same splits regardless of scale. This carries over to random forests and gradient boosting, which is one reason tree ensembles are so popular for heterogeneous tabular data — you can throw raw features at them without worrying about scaling.

    sklearn provides three main scalers:

    - **StandardScaler**: subtracts the mean and divides by the standard deviation. Each feature gets mean 0 and variance 1. This is the default choice for most algorithms.
    - **MinMaxScaler**: rescales each feature to a fixed range, typically [0, 1]. Useful when you need bounded values (e.g., image pixels for neural networks).
    - **RobustScaler**: uses the median and interquartile range instead of mean and standard deviation. Robust to outliers — if your data has extreme values that would distort the mean, this is a better choice.
    """)
    return


@app.cell
def _(np, plt, StandardScaler, MinMaxScaler, RobustScaler):
    def _run():
        # Generate skewed data with outliers to show scaler differences
        rng = np.random.default_rng(42)
        n = 300
        # Feature 1: roughly normal
        x1 = rng.normal(50, 15, n)
        # Feature 2: skewed with outliers
        x2 = np.concatenate([rng.exponential(10, n - 5), rng.uniform(150, 200, 5)])
        rng.shuffle(x2)
        X_raw = np.column_stack([x1, x2])

        scalers = {
            "StandardScaler": StandardScaler(),
            "MinMaxScaler": MinMaxScaler(),
            "RobustScaler": RobustScaler(),
        }

        fig, axes = plt.subplots(1, 4, figsize=(14, 3))
        axes[0].hist(X_raw[:, 1], bins=30, color="gray", edgecolor="k", alpha=0.7)
        axes[0].set_title("Raw (Feature 2)")
        axes[0].set_xlabel("Value")

        for ax, (name, scaler) in zip(axes[1:], scalers.items()):
            X_s = scaler.fit_transform(X_raw)
            ax.hist(X_s[:, 1], bins=30, color="steelblue", edgecolor="k", alpha=0.7)
            ax.set_title(name)
            ax.set_xlabel("Scaled value")

        plt.suptitle("Effect of different scalers on skewed data with outliers", y=1.02)
        plt.tight_layout()
        fig


    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interactive: Compare Scalers

    Use the dropdown to switch between scalers and see how each one reshapes the same 2D feature distribution. Pay attention to how outliers are handled differently.
    """)
    return


@app.cell
def _(mo):
    scaler_dropdown = mo.ui.dropdown(
        options=["StandardScaler", "MinMaxScaler", "RobustScaler"],
        value="StandardScaler",
        label="Scaler"
    )
    scaler_dropdown
    return (scaler_dropdown,)


@app.cell
def _(scaler_dropdown, np, plt, StandardScaler, MinMaxScaler, RobustScaler):
    def _run():
        rng = np.random.default_rng(42)
        n = 400
        # Two features: one roughly normal, one skewed with outliers
        x1 = rng.normal(50, 15, n)
        x2 = np.concatenate([rng.exponential(10, n - 8), rng.uniform(120, 200, 8)])
        rng.shuffle(x2)
        X_raw = np.column_stack([x1, x2])

        scaler_map = {
            "StandardScaler": StandardScaler(),
            "MinMaxScaler": MinMaxScaler(),
            "RobustScaler": RobustScaler(),
        }
        scaler = scaler_map[scaler_dropdown.value]
        X_scaled = scaler.fit_transform(X_raw)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].scatter(X_raw[:, 0], X_raw[:, 1], alpha=0.4, s=15, c="gray")
        axes[0].set_title("Raw features")
        axes[0].set_xlabel("Feature 1 (normal)")
        axes[0].set_ylabel("Feature 2 (skewed + outliers)")

        axes[1].scatter(X_scaled[:, 0], X_scaled[:, 1], alpha=0.4, s=15, c="steelblue")
        axes[1].set_title(f"After {scaler_dropdown.value}")
        axes[1].set_xlabel("Scaled Feature 1")
        axes[1].set_ylabel("Scaled Feature 2")

        # Show statistics
        if scaler_dropdown.value == "StandardScaler":
            axes[1].axhline(0, color="k", ls="--", lw=0.5)
            axes[1].axvline(0, color="k", ls="--", lw=0.5)
        elif scaler_dropdown.value == "MinMaxScaler":
            axes[1].set_xlim(-0.1, 1.1)
            axes[1].set_ylim(-0.1, 1.1)

        plt.tight_layout()
        fig


    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice how StandardScaler centers the data but the outliers in Feature 2 stretch the axis, compressing the bulk of the points. MinMaxScaler maps everything to [0, 1], but the outliers claim most of the range, squashing the non-outlier points into a narrow band. RobustScaler handles this best — it uses the median and IQR, so the outliers sit outside the main cloud without distorting the scaling of the majority.

    ### The Dramatic Effect of Scaling on SVMs

    Here is a concrete demonstration that scaling is not optional for certain models. We fit an SVM with an RBF kernel on the same 2D data, once with raw features and once with standardized features. The difference in decision boundaries is striking.
    """)
    return


@app.cell
def _(np, plt, make_classification, SVC, StandardScaler):
    def _run():
        # Create data where features have very different scales
        rng = np.random.default_rng(42)
        X_svm, y_svm = make_classification(
            n_samples=200, n_features=2, n_informative=2,
            n_redundant=0, n_clusters_per_class=1, random_state=42
        )
        # Stretch feature 0 to a much larger scale
        X_svm[:, 0] = X_svm[:, 0] * 100
        X_svm[:, 1] = X_svm[:, 1] * 1

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, do_scale, title in zip(
            axes, [False, True], ["WITHOUT scaling", "WITH scaling"]
        ):
            Xp = StandardScaler().fit_transform(X_svm) if do_scale else X_svm.copy()
            svm = SVC(kernel="rbf", gamma="scale", C=1.0)
            svm.fit(Xp, y_svm)
            acc = svm.score(Xp, y_svm)

            x_min, x_max = Xp[:, 0].min() - 0.5, Xp[:, 0].max() + 0.5
            y_min, y_max = Xp[:, 1].min() - 0.5, Xp[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                                  np.linspace(y_min, y_max, 200))
            Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

            ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
            ax.scatter(Xp[y_svm == 0, 0], Xp[y_svm == 0, 1], c="steelblue",
                       edgecolors="k", s=20)
            ax.scatter(Xp[y_svm == 1, 0], Xp[y_svm == 1, 1], c="tomato",
                       edgecolors="k", s=20)
            ax.set_title(f"{title}  (accuracy: {acc:.3f})")
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")

        plt.tight_layout()
        fig


    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The unscaled SVM is essentially fitting a boundary in a space where one axis spans hundreds of units and the other spans single digits. The RBF kernel's notion of "distance" is dominated by the large-scale feature. After standardization, both features contribute equally, and the SVM finds a much better boundary.

    ### 3.2 Encoding Categorical Variables

    Machine learning algorithms operate on numbers. If your data has a "color" column with values like "red", "green", "blue", you need to convert these to a numerical representation. The choice of encoding matters more than people realize.

    **OrdinalEncoder** assigns an integer to each category: red=0, green=1, blue=2. This is appropriate when the categories have a natural ordering — education level (high school < bachelor's < master's < PhD), satisfaction rating (low < medium < high). The integers preserve the ordinal relationship.

    But for nominal categories with no natural ordering — colors, country names, blood types — ordinal encoding introduces a false structure. The model sees that green (1) is "between" red (0) and blue (2), and "closer" to red than blue. This is meaningless and can hurt performance.

    **OneHotEncoder** creates a binary column for each category. "red" becomes [1, 0, 0], "green" becomes [0, 1, 0], "blue" becomes [0, 0, 1]. No false ordering is imposed. This is the default choice for nominal categories.

    **The dummy variable trap.** If you have K categories, you only need K-1 binary columns. The K-th is redundant because it is perfectly determined by the others (if it is not red and not green, it must be blue). Including all K columns creates perfect multicollinearity, which causes problems for linear models. Use `drop='first'` in OneHotEncoder to produce K-1 columns.
    """)
    return


@app.cell
def _(np, OrdinalEncoder, OneHotEncoder):
    def _run():
        # Ordinal vs OneHot encoding
        categories = np.array([["red"], ["green"], ["blue"], ["red"], ["blue"]])

        # Ordinal: assigns integers
        oe = OrdinalEncoder()
        print("OrdinalEncoder:")
        print(f"  Input:  {categories.ravel()}")
        print(f"  Output: {oe.fit_transform(categories).ravel()}")
        print(f"  Problem: implies green is 'between' red and blue\n")

        # OneHot: binary columns
        ohe = OneHotEncoder(sparse_output=False)
        encoded = ohe.fit_transform(categories)
        print("OneHotEncoder:")
        print(f"  Categories: {ohe.categories_[0]}")
        print(f"  Encoded:\n{encoded}")

        # With drop='first': K-1 columns
        ohe_drop = OneHotEncoder(sparse_output=False, drop="first")
        encoded_drop = ohe_drop.fit_transform(categories)
        print(f"\n  With drop='first' (K-1 columns):\n{encoded_drop}")
        print(f"  Dropped category: {ohe_drop.drop_idx_}")


    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.3 Handling Missing Values

    Real-world data has missing values. sklearn's `SimpleImputer` fills them in using one of four strategies:

    - **mean**: replace missing values with the feature mean. Simple, reasonable for roughly symmetric distributions.
    - **median**: replace with the feature median. More robust to outliers and skewed distributions.
    - **most_frequent**: replace with the mode. The default choice for categorical features.
    - **constant**: replace with a fixed value you specify. Useful when missingness is informative (you might impute with -999 and let the model learn that this value is special).

    When is imputation acceptable? When the data is **missing at random** — the probability of a value being missing does not depend on the missing value itself. If the missingness is systematic (e.g., high-income individuals refuse to report income), imputation can mask a real pattern and lead to biased models. In such cases, you should consider creating a binary "is_missing" indicator feature alongside the imputation, so the model can learn to distinguish imputed values from real ones.
    """)
    return


@app.cell
def _(np, SimpleImputer):
    def _run():
        # SimpleImputer strategies
        X_miss = np.array([
            [1.0,  np.nan, 7.0],
            [2.0,  6.0,    np.nan],
            [np.nan, 3.0,  9.0],
            [4.0,  8.0,    8.0],
            [5.0,  5.0,    10.0],
        ])

        strategies = ["mean", "median", "most_frequent", "constant"]
        for strat in strategies:
            kwargs = {"fill_value": -999} if strat == "constant" else {}
            imp = SimpleImputer(strategy=strat, **kwargs)
            filled = imp.fit_transform(X_miss)
            print(f"Strategy: {strat:15s} -> col means after: "
                  f"{filled.mean(axis=0).round(2)}")
            if strat == "mean":
                print(f"  Imputed values: col0={imp.statistics_[0]:.1f}, "
                      f"col1={imp.statistics_[1]:.1f}, col2={imp.statistics_[2]:.1f}")

        print("\nOriginal (NaN positions):")
        print(X_miss)


    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4. Pipelines — The Central Abstraction

    This section contains the single most important concept in this module. If you remember nothing else, remember this: **preprocessing and modeling must be bundled together, and the bundle must respect the train/test split.**

    ### 4.1 The Data Leakage Problem

    Data leakage means using information from the test set during training. The most common form is subtle: you fit a preprocessor on the entire dataset (including the test set) before splitting. Here is a concrete example.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The WRONG Way: Scale First, Then Split
    """)
    return


@app.cell
def _(np, make_classification, train_test_split, cross_val_score, StandardScaler, SVC):
    def _run():
        # Generate a moderately hard classification problem
        X_leak, y_leak = make_classification(
            n_samples=200, n_features=20, n_informative=5,
            n_redundant=5, random_state=42
        )

        # === WRONG: fit scaler on ALL data, then cross-validate ===
        scaler_bad = StandardScaler()
        X_leaked = scaler_bad.fit_transform(X_leak)  # <-- uses ALL data
        scores_leaked = cross_val_score(SVC(), X_leaked, y_leak, cv=5,
                                         scoring="accuracy")

        # === RIGHT: use a Pipeline so scaling happens INSIDE each fold ===
        from sklearn.pipeline import make_pipeline
        pipe_clean = make_pipeline(StandardScaler(), SVC())
        scores_clean = cross_val_score(pipe_clean, X_leak, y_leak, cv=5,
                                        scoring="accuracy")

        print("=== Data Leakage Demonstration ===\n")
        print(f"WRONG (scale all, then CV):  {scores_leaked.mean():.4f} "
              f"+/- {scores_leaked.std():.4f}")
        print(f"RIGHT (Pipeline inside CV):  {scores_clean.mean():.4f} "
              f"+/- {scores_clean.std():.4f}")
        print(f"\nDifference: {(scores_leaked.mean() - scores_clean.mean()):.4f}")
        print("\nThe leaked scores are optimistically biased. The scaler saw")
        print("the test fold's statistics, giving the model information it")
        print("should not have. On this dataset the difference is small,")
        print("but on high-dimensional or small datasets it can be enormous.")


    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Why does this happen? When you call `StandardScaler().fit_transform(X)` on the entire dataset, the scaler computes the mean and standard deviation using *all* samples — including the ones that will later end up in the test fold. When cross-validation then evaluates on the test fold, those test samples have already been centered and scaled using statistics that were partially computed from themselves. The test set is no longer truly unseen.

    The fix is a Pipeline. Inside cross-validation, sklearn calls `pipe.fit(X_train_fold, y_train_fold)`, which internally calls `scaler.fit_transform(X_train_fold)` — using only the training fold. Then `pipe.predict(X_test_fold)` calls `scaler.transform(X_test_fold)` — applying the training fold's statistics to the test fold, without re-fitting. The test fold remains untouched during fitting.

    ### 4.2 Building a Pipeline

    A Pipeline is a list of (name, estimator) tuples. All but the last must be Transformers (they need `.transform()`). The last can be any Estimator (Transformer or Predictor). When you call `.fit()` on the Pipeline, it calls `.fit_transform()` on each step sequentially, feeding the transformed output to the next step. The last step gets `.fit()` only.
    """)
    return


@app.cell
def _(Pipeline, StandardScaler, SVC, make_moons, train_test_split, np):
    def _run():
        X_pipe, y_pipe = make_moons(n_samples=300, noise=0.2, random_state=42)
        X_tr, X_te, y_tr, y_te = train_test_split(X_pipe, y_pipe, random_state=42)

        # Explicit Pipeline construction
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0))
        ])

        # What happens internally:
        # pipe.fit(X_tr, y_tr):
        #   1. scaler.fit_transform(X_tr) -> X_tr_scaled
        #   2. clf.fit(X_tr_scaled, y_tr)
        pipe.fit(X_tr, y_tr)

        # pipe.predict(X_te):
        #   1. scaler.transform(X_te) -> X_te_scaled  (NOT fit_transform!)
        #   2. clf.predict(X_te_scaled)
        test_acc = pipe.score(X_te, y_te)

        print(f"Pipeline test accuracy: {test_acc:.3f}")
        print(f"\nPipeline steps: {[name for name, _ in pipe.steps]}")
        print(f"Scaler mean (learned from training set): {pipe['scaler'].mean_.round(3)}")
        print(f"SVM support vectors: {pipe['clf'].n_support_}")


    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.3 The make_pipeline Shortcut and the __ Notation

    `make_pipeline` constructs a Pipeline and auto-generates names from the class names (lowercased). This is convenient for quick experiments:

    ```python
    # These are equivalent:
    Pipeline([("standardscaler", StandardScaler()), ("svc", SVC())])
    make_pipeline(StandardScaler(), SVC())
    ```

    The **double underscore** (dunder) notation lets you access parameters of nested estimators inside a Pipeline. This is essential for hyperparameter tuning (covered in Module 5B). The pattern is `<step_name>__<parameter>`:

    ```python
    pipe.set_params(svc__C=10.0, svc__kernel="linear")
    ```

    This sets the `C` and `kernel` parameters of the step named `svc` inside the Pipeline. Without this notation, there would be no way to tune internal parameters from the outside — and since GridSearchCV operates on the outermost estimator, this is how you tell it which knobs to turn.
    """)
    return


@app.cell
def _(make_pipeline, StandardScaler, SVC, PCA, np, make_moons, train_test_split):
    def _run():
        X_mp, y_mp = make_moons(n_samples=300, noise=0.2, random_state=42)
        X_tr, X_te, y_tr, y_te = train_test_split(X_mp, y_mp, random_state=42)

        # make_pipeline auto-names steps
        pipe = make_pipeline(StandardScaler(), PCA(n_components=2), SVC())
        pipe.fit(X_tr, y_tr)
        print(f"Auto-generated step names: {[name for name, _ in pipe.steps]}")
        print(f"Test accuracy: {pipe.score(X_te, y_te):.3f}")

        # Access/modify nested params with __ notation
        print(f"\nCurrent SVC C: {pipe.get_params()['svc__C']}")
        pipe.set_params(svc__C=10.0)
        pipe.fit(X_tr, y_tr)
        print(f"After set_params(svc__C=10): {pipe.score(X_te, y_te):.3f}")

        # List all parameters (useful for hyperparameter tuning)
        all_params = pipe.get_params()
        print(f"\nAll tunable parameters:")
        for k, v in sorted(all_params.items()):
            if "__" in k and not k.endswith("_"):
                print(f"  {k}: {v}")


    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 5. ColumnTransformer — Real-World Data

    So far, every example has used clean, purely numerical data. Real datasets are not like that. A typical table might have:

    - Numerical features: age, income, account_balance (need scaling, possibly imputation)
    - Categorical features: occupation, state, account_type (need encoding, possibly imputation)
    - Features that should be left alone: binary indicators that are already 0/1

    You cannot apply a single StandardScaler to the entire table — it would choke on string columns. And you cannot apply a OneHotEncoder to numerical columns. You need **different transformations for different subsets of columns.** That is what ColumnTransformer does.

    A ColumnTransformer takes a list of (name, transformer, columns) tuples. Each transformer is applied only to its specified columns. The results are concatenated horizontally into a single output matrix.

    Let us build a realistic example: a preprocessing pipeline for a synthetic loan-approval dataset with mixed types.
    """)
    return


@app.cell
def _(np, pd):
    # Create a synthetic mixed-type dataset (loan approval)
    rng_loan = np.random.default_rng(42)
    n_loan = 500

    loan_data = pd.DataFrame({
        "age": rng_loan.integers(18, 70, n_loan).astype(float),
        "income": rng_loan.lognormal(10.5, 0.8, n_loan).round(0),
        "debt_ratio": rng_loan.uniform(0, 1, n_loan).round(3),
        "employment": rng_loan.choice(
            ["employed", "self-employed", "unemployed", "retired"], n_loan
        ),
        "education": rng_loan.choice(
            ["high_school", "bachelors", "masters", "phd"], n_loan
        ),
        "home_ownership": rng_loan.choice(
            ["rent", "own", "mortgage"], n_loan
        ),
    })

    # Inject some missing values
    for col in ["age", "income", "employment"]:
        mask = rng_loan.random(n_loan) < 0.05
        loan_data.loc[mask, col] = np.nan

    # Create target: higher income + employed + higher education -> approved
    score = (
        (loan_data["income"].fillna(30000) > 40000).astype(float) * 0.3
        + (loan_data["employment"].fillna("unemployed") == "employed").astype(float) * 0.3
        + (loan_data["education"].isin(["masters", "phd"])).astype(float) * 0.2
        + (loan_data["debt_ratio"] < 0.5).astype(float) * 0.2
        + rng_loan.normal(0, 0.15, n_loan)
    )
    loan_data["approved"] = (score > 0.5).astype(int)

    print(f"Loan dataset: {loan_data.shape[0]} samples, {loan_data.shape[1]} columns")
    print(f"Approval rate: {loan_data['approved'].mean():.1%}")
    print(f"\nMissing values:\n{loan_data.isnull().sum()}")
    print(f"\nFirst 5 rows:")
    print(loan_data.head())
    return (loan_data, rng_loan, n_loan)


@app.cell
def _(loan_data, Pipeline, ColumnTransformer, StandardScaler, SimpleImputer,
      OneHotEncoder, LogisticRegression, train_test_split, cross_val_score, np):
    def _run():
        X_loan = loan_data.drop("approved", axis=1)
        y_loan = loan_data["approved"]

        # Define column groups
        num_cols = ["age", "income", "debt_ratio"]
        cat_cols = ["employment", "education", "home_ownership"]

        # Build sub-pipelines for each column type
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(drop="first", sparse_output=False,
                                       handle_unknown="ignore")),
        ])

        # ColumnTransformer: apply different transforms to different columns
        preprocessor = ColumnTransformer([
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ])

        # Full pipeline: preprocessor + classifier
        full_pipe = Pipeline([
            ("preprocess", preprocessor),
            ("clf", LogisticRegression(max_iter=1000)),
        ])

        # Cross-validate the entire pipeline — no leakage possible
        scores = cross_val_score(full_pipe, X_loan, y_loan, cv=5,
                                  scoring="accuracy")
        print(f"5-fold CV accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")

        # Fit on all data to inspect the preprocessor
        full_pipe.fit(X_loan, y_loan)

        # Get feature names from the fitted pipeline
        feature_names = full_pipe["preprocess"].get_feature_names_out()
        print(f"\nFeature names after preprocessing ({len(feature_names)} total):")
        for name in feature_names:
            print(f"  {name}")

        # Access nested parameters
        print(f"\nNested param access:")
        print(f"  Numerical scaler mean: "
              f"{full_pipe['preprocess'].named_transformers_['num']['scaler'].mean_.round(1)}")
        print(f"  Classifier coefs shape: "
              f"{full_pipe['clf'].coef_.shape}")


    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is the pattern you will use in nearly every real project. Let us trace through what happens when you call `full_pipe.fit(X_train, y_train)`:

    1. The ColumnTransformer selects `["age", "income", "debt_ratio"]` and sends them to the numerical pipeline:
       - SimpleImputer fits on these columns (learns median of each), then transforms (fills NaN with median)
       - StandardScaler fits on the imputed values (learns mean and std), then transforms (standardizes)
    2. The ColumnTransformer selects `["employment", "education", "home_ownership"]` and sends them to the categorical pipeline:
       - SimpleImputer fits (learns mode of each), then transforms (fills missing)
       - OneHotEncoder fits (learns categories), then transforms (creates binary columns)
    3. The ColumnTransformer concatenates the numerical and categorical outputs horizontally
    4. The LogisticRegression fits on the fully preprocessed feature matrix

    When you call `full_pipe.predict(X_test)`, the same sequence happens — but every transformer calls `.transform()` instead of `.fit_transform()`. The test data is imputed, scaled, and encoded using statistics learned from the training data only. This is data-leakage-free by construction.

    The `get_feature_names_out()` method on the ColumnTransformer returns human-readable names for every column in the output. This is critical for interpretability — without it, you are staring at an anonymous matrix of floats, unable to tell which coefficient corresponds to which feature.

    ### Swapping Models Is Trivial

    Because the Pipeline separates preprocessing from modeling, you can swap classifiers without touching the preprocessing code:
    """)
    return


@app.cell
def _(loan_data, Pipeline, ColumnTransformer, StandardScaler, SimpleImputer,
      OneHotEncoder, LogisticRegression, RandomForestClassifier,
      GradientBoostingClassifier, SVC, cross_val_score):
    def _run():
        X_loan = loan_data.drop("approved", axis=1)
        y_loan = loan_data["approved"]

        num_cols = ["age", "income", "debt_ratio"]
        cat_cols = ["employment", "education", "home_ownership"]

        preprocessor = ColumnTransformer([
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(drop="first", sparse_output=False,
                                           handle_unknown="ignore")),
            ]), cat_cols),
        ])

        # Try multiple classifiers — same preprocessing, only the last step changes
        classifiers = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "SVM (RBF)": SVC(kernel="rbf"),
        }

        print("Same preprocessing, different classifiers:\n")
        for name, clf in classifiers.items():
            pipe = Pipeline([
                ("preprocess", preprocessor),
                ("clf", clf),
            ])
            scores = cross_val_score(pipe, X_loan, y_loan, cv=5, scoring="accuracy")
            print(f"  {name:25s}  {scores.mean():.3f} +/- {scores.std():.3f}")

        print("\nThe preprocessing pipeline is written once and reused everywhere.")


    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.1 The remainder Parameter and passthrough

    By default, ColumnTransformer drops any column not explicitly listed in one of the transformer tuples. This is a deliberate safety choice — it prevents you from accidentally passing raw, unprocessed columns to a model. But sometimes you have columns that need no transformation at all: binary indicators that are already 0/1, or features that have been pre-processed upstream.

    The `remainder` parameter controls what happens to unlisted columns:

    - `remainder="drop"` (default): unlisted columns are silently dropped
    - `remainder="passthrough"`: unlisted columns are included unchanged in the output
    - `remainder=SomeTransformer()`: unlisted columns are sent through the specified transformer

    You can also use the string `"passthrough"` in place of a transformer in the transformer list to pass specific columns through unchanged. This is useful for binary features that should not be scaled or encoded.

    ### 5.2 Pipeline Serialization and Reproducibility

    A fitted Pipeline captures everything needed to reproduce a prediction: the learned scaler statistics, the encoder categories, the model weights. You can serialize the entire pipeline to disk with `joblib` and load it later — on a different machine, months after training:

    ```python
    import joblib
    joblib.dump(full_pipe, "model_v1.pkl")

    # Later, on a different machine:
    loaded_pipe = joblib.load("model_v1.pkl")
    loaded_pipe.predict(new_data)  # uses the exact same preprocessing
    ```

    This is the practical payoff of bundling preprocessing with modeling. Without a Pipeline, you would need to separately serialize the scaler, the encoder, and the model, then remember to apply them in the right order. With a Pipeline, one file contains the entire inference pathway.

    ### 5.3 When Pipelines Are Not Enough

    Pipelines handle the most common workflow: sequential transformations followed by a model. But some workflows do not fit this pattern:

    - **Feature engineering that depends on the target** (e.g., target encoding) requires careful wrapping to avoid leakage
    - **Multi-output models** where different outputs need different preprocessing
    - **Custom transformers** that need to access multiple columns simultaneously

    For the first case, sklearn provides `TargetEncoder` (since 1.3). For the second, you may need a `MultiOutputClassifier` wrapper. For the third, you can write a custom transformer by subclassing `BaseEstimator` and `TransformerMixin` and implementing `fit` and `transform`. The sklearn API is deliberately extensible — as long as your custom class follows the Estimator contract, it snaps into Pipelines and cross-validation like any built-in class.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Summary

    | Concept | What It Does | Why It Matters |
    |---|---|---|
    | Estimator API | `.fit()` / `.transform()` / `.predict()` | Uniform interface across all algorithms |
    | StandardScaler | Z-score normalization | Required for gradient/distance-based models |
    | OneHotEncoder | K categories to K-1 binary columns | Correct representation of nominal categories |
    | SimpleImputer | Fill missing values | Handles real-world data without ad-hoc code |
    | Pipeline | Chains transformers + estimator | Prevents data leakage, ensures reproducibility |
    | ColumnTransformer | Different transforms per column type | Handles mixed-type datasets cleanly |

    The key insight of this module is that **the Pipeline is not just a convenience — it is a correctness tool.** Without it, you will eventually introduce data leakage. The Pipeline makes the right thing easy and the wrong thing impossible. Every model you build from this point forward should be a Pipeline.

    **Key references:**
    - [Geron Ch 2-3](file:///C:/Users/landa/ml-course/textbooks/Geron.pdf) — End-to-end sklearn project walkthrough
    - [ISLR Ch 5](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf) — Cross-validation theory (motivates why Pipelines matter)
    - Buitinck et al. (2013), "API design for machine learning software" — The design philosophy paper
    - [ESL Ch 7.10](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf) — Data leakage and the wrong/right way to cross-validate

    ---

    ## Practice Exercises

    **Pencil and Paper:**

    1. What does `fit_transform(X)` do that `fit(X)` followed by `transform(X)` does not? (Answer: nothing — it is a convenience method. For some transformers like PCA there is a computational shortcut, but the output is identical.)

    2. When would you NOT want to scale features? Give two specific scenarios and explain why scaling would be unnecessary or harmful.

    3. Suppose you have a dataset with 1000 samples, 50 features, and you use 5-fold cross-validation. You fit a StandardScaler on all 1000 samples before splitting into folds. How many test-fold samples have their own statistics mixed into the scaler's mean? What fraction of the scaler's training data is "contaminated" by the test fold?

    4. A colleague encodes a "color" feature with 5 categories using `OneHotEncoder(drop=None)`, creating 5 binary columns. They then fit a `LinearRegression`. Explain the mathematical problem this causes and how `drop='first'` fixes it.

    **Coding:**

    5. Load the wine dataset (`sklearn.datasets.load_wine`). Build a Pipeline that standardizes features and classifies with KNN. Compare 5-fold CV accuracy with k=1, 5, 10, 20.

    6. Using the Iris dataset, demonstrate data leakage by comparing two approaches: (a) fit StandardScaler on all data, then cross-validate an SVM, vs. (b) cross-validate a Pipeline containing StandardScaler + SVM. Report the score difference.

    7. Create a ColumnTransformer for a DataFrame with columns `["height", "weight", "city", "has_pet"]` where `height` and `weight` are numerical, `city` is categorical, and `has_pet` is already binary. Apply appropriate transformations to each group and leave `has_pet` untouched (hint: use `"passthrough"`).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Code It: Implementation Exercises

    Now build it yourself. Each exercise gives you starter code with TODOs to fill in.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 1: Build a Pipeline for Iris Classification

    Build a Pipeline that standardizes features and then classifies with an SVM. Evaluate with 5-fold cross-validation. Then swap the SVM for a RandomForest and compare — the preprocessing code should not change.
    """)
    return


@app.cell
def _(load_iris, Pipeline, StandardScaler, SVC, RandomForestClassifier, cross_val_score):
    def _run():
        iris = load_iris()
        X_iris, y_iris = iris.data, iris.target

        # TODO: build a Pipeline with StandardScaler and SVC
        pipe_svm = None  # replace this

        # TODO: evaluate with 5-fold cross-validation
        # scores_svm = cross_val_score(pipe_svm, X_iris, y_iris, cv=5)
        # print(f"SVM Pipeline:  {scores_svm.mean():.3f} +/- {scores_svm.std():.3f}")

        # TODO: build a second Pipeline swapping SVC for RandomForestClassifier
        pipe_rf = None  # replace this

        # TODO: evaluate with 5-fold cross-validation
        # scores_rf = cross_val_score(pipe_rf, X_iris, y_iris, cv=5)
        # print(f"RF Pipeline:   {scores_rf.mean():.3f} +/- {scores_rf.std():.3f}")

        print("Exercise 1 skeleton ready — fill in the TODOs")


    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 2: Create a ColumnTransformer for Mixed-Type Data

    Build a full preprocessing + classification pipeline for a mixed-type dataset. Use different transformers for numerical and categorical columns.
    """)
    return


@app.cell
def _(np, pd, Pipeline, ColumnTransformer, StandardScaler, SimpleImputer,
      OneHotEncoder, LogisticRegression, cross_val_score):
    def _run():
        # Synthetic mixed-type data
        rng_ex = np.random.default_rng(123)
        n_ex = 300
        df_ex = pd.DataFrame({
            "feature_1": rng_ex.normal(50, 10, n_ex),
            "feature_2": rng_ex.uniform(0, 100, n_ex),
            "category_a": rng_ex.choice(["X", "Y", "Z"], n_ex),
            "category_b": rng_ex.choice(["low", "medium", "high"], n_ex),
        })
        y_ex = (df_ex["feature_1"] > 50).astype(int) ^ (df_ex["category_a"] == "X").astype(int)

        # Inject missing values
        df_ex.loc[rng_ex.choice(n_ex, 15, replace=False), "feature_1"] = np.nan
        df_ex.loc[rng_ex.choice(n_ex, 10, replace=False), "category_a"] = np.nan

        num_cols_ex = ["feature_1", "feature_2"]
        cat_cols_ex = ["category_a", "category_b"]

        # TODO: build a Pipeline for numerical columns (impute + scale)
        num_pipe = None  # replace this

        # TODO: build a Pipeline for categorical columns (impute + one-hot encode)
        cat_pipe = None  # replace this

        # TODO: build a ColumnTransformer combining both
        preprocessor_ex = None  # replace this

        # TODO: build a full Pipeline with preprocessor + LogisticRegression
        full_pipe_ex = None  # replace this

        # TODO: evaluate with cross_val_score
        # scores_ex = cross_val_score(full_pipe_ex, df_ex, y_ex, cv=5)
        # print(f"CV accuracy: {scores_ex.mean():.3f} +/- {scores_ex.std():.3f}")

        print("Exercise 2 skeleton ready — fill in the TODOs")


    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 3: Prove Data Leakage Exists

    Demonstrate that fitting a scaler on all data before cross-validation gives different (optimistically biased) scores compared to putting the scaler inside a Pipeline. Use a small, high-dimensional dataset where the effect is more pronounced.
    """)
    return


@app.cell
def _(np, make_classification, StandardScaler, SVC, cross_val_score, Pipeline):
    def _run():
        # Small, high-dimensional data where leakage is most visible
        X_leak_ex, y_leak_ex = make_classification(
            n_samples=100, n_features=50, n_informative=5,
            n_redundant=10, random_state=42
        )

        # TODO (WRONG way): fit StandardScaler on ALL data, then cross-validate SVC
        # scaler_all = StandardScaler()
        # X_leaked = scaler_all.fit_transform(X_leak_ex)
        # scores_wrong = cross_val_score(SVC(), X_leaked, y_leak_ex, cv=5)
        # print(f"WRONG (leaked):  {scores_wrong.mean():.4f} +/- {scores_wrong.std():.4f}")

        # TODO (RIGHT way): Pipeline(StandardScaler, SVC), then cross-validate
        # pipe_right = Pipeline([("scaler", StandardScaler()), ("clf", SVC())])
        # scores_right = cross_val_score(pipe_right, X_leak_ex, y_leak_ex, cv=5)
        # print(f"RIGHT (clean):   {scores_right.mean():.4f} +/- {scores_right.std():.4f}")

        # TODO: print the difference
        # print(f"Difference:      {(scores_wrong.mean() - scores_right.mean()):.4f}")

        print("Exercise 3 skeleton ready — fill in the TODOs")


    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    > **Next**: [5B — Model Selection in Practice](5b_model_selection.html)
    """)
    return


if __name__ == "__main__":
    app.run()
