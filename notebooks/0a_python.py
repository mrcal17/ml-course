import marimo

app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Module 0A: Python & Data Science Stack

    You already know how to program. You've written Python before -- maybe even seriously -- but it's been a while, and the details have gone fuzzy. That's fine. This module is not "Introduction to Programming." It's a targeted refresher on the parts of Python and its ecosystem that you will use constantly in machine learning work. I'll move quickly through things you'll recognize and slow down where the ML-specific patterns diverge from general-purpose programming.

    By the end of this lecture, you should be comfortable reading and writing the kind of Python that appears in ML codebases, research papers' companion code, and your own experiments.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 1. Why Python for ML

    I'll keep this short because the answer is pragmatic, not philosophical. Python won the ML ecosystem war. The reasons:

    - **NumPy, SciPy, and the numeric stack** give you fast array operations backed by C/Fortran without writing C/Fortran.
    - **PyTorch and TensorFlow** are Python-first. Their APIs are designed around Python idioms.
    - **scikit-learn** remains the gold standard for classical ML, and it's Python.
    - **Community mass**: when a new paper drops on arXiv, the reference implementation is almost always Python.

    Python itself is slow. That's a real concern, and we'll address it (vectorization is the answer for 90% of cases). But the ecosystem leverage is overwhelming. You'd need a very good reason to choose anything else for ML research or applied work today.

    For a broader perspective on the mathematical tools we'll be building toward, see [MML Chapter 2 -- Linear Algebra](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) -- much of what makes Python powerful for ML is how directly it maps to the linear algebra and probability operations that underpin everything.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 2. Python Refresher: The Parts That Matter

    ### Data Types and Containers

    You remember lists, dicts, tuples, sets. Here's what matters for ML specifically:
    """)
    return


@app.cell
def _():
    # Lists -- you'll use these for collecting results, hyperparameters, etc.
    accuracies = [0.82, 0.85, 0.87, 0.91]

    # Dicts -- configuration is everywhere in ML
    config = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "hidden_dims": [128, 64, 32],
    }

    # Tuples -- immutable, used for shapes constantly
    batch_size = 32
    channels, height, width = 3, 224, 224
    shape = (batch_size, channels, height, width)  # you'll see this a lot

    # Sets -- useful for vocabulary, unique labels
    classes = {"cat", "dog", "bird"}

    print(f"accuracies: {accuracies}")
    print(f"config: {config}")
    print(f"shape: {shape}")
    print(f"classes: {classes}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### List Comprehensions and Generators

    List comprehensions are dense but readable once you're used to them. In ML code, you see them constantly:
    """)
    return


@app.cell
def _():
    # Filter and transform in one shot
    all_losses = [0.5, 1.2, 0.3, 0.8, 1.5, 0.2]
    threshold = 1.0
    valid_losses = [loss for loss in all_losses if loss < threshold]
    print(f"valid_losses: {valid_losses}")

    # Nested comprehension -- flattening batches
    batches = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    all_predictions = [pred for batch in batches for pred in batch]
    print(f"all_predictions: {all_predictions}")

    # Dict comprehension -- renaming model layers
    state_dict = {"module.layer1": "w1", "module.layer2": "w2"}
    renamed = {k.replace("module.", ""): v for k, v in state_dict.items()}
    print(f"renamed: {renamed}")
    return


@app.cell
def _():
    # Generators matter when your dataset doesn't fit in memory (which happens often)
    def data_stream(items):
        for item in items:
            yield item * 2  # yields one item at a time, no full list in memory

    # Generator expression -- lazy evaluation
    squares = (x**2 for x in range(10_000_000))  # uses almost no memory
    print(f"First 5 squares: {[next(squares) for _ in range(5)]}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Functions, Lambda, *args/**kwargs
    """)
    return


@app.cell
def _():
    # Default arguments -- common for configurable ML functions
    def train(model, data, lr=0.001, epochs=10, verbose=True):
        if verbose:
            print(f"Training with lr={lr}, epochs={epochs}")

    # Lambda -- short throwaway functions, often for sorting or key functions
    model_layers = [{"name": "fc1", "params": 1000}, {"name": "fc2", "params": 500}]
    layers = sorted(model_layers, key=lambda x: x["params"])
    print(f"Sorted layers: {layers}")

    # *args and **kwargs -- you'll see these in wrapper functions and decorators
    def log_experiment(name, **hyperparams):
        """Log any set of hyperparameters without defining them all upfront."""
        print(f"Experiment: {name}")
        for k, v in hyperparams.items():
            print(f"  {k}: {v}")

    log_experiment("run_42", lr=0.001, batch_size=64, dropout=0.3)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The `**kwargs` pattern is especially important because ML frameworks use it heavily. PyTorch's `nn.Module.__init__` passes keyword arguments up the inheritance chain this way.

    ### Classes -- Just Enough for PyTorch

    You don't need to be an OOP expert, but you need to understand class basics because every PyTorch model is a class:
    """)
    return


@app.cell
def _():
    import numpy as np

    class LinearRegressor:
        def __init__(self, n_features):
            self.weights = np.zeros(n_features)
            self.bias = 0.0

        def predict(self, X):
            return X @ self.weights + self.bias

        def __repr__(self):
            return f"LinearRegressor(n_features={len(self.weights)})"

    model = LinearRegressor(5)
    print(model)
    X_test = np.random.randn(3, 5)
    print(f"Predictions: {model.predict(X_test)}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    The pattern you'll use constantly in PyTorch:

    ```python
    import torch.nn as nn

    class MyModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()  # always call this
            self.layer1 = nn.Linear(input_dim, hidden_dim)
            self.layer2 = nn.Linear(hidden_dim, output_dim)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.layer1(x))
            return self.layer2(x)
    ```

    That's the pattern. `__init__` defines the layers, `forward` defines the computation. Inheritance from `nn.Module` gives you parameter tracking, GPU movement, saving/loading -- all for free.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### F-strings, Context Managers, Decorators

    **F-strings** -- use them everywhere for readable logging:
    """)
    return


@app.cell
def _():
    # F-string format specifiers worth memorizing
    epoch = 7
    loss = 0.3421
    accuracy = 0.9120
    print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Acc: {accuracy:.2%}")
    # Output: Epoch 007 | Loss: 0.3421 | Acc: 91.20%
    return


@app.cell
def _(mo):
    mo.md(r"""
    The format specifiers (`:03d`, `:.4f`, `:.2%`) are worth memorizing. You'll print training metrics thousands of times.

    **Context managers** -- the `with` statement handles resource cleanup:

    ```python
    # File I/O
    with open("results.json", "w") as f:
        json.dump(results, f)

    # PyTorch: disabling gradient computation during evaluation
    with torch.no_grad():
        predictions = model(test_data)
    ```

    **Decorators** -- you'll mostly consume them, not write them:

    ```python
    @torch.no_grad()          # same as the context manager, but for a whole function
    def evaluate(model, data):
        ...

    @staticmethod             # method that doesn't need self
    def compute_loss(pred, target):
        ...
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 3. NumPy -- The Foundation of Everything

    NumPy is the single most important library in the Python ML stack. PyTorch tensors are designed to feel like NumPy arrays. If you understand NumPy deeply, PyTorch will feel natural. If you don't, everything will feel like fighting the framework.

    For the mathematical foundation of array and matrix operations, see [MML Chapter 2 -- Linear Algebra](file:///C:/Users/landa/ml-course/textbooks/MML.pdf).

    ### Why Vectorization Matters

    This is the most important conceptual point in this entire module. Python loops are slow. NumPy operations on arrays are fast. The difference is not 2x -- it's 100x or more.
    """)
    return


@app.cell
def _():
    import numpy as np
    import time

    n = 1_000_000

    # The slow way: Python loop
    a = list(range(n))
    b = list(range(n))

    start = time.time()
    c = [a[i] + b[i] for i in range(n)]
    loop_time = time.time() - start

    # The fast way: NumPy vectorized
    a_np = np.arange(n)
    b_np = np.arange(n)

    start = time.time()
    c_np = a_np + b_np
    vec_time = time.time() - start

    print(f"Loop: {loop_time:.4f}s | Vectorized: {vec_time:.4f}s | Speedup: {loop_time/vec_time:.0f}x")
    return


@app.cell
def _(mo):
    mo.md(r"""
    The rule: **if you're writing a for-loop over array elements in ML code, you're probably doing it wrong.** Think in terms of whole-array operations. This is a mindset shift, and it takes practice, but it's essential.

    ### Array Creation, Indexing, Slicing, Reshaping
    """)
    return


@app.cell
def _():
    import numpy as np

    # Creation
    zeros = np.zeros((3, 4))           # 3x4 matrix of zeros
    ones = np.ones((2, 3))             # 2x3 matrix of ones
    identity = np.eye(4)               # 4x4 identity matrix
    range_arr = np.arange(0, 10, 0.5)  # like range() but supports floats
    linspace = np.linspace(0, 1, 100)  # 100 evenly spaced points in [0, 1]
    random = np.random.randn(3, 4)     # 3x4 matrix from standard normal

    print(f"zeros shape: {zeros.shape}")
    print(f"ones shape: {ones.shape}")
    print(f"identity:\n{identity}")
    print(f"range_arr: {range_arr[:10]}...")
    print(f"random:\n{random}")
    return


@app.cell
def _():
    import numpy as np

    # Indexing and slicing -- same as Python lists, but multi-dimensional
    A = np.arange(20).reshape(4, 5)
    print(f"A:\n{A}\n")

    print(f"A[0, :] (first row): {A[0, :]}")
    print(f"A[:, 0] (first column): {A[:, 0]}")
    print(f"A[1:3, 2:4] (submatrix):\n{A[1:3, 2:4]}\n")

    # Boolean indexing -- very common for filtering
    mask = A > 10
    print(f"A[A > 10]: {A[mask]}")

    # Fancy indexing -- selecting specific rows/columns
    print(f"A[[0, 2, 3], :] (rows 0, 2, 3):\n{A[[0, 2, 3], :]}")
    return


@app.cell
def _():
    import numpy as np

    # Reshaping
    v = np.arange(12)
    print(f"v: {v}")
    print(f"v.reshape(3, 4):\n{v.reshape(3, 4)}")
    print(f"v.reshape(3, -1) (same as 3, 4):\n{v.reshape(3, -1)}")
    print(f"v.reshape(-1, 1) (column vector):\n{v.reshape(-1, 1).T} (shown transposed)")

    # Transpose
    A = np.arange(12).reshape(3, 4)
    print(f"\nA:\n{A}")
    print(f"A.T:\n{A.T}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Critical detail**: NumPy slices are *views*, not copies. Modifying a slice modifies the original array. Use `.copy()` when you need independence.
    """)
    return


@app.cell
def _():
    import numpy as np

    A = np.arange(20).reshape(4, 5)
    print(f"Original A[0, 0]: {A[0, 0]}")

    row = A[0, :]
    row[0] = 999      # this ALSO changes A[0, 0] to 999!
    print(f"After modifying slice, A[0, 0]: {A[0, 0]}")

    A = np.arange(20).reshape(4, 5)
    row = A[0, :].copy()  # now row is independent
    row[0] = 999
    print(f"After modifying copy, A[0, 0]: {A[0, 0]}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Broadcasting Rules

    Broadcasting is how NumPy handles operations between arrays of different shapes. It is elegant when you understand it and baffling when you don't. Here are the rules:

    1. If arrays have different numbers of dimensions, pad the smaller shape with 1s **on the left**.
    2. Arrays with size 1 along a dimension act as if they had the size of the other array along that dimension (the values are repeated).
    3. If sizes disagree along a dimension and neither is 1, it's an error.
    """)
    return


@app.cell
def _():
    import numpy as np

    # Example 1: scalar + array (trivial broadcasting)
    A = np.ones((3, 4))
    result1 = A + 5  # 5 is broadcast to (3, 4)
    print(f"A + 5:\n{result1}\n")

    # Example 2: row vector + matrix
    A = np.ones((3, 4))        # shape (3, 4)
    row = np.array([1, 2, 3, 4])  # shape (4,) -> treated as (1, 4) -> broadcast to (3, 4)
    result2 = A + row  # adds [1,2,3,4] to every row
    print(f"A + row:\n{result2}\n")

    # Example 3: column vector + matrix
    col = np.array([[10], [20], [30]])  # shape (3, 1) -> broadcast to (3, 4)
    result3 = A + col  # adds 10 to row 0, 20 to row 1, 30 to row 2
    print(f"A + col:\n{result3}\n")

    # Example 4: outer product via broadcasting
    x = np.array([1, 2, 3]).reshape(-1, 1)  # shape (3, 1)
    y = np.array([4, 5, 6]).reshape(1, -1)  # shape (1, 3)
    outer = x * y  # shape (3, 3) -- outer product!
    print(f"Outer product:\n{outer}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    The outer product example is worth staring at. Broadcasting lets you avoid explicit loops for operations that would otherwise require nested iteration. In ML, you'll use this pattern for computing pairwise distances, attention scores, and many other things.

    ### Linear Algebra Operations

    This is where NumPy connects directly to the math you'll be doing. See [MML Section 2.3 -- Solving Systems of Linear Equations](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) for the underlying theory.
    """)
    return


@app.cell
def _():
    import numpy as np

    A = np.random.randn(3, 3)
    b = np.random.randn(3)

    # Matrix-vector product
    print(f"A @ b: {A @ b}")
    print(f"np.dot(A, b): {np.dot(A, b)}")

    # Matrix-matrix product
    B = np.random.randn(3, 4)
    C = A @ B                # (3,3) @ (3,4) = (3,4)
    print(f"\nA @ B shape: {C.shape}")

    # Transpose
    print(f"A.T:\n{A.T}")

    # Inverse
    A_inv = np.linalg.inv(A)
    print(f"\nA @ A_inv (should be identity):\n{np.round(A @ A_inv, 10)}")

    # Solve linear system Ax = b (better than computing inverse explicitly)
    x = np.linalg.solve(A, b)
    print(f"\nSolution x: {x}")
    print(f"A @ x (should equal b): {A @ x}")
    print(f"b: {b}")
    return


@app.cell
def _():
    import numpy as np

    A = np.random.randn(3, 3)
    b = np.random.randn(3)

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print(f"Eigenvalues: {eigenvalues}")

    # Singular Value Decomposition -- you'll use this a lot
    U, S, Vt = np.linalg.svd(A)
    print(f"Singular values: {S}")

    # Determinant, rank, norm
    print(f"\nDeterminant: {np.linalg.det(A):.4f}")
    print(f"Rank: {np.linalg.matrix_rank(A)}")
    print(f"L2 norm of b: {np.linalg.norm(b):.4f}")
    print(f"L1 norm of b: {np.linalg.norm(b, ord=1):.4f}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Practical tip**: never compute `np.linalg.inv(A) @ b` when you can use `np.linalg.solve(A, b)`. The latter is faster and more numerically stable. SVD will become very important when we cover dimensionality reduction (PCA is just truncated SVD). See [ISLR Section 6.3 -- Dimension Reduction Methods](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf) for more context.

    ### Random Number Generation
    """)
    return


@app.cell
def _():
    import numpy as np

    rng = np.random.default_rng(seed=42)  # modern API -- use this, not np.random.seed()

    print(f"Standard normal:\n{rng.standard_normal((3, 4))}\n")
    print(f"Uniform [0, 1):\n{rng.uniform(0, 1, size=(3, 4))}\n")
    print(f"Random integers: {rng.integers(0, 10, size=5)}")
    print(f"Choice without replacement: {rng.choice([1, 2, 3, 4], size=2, replace=False)}")
    print(f"Permutation: {rng.permutation(10)}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    Always seed your random number generators for reproducibility. When comparing two models, you need identical train/test splits. Unseeded randomness makes debugging nearly impossible.

    ---

    ## 4. Pandas -- Data Manipulation

    Pandas is how you'll load, clean, explore, and preprocess tabular data before feeding it to models. For statistical perspectives on data handling, see [ISLR Chapter 2 -- Statistical Learning Overview](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf).

    ### DataFrames and Series
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np

    # Creating a DataFrame
    df = pd.DataFrame({
        "feature_1": [1.2, 3.4, 5.6, 7.8],
        "feature_2": [0.1, 0.4, 0.2, 0.8],
        "label": ["cat", "dog", "cat", "bird"],
    })

    print(f"Shape: {df.shape}")
    print(f"\nDtypes:\n{df.dtypes}")
    print(f"\nDescribe:\n{df.describe()}")
    print(f"\nHead:\n{df.head()}")
    print(f"\nValue counts:\n{df['label'].value_counts()}")
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""
    A `Series` is a single column. A `DataFrame` is a collection of Series sharing an index.

    ### Indexing: loc vs iloc

    This trips people up. The rule is simple:

    - **`loc`** -- label-based indexing (uses the actual index values and column names)
    - **`iloc`** -- integer position-based indexing (uses 0-based positions, like NumPy)
    """)
    return


@app.cell
def _(df):
    # loc: by label
    print("loc[0:2, 'feature_1']:")
    print(df.loc[0:2, "feature_1"])

    print("\nloc where label == 'cat':")
    print(df.loc[df["label"] == "cat", :])

    # iloc: by position
    print("\niloc[0:2, 0] (first 2 rows, first column):")
    print(df.iloc[0:2, 0])

    print("\niloc[:, -1] (last column):")
    print(df.iloc[:, -1])

    # Boolean indexing -- arguably the most common pattern
    mask = (df["feature_1"] > 3.0) & (df["label"] != "bird")
    filtered = df[mask]
    print(f"\nFiltered (feature_1 > 3.0 and not bird):\n{filtered}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Common gotcha**: `loc[0:2]` includes row 2. `iloc[0:2]` excludes row 2. This is because `loc` uses inclusive-on-both-ends label slicing, while `iloc` follows Python's half-open convention.

    ### groupby, merge, apply
    """)
    return


@app.cell
def _(df):
    import pandas as pd

    # groupby -- split-apply-combine
    print("Mean of feature_1 per class:")
    print(df.groupby("label")["feature_1"].mean())

    # merge -- joining tables (SQL-style)
    features_df = pd.DataFrame({"id": [1, 2, 3], "x": [10, 20, 30]})
    labels_df = pd.DataFrame({"id": [1, 2, 3], "y": [0, 1, 1]})
    merged = pd.merge(features_df, labels_df, on="id", how="inner")
    print(f"\nMerged:\n{merged}")

    # Vectorized operations (prefer over .apply for speed)
    df_copy = df.copy()
    df_copy["feature_1_scaled"] = (df_copy["feature_1"] - df_copy["feature_1"].mean()) / df_copy["feature_1"].std()
    print(f"\nScaled:\n{df_copy}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Handling Missing Data

    Real datasets have missing values. Always check and handle them before modeling.
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np

    # Create sample data with missing values
    df_missing = pd.DataFrame({
        "feature_1": [1.0, np.nan, 3.0, 4.0],
        "feature_2": [0.1, 0.2, np.nan, 0.4],
        "label": ["cat", "dog", None, "bird"],
    })

    print(f"Missing values per column:\n{df_missing.isnull().sum()}\n")
    print(f"After dropna:\n{df_missing.dropna()}\n")

    df_filled = df_missing.copy()
    df_filled["feature_1"] = df_filled["feature_1"].fillna(df_filled["feature_1"].median())
    print(f"After fillna (median):\n{df_filled}\n")

    # Check for implicit missing values too
    print(f"Unique labels: {df_missing['label'].unique()}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 5. Matplotlib -- Visualization

    Visualization is how you build intuition about your data and diagnose model behavior. Don't skip this -- the ability to quickly plot things will save you hours of confusion.

    ### Basic Plots
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    # Line plot -- training curves
    epochs = range(1, 51)
    rng = np.random.default_rng(42)
    train_loss = np.exp(-np.linspace(0, 3, 50)) + 0.1 * rng.standard_normal(50) * 0.05
    val_loss = np.exp(-np.linspace(0, 2.5, 50)) + 0.15 * rng.standard_normal(50) * 0.05

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, label="Train Loss", color="blue")
    ax.plot(epochs, val_loss, label="Val Loss", color="red", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    # Scatter plot -- 2D data visualization
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 2))
    labels = (X[:, 0] + X[:, 1] > 0).astype(int)

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="bwr", alpha=0.6, edgecolors="k", linewidth=0.5)
    fig.colorbar(scatter, ax=ax, label="Class")
    ax.set_title("2D Classification Data")
    fig
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    # Histogram -- distribution inspection
    rng = np.random.default_rng(42)
    data = rng.standard_normal(10000)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(data, bins=50, density=True, alpha=0.7, color="steelblue", edgecolor="white")
    ax.set_title("Standard Normal Distribution")
    fig
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    # Heatmap -- correlation matrices, confusion matrices
    rng = np.random.default_rng(42)
    corr_matrix = np.corrcoef(rng.standard_normal((5, 100)))

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set_title("Correlation Matrix")
    fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Subplots and Figure Customization
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(42)
    epochs = range(1, 51)
    train_loss = np.exp(-np.linspace(0, 3, 50)) + 0.1 * rng.standard_normal(50) * 0.05
    data = rng.standard_normal(10000)
    X = rng.standard_normal((200, 2))
    labels = (X[:, 0] + X[:, 1] > 0).astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, train_loss)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")

    axes[1].hist(data, bins=50, color="coral")
    axes[1].set_title("Data Distribution")

    axes[2].scatter(X[:, 0], X[:, 1], c=labels, cmap="bwr", s=10)
    axes[2].set_title("Decision Boundary Data")

    plt.tight_layout()
    fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### The Object-Oriented API

    The pyplot interface (`plt.plot(...)`) is fine for quick exploration, but the object-oriented API gives you more control. The pattern above with `fig, axes = plt.subplots(...)` is the OO approach. Use it for anything beyond a single throwaway plot.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(42)
    epochs = range(1, 51)
    train_loss = np.exp(-np.linspace(0, 3, 50)) + 0.1 * rng.standard_normal(50) * 0.05
    val_loss = np.exp(-np.linspace(0, 2.5, 50)) + 0.15 * rng.standard_normal(50) * 0.05

    # OO approach -- preferred for production/reusable code
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, label="Train")
    ax.plot(epochs, val_loss, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("My Plot")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 1.5)
    fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    The key distinction: `plt.xlabel(...)` implicitly acts on "the current axes." `ax.set_xlabel(...)` explicitly targets a specific axes object. When you have multiple subplots, explicit is better than implicit.

    ---

    ## 6. Jupyter Notebooks -- The ML Workbench

    ### Why Notebooks for Exploration

    Jupyter notebooks let you run code in cells, see outputs inline (including plots), and mix code with markdown documentation. They are the standard environment for:

    - **Exploratory data analysis**: load data, make plots, compute statistics, iterate
    - **Model prototyping**: try architectures, tune hyperparameters, visualize results
    - **Communication**: share findings with collaborators who can re-run your analysis

    They are *not* good for production code, version control, or large software projects. Use `.py` files for those. But for the experimental, iterative nature of ML work, notebooks are the right tool.

    ### Getting Started

    ```bash
    pip install jupyterlab
    jupyter lab
    ```

    This opens a browser-based IDE. Create a new notebook and you get a sequence of cells.

    ### Magic Commands

    Jupyter has special "magic" commands prefixed with `%` (line magic) or `%%` (cell magic):

    ```python
    %timeit np.dot(a, b)           # time a single line (runs multiple times for accuracy)
    %%timeit                        # time an entire cell
    result = np.linalg.svd(large_matrix)

    %matplotlib inline              # show plots inline (usually set once at top of notebook)

    %who                            # list all variables in namespace
    %whos                           # list with types and sizes

    %%writefile my_module.py        # write cell contents to a file
    def helper_function():
        return 42

    %load_ext autoreload            # auto-reload imported modules when they change
    %autoreload 2                   # (very useful during development)
    ```

    The `autoreload` magic is especially valuable: you can define functions in `.py` files, import them in your notebook, edit the files, and the notebook picks up the changes without restarting the kernel.

    ### Markdown Cells

    Switch a cell to Markdown mode (press `M` in command mode or select from the dropdown) to write formatted notes, equations, and documentation alongside your code:

    ```markdown
    ## Experiment 3: Learning Rate Sweep

    Testing learning rates: $\eta \in \{0.1, 0.01, 0.001\}$

    **Hypothesis**: Lower learning rates will converge more slowly but to a better minimum.

    Results:
    | LR    | Final Loss | Epochs to Converge |
    |-------|------------|--------------------|
    | 0.1   | 0.342      | 12                 |
    | 0.01  | 0.298      | 47                 |
    | 0.001 | 0.285      | 180                |
    ```

    Jupyter renders LaTeX math, which you'll use heavily when documenting the math behind your models.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Practice Exercises

    These are designed to exercise the patterns you'll actually use in ML work.

    ### Exercise 1: NumPy Fundamentals

    Create a function that takes a matrix $X$ of shape $(n, d)$ and returns the matrix of pairwise Euclidean distances of shape $(n, n)$. Use broadcasting -- no loops allowed. Hint: $\|x_i - x_j\|^2 = \|x_i\|^2 + \|x_j\|^2 - 2 x_i \cdot x_j$.
    """)
    return


@app.cell
def _():
    import numpy as np

    def pairwise_distances(X):
        """Compute pairwise Euclidean distances without loops.
        X: shape (n, d)
        Returns: shape (n, n)
        """
        # Your code here
        pass

    # Test: should give 0 on the diagonal
    X = np.random.randn(100, 5)
    D = pairwise_distances(X)
    # assert D.shape == (100, 100)
    # assert np.allclose(np.diag(D), 0)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Exercise 2: Data Exploration with Pandas

    Load any CSV dataset (try `sklearn.datasets.load_iris()` or `pd.read_csv()` on a Kaggle dataset). Perform a complete exploration:
    1. Check shape, dtypes, and missing values
    2. Compute summary statistics per class
    3. Create a correlation matrix and visualize it as a heatmap
    4. Identify which features have the highest variance

    ### Exercise 3: Visualization

    Using the same dataset from Exercise 2:
    1. Create a 2x2 subplot grid showing: (a) histogram of each feature, (b) scatter plot of the two most correlated features, (c) box plot per class, (d) a bar chart of class frequencies
    2. Use the OO matplotlib API throughout
    3. Make sure all plots have labeled axes and titles

    ### Exercise 4: Vectorization Timing

    Implement the following three ways and time each with `%timeit`:
    - Given a matrix $X$ of shape $(n, d)$, compute the mean of each row
      1. Using a Python for-loop
      2. Using `np.mean(X, axis=1)`
      3. Using `X @ np.ones(d) / d`

    How do the speeds compare for $n=10000, d=1000$?

    ### Exercise 5: Broadcasting Workout

    Without running the code, predict the output shape of each operation. Then verify:
    """)
    return


@app.cell
def _():
    import numpy as np

    A = np.ones((3, 4, 5))
    B = np.ones((4, 1))
    C = np.ones((5,))
    D = np.ones((3, 1, 1))

    # What shape is each result?
    print(f"A + B -> {(A + B).shape}")
    print(f"A + C -> {(A + C).shape}")
    print(f"A + D -> {(A + D).shape}")
    print(f"A * B * C -> {(A * B * C).shape}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Exercise 6: End-to-End Mini Pipeline

    Write a complete mini-pipeline that:
    1. Generates synthetic 2D classification data (two clusters with Gaussian noise)
    2. Splits it into train/test (80/20) using only NumPy (no sklearn)
    3. Implements a nearest-centroid classifier (compute class means, assign test points to the nearest mean)
    4. Computes accuracy on the test set
    5. Plots the decision boundary overlaid on the data

    This exercise combines everything: NumPy array operations, broadcasting, matplotlib visualization, and the kind of structured thinking you need for ML pipelines.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Summary

    Here's what you should have internalized from this module:

    - **Vectorize everything.** Think in arrays, not loops. This is the single biggest shift in going from general-purpose programming to ML programming.
    - **NumPy is the foundation.** PyTorch tensors mirror NumPy's API. Master NumPy and you master the language of ML computation.
    - **Broadcasting is your friend** once you understand the rules. When you get shape mismatch errors (and you will), draw out the shapes and apply the broadcasting rules manually.
    - **Pandas for data wrangling**, Matplotlib for visualization, Jupyter for experimentation. These tools form the workbench on which you'll build everything else.

    In the next module, we'll start building on this foundation with the mathematical prerequisites: linear algebra, calculus, and probability -- the three pillars that everything in ML rests on.

    For a comprehensive reference on the mathematical foundations ahead, see [MML -- Mathematics for Machine Learning, full text](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) and [Bishop PRML -- Introduction](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) for the probabilistic perspective we'll be developing throughout the course.
    """)
    return


if __name__ == "__main__":
    app.run()
