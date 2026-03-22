import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
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

    For a broader perspective on the mathematical tools we'll be building toward, see [MML Chapter 2 -- Linear Algebra](../textbooks/MML.pdf) -- much of what makes Python powerful for ML is how directly it maps to the linear algebra and probability operations that underpin everything.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 2. Python Refresher: The Parts That Matter

    ### Data Types and Containers

    You remember lists, dicts, tuples, sets. But in ML code, each of these containers shows up in very specific roles, and understanding *why* each one is used where it is will help you read ML codebases much faster.

    - **Lists** are your go-to for collecting things that grow over time -- training losses per epoch, accuracy scores across folds, batches of predictions. They're ordered, mutable, and you can append to them in a training loop.

    - **Dicts** are how virtually every ML experiment is configured. When you see a config dict with keys like `"learning_rate"`, `"batch_size"`, `"epochs"` -- that's the standard pattern. Libraries like Weights & Biases and MLflow expect hyperparameters as dicts. PyTorch model state dicts (the saved weights of a model) are literally Python dicts mapping layer names to tensors.

    - **Tuples** represent shapes, and shapes are everywhere. When you see `(32, 3, 224, 224)`, that's a batch of 32 images, each with 3 color channels, at 224x224 resolution. Tuples are immutable for a reason here -- a tensor's shape is a fixed structural property, not something you should be casually mutating.

    - **Sets** are perfect for vocabulary management in NLP (unique tokens), tracking unique class labels, or computing intersections between prediction sets. If you need to check "is this word in my vocabulary?", a set gives you O(1) lookup vs O(n) for a list.
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
    Notice how the tuple unpacking `channels, height, width = 3, 224, 224` works -- you'll see this constantly when extracting dimensions from tensor shapes. In PyTorch, you'll write things like `batch_size, seq_len, hidden_dim = x.shape` to destructure a tensor's shape into named variables. It makes the code self-documenting: instead of `x.shape[2]`, you write `hidden_dim`.

    Also notice that the config dict has a nested list inside it (`"hidden_dims": [128, 64, 32]`). ML configs are often deeply nested -- dicts of dicts, dicts containing lists of dicts. Libraries like Hydra and OmegaConf exist specifically to manage this complexity.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### List Comprehensions and Generators

    List comprehensions are dense but readable once you're used to them. In ML code, you see them constantly for data transformations, filtering, and reshaping. They're not just syntactic sugar -- they're often significantly faster than equivalent for-loops because Python optimizes them internally.

    Pay attention to the dict comprehension example below. Renaming keys in a state dict is something you will literally do when loading pretrained model weights -- different frameworks or training setups often save weights with different naming conventions, and you need to fix the keys before you can load them.
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
def _(mo):
    mo.md(r"""
    The nested comprehension syntax (`for batch in batches for pred in batch`) reads left-to-right in the same order you'd write nested for-loops. It trips people up at first, but it becomes second nature. You'll use this pattern when you need to flatten lists of predictions from multiple batches into a single list for computing metrics.

    Now let's look at generators -- they become critical when your data doesn't fit in memory, which happens more often than you'd think in ML.
    """)
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
    The key insight with generators: the `squares` generator above represents 10 million values but uses almost zero memory. It computes each value only when you ask for it with `next()`. This is exactly how PyTorch's `DataLoader` works under the hood -- it doesn't load your entire dataset into memory at once. It yields batches lazily, one at a time, which is the only way to train on datasets that are larger than your RAM.

    If you see `yield` in ML code, the author is building a data pipeline that processes data incrementally rather than materializing everything at once.

    ### Functions, Lambda, *args/**kwargs

    ML code is full of configurable functions with default arguments, short lambda functions for sorting and filtering, and `**kwargs` for passing flexible sets of hyperparameters. Let's look at the patterns you'll encounter most.
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
    A few things to notice here:

    - The `train` function signature with keyword defaults (`lr=0.001, epochs=10`) is the canonical pattern for ML functions. You'll see this in every training script. The defaults serve as documentation -- they tell you what reasonable values look like.

    - The `lambda` for sorting is a one-liner you'll use when comparing models by performance, sorting layers by parameter count, or ranking features by importance. It's just a function without a name.

    - The `**kwargs` pattern in `log_experiment` is especially important because ML frameworks use it heavily. PyTorch's `nn.Module.__init__` passes keyword arguments up the inheritance chain this way. You also see it in experiment tracking -- you don't know upfront which hyperparameters you'll want to log, so you accept arbitrary keyword arguments.

    **Gotcha with default arguments**: Never use a mutable default like `def f(x, data=[])`. The list is shared across all calls. Use `None` as the default and create the list inside the function. This is a classic Python pitfall that causes subtle bugs.

    ### Classes -- Just Enough for PyTorch

    You don't need to be an OOP expert, but you need to understand class basics because every PyTorch model is a class. The pattern below is a simplified version of what you'll write dozens of times in this course.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here's a minimal class that implements a linear regressor using NumPy. Pay attention to the `@` operator in the `predict` method -- that's matrix multiplication, and it's doing all the real work. The `__repr__` method is a nice-to-have that makes debugging easier; when you print a model, you want to see something informative, not `<LinearRegressor object at 0x...>`.
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

    The `super().__init__()` call is non-negotiable -- skip it and PyTorch won't track your parameters. The `forward` method is called automatically when you do `model(x)` (PyTorch overrides `__call__` to add hooks and gradient tracking around your `forward`).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### F-strings, Context Managers, Decorators

    **F-strings** -- use them everywhere for readable logging. In ML, you will print training metrics thousands of times (every epoch, sometimes every batch), so it's worth memorizing the format specifiers that make output clean and scannable.
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
    Let's break down those format specifiers because you'll use them constantly:

    - `:03d` -- pad the integer to 3 digits with leading zeros. Makes log output align nicely when epochs go from 1 to 100.
    - `:.4f` -- show exactly 4 decimal places. Losses are usually in the range 0.0001 to 10.0, so 4 decimals gives you enough precision to see whether your model is still improving.
    - `:.2%` -- format as a percentage with 2 decimal places. Multiplies by 100 and adds the `%` sign automatically. So `0.9120` becomes `91.20%`.

    When you're monitoring a training run that takes hours, clean log output is the difference between quickly spotting a problem and staring at a wall of numbers.

    **Context managers** -- the `with` statement handles resource cleanup. In ML, two context managers come up constantly:

    ```python
    # File I/O
    with open("results.json", "w") as f:
        json.dump(results, f)

    # PyTorch: disabling gradient computation during evaluation
    with torch.no_grad():
        predictions = model(test_data)
    ```

    The `torch.no_grad()` context manager is critical for evaluation -- it tells PyTorch not to build a computation graph, which saves memory and speeds things up significantly. Without it, your GPU will run out of memory during evaluation on large models.

    **Decorators** -- you'll mostly consume them, not write them:

    ```python
    @torch.no_grad()          # same as the context manager, but for a whole function
    def evaluate(model, data):
        ...

    @staticmethod             # method that doesn't need self
    def compute_loss(pred, target):
        ...
    ```

    A decorator is just a function that wraps another function. `@torch.no_grad()` on a function is equivalent to putting the entire function body inside `with torch.no_grad():`. Use the decorator when the whole function should run without gradients; use the context manager when only part of it should.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 3. NumPy -- The Foundation of Everything

    NumPy is the single most important library in the Python ML stack. PyTorch tensors are designed to feel like NumPy arrays. If you understand NumPy deeply, PyTorch will feel natural. If you don't, everything will feel like fighting the framework.

    For the mathematical foundation of array and matrix operations, see [MML Chapter 2 -- Linear Algebra](../textbooks/MML.pdf).

    ### Why Vectorization Matters

    This is the most important conceptual point in this entire module. Python loops are slow. NumPy operations on arrays are fast. The difference is not 2x -- it's 100x or more.

    Why? When you write a Python for-loop, each iteration involves:
    1. Python interpreter overhead (type checking, reference counting, etc.)
    2. Fetching the next element from the list (which can be any Python object)
    3. Performing the operation through Python's generic dispatch

    When you write `a + b` on NumPy arrays, the entire operation happens in a single call to optimized C code that operates on contiguous blocks of memory. The data is stored as raw numbers (not Python objects), and the CPU can use SIMD instructions to process multiple elements simultaneously.

    The bottom line: **every minute you spend learning to think in vectorized operations will save you hours of waiting for code to run.**
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
    Run that cell and look at the speedup number. On most machines, you'll see somewhere between 50x and 200x. Now imagine that speedup applied to every matrix operation in a training loop that runs for hours. Vectorization isn't a nice-to-have -- it's the difference between an experiment taking 5 minutes and taking 8 hours.

    The rule: **if you're writing a for-loop over array elements in ML code, you're probably doing it wrong.** Think in terms of whole-array operations. This is a mindset shift, and it takes practice, but it's essential.

    ### Array Creation, Indexing, Slicing, Reshaping

    Let's go through the array creation functions you'll use most. Each one exists for a reason, and you'll reach for different ones in different situations.

    - `np.zeros` and `np.ones` -- initializing weight matrices, creating masks, placeholder arrays
    - `np.eye` -- identity matrix, used in regularization and as a baseline transformation
    - `np.arange` -- like Python's `range()` but returns an array and supports floats
    - `np.linspace` -- evenly spaced points in an interval, essential for plotting smooth curves and creating learning rate schedules
    - `np.random.randn` -- samples from the standard normal distribution, used for weight initialization (more on this when we cover neural networks)
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
def _(mo):
    mo.md(r"""
    Now indexing and slicing. If you've used Python lists, you know the basics, but NumPy extends this to multiple dimensions and adds two powerful features: **boolean indexing** and **fancy indexing**.

    - **Boolean indexing** lets you select elements based on a condition. This is how you filter data points by class, select predictions above a confidence threshold, or mask out padding tokens.
    - **Fancy indexing** lets you select specific rows/columns by passing a list of indices. This is how you select a subset of training examples (e.g., for a mini-batch) or reorder rows.

    Watch the shapes carefully as you read the output -- understanding how slicing affects dimensionality is crucial for avoiding shape mismatch errors.
    """)
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
def _(mo):
    mo.md(r"""
    Now reshaping -- this is where most beginners hit errors, and it's worth understanding deeply.

    `reshape` changes the shape of an array without changing its data. The data stays in the same order in memory; you're just telling NumPy to interpret the same block of numbers as a different-shaped grid.

    The `-1` in `reshape(3, -1)` means "figure out this dimension automatically." NumPy divides the total number of elements by the other specified dimensions. So for 12 elements, `reshape(3, -1)` becomes `reshape(3, 4)`. You'll use `-1` constantly because it saves you from computing dimensions by hand.

    `reshape(-1, 1)` turns a 1D vector into a column vector (2D array with one column). This comes up when you need to make a vector compatible with a matrix operation via broadcasting.

    Transpose (`.T`) swaps rows and columns. For a matrix $A$ of shape $(m, n)$, $A^T$ has shape $(n, m)$. You'll use this for things like computing $X^T X$ (the Gram matrix), which appears in linear regression, PCA, and many other places.
    """)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Critical detail**: NumPy slices are *views*, not copies. Modifying a slice modifies the original array. This is a performance optimization (no data is copied), but it's a source of subtle bugs if you're not aware of it. Use `.copy()` when you need independence.

    This matters in ML when you're preprocessing data -- if you normalize a slice of your dataset, you might accidentally modify the original. The cell below demonstrates this behavior.
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
    Takeaway: if you slice an array and then modify the slice, check whether you intended to modify the original. In training pipelines, this is a common source of data leakage bugs -- you normalize your test data, which accidentally normalizes the same underlying memory as your training data.

    ### Broadcasting Rules

    Broadcasting is how NumPy handles operations between arrays of different shapes. It is elegant when you understand it and baffling when you don't. Here are the rules:

    1. If arrays have different numbers of dimensions, pad the smaller shape with 1s **on the left**.
    2. Arrays with size 1 along a dimension act as if they had the size of the other array along that dimension (the values are repeated).
    3. If sizes disagree along a dimension and neither is 1, it's an error.

    Why does this matter for ML? Because almost every operation you do involves arrays of different shapes. Adding a bias vector to every row of a matrix? Broadcasting. Normalizing each feature column by its mean? Broadcasting. Computing pairwise distances between two sets of points? Broadcasting. Without it, you'd need explicit loops or manual `np.tile` calls for all of these.

    Let's walk through four examples, from simple to subtle.
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
    Let's trace through Example 4 step by step, because this is the pattern that unlocks so much in ML:

    - `x` has shape `(3, 1)` and `y` has shape `(1, 3)`
    - Broadcasting rule 2: `x` is stretched along axis 1 to `(3, 3)`, and `y` is stretched along axis 0 to `(3, 3)`
    - Now they're both `(3, 3)` and element-wise multiplication proceeds

    The result is the outer product $x y^T$, computed without any loops. This exact pattern shows up when computing:
    - **Pairwise distances** between points (for k-nearest neighbors, kernel methods)
    - **Attention scores** in transformers ($Q K^T$)
    - **Gram matrices** for kernel methods
    - **Outer products** in gradient computations

    When you get shape mismatch errors (and you will, frequently), draw out the shapes on paper and apply the broadcasting rules manually. It becomes second nature with practice.

    ### Linear Algebra Operations

    This is where NumPy connects directly to the math you'll be doing. Matrix multiplication is *the* core operation of machine learning -- every neural network layer, every linear regression, every PCA computation boils down to matrix multiplications and element-wise operations.

    The `@` operator was added to Python specifically for this. Before Python 3.5, you had to write `np.dot(A, B)` or `A.dot(B)`, which was harder to read. Now `A @ B` makes matrix multiplication as clean as addition.

    See [MML Section 2.3 -- Solving Systems of Linear Equations](../textbooks/MML.pdf) for the underlying theory.
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
def _(mo):
    mo.md(r"""
    A key point about `np.linalg.solve` vs `np.linalg.inv`: computing $A^{-1}b$ requires first computing $A^{-1}$ (expensive) and then multiplying (another operation). `solve` uses LU decomposition to find $x$ directly, which is both faster and more numerically stable. In practice, you almost never need the actual inverse matrix -- you need the *result* of applying the inverse, which `solve` gives you directly.

    This is a good example of a broader principle: **the mathematically equivalent approach and the computationally best approach are often different.**

    Now let's look at decompositions. These are the workhorses of dimensionality reduction, data compression, and understanding matrix structure.
    """)
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
    Why these operations matter for ML:

    - **SVD (Singular Value Decomposition)**: PCA is literally truncated SVD. When you reduce 1000-dimensional data to 50 dimensions, you're keeping the top 50 singular values/vectors. You'll also see SVD in matrix factorization for recommendation systems, low-rank approximations for compressing neural networks, and computing pseudoinverses.

    - **Eigenvalues/Eigenvectors**: These tell you about the "natural axes" of a linear transformation. In PCA, the eigenvectors of the covariance matrix are the principal components. In spectral clustering, you use eigenvectors of the graph Laplacian. The eigenvalues tell you how much variance each axis captures.

    - **Norms**: The L2 norm $\|x\|_2 = \sqrt{\sum x_i^2}$ is the standard Euclidean distance. The L1 norm $\|x\|_1 = \sum |x_i|$ promotes sparsity (this is the core idea behind Lasso regularization). You'll use norms for regularization, gradient clipping, and measuring distances.

    **Practical tip**: never compute `np.linalg.inv(A) @ b` when you can use `np.linalg.solve(A, b)`. The latter is faster and more numerically stable. SVD will become very important when we cover dimensionality reduction (PCA is just truncated SVD). See [ISLR Section 6.3 -- Dimension Reduction Methods](../textbooks/ISLR.pdf) for more context.

    ### Random Number Generation

    Randomness is everywhere in ML: weight initialization, data shuffling, dropout, stochastic gradient descent, train/test splitting. Getting randomness right -- and being able to reproduce it exactly -- is essential for scientific work.
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
    A few important points about random number generation in ML:

    - **Always seed your RNG for reproducibility.** When comparing two models, you need identical train/test splits and identical weight initializations. Unseeded randomness makes debugging nearly impossible and makes your experiments non-reproducible.

    - **Use the modern `default_rng` API**, not the legacy `np.random.seed()` / `np.random.randn()`. The legacy API uses a global state, which means any library you import could silently change your random state. `default_rng` creates an independent generator object that only you control.

    - **`standard_normal` vs `uniform`**: You'll use normal distributions for weight initialization (with appropriate scaling), and uniform distributions for things like random search over hyperparameters.

    - **`permutation`** is how you shuffle data. Never sort your training data by label -- models will learn the label order instead of the features.

    ---

    ## 4. Pandas -- Data Manipulation

    Pandas is how you'll load, clean, explore, and preprocess tabular data before feeding it to models. It's the first tool you reach for when you get a new dataset: What does it look like? How much data is there? Are there missing values? What's the distribution of each feature?

    For statistical perspectives on data handling, see [ISLR Chapter 2 -- Statistical Learning Overview](../textbooks/ISLR.pdf).

    ### DataFrames and Series

    A DataFrame is essentially a table -- rows are samples (data points), columns are features (variables). This maps directly to the $X$ matrix in ML notation, where each row $x_i$ is a training example and each column $j$ is a feature dimension.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Get in the habit of running these five things on every new dataset:

    1. `.shape` -- How much data do you have? Is it enough? (100 samples and 10,000 features is a very different problem than 10,000 samples and 100 features.)
    2. `.dtypes` -- Are your numeric columns actually numeric, or are they stored as strings? Pandas silently stores mixed-type columns as `object`, which will cause errors downstream.
    3. `.describe()` -- What's the range, mean, std of each feature? Are there any obvious outliers (e.g., a negative age)?
    4. `.head()` -- What does the raw data actually look like? You'd be surprised how often the first few rows reveal data quality issues.
    5. `.value_counts()` on the label column -- Is the dataset balanced? If 95% of your labels are "normal" and 5% are "anomaly," you have a class imbalance problem that needs specific handling.

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
    **Common gotcha**: `loc[0:2]` includes row 2. `iloc[0:2]` excludes row 2. This is because `loc` uses inclusive-on-both-ends label slicing, while `iloc` follows Python's half-open convention. This will bite you at least once -- now you know why.

    Also notice the boolean indexing uses `&` (bitwise AND), not `and` (logical AND). Each condition must be in parentheses. This is a Pandas quirk due to Python's operator precedence rules. `|` is OR, `~` is NOT.

    ### groupby, merge, apply

    These three operations cover most of what you'll do with tabular data in the preprocessing stage.

    - **groupby** -- compute per-class statistics (mean accuracy per class, loss per batch, etc.)
    - **merge** -- join tables together (features table + labels table, train metadata + evaluation results)
    - **vectorized operations** -- always prefer these over `.apply()` for performance. The same vectorization principle from NumPy applies here.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    That last example -- subtracting the mean and dividing by the standard deviation -- is called **z-score normalization** (or standardization). It transforms each feature to have mean 0 and standard deviation 1. This is one of the most common preprocessing steps in ML because many algorithms (gradient descent, SVMs, k-nearest neighbors) perform poorly or converge slowly when features are on very different scales. If one feature ranges from 0 to 1 and another from 0 to 1,000,000, the large-scale feature will dominate the distance calculations.

    Notice that we used vectorized Pandas operations (`df["col"] - df["col"].mean()`) rather than `.apply(lambda x: ...)`. The vectorized version is 10-100x faster because it drops down to NumPy under the hood.

    ### Handling Missing Data

    Real datasets have missing values. Always check and handle them before modeling. A model trained on data with silently propagated NaNs will produce garbage without any error messages -- one of the most insidious bugs in ML pipelines.
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
    Two strategies for missing data, each with tradeoffs:

    - **`dropna()`** -- remove rows with any missing values. Simple but dangerous: if 30% of rows have at least one missing value, you just threw away 30% of your data. Only use this when missing values are rare and random.

    - **`fillna()`** -- replace missing values with a default (mean, median, mode, or a sentinel value). The median is more robust than the mean when your data has outliers. More sophisticated approaches include model-based imputation (predicting missing values from other features), but the median is a reasonable starting point.

    **Gotcha**: always compute fill statistics (mean, median) from the *training* data only, then apply them to both train and test. If you compute the median including test data, you've leaked test information into your preprocessing -- a form of data leakage that inflates your accuracy estimates.

    Also watch for "implicit" missing values -- things like empty strings `""`, the word `"NA"`, or `-999` used as sentinels. `isnull()` won't catch these; you need to inspect `unique()` values manually.

    ---

    ## 5. Matplotlib -- Visualization

    Visualization is how you build intuition about your data and diagnose model behavior. Don't skip this -- the ability to quickly plot things will save you hours of confusion. When your model isn't learning, the first thing you should do is plot:
    - The training and validation loss curves (are they diverging? plateauing?)
    - The data itself (is there a clear pattern? are classes separable?)
    - The distribution of predictions (is the model predicting the same class for everything?)

    ### Basic Plots

    Let's cover the four plot types you'll use most in ML work.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Training curves** are the plot you'll look at most often. The x-axis is the epoch (one full pass through the training data), and the y-axis is the loss (how wrong the model is). A healthy training curve goes down. If the training loss goes down but the validation loss goes up, your model is overfitting -- memorizing the training data instead of learning generalizable patterns.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Scatter plots** let you visualize 2D data with class labels. This is how you'll check whether your classes are separable, spot clusters, and visualize learned embeddings. The `c=labels` parameter colors points by class, and `cmap="bwr"` gives you a blue/white/red colormap.
    """)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Histograms** show the distribution of your data. You'll use these to check whether features are normally distributed (many algorithms assume this), spot outliers, and verify that your data augmentation is working as expected. The `density=True` parameter normalizes the histogram so the area under the curve equals 1, making it comparable to probability density functions.
    """)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Heatmaps** visualize matrices. The two most common uses in ML:
    - **Correlation matrices**: show how features are related to each other. Highly correlated features are redundant; you might drop one to reduce dimensionality.
    - **Confusion matrices**: show where your classifier makes mistakes. A perfect classifier has all its weight on the diagonal; off-diagonal entries tell you which classes get confused with each other.
    """)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Subplots and Figure Customization

    In practice, you'll almost always want multiple plots side by side -- comparing metrics, showing different views of the same data, or displaying results across experiments. The `subplots` function gives you a grid of axes objects that you can fill independently.
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

    The key distinction: `plt.xlabel(...)` implicitly acts on "the current axes." `ax.set_xlabel(...)` explicitly targets a specific axes object. When you have multiple subplots, explicit is better than implicit. The implicit API becomes ambiguous and error-prone with multiple subplots -- you'll set a label and it'll end up on the wrong plot.
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
    A few practical tips for matplotlib in ML work:

    - **Always label your axes and add a title.** When you come back to a plot three weeks later, you won't remember what axis 1 was.
    - **Use `tight_layout()`** or `constrained_layout=True` to prevent labels from getting cut off.
    - **Save figures with `fig.savefig("plot.png", dpi=150, bbox_inches="tight")`** for publications and reports. The `bbox_inches="tight"` prevents clipping.
    - **Use consistent color schemes** across related plots. Matplotlib's default colors are fine for exploration; for papers, consider colorblind-friendly palettes like `tab10` or `Set2`.

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


@app.cell(hide_code=True)
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
def _(mo):
    mo.md(r"""
    This exercise is designed to test your broadcasting skills. The hint gives you the key identity:

    $$\|x_i - x_j\|^2 = \|x_i\|^2 + \|x_j\|^2 - 2 x_i^T x_j$$

    Think about the shapes:
    - $\|x_i\|^2$ for all $i$ is a vector of shape $(n,)$ -- you'll want to reshape it to $(n, 1)$ for broadcasting
    - $\|x_j\|^2$ for all $j$ is a vector of shape $(n,)$ -- reshape to $(1, n)$
    - $x_i^T x_j$ for all pairs is the matrix $X X^T$, which has shape $(n, n)$

    Broadcasting handles the rest. This is exactly the outer-product-via-broadcasting pattern from earlier, applied to a real ML problem (pairwise distances appear in k-NN, kernel methods, and t-SNE).
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Before you run the cell below, try to work out each shape on paper using the broadcasting rules:

    - `A` is `(3, 4, 5)` and `B` is `(4, 1)`. Rule 1: pad B to `(1, 4, 1)`. Then broadcast: `(3, 4, 5)`.
    - `A` is `(3, 4, 5)` and `C` is `(5,)`. Rule 1: pad C to `(1, 1, 5)`. Then broadcast: `(3, 4, 5)`.
    - `A` is `(3, 4, 5)` and `D` is `(3, 1, 1)`. Already 3D, broadcast: `(3, 4, 5)`.
    - `A * B * C` chains the broadcasts: `(3, 4, 5)`.

    Being able to do this in your head is a skill worth developing -- shape errors are the most common bug in ML code, and the fastest debuggers are the ones who can trace shapes mentally.
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

    For a comprehensive reference on the mathematical foundations ahead, see [MML -- Mathematics for Machine Learning, full text](../textbooks/MML.pdf) and [Bishop PRML -- Introduction](../textbooks/Bishop-PRML.pdf) for the probabilistic perspective we'll be developing throughout the course.
    """)
    return


if __name__ == "__main__":
    app.run()
