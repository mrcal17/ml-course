import marimo

app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Optimization

    Everything in machine learning — every model you'll train, every parameter you'll fit — comes down to optimization. This module puts that machinery on solid ground. You've seen gradient descent before and have some intuition for convexity, but now we're going to build the whole picture systematically, armed with the calculus and linear algebra you just refreshed.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 1. Why Optimization Is the Engine of ML

    Here's the pattern that repeats across nearly every ML algorithm:

    1. **Define a model** with parameters $\theta$.
    2. **Define a loss function** $\mathcal{L}(\theta)$ that measures how badly the model fits the data.
    3. **Find the $\theta$ that minimizes $\mathcal{L}(\theta)$.**

    Step 3 is optimization. That's it. Linear regression, logistic regression, neural networks, SVMs, decision trees (yes, even those) — they all follow this template. The differences lie in what model you choose and what loss function you use, but the optimization step is always there.

    You'll hear several names for the thing being minimized: **loss function**, **cost function**, **objective function**, **energy function**. These are, for our purposes, the same concept. Some authors use "loss" for per-sample error and "cost" for the average over the dataset, but don't get hung up on terminology.

    **The landscape metaphor.** Think of $\mathcal{L}(\theta)$ as defining a surface over parameter space. For two parameters, you can literally picture a terrain — hills, valleys, ridges. Optimization is the process of finding the lowest point in that terrain. The shape of the terrain determines whether this is easy or hard, and most of this module is about understanding that shape and choosing the right strategy to navigate it.

    See [MML Ch 7](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) for a thorough treatment of optimization in the ML context.
    """)
    return


@app.cell
def _():
    import numpy as np

    # The ML optimization pattern in code:
    # 1. Model: linear prediction y_hat = X @ theta
    # 2. Loss: mean squared error L(theta) = (1/N) * ||y - X@theta||^2
    # 3. Optimize: find theta that minimizes L

    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(50), rng.standard_normal(50)])  # 50 samples, 2 features
    theta_true = np.array([3.0, 1.5])
    y = X @ theta_true + 0.3 * rng.standard_normal(50)

    # Loss function: maps parameters -> scalar
    def mse_loss(theta, X, y):
        residuals = y - X @ theta
        return np.mean(residuals ** 2)

    # Evaluate loss at a few guesses
    for guess in [np.array([0.0, 0.0]), np.array([3.0, 1.0]), theta_true]:
        print(f"theta={guess} -> loss={mse_loss(guess, X, y):.4f}")
    return (mse_loss,)


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 2. Convexity — When Optimization Is "Easy"

    Convexity is the single most important structural property in optimization. When you have it, life is good. When you don't, you need tricks.

    ### Convex Sets and Functions

    A set $C \subseteq \mathbb{R}^n$ is **convex** if for any two points $x, y \in C$ and any $\lambda \in [0, 1]$:

    $$\lambda x + (1 - \lambda) y \in C$$

    Geometrically: draw a line segment between any two points in the set, and the entire segment stays inside the set. A circle is convex. A star shape is not.

    A function $f : \mathbb{R}^n \to \mathbb{R}$ is **convex** if its domain is a convex set and for all $x, y$ in the domain and $\lambda \in [0, 1]$:

    $$f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda) f(y)$$

    Geometrically: the function lies *below* (or on) the line segment connecting any two points on its graph. It "curves upward." A bowl, not a saddle.

    ### Why Convexity Matters

    The fundamental theorem that makes convex optimization tractable:

    > **For a convex function, every local minimum is a global minimum.**

    This is enormous. It means you can use *local* search methods — methods that only look at the function in a neighborhood of the current point — and be guaranteed to find the *globally* best solution. No need to worry about getting trapped in a bad local minimum.

    For a formal treatment and proofs, see [Boyd & Vandenberghe, Ch 3](file:///C:/Users/landa/ml-course/textbooks/Boyd-ConvexOptimization.pdf).

    ### Checking Convexity with the Hessian

    You refreshed the Hessian in Module 0B. Here's the payoff: a twice-differentiable function $f$ is convex if and only if its Hessian $\mathbf{H}(x) = \nabla^2 f(x)$ is **positive semi-definite** for all $x$ in its domain:

    $$\nabla^2 f(x) \succeq 0 \quad \forall x$$

    This connects directly to the eigenvalue work from Module 0C — positive semi-definite means all eigenvalues of the Hessian are $\geq 0$ everywhere.

    **Example — linear regression is convex.** The ordinary least squares loss is:

    $$\mathcal{L}(\theta) = \|y - X\theta\|^2 = (y - X\theta)^\top(y - X\theta)$$

    The Hessian is $\nabla^2 \mathcal{L} = 2X^\top X$, which is always positive semi-definite (since $v^\top X^\top X v = \|Xv\|^2 \geq 0$ for any $v$). So OLS is convex — gradient descent will always find the global optimum.

    **Example — neural networks are NOT convex.** The loss surface of a neural network is highly non-convex, full of saddle points and local minima. This is why training neural networks is fundamentally harder than fitting linear regression, and why we need all the tricks in Sections 4–6.

    ### Strong Convexity

    A function is **strongly convex** with parameter $m > 0$ if:

    $$f(y) \geq f(x) + \nabla f(x)^\top (y - x) + \frac{m}{2}\|y - x\|^2$$

    This means the function curves upward *at least* as fast as a quadratic bowl. Strong convexity gives you:
    - A **unique** global minimum (not just "every local min is global" but "there's exactly one")
    - **Faster convergence** guarantees for gradient descent

    Adding $\ell_2$ regularization $\lambda \|\theta\|^2$ to any convex loss makes it strongly convex — one of the theoretical reasons regularization is so useful.

    See [Boyd, Ch 9](file:///C:/Users/landa/ml-course/textbooks/Boyd-ConvexOptimization.pdf) for convergence rate analysis that depends on strong convexity.
    """)
    return


@app.cell
def _():
    def _run():

        # Checking convexity: verify f(lambda*x + (1-lambda)*y) <= lambda*f(x) + (1-lambda)*f(y)
        f = lambda x: x**2  # convex
        g = lambda x: -x**2  # NOT convex (concave)

        x, y = -2.0, 3.0
        lam = 0.4
        mid = lam * x + (1 - lam) * y  # point on line segment

        print("f(x) = x^2 (convex):")
        print(f"  f(midpoint)     = {f(mid):.4f}")
        print(f"  weighted average = {lam * f(x) + (1 - lam) * f(y):.4f}")
        print(f"  f(mid) <= avg?   {f(mid) <= lam * f(x) + (1 - lam) * f(y)}")

        print("\ng(x) = -x^2 (concave):")
        print(f"  g(midpoint)     = {g(mid):.4f}")
        print(f"  weighted average = {lam * g(x) + (1 - lam) * g(y):.4f}")
        print(f"  g(mid) <= avg?   {g(mid) <= lam * g(x) + (1 - lam) * g(y)}")


    _run()
    return

@app.cell
def _():
    def _run():

        # Hessian check: OLS Hessian is 2*X^T*X, should be PSD
        rng = np.random.default_rng(1)
        X = rng.standard_normal((20, 3))
        H_ols = 2 * X.T @ X  # Hessian of ||y - Xθ||^2

        eigenvalues = np.linalg.eigvalsh(H_ols)
        print(f"OLS Hessian eigenvalues: {eigenvalues.round(4)}")
        print(f"All >= 0? {np.all(eigenvalues >= -1e-10)}  => OLS is convex")

        # Add L2 regularization: Hessian becomes 2(X^T X + lambda*I)
        lam = 1.0
        H_ridge = 2 * (X.T @ X + lam * np.eye(3))
        eig_ridge = np.linalg.eigvalsh(H_ridge)
        print(f"\nRidge Hessian eigenvalues: {eig_ridge.round(4)}")
        print(f"All > 0?  {np.all(eig_ridge > 0)}  => Ridge is strongly convex")


    _run()
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3. Gradient Descent — The Workhorse

    You've seen this before, but let's be precise.

    ### The Algorithm

    Given a differentiable function $f(\theta)$, gradient descent iterates:

    $$\theta_{t+1} = \theta_t - \alpha \nabla f(\theta_t)$$

    where $\alpha > 0$ is the **learning rate** (or step size).

    ```
    Algorithm: Gradient Descent
    ─────────────────────────────
    Input: f(θ), initial θ₀, learning rate α, tolerance ε
    t ← 0
    repeat:
        g ← ∇f(θ_t)              # compute gradient
        θ_{t+1} ← θ_t − α · g    # take step
        t ← t + 1
    until ‖g‖ < ε or t > max_iterations
    return θ_t
    ```

    The gradient $\nabla f(\theta_t)$ points in the direction of steepest *ascent*. We subtract it to go *downhill*.

    ### The Learning Rate

    The learning rate $\alpha$ is your first encounter with a hyperparameter that can make or break training:

    - **Too large:** The updates overshoot the minimum, and the iterates oscillate or diverge entirely. Imagine taking giant leaps across a valley — you jump right over the bottom and land on the opposite slope, then jump back even further.
    - **Too small:** Convergence is painfully slow. You're taking tiny, cautious steps when you could afford to stride.
    - **Just right:** Steady progress toward the minimum.
    """)
    return


@app.cell
def _():
    def _run():

        # Gradient descent from scratch on f(x) = x^2
        # gradient: f'(x) = 2x
        # update:   x_{t+1} = x_t - alpha * 2 * x_t

        def gradient_descent_1d(f_grad, x0, lr, n_steps):
            """Vanilla GD: theta <- theta - lr * grad."""
            x = x0
            history = [x]
            for _ in range(n_steps):
                g = f_grad(x)       # compute gradient
                x = x - lr * g      # take step downhill
                history.append(x)
            return np.array(history)

        f_grad = lambda x: 2 * x  # gradient of x^2
        path = gradient_descent_1d(f_grad, x0=5.0, lr=0.1, n_steps=20)
        print("GD on f(x)=x^2, lr=0.1, start=5.0:")
        print(f"  Steps 0-4: {path[:5].round(4)}")
        print(f"  Final x = {path[-1]:.6f}, f(x) = {path[-1]**2:.6f}")


    _run()
    return

@app.cell
def _():
    def _run():

        # Learning rate comparison: too small, just right, too large
        f_grad = lambda x: 2 * x  # gradient of x^2

        for lr, label in [(0.01, "too small"), (0.1, "just right"), (1.05, "too large")]:
            x = 5.0
            for t in range(20):
                x = x - lr * f_grad(x)
                if abs(x) > 1e6:
                    print(f"lr={lr:5.2f} ({label:>10s}): DIVERGED at step {t}")
                    break
            else:
                print(f"lr={lr:5.2f} ({label:>10s}): x={x:.6f}, f(x)={x**2:.6f}")


    _run()
    return

@app.cell
def _(mo):
    mo.md(r"""
    ### Interactive: Learning Rate Explorer

    Use the slider to change the learning rate and watch how gradient descent behaves on $f(x, y) = x^2 + 4y^2$. Too small = slow convergence. Too large = divergence. Just right = fast convergence.
    """)
    return


@app.cell
def _(mo):
    lr_slider = mo.ui.slider(start=0.01, stop=2.0, step=0.01, value=0.1, label="Learning rate alpha")
    lr_slider
    return (lr_slider,)


@app.cell
def _(lr_slider):
    def _run():
        import matplotlib.pyplot as plt

        def gradient_descent_2d(lr, n_steps=50):
            """Run gradient descent on f(x,y) = x^2 + 4y^2."""
            path = [(4.0, 4.0)]
            x, y = 4.0, 4.0
            for _ in range(n_steps):
                gx = 2 * x
                gy = 8 * y
                x = x - lr * gx
                y = y - lr * gy
                path.append((x, y))
                if x**2 + y**2 > 1e6:  # diverged
                    break
            return np.array(path)

        _lr = lr_slider.value
        _path = gradient_descent_2d(_lr)

        fig_lr, ax_lr = plt.subplots(figsize=(8, 6))

        # Contour plot
        _xx = np.linspace(-6, 6, 200)
        _yy = np.linspace(-6, 6, 200)
        _XX, _YY = np.meshgrid(_xx, _yy)
        _ZZ = _XX**2 + 4 * _YY**2

        ax_lr.contour(_XX, _YY, _ZZ, levels=20, cmap='viridis', alpha=0.5)
        ax_lr.contourf(_XX, _YY, _ZZ, levels=20, cmap='viridis', alpha=0.2)

        # Clip path for plotting
        _path_clipped = _path[np.abs(_path).max(axis=1) < 10]

        if len(_path_clipped) > 1:
            ax_lr.plot(_path_clipped[:, 0], _path_clipped[:, 1], 'ro-', markersize=3, linewidth=1)
            ax_lr.plot(_path_clipped[0, 0], _path_clipped[0, 1], 'gs', markersize=10, label='Start')
            ax_lr.plot(_path_clipped[-1, 0], _path_clipped[-1, 1], 'r*', markersize=15, label='End')

        _final_loss = _path[-1][0]**2 + 4 * _path[-1][1]**2 if len(_path) > 0 else float('inf')
        _status = "DIVERGED" if _final_loss > 100 else f"Loss = {_final_loss:.4f}"

        ax_lr.set_title(f"Gradient Descent on f(x,y) = x² + 4y² | lr={_lr:.2f} | {_status}")
        ax_lr.set_xlabel("x")
        ax_lr.set_ylabel("y")
        ax_lr.set_xlim(-6, 6)
        ax_lr.set_ylim(-6, 6)
        ax_lr.legend()
        ax_lr.set_aspect('equal')
        plt.tight_layout()
        fig_lr


    _run()
    return

@app.cell
def _(mo):
    mo.image(src="../animations/rendered/GradientDescentContour.gif")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Convergence Guarantees

    For a convex function with $L$-Lipschitz continuous gradients (i.e., $\|\nabla f(x) - \nabla f(y)\| \leq L\|x - y\|$), gradient descent with $\alpha = 1/L$ satisfies:

    $$f(\theta_t) - f(\theta^*) \leq \frac{L \|\theta_0 - \theta^*\|^2}{2t}$$

    This is $O(1/t)$ convergence. For **strongly convex** functions with parameter $m$, you get **linear** (exponential) convergence:

    $$f(\theta_t) - f(\theta^*) \leq \left(1 - \frac{m}{L}\right)^t \cdot (f(\theta_0) - f(\theta^*))$$

    The ratio $\kappa = L/m$ is the **condition number** — it measures how "elongated" the loss surface is. A large condition number means narrow ravines in the landscape, and gradient descent struggles because it zigzags back and forth across the ravine instead of sliding down it. This is exactly the pathology that momentum methods (Section 5) fix.

    ### The Contour Plot Perspective

    Visualize the loss function via its contour lines (level sets). For a well-conditioned quadratic, the contours are nearly circular and gradient descent takes a nearly straight path to the center. For an ill-conditioned one, the contours are highly elongated ellipses, and gradient descent zigzags because the gradient is perpendicular to the contour — it points toward the nearest contour wall, not toward the center of the ellipses.

    See [MML 7.1](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) for gradient descent with worked examples and convergence discussion.
    """)
    return


@app.cell
def _():
    def _run():

        # Convergence rate depends on condition number kappa = L / m
        # Well-conditioned (kappa=2) vs ill-conditioned (kappa=50)

        def gd_quadratic(A, x0, n_steps=50):
            """GD on f(x) = 0.5 * x^T A x with lr = 1/L."""
            L = np.max(np.linalg.eigvalsh(A))  # Lipschitz constant
            lr = 1.0 / L
            x = x0.copy()
            losses = [0.5 * x @ A @ x]
            for _ in range(n_steps):
                g = A @ x              # gradient of 0.5 * x^T A x
                x = x - lr * g         # GD step with optimal fixed lr
                losses.append(0.5 * x @ A @ x)
            return np.array(losses)

        x0 = np.array([10.0, 10.0])
        # Well-conditioned: eigenvalues 1 and 2 -> kappa = 2
        losses_good = gd_quadratic(np.diag([1.0, 2.0]), x0)
        # Ill-conditioned: eigenvalues 1 and 50 -> kappa = 50
        losses_bad = gd_quadratic(np.diag([1.0, 50.0]), x0)

        print("Steps to reach loss < 0.01:")
        good_steps = np.argmax(losses_good < 0.01)
        bad_steps = np.argmax(losses_bad < 0.01)
        print(f"  kappa=2:  {good_steps} steps")
        print(f"  kappa=50: {bad_steps} steps")
        print(f"High condition number -> {bad_steps/good_steps:.1f}x slower convergence")


    _run()
    return

@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 4. Stochastic Gradient Descent (SGD)

    ### The Problem with Full Gradients

    In ML, the loss function is typically a sum over the entire dataset:

    $$\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \ell(\theta; x_i, y_i)$$

    Computing $\nabla \mathcal{L}(\theta)$ requires touching every data point. When $N$ is millions or billions, this is prohibitively expensive for a single update step.

    ### The Stochastic Approximation

    SGD replaces the full gradient with a **noisy estimate** computed on a random subset (mini-batch) $\mathcal{B} \subset \{1, \ldots, N\}$:

    $$g_t = \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \nabla \ell(\theta_t; x_i, y_i)$$

    $$\theta_{t+1} = \theta_t - \alpha_t \, g_t$$

    The key insight: $\mathbb{E}[g_t] = \nabla \mathcal{L}(\theta_t)$. The stochastic gradient is an **unbiased estimator** of the true gradient. On average, you're going in the right direction — you're just noisy about it.

    ```
    Algorithm: Stochastic Gradient Descent
    ──────────────────────────────────────────
    Input: dataset D, loss ℓ(θ; x, y), initial θ₀, batch size B
    t ← 0
    for each epoch:
        shuffle D
        for each mini-batch B ⊂ D of size B:
            g ← (1/B) Σ_{(x,y)∈B} ∇ℓ(θ_t; x, y)
            θ_{t+1} ← θ_t − α_t · g
            t ← t + 1
    return θ_t
    ```

    ### The Noise Is Actually Helpful

    This is counterintuitive but important. The noise in SGD:

    - **Helps escape shallow local minima** — the random fluctuations can kick you out of a bad basin.
    - **Provides implicit regularization** — SGD tends to find "flat" minima (minima surrounded by a broad, shallow basin), which generalize better than "sharp" minima. This is an active area of research, but empirically it's well established.

    ### Mini-Batch Size Tradeoffs

    | Batch size | Gradient estimate | Per-step cost | Parallelism | Noise |
    |---|---|---|---|---|
    | 1 (true SGD) | Very noisy | Cheap | Low | High |
    | 32–256 (typical) | Moderate noise | Moderate | Good GPU utilization | Moderate |
    | Full dataset (GD) | Exact | Expensive | Maximal | None |

    In practice, batch sizes of 32–256 hit the sweet spot: enough parallelism to utilize GPUs efficiently, enough noise to help generalization.

    ### Learning Rate Schedules

    With full-batch gradient descent, a fixed learning rate works fine. With SGD, you generally need to **decrease the learning rate** over time. Why? The noise floor. Even when you're near the optimum, the stochastic gradient still has variance, so you'll oscillate around the minimum rather than converging to it — unless you shrink the step size.

    Common schedules:
    - **Step decay:** Multiply $\alpha$ by 0.1 every $k$ epochs.
    - **Cosine annealing:** $\alpha_t = \alpha_0 \cdot \frac{1}{2}(1 + \cos(\pi t / T))$
    - **Linear warmup + decay:** Start small (warmup), increase to peak, then decay. Common in transformer training.

    The theoretical requirement is $\sum_t \alpha_t = \infty$ and $\sum_t \alpha_t^2 < \infty$ (the Robbins-Monro conditions). The learning rate must shrink but not too fast.
    """)
    return


@app.cell
def _():
    def _run():

        # SGD vs full-batch GD on linear regression
        rng = np.random.default_rng(42)
        N, d = 200, 2
        X = rng.standard_normal((N, d))
        theta_true = np.array([3.0, -1.0])
        y = X @ theta_true + 0.5 * rng.standard_normal(N)

        # Full gradient: average over ALL samples
        def full_gradient(theta, X, y):
            return -(2 / len(y)) * X.T @ (y - X @ theta)

        # Mini-batch gradient: average over a random subset
        def minibatch_gradient(theta, X, y, batch_size=32):
            idx = rng.choice(len(y), batch_size, replace=False)
            return -(2 / batch_size) * X[idx].T @ (y[idx] - X[idx] @ theta)

        theta_gd = np.zeros(d)
        theta_sgd = np.zeros(d)
        lr = 0.01

        for t in range(200):
            theta_gd  -= lr * full_gradient(theta_gd, X, y)
            theta_sgd -= lr * minibatch_gradient(theta_sgd, X, y, batch_size=16)

        print(f"True theta:      {theta_true}")
        print(f"Full-batch GD:   {theta_gd.round(4)}")
        print(f"SGD (B=16):      {theta_sgd.round(4)}")


    _run()
    return

@app.cell
def _():
    def _run():

        # Learning rate schedules in code
        T = 100  # total steps

        def step_decay(t, lr0=0.1, drop_every=30, factor=0.1):
            return lr0 * (factor ** (t // drop_every))

        def cosine_anneal(t, lr0=0.1, T=100):
            return lr0 * 0.5 * (1 + np.cos(np.pi * t / T))

        def warmup_linear(t, lr0=0.1, warmup=10, T=100):
            if t < warmup:
                return lr0 * t / warmup  # linear warmup
            return lr0 * (1 - (t - warmup) / (T - warmup))  # linear decay

        steps = np.arange(T)
        for name, fn in [("Step decay", step_decay), ("Cosine", cosine_anneal), ("Warmup+decay", warmup_linear)]:
            lrs = [fn(t) for t in steps]
            print(f"{name:>14s}: lr[0]={lrs[0]:.4f}, lr[50]={lrs[50]:.4f}, lr[99]={lrs[99]:.4f}")


    _run()
    return

@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 5. Momentum Methods

    ### The Problem Momentum Solves

    Recall the ill-conditioned landscape: elongated contours cause gradient descent to zigzag. Each step oscillates across the narrow direction while making slow progress along the long direction. The gradient components across the ravine keep flipping sign, while the components along the ravine are consistently in the same direction.

    ### Classical Momentum

    Accumulate a **velocity** term that averages recent gradients:

    $$v_{t+1} = \beta v_t + \nabla f(\theta_t)$$
    $$\theta_{t+1} = \theta_t - \alpha \, v_{t+1}$$

    where $\beta \in [0, 1)$ is the momentum coefficient (typically 0.9).

    ```
    Algorithm: SGD with Momentum
    ──────────────────────────────
    Input: initial θ₀, learning rate α, momentum β
    v ← 0
    for t = 0, 1, 2, …:
        g ← ∇f(θ_t)        # or stochastic gradient
        v ← β · v + g       # accumulate velocity
        θ ← θ − α · v       # update parameters
    ```

    **Physics analogy:** Imagine a heavy ball rolling down the loss surface. It accumulates velocity in consistent downhill directions (along the ravine) while the oscillatory components (across the ravine) cancel out over successive steps. The ball builds speed in the right direction and dampens motion in the wrong directions.

    ### Nesterov Accelerated Gradient (NAG)

    Nesterov's insight: if you're going to take a momentum step anyway, **first look ahead** to where momentum would take you, compute the gradient *there*, and then correct:

    $$v_{t+1} = \beta v_t + \nabla f(\theta_t - \alpha \beta v_t)$$
    $$\theta_{t+1} = \theta_t - \alpha \, v_{t+1}$$

    This "lookahead" provides a corrective factor — if momentum is carrying you too far, the gradient at the lookahead point will push you back. NAG converges at the optimal rate $O(1/t^2)$ for convex functions, provably faster than vanilla gradient descent.

    See [MML 7.1.2](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) and [Boyd 9.3](file:///C:/Users/landa/ml-course/textbooks/Boyd-ConvexOptimization.pdf) for convergence analysis.
    """)
    return


@app.cell
def _():
    def _run():

        # Momentum from scratch on f(x,y) = x^2 + 50*y^2 (ill-conditioned, kappa=50)
        def gd_no_momentum(lr=0.01, n_steps=100):
            x = np.array([10.0, 10.0])
            losses = []
            for _ in range(n_steps):
                g = np.array([2*x[0], 100*x[1]])  # grad of x^2 + 50*y^2
                x = x - lr * g
                losses.append(x[0]**2 + 50*x[1]**2)
            return losses

        def gd_with_momentum(lr=0.01, beta=0.9, n_steps=100):
            x = np.array([10.0, 10.0])
            v = np.zeros(2)  # velocity initialized to zero
            losses = []
            for _ in range(n_steps):
                g = np.array([2*x[0], 100*x[1]])
                v = beta * v + g       # accumulate velocity
                x = x - lr * v         # step with momentum
                losses.append(x[0]**2 + 50*x[1]**2)
            return losses

        losses_plain = gd_no_momentum()
        losses_mom = gd_with_momentum()

        print("Steps to reach loss < 1.0:")
        plain_s = next((i for i, l in enumerate(losses_plain) if l < 1.0), None)
        mom_s = next((i for i, l in enumerate(losses_mom) if l < 1.0), None)
        print(f"  Vanilla GD:      {plain_s if plain_s else '>100'}")
        print(f"  GD + Momentum:   {mom_s if mom_s else '>100'}")
        print(f"  Final loss (GD):       {losses_plain[-1]:.4f}")
        print(f"  Final loss (Momentum): {losses_mom[-1]:.6f}")


    _run()
    return

@app.cell
def _(mo):
    mo.image(src="../animations/rendered/MomentumVsSGD.gif")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 6. Adaptive Learning Rate Methods

    ### The Core Idea

    Different parameters often live on very different scales. A feature that appears rarely produces sparse, large gradients; a feature that appears constantly produces dense, small gradients. A single global learning rate can't serve both well. Adaptive methods give each parameter its own effective learning rate.

    ### AdaGrad

    Accumulate the sum of squared gradients for each parameter and scale the learning rate inversely:

    $$s_{t+1} = s_t + g_t \odot g_t$$
    $$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{s_{t+1}} + \epsilon} \odot g_t$$

    where $\odot$ is element-wise multiplication, and $\epsilon \approx 10^{-8}$ prevents division by zero.

    **The good:** Parameters with large accumulated gradients get a smaller effective learning rate (they've already learned a lot), while parameters with small accumulated gradients retain a larger effective learning rate (they still need to learn).

    **The bad:** The accumulator $s_t$ only grows. Over time, the effective learning rate shrinks to zero for all parameters, and learning stalls. This makes AdaGrad poorly suited for non-convex problems like neural network training.

    ### RMSProp

    Geoffrey Hinton's fix: use an **exponential moving average** of squared gradients instead of a cumulative sum:

    $$s_{t+1} = \gamma \, s_t + (1 - \gamma) \, g_t \odot g_t$$
    $$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{s_{t+1}} + \epsilon} \odot g_t$$

    where $\gamma \approx 0.9$. The moving average forgets old gradients, so the effective learning rate doesn't monotonically decay. This was proposed in a Coursera lecture, not a paper — one of the most influential unpublished ideas in deep learning.

    ### Adam (Adaptive Moment Estimation)

    Adam combines momentum (first moment) with RMSProp (second moment), plus bias correction:

    $$m_{t+1} = \beta_1 m_t + (1 - \beta_1) g_t \quad \text{(momentum)}$$
    $$s_{t+1} = \beta_2 s_t + (1 - \beta_2) g_t \odot g_t \quad \text{(RMSProp)}$$
    $$\hat{m} = \frac{m_{t+1}}{1 - \beta_1^{t+1}}, \quad \hat{s} = \frac{s_{t+1}}{1 - \beta_2^{t+1}} \quad \text{(bias correction)}$$
    $$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{s}} + \epsilon} \odot \hat{m}$$

    ```
    Algorithm: Adam
    ─────────────────
    Input: initial θ₀, α = 0.001, β₁ = 0.9, β₂ = 0.999, ε = 1e-8
    m ← 0, s ← 0, t ← 0
    repeat:
        t ← t + 1
        g ← ∇f(θ)              # stochastic gradient
        m ← β₁·m + (1−β₁)·g    # update first moment
        s ← β₂·s + (1−β₂)·g²   # update second moment (element-wise)
        m_hat ← m / (1 − β₁^t)      # bias-corrected first moment
        s_hat ← s / (1 − β₂^t)      # bias-corrected second moment
        θ ← θ − α · m_hat / (√s_hat + ε)
    ```

    The bias correction terms compensate for the fact that $m$ and $s$ are initialized at zero and are biased toward zero in early iterations.

    ### When to Use What

    - **Adam** is the default choice. It works well across a wide range of problems with minimal hyperparameter tuning. Start here.
    - **SGD + momentum** often achieves better *generalization* on tasks where you can afford to tune the learning rate schedule carefully (e.g., image classification with ResNets). Many state-of-the-art results use SGD + momentum + cosine schedule.
    - **AdaGrad** is useful for sparse problems (NLP with large vocabularies, recommender systems).

    See [Murphy PML1 8.4](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf) for a comparative treatment of all these methods.
    """)
    return


@app.cell
def _():
    def _run():

        # Adam from scratch on f(x,y) = x^2 + 50*y^2
        def adam_optimize(f_grad, x0, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, n_steps=100):
            x = x0.copy()
            m = np.zeros_like(x)  # first moment (mean of gradients)
            s = np.zeros_like(x)  # second moment (mean of squared gradients)
            losses = []
            for t in range(1, n_steps + 1):
                g = f_grad(x)
                m = beta1 * m + (1 - beta1) * g       # update biased first moment
                s = beta2 * s + (1 - beta2) * g**2     # update biased second moment
                m_hat = m / (1 - beta1**t)             # bias correction
                s_hat = s / (1 - beta2**t)             # bias correction
                x = x - lr * m_hat / (np.sqrt(s_hat) + eps)  # adaptive step
                losses.append(x[0]**2 + 50*x[1]**2)
            return losses

        grad_f = lambda x: np.array([2*x[0], 100*x[1]])
        x0 = np.array([10.0, 10.0])

        losses_adam = adam_optimize(grad_f, x0, lr=0.5)
        print(f"Adam: loss after 20 steps = {losses_adam[19]:.6f}")
        print(f"Adam: loss after 50 steps = {losses_adam[49]:.6f}")
        print(f"Adam: loss after 100 steps = {losses_adam[99]:.8f}")


    _run()
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Interactive: Optimizer Comparison

    Select an optimizer and watch it navigate the loss surface of $f(x, y) = x^2 + 4y^2$.
    """)
    return


@app.cell
def _(mo):
    optimizer_dropdown = mo.ui.dropdown(
        options=["SGD", "Momentum", "Adam"],
        value="SGD",
        label="Optimizer",
    )
    optimizer_dropdown
    return (optimizer_dropdown,)


@app.cell
def _(optimizer_dropdown):
    def _run():

        def run_optimizer(name, n_steps=100, lr=0.05):
            """Run optimizer on f(x,y) = x^2 + 4y^2."""
            x, y = 4.0, 4.0
            path = [(x, y)]

            # State variables
            vx, vy = 0.0, 0.0  # velocity (momentum)
            mx, my = 0.0, 0.0  # first moment (Adam)
            sx, sy = 0.0, 0.0  # second moment (Adam)
            beta1, beta2, eps = 0.9, 0.999, 1e-8

            for t in range(1, n_steps + 1):
                gx = 2 * x
                gy = 8 * y

                if name == "SGD":
                    x -= lr * gx
                    y -= lr * gy
                elif name == "Momentum":
                    vx = 0.9 * vx + gx
                    vy = 0.9 * vy + gy
                    x -= lr * vx
                    y -= lr * vy
                elif name == "Adam":
                    mx = beta1 * mx + (1 - beta1) * gx
                    my = beta1 * my + (1 - beta1) * gy
                    sx = beta2 * sx + (1 - beta2) * gx**2
                    sy = beta2 * sy + (1 - beta2) * gy**2
                    mx_hat = mx / (1 - beta1**t)
                    my_hat = my / (1 - beta1**t)
                    sx_hat = sx / (1 - beta2**t)
                    sy_hat = sy / (1 - beta2**t)
                    x -= lr * mx_hat / (np.sqrt(sx_hat) + eps)
                    y -= lr * my_hat / (np.sqrt(sy_hat) + eps)

                path.append((x, y))
                if x**2 + y**2 > 1e6:
                    break

            return np.array(path)

        _name = optimizer_dropdown.value
        _lr_map = {"SGD": 0.05, "Momentum": 0.02, "Adam": 0.3}
        _path = run_optimizer(_name, lr=_lr_map[_name])

        fig_opt, ax_opt = plt.subplots(figsize=(8, 6))

        _xx = np.linspace(-6, 6, 200)
        _yy = np.linspace(-6, 6, 200)
        _XX, _YY = np.meshgrid(_xx, _yy)
        _ZZ = _XX**2 + 4 * _YY**2

        ax_opt.contour(_XX, _YY, _ZZ, levels=20, cmap='viridis', alpha=0.5)
        ax_opt.contourf(_XX, _YY, _ZZ, levels=20, cmap='viridis', alpha=0.2)

        _p = _path[np.abs(_path).max(axis=1) < 10]
        if len(_p) > 1:
            ax_opt.plot(_p[:, 0], _p[:, 1], 'ro-', markersize=3, linewidth=1)
            ax_opt.plot(_p[0, 0], _p[0, 1], 'gs', markersize=10, label='Start')
            ax_opt.plot(_p[-1, 0], _p[-1, 1], 'r*', markersize=15, label='End')

        _final = _path[-1][0]**2 + 4 * _path[-1][1]**2
        ax_opt.set_title(f"{_name} on f(x,y) = x² + 4y² | Steps={len(_path)-1} | Final loss={_final:.4f}")
        ax_opt.set_xlabel("x")
        ax_opt.set_ylabel("y")
        ax_opt.set_xlim(-6, 6)
        ax_opt.set_ylim(-6, 6)
        ax_opt.legend()
        ax_opt.set_aspect('equal')
        plt.tight_layout()
        fig_opt


    _run()
    return

@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 7. Constrained Optimization

    Most of ML uses *unconstrained* optimization (possibly with regularization). But understanding constrained optimization connects several important ideas.

    ### Lagrange Multipliers

    Suppose you want to minimize $f(\theta)$ subject to $g(\theta) = 0$. The **Lagrangian** is:

    $$\mathcal{L}(\theta, \lambda) = f(\theta) + \lambda \, g(\theta)$$

    At the constrained optimum, $\nabla_\theta f = -\lambda \nabla_\theta g$ — the gradient of the objective is parallel to the gradient of the constraint. The scalar $\lambda$ is called the **Lagrange multiplier**, and it measures the sensitivity of the optimal value to relaxing the constraint.

    ### The Connection to Regularization

    This is the key insight for ML: constrained optimization and penalized optimization are **dual** formulations of the same problem. Specifically:

    $$\min_\theta \mathcal{L}(\theta) \quad \text{s.t.} \|\theta\|^2 \leq C$$

    is equivalent (for some corresponding $\lambda$) to:

    $$\min_\theta \mathcal{L}(\theta) + \lambda \|\theta\|^2$$

    The first is $\ell_2$-constrained optimization. The second is $\ell_2$-regularized optimization (Ridge regression). They give the same solutions — there's a one-to-one mapping between the constraint bound $C$ and the regularization weight $\lambda$.

    This is why regularization "works" — it's enforcing a constraint on model complexity, just expressed differently.

    ### KKT Conditions

    For inequality constraints $g_i(\theta) \leq 0$, the **Karush-Kuhn-Tucker (KKT) conditions** generalize Lagrange multipliers. The key addition is **complementary slackness**: $\lambda_i g_i(\theta) = 0$, meaning either the constraint is active ($g_i = 0$) or the multiplier is zero ($\lambda_i = 0$). You'll encounter KKT conditions prominently when we derive Support Vector Machines.

    See [Boyd, Ch 5](file:///C:/Users/landa/ml-course/textbooks/Boyd-ConvexOptimization.pdf) for a complete treatment of duality and KKT conditions.
    """)
    return


@app.cell
def _():
    def _run():

        # Lagrange multipliers: minimize f(x,y) = x + y subject to x^2 + y^2 = 1
        # Lagrangian: L = x + y + lambda*(x^2 + y^2 - 1)
        # Grad_x L = 1 + 2*lambda*x = 0 => x = -1/(2*lambda)
        # Grad_y L = 1 + 2*lambda*y = 0 => y = -1/(2*lambda)
        # Constraint: x^2 + y^2 = 1 => 2/(4*lambda^2) = 1 => lambda = +/- 1/sqrt(2)

        lam = 1 / np.sqrt(2)  # take positive root (minimizer)
        x_star = -1 / (2 * lam)
        y_star = -1 / (2 * lam)

        print("Minimize f(x,y) = x + y  subject to  x^2 + y^2 = 1")
        print(f"Solution: x* = {x_star:.4f}, y* = {y_star:.4f}")
        print(f"f(x*,y*) = {x_star + y_star:.4f}")
        print(f"Constraint check: x*^2 + y*^2 = {x_star**2 + y_star**2:.4f}")
        print(f"Lagrange multiplier lambda = {lam:.4f}")


    _run()
    return

@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 8. Second-Order Methods

    Everything so far uses only the gradient (first derivative). Second-order methods also use curvature (the Hessian).

    ### Newton's Method

    Newton's method uses a second-order Taylor approximation to the objective and jumps directly to the minimum of that approximation:

    $$\theta_{t+1} = \theta_t - [\nabla^2 f(\theta_t)]^{-1} \nabla f(\theta_t)$$

    For a quadratic function, this converges in **one step** (the Taylor approximation is exact). For general smooth functions, it converges **quadratically** near the optimum — each step roughly squares the accuracy. Compare this to gradient descent's linear convergence.

    **The cost:** You need to compute and invert the $d \times d$ Hessian at each step. Computing the Hessian is $O(d^2)$ in space and inverting it is $O(d^3)$ in time. For a neural network with millions of parameters, this is completely infeasible.

    ### Quasi-Newton Methods (L-BFGS)

    Quasi-Newton methods approximate the inverse Hessian using only gradient information accumulated over the last few steps. **L-BFGS** (Limited-memory BFGS) is the most practical variant — it stores only the last $m$ gradient differences (typically $m \approx 10$) and uses them to implicitly represent the inverse Hessian approximation.

    L-BFGS is the method of choice for medium-scale smooth optimization problems (thousands to low millions of parameters) where you can afford full-batch gradients — logistic regression, CRFs, some classical NLP models. It typically converges in far fewer iterations than gradient descent.

    ### Why Deep Learning Uses First-Order Methods

    Three reasons:
    1. **Scale:** Modern networks have billions of parameters. Even storing the Hessian ($d^2$ entries) is impossible.
    2. **Stochasticity:** Second-order methods don't combine naturally with mini-batch gradients. The Hessian estimate from a mini-batch is extremely noisy.
    3. **Non-convexity:** Newton's method can be attracted to saddle points (where the Hessian has negative eigenvalues), making it unreliable in non-convex landscapes.

    See [Boyd 9.5](file:///C:/Users/landa/ml-course/textbooks/Boyd-ConvexOptimization.pdf) for Newton's method and [MML 7.3](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) for second-order methods in the ML context.
    """)
    return


@app.cell
def _():
    def _run():

        # Newton's method vs GD on f(x) = 0.5 * x^T A x
        # Newton: x <- x - H^{-1} g  (one step for quadratics!)
        A = np.array([[10.0, 2.0], [2.0, 3.0]])
        x0 = np.array([10.0, 10.0])

        # Newton's method
        x_newton = x0.copy()
        g = A @ x_newton
        H_inv = np.linalg.inv(A)
        x_newton = x_newton - H_inv @ g  # single Newton step
        print(f"Newton after 1 step:  x = {x_newton.round(6)}, loss = {0.5 * x_newton @ A @ x_newton:.6f}")

        # Gradient descent needs many steps
        x_gd = x0.copy()
        L = np.max(np.linalg.eigvalsh(A))
        lr = 1.0 / L
        for t in range(50):
            x_gd = x_gd - lr * (A @ x_gd)
        print(f"GD after 50 steps:    x = {x_gd.round(6)}, loss = {0.5 * x_gd @ A @ x_gd:.6f}")
        print("\nNewton solves quadratics in one step; GD needs many iterations.")


    _run()
    return

@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 9. The Optimization Landscape of Neural Networks (Preview)

    Everything above sets the stage for understanding why training neural networks works *at all*, despite being a non-convex optimization problem.

    ### Non-Convex but Not Catastrophic

    The loss surface of a neural network is highly non-convex. But empirical and theoretical evidence suggests it's more benign than a generic non-convex function:

    - **Symmetries create equivalent minima.** Permuting hidden neurons gives functionally identical networks with different parameter values. So there are many global minima, and they're spread throughout parameter space.
    - **Most critical points are saddle points, not local minima.** In high-dimensional spaces, for a critical point to be a local minimum, the Hessian must be positive definite — all $d$ eigenvalues must be positive. Random matrix theory suggests this becomes exponentially unlikely as $d$ grows. Most critical points have a mix of positive and negative eigenvalues (saddle points), which gradient-based methods can escape.

    ### Saddle Points vs. Local Minima

    In low dimensions (2D, 3D), local minima are the main worry. In high dimensions (millions of parameters), **saddle points** are far more prevalent. SGD with momentum naturally escapes saddle points — the noise and momentum push you off the saddle along directions of negative curvature.

    The local minima that *do* exist tend to have loss values very close to the global minimum. So even getting "stuck" in a local minimum isn't catastrophic — you're almost as good as the best possible solution.

    ### Overparameterization Helps

    A surprising finding: making the network *larger* (more parameters) often makes optimization *easier*, not harder. Overparameterized networks (more parameters than data points) tend to have:
    - More paths through parameter space from initialization to good solutions
    - Smoother loss landscapes
    - No spurious local minima (under certain conditions)

    This is one of the great puzzles of modern deep learning — classical theory says overparameterization should cause overfitting, but in practice it often helps both optimization and generalization.

    For a probabilistic perspective on optimization in the context of neural networks, see [Bishop 5.2](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) and [Murphy PML1 Ch 13](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf).
    """)
    return


@app.cell
def _():
    def _run():

        # Saddle points vs local minima in high dimensions
        # At a critical point, the Hessian has d eigenvalues.
        # Probability ALL are positive (local min) shrinks exponentially with d.

        rng = np.random.default_rng(7)
        for d in [2, 10, 50, 200]:
            # Simulate: random symmetric matrix eigenvalues (like a random critical point)
            n_trials = 1000
            n_local_min = 0
            for _ in range(n_trials):
                eigs = rng.standard_normal(d)  # random eigenvalues
                if np.all(eigs > 0):       # local min requires ALL positive
                    n_local_min += 1
            frac = n_local_min / n_trials
            print(f"d={d:>3d}: P(local min) ~ {frac:.4f}  (saddle points dominate as d grows)")


    _run()
    return

@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Summary: The Optimization Toolkit

    | Method | Order | Per-step cost | Memory | Best for |
    |---|---|---|---|---|
    | Gradient Descent | 1st | $O(Nd)$ | $O(d)$ | Small datasets, convex problems |
    | SGD | 1st | $O(Bd)$ | $O(d)$ | Large datasets |
    | SGD + Momentum | 1st | $O(Bd)$ | $O(d)$ | Ill-conditioned landscapes |
    | Adam | 1st | $O(Bd)$ | $O(d)$ | General default |
    | L-BFGS | ~2nd | $O(Nd)$ | $O(md)$ | Medium-scale, smooth, full-batch |
    | Newton | 2nd | $O(Nd^2)$ | $O(d^2)$ | Small-scale, smooth |

    Where $N$ = dataset size, $B$ = batch size, $d$ = number of parameters, $m$ = L-BFGS memory size.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Practice Exercises

    **Conceptual:**

    1. Explain why adding $\ell_2$ regularization to a convex loss function makes it strongly convex. What does this imply about the uniqueness of the solution?

    2. Suppose you're training a linear regression model and a neural network on the same dataset. Both use gradient descent. Why can you guarantee that the linear regression will converge to the global optimum but not the neural network?

    3. The condition number $\kappa = L/m$ controls gradient descent convergence speed. What does a large $\kappa$ mean geometrically about the loss surface? Which method from this module is specifically designed to handle large $\kappa$?

    4. Adam uses bias correction terms $1/(1 - \beta_1^t)$ and $1/(1 - \beta_2^t)$. Why are these necessary? What would happen in the first few iterations without them? (Hint: what are $m$ and $s$ initialized to?)

    5. Explain the connection between the constrained problem $\min \mathcal{L}(\theta)$ s.t. $\|\theta\|_1 \leq C$ and $\ell_1$-regularized optimization (Lasso). Why does $\ell_1$ produce sparse solutions while $\ell_2$ does not? (Hint: think about the shape of the constraint region.)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Computational (pen and paper):**

    6. Consider $f(x) = \frac{1}{2}x^\top A x - b^\top x$ where $A = \begin{pmatrix} 10 & 0 \\ 0 & 1 \end{pmatrix}$ and $b = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$. Starting from $x_0 = (10, 10)^\top$:
       - What is the condition number?
       - Perform 3 iterations of gradient descent with $\alpha = 1/L$ where $L$ is the largest eigenvalue of $A$.
       - How does the convergence differ between the two coordinates?
    """)
    return


@app.cell
def _():
    def _run():

        # Exercise 6: Gradient descent on quadratic
        A = np.array([[10, 0], [0, 1]])
        b = np.array([0, 0])
        x = np.array([10.0, 10.0])

        L = 10  # largest eigenvalue
        m = 1   # smallest eigenvalue
        kappa = L / m
        alpha = 1 / L

        print(f"Condition number kappa = {kappa}")
        print(f"Learning rate alpha = 1/L = {alpha}")
        print(f"\nInitial x = {x}")
        print(f"Initial f(x) = {0.5 * x @ A @ x - b @ x}")

        for t in range(1, 4):
            grad = A @ x - b
            x = x - alpha * grad
            f_val = 0.5 * x @ A @ x - b @ x
            print(f"\nStep {t}: x = [{x[0]:.4f}, {x[1]:.4f}], f(x) = {f_val:.4f}")
            print(f"  x1 shrinks by factor {1 - alpha * 10:.1f}, x2 shrinks by factor {1 - alpha * 1:.1f}")


    _run()
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    7. Verify that the Hessian of the logistic regression loss $\mathcal{L}(\theta) = -\sum_{i} [y_i \log \sigma(\theta^\top x_i) + (1-y_i)\log(1 - \sigma(\theta^\top x_i))]$ is positive semi-definite, confirming convexity. (Hint: the Hessian is $X^\top D X$ where $D$ is diagonal with $D_{ii} = \sigma(\theta^\top x_i)(1 - \sigma(\theta^\top x_i))$.)
    """)
    return


@app.cell
def _():
    def _run():

        # Verify numerically that the Hessian of logistic regression is PSD
        rng = np.random.default_rng(42)
        n, d = 50, 3
        X = rng.standard_normal((n, d))
        theta = rng.standard_normal(d)

        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        # Compute Hessian: H = X^T D X
        z = X @ theta
        s = sigmoid(z)
        D = np.diag(s * (1 - s))
        H = X.T @ D @ X

        # Check eigenvalues
        eigenvalues = np.linalg.eigvalsh(H)
        print(f"Hessian eigenvalues: {eigenvalues}")
        print(f"All non-negative? {np.all(eigenvalues >= -1e-10)}")
        print(f"=> Logistic regression loss is convex!")


    _run()
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    8. Write out three iterations of Adam on the 1D function $f(x) = x^2$ starting from $x_0 = 10$ with $\alpha = 0.1$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$. Track $m$, $s$, $\hat{m}$, $\hat{s}$, and $x$ at each step.
    """)
    return


@app.cell
def _():
    def _run():

        # Exercise 8: Adam iterations on f(x) = x^2
        x = 10.0
        alpha = 0.1
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        m = 0.0
        s = 0.0

        print(f"Initial: x = {x:.6f}, f(x) = {x**2:.6f}")
        print()

        for t in range(1, 4):
            g = 2 * x  # gradient of x^2
            m = beta1 * m + (1 - beta1) * g
            s = beta2 * s + (1 - beta2) * g**2
            m_hat = m / (1 - beta1**t)
            s_hat = s / (1 - beta2**t)
            x = x - alpha * m_hat / (np.sqrt(s_hat) + eps)

            print(f"Step {t}:")
            print(f"  gradient = {g:.6f}")
            print(f"  m = {m:.6f}, s = {s:.6f}")
            print(f"  m_hat = {m_hat:.6f}, s_hat = {s_hat:.6f}")
            print(f"  x = {x:.6f}, f(x) = {x**2:.6f}")
            print()


    _run()
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Code It: Implementation Exercises

    Put the optimization theory into practice. Each exercise gives you a problem and starter code — fill in the missing pieces.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Exercise 1: Implement Gradient Descent for Linear Regression

    Implement GD to fit a linear regression model. The loss is $\mathcal{L}(\theta) = \frac{1}{N}\|y - X\theta\|^2$ and the gradient is $\nabla \mathcal{L} = -\frac{2}{N} X^\top(y - X\theta)$.

    Compare your result against the closed-form solution $\theta^* = (X^\top X)^{-1} X^\top y$.
    """)
    return


@app.cell
def _():
    def _run():

        rng = np.random.default_rng(99)
        N, d = 100, 3
        X = rng.standard_normal((N, d))
        theta_true = np.array([2.0, -1.0, 0.5])
        y = X @ theta_true + 0.2 * rng.standard_normal(N)

        # TODO: implement gradient descent
        theta = np.zeros(d)
        lr = 0.01
        for t in range(500):
            # Compute gradient: grad = -(2/N) * X^T @ (y - X @ theta)
            grad = ...  # FILL IN
            theta = ...  # FILL IN: GD update

        # Closed-form solution for comparison
        theta_exact = np.linalg.solve(X.T @ X, X.T @ y)

        print(f"Your GD result:      {theta.round(4) if not isinstance(theta, type(...)) else 'NOT IMPLEMENTED'}")
        print(f"Closed-form result:  {theta_exact.round(4)}")
        print(f"True theta:          {theta_true}")


    _run()
    return

@app.cell
def _(mo):
    mo.md(r"""
    ### Exercise 2: Implement SGD with Mini-Batches

    Modify your GD implementation to use mini-batch stochastic gradients. Run for 3 epochs and compare the convergence path to full-batch GD.
    """)
    return


@app.cell
def _():
    def _run():

        rng = np.random.default_rng(42)
        N, d = 200, 3
        X = rng.standard_normal((N, d))
        theta_true = np.array([1.0, -2.0, 3.0])
        y = X @ theta_true + 0.3 * rng.standard_normal(N)

        theta = np.zeros(d)
        lr = 0.01
        batch_size = 32

        losses = []
        for epoch in range(3):
            # TODO: shuffle the data each epoch
            perm = ...  # FILL IN: random permutation of indices

            for start in range(0, N, batch_size):
                # TODO: select mini-batch
                idx = ...  # FILL IN: slice from perm
                X_batch = ...  # FILL IN
                y_batch = ...  # FILL IN

                # TODO: compute mini-batch gradient and update
                grad = ...  # FILL IN
                theta = ...  # FILL IN

                loss = np.mean((y - X @ theta) ** 2)
                losses.append(loss)

        print(f"Final theta: {theta.round(4) if not isinstance(theta, type(...)) else 'NOT IMPLEMENTED'}")
        print(f"True theta:  {theta_true}")
        print(f"Final loss:  {losses[-1]:.4f}" if losses else "No losses recorded")


    _run()
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 3: Implement Momentum from Scratch

    Add momentum to gradient descent and compare convergence on an ill-conditioned quadratic $f(x) = \frac{1}{2} x^\top A x$ where $A$ has condition number 50.
    """)
    return


@app.cell
def _():
    def _run():

        A = np.diag([1.0, 50.0])  # condition number = 50
        x0 = np.array([10.0, 10.0])
        lr = 0.01
        beta = 0.9
        n_steps = 200

        # Vanilla GD
        x_gd = x0.copy()
        losses_gd = []
        for _ in range(n_steps):
            g = A @ x_gd
            x_gd = x_gd - lr * g
            losses_gd.append(0.5 * x_gd @ A @ x_gd)

        # TODO: GD with momentum
        x_mom = x0.copy()
        v = np.zeros(2)  # velocity
        losses_mom = []
        for _ in range(n_steps):
            g = A @ x_mom
            # TODO: update velocity and parameters
            v = ...  # FILL IN: v = beta * v + g
            x_mom = ...  # FILL IN: x = x - lr * v
            losses_mom.append(0.5 * x_mom @ A @ x_mom)

        print(f"After {n_steps} steps:")
        print(f"  Vanilla GD loss:  {losses_gd[-1]:.6f}")
        print(f"  Momentum loss:    {losses_mom[-1]:.6f}" if not isinstance(v, type(...)) else "  Momentum: NOT IMPLEMENTED")


    _run()
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 4: Implement Adam from Scratch

    Implement the full Adam optimizer and test it on the Rosenbrock function $f(x, y) = (1-x)^2 + 100(y-x^2)^2$, a classic non-convex test function.
    """)
    return


@app.cell
def _():

    def rosenbrock(x):
        """f(x,y) = (1-x)^2 + 100*(y-x^2)^2. Minimum at (1,1)."""
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

    def rosenbrock_grad(x):
        """Gradient of the Rosenbrock function."""
        dx = -2*(1 - x[0]) + 100 * 2*(x[1] - x[0]**2)*(-2*x[0])
        dy = 100 * 2*(x[1] - x[0]**2)
        return np.array([dx, dy])

    x = np.array([-1.0, 1.0])
    lr, beta1, beta2, eps = 0.001, 0.9, 0.999, 1e-8
    m = np.zeros(2)
    s = np.zeros(2)

    for t in range(1, 5001):
        g = rosenbrock_grad(x)
        # TODO: implement Adam update
        m = ...  # FILL IN: first moment update
        s = ...  # FILL IN: second moment update
        m_hat = ...  # FILL IN: bias-corrected first moment
        s_hat = ...  # FILL IN: bias-corrected second moment
        x = ...  # FILL IN: parameter update

        if t % 1000 == 0:
            f_val = rosenbrock(x) if not isinstance(x, type(...)) else float('nan')
            print(f"Step {t:>5d}: f(x) = {f_val:.6f}, x = {x}" if not isinstance(x, type(...)) else f"Step {t}: NOT IMPLEMENTED")

    print(f"\nTarget: minimum at (1, 1) with f = 0")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 5: Optimizer Showdown

    Run vanilla GD, momentum, and Adam on the same problem and plot the loss curves. Use $f(x) = \frac{1}{2} x^\top A x$ with $A = \text{diag}(1, 5, 25, 50, 100)$.
    """)
    return


@app.cell
def _():
    def _run():

        A = np.diag([1.0, 5.0, 25.0, 50.0, 100.0])
        x0 = np.ones(5) * 10.0
        n_steps = 300

        def run_all_optimizers(A, x0, n_steps):
            results = {}
            d = len(x0)

            # --- Vanilla GD ---
            x = x0.copy()
            L = np.max(np.diag(A))
            lr_gd = 1.0 / L
            losses = []
            for _ in range(n_steps):
                x = x - lr_gd * (A @ x)
                losses.append(0.5 * x @ A @ x)
            results["GD"] = losses

            # --- Momentum ---
            # TODO: implement and collect losses
            # results["Momentum"] = losses

            # --- Adam ---
            # TODO: implement and collect losses
            # results["Adam"] = losses

            return results

        results = run_all_optimizers(A, x0, n_steps)

        fig, ax = plt.subplots(figsize=(8, 5))
        for name, losses in results.items():
            ax.semilogy(losses, label=name)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss (log scale)")
        ax.set_title("Optimizer Showdown on ill-conditioned quadratic")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig


    _run()
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Further Reading

    - [Boyd & Vandenberghe, Ch 1–3](file:///C:/Users/landa/ml-course/textbooks/Boyd-ConvexOptimization.pdf): The definitive reference on convexity — definitions, properties, and examples
    - [Boyd & Vandenberghe, Ch 9](file:///C:/Users/landa/ml-course/textbooks/Boyd-ConvexOptimization.pdf): Unconstrained minimization — gradient descent, Newton's method, convergence theory
    - [MML, Ch 7](file:///C:/Users/landa/ml-course/textbooks/MML.pdf): Continuous optimization with an ML focus — gradient descent, convexity, and connections to model fitting
    - [Murphy PML1, Ch 8](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf): Optimization algorithms used in ML, including Adam and learning rate schedules
    - [Bishop PRML, 5.2](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf): Optimization in the context of neural network training
    """)
    return


if __name__ == "__main__":
    app.run()
