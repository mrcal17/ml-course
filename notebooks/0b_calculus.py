import marimo

app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Calculus Refresh

    ## Why Calculus Matters for ML

    Let me be direct: calculus is the language that machine learning is written in. Not in the sense that you'll be hand-computing integrals all day -- you won't -- but in the sense that if you don't understand what a gradient is, you're going to be pattern-matching formulas without any idea why they work.

    Three pillars of ML rest squarely on calculus:

    1. **Optimization.** Training a model means minimizing a loss function. Gradient descent -- the workhorse algorithm -- is literally "walk downhill on the loss surface." That requires derivatives.
    2. **Probability.** Continuous probability distributions are defined via integrals. Marginalizing out variables, computing expectations, deriving maximum likelihood estimators -- all calculus.
    3. **Neural networks.** Backpropagation is the chain rule applied systematically through a computational graph. That's it. If you understand the multivariable chain rule, you understand backprop.

    This lecture assumes you once knew single-variable calculus well and taught yourself the multivariable basics. We're going to rebuild that foundation, but this time with ML as the motivating context. Everything here serves a purpose downstream.

    For a thorough treatment of the mathematics, see [MML Chapter 5: Vector Calculus](file:///C:/Users/landa/ml-course/textbooks/MML.pdf).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 1. Single-Variable Review

    You've done this before, so I'll be quick. The goal is to make sure the machinery is loaded back into working memory.

    ### Derivatives as Rates of Change

    The derivative $f'(x)$ tells you how fast $f$ changes as $x$ changes. Formally:

    $$f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$$

    In ML terms: if $x$ is a model parameter and $f(x)$ is the loss, then $f'(x)$ tells you whether increasing $x$ makes the loss go up or down, and by how much.

    ### Core Rules

    | Rule | Formula |
    |------|---------|
    | Power rule | $\frac{d}{dx} x^n = n x^{n-1}$ |
    | Product rule | $(fg)' = f'g + fg'$ |
    | Chain rule | $\frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x)$ |
    | Quotient rule | $\left(\frac{f}{g}\right)' = \frac{f'g - fg'}{g^2}$ |

    The **chain rule** is the one you'll use most. It says: to differentiate a composition, differentiate the outer function, then multiply by the derivative of the inner function. You'll see this scaled up to hundreds of dimensions in backpropagation.

    ### Common Derivatives

    | Function | Derivative | ML relevance |
    |----------|-----------|--------------|
    | $x^n$ | $nx^{n-1}$ | Polynomial features |
    | $e^x$ | $e^x$ | Softmax, probability models |
    | $\ln x$ | $1/x$ | Log-likelihood, cross-entropy |
    | $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\sigma(x)(1 - \sigma(x))$ | Logistic regression, neural net activations |
    | $\tanh(x)$ | $1 - \tanh^2(x)$ | RNN activations |
    | $\max(0, x)$ (ReLU) | $\begin{cases} 1 & x > 0 \\ 0 & x < 0 \end{cases}$ | Most common neural net activation |
    """)
    return


@app.cell
def _():
    import numpy as np

    # Numerical derivative using the limit definition: f'(x) = lim_{h->0} [f(x+h) - f(x)] / h
    def numerical_derivative(f, x, h=1e-7):
        return (f(x + h) - f(x)) / h

    # Power rule: d/dx x^n = n * x^(n-1)
    x = 3.0
    n = 4
    f_power = lambda x: x**n
    analytic = n * x**(n - 1)       # 4 * 3^3 = 108
    numeric = numerical_derivative(f_power, x)
    print(f"Power rule: d/dx x^{n} at x={x}")
    print(f"  Analytic: {analytic:.6f}")
    print(f"  Numeric:  {numeric:.6f}")

    # Chain rule: d/dx f(g(x)) = f'(g(x)) * g'(x)
    # Example: d/dx sin(x^2) = cos(x^2) * 2x
    f_composed = lambda x: np.sin(x**2)
    analytic_chain = np.cos(x**2) * 2 * x  # f'(g(x)) * g'(x)
    numeric_chain = numerical_derivative(f_composed, x)
    print(f"\nChain rule: d/dx sin(x^2) at x={x}")
    print(f"  Analytic: {analytic_chain:.6f}")
    print(f"  Numeric:  {numeric_chain:.6f}")
    return


@app.cell
def _():
    import numpy as np

    # Common ML activation functions and their derivatives
    x_vals = np.linspace(-4, 4, 9)

    # Sigmoid: sigma(x) = 1/(1+e^{-x}), derivative = sigma(x)*(1-sigma(x))
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    sigmoid_deriv = lambda x: sigmoid(x) * (1 - sigmoid(x))

    # tanh: derivative = 1 - tanh^2(x)
    tanh_deriv = lambda x: 1 - np.tanh(x)**2

    # ReLU: derivative = 1 if x>0, else 0
    relu = lambda x: np.maximum(0, x)
    relu_deriv = lambda x: (x > 0).astype(float)

    print("x       | sigmoid  | sig'(x)  | tanh(x)  | tanh'(x) | ReLU   | ReLU'")
    print("-" * 80)
    for xi in x_vals:
        print(f"{xi:+6.2f}  | {sigmoid(xi):.4f}  | {sigmoid_deriv(xi):.4f}  "
              f"| {np.tanh(xi):+.4f}  | {tanh_deriv(xi):.4f}  "
              f"| {relu(xi):.4f} | {relu_deriv(xi):.1f}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### The Exponential and Logarithm: Why They're Everywhere

    You might wonder why $e^x$ and $\ln x$ dominate ML. There are deep reasons:

    **The exponential** $e^x$ is the unique function that is its own derivative. This makes it algebraically clean -- when you differentiate expressions involving $e^x$, things simplify instead of getting worse. The softmax function, the Gaussian distribution, the Boltzmann distribution -- they all use $e^x$ because it converts additive quantities (log-odds, energies) into positive, normalizable quantities.

    **The natural logarithm** $\ln x$ is the inverse of $e^x$, and it turns products into sums: $\ln(ab) = \ln a + \ln b$. This is why we maximize the **log**-likelihood instead of the likelihood -- a product of $n$ probabilities becomes a sum, which is numerically stable and easier to differentiate. Almost every loss function you'll encounter is a log of something.

    **Worked example -- log-likelihood of a Bernoulli:**

    If $p(y=1) = \sigma(w \cdot x)$ where $\sigma$ is the sigmoid, the log-likelihood of a single observation $y \in \{0,1\}$ is:

    $$\ell(w) = y \ln \sigma(w \cdot x) + (1-y) \ln(1 - \sigma(w \cdot x))$$

    Differentiating this (using the chain rule and the fact that $\sigma' = \sigma(1-\sigma)$) gives you the gradient for logistic regression. We'll get there.
    """)
    return


@app.cell
def _():
    import numpy as np

    # Bernoulli log-likelihood: l(w) = y*ln(sigma(w*x)) + (1-y)*ln(1 - sigma(w*x))
    sigmoid = lambda z: 1 / (1 + np.exp(-z))

    w, x, y = 0.5, 2.0, 1  # weight, feature, true label
    z = w * x

    # Log-likelihood value
    log_lik = y * np.log(sigmoid(z)) + (1 - y) * np.log(1 - sigmoid(z))

    # Analytic gradient: dl/dw = (y - sigma(w*x)) * x
    grad_analytic = (y - sigmoid(z)) * x

    # Numerical gradient for verification
    eps = 1e-7
    z_plus = (w + eps) * x
    log_lik_plus = y * np.log(sigmoid(z_plus)) + (1 - y) * np.log(1 - sigmoid(z_plus))
    grad_numeric = (log_lik_plus - log_lik) / eps

    print(f"Bernoulli log-likelihood at w={w}, x={x}, y={y}")
    print(f"  sigma(w*x) = {sigmoid(z):.4f}  (predicted probability)")
    print(f"  log-lik    = {log_lik:.4f}")
    print(f"  dl/dw (analytic) = {grad_analytic:.6f}")
    print(f"  dl/dw (numeric)  = {grad_numeric:.6f}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 2. Partial Derivatives

    ### Functions of Multiple Variables

    In ML, your loss function almost never depends on a single parameter. A linear regression with 100 features has 101 parameters. A neural network might have millions. The loss $L(\theta_1, \theta_2, \ldots, \theta_n)$ is a function of all of them simultaneously.

    Think of it geometrically. With one parameter, the loss is a curve -- you can see hills and valleys. With two parameters, the loss is a **surface** -- a landscape with peaks, valleys, ridges, and saddle points. With more parameters, you can't visualize it, but the mathematics works the same way.

    ### Taking Partial Derivatives

    A partial derivative $\frac{\partial f}{\partial x_i}$ asks: "If I nudge $x_i$ slightly while holding every other variable fixed, how does $f$ change?"

    Mechanically, it's straightforward: treat every other variable as a constant and differentiate with respect to $x_i$ using your single-variable rules.

    **Example 1:** Let $f(x, y) = x^2 y + 3xy^2 - 7y$.

    $$\frac{\partial f}{\partial x} = 2xy + 3y^2$$
    $$\frac{\partial f}{\partial y} = x^2 + 6xy - 7$$

    **Example 2 (ML-flavored):** Mean squared error for simple linear regression $\hat{y} = wx + b$:

    $$L(w, b) = \frac{1}{n} \sum_{i=1}^n (y_i - wx_i - b)^2$$

    $$\frac{\partial L}{\partial w} = \frac{-2}{n} \sum_{i=1}^n x_i(y_i - wx_i - b)$$

    $$\frac{\partial L}{\partial b} = \frac{-2}{n} \sum_{i=1}^n (y_i - wx_i - b)$$

    These are exactly the update rules for gradient descent on linear regression. You'll derive them from scratch in a later module.

    See [MML Section 5.1: Differentiation of Univariate Functions](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) and [MML Section 5.2: Partial Differentiation and Gradients](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) for the formal treatment.
    """)
    return


@app.cell
def _():
    import numpy as np

    # Partial derivatives of f(x,y) = x^2*y + 3*x*y^2 - 7*y
    def f(x, y):
        return x**2 * y + 3 * x * y**2 - 7 * y

    # Analytic partial derivatives
    df_dx = lambda x, y: 2 * x * y + 3 * y**2       # df/dx
    df_dy = lambda x, y: x**2 + 6 * x * y - 7       # df/dy

    # Numerical partial derivatives (nudge one variable, hold the other)
    h = 1e-7
    x0, y0 = 2.0, 3.0
    num_df_dx = (f(x0 + h, y0) - f(x0, y0)) / h
    num_df_dy = (f(x0, y0 + h) - f(x0, y0)) / h

    print(f"f(x,y) = x^2*y + 3*x*y^2 - 7*y  at ({x0}, {y0})")
    print(f"  df/dx: analytic = {df_dx(x0,y0):.4f}, numeric = {num_df_dx:.4f}")
    print(f"  df/dy: analytic = {df_dy(x0,y0):.4f}, numeric = {num_df_dy:.4f}")
    return


@app.cell
def _():
    import numpy as np

    # MSE partial derivatives for simple linear regression: y_hat = w*x + b
    rng = np.random.default_rng(42)
    n = 50
    x_data = rng.standard_normal(n)
    y_data = 2.5 * x_data + 1.0 + 0.3 * rng.standard_normal(n)  # true w=2.5, b=1.0

    w, b = 1.0, 0.0  # initial guess
    residuals = y_data - w * x_data - b

    # Analytic gradients: dL/dw = (-2/n) * sum(x_i * residual_i)
    dL_dw = (-2 / n) * np.sum(x_data * residuals)
    dL_db = (-2 / n) * np.sum(residuals)

    # Numerical check
    h = 1e-7
    mse = lambda w, b: np.mean((y_data - w * x_data - b)**2)
    num_dL_dw = (mse(w + h, b) - mse(w, b)) / h
    num_dL_db = (mse(w, b + h) - mse(w, b)) / h

    print(f"MSE gradient at w={w}, b={b}:")
    print(f"  dL/dw: analytic = {dL_dw:.6f}, numeric = {num_dL_dw:.6f}")
    print(f"  dL/db: analytic = {dL_db:.6f}, numeric = {num_dL_db:.6f}")
    print(f"\nGradient is negative => increasing w,b would decrease loss")
    print(f"(Makes sense: true values are w=2.5, b=1.0 which are larger)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3. The Gradient

    ### The Gradient Vector

    The gradient collects all partial derivatives into a single vector:

    $$\nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

    This vector has a crucial geometric property: **it points in the direction of steepest ascent** of $f$. Its magnitude tells you how steep that ascent is.

    ### Geometric Intuition

    Imagine standing on a hilly landscape (the loss surface). The gradient at your feet is an arrow that points uphill in the steepest direction. If you're on a flat plateau, the gradient is approximately zero. If you're at a sharp peak, the gradient has large magnitude.

    Now picture contour plots -- the topographic-map view. Contour lines connect points of equal function value. The gradient is always **perpendicular to the contour lines**, pointing toward higher values. Where contour lines are packed tightly together, the gradient is large (steep slope). Where they're spread out, the gradient is small (gentle slope).

    ### Why Gradient Descent Goes Against the Gradient

    This is simple but worth stating clearly. If $\nabla f$ points toward the **steepest increase**, then $-\nabla f$ points toward the **steepest decrease**. Since we want to minimize the loss, we update:

    $$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \nabla f(\mathbf{x}_t)$$

    where $\eta$ is the learning rate. That's gradient descent. The negative sign is the whole idea: go downhill.

    The learning rate $\eta$ controls step size. Too large and you overshoot the minimum; too small and convergence is glacially slow. This tension drives much of the optimizer design in deep learning (Adam, RMSProp, learning rate schedules).

    See [MML Section 5.2.1: Gradients](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) and [Boyd Section 9.3: Gradient Descent Method](file:///C:/Users/landa/ml-course/textbooks/Boyd-ConvexOptimization.pdf).
    """)
    return


@app.cell
def _(mo):
    mo.image(src="../animations/rendered/GradientVector2D.gif")
    return


@app.cell
def _():
    import numpy as np

    # Gradient descent on f(x,y) = (x-3)^2 + (y+1)^2
    # Gradient: [2(x-3), 2(y+1)]
    # Minimum at (3, -1)
    f = lambda x, y: (x - 3)**2 + (y + 1)**2
    grad_f = lambda x, y: np.array([2 * (x - 3), 2 * (y + 1)])

    eta = 0.1  # learning rate
    pos = np.array([0.0, 0.0])

    print(f"Gradient descent: f(x,y) = (x-3)^2 + (y+1)^2, lr={eta}")
    print(f"{'Step':>4}  {'x':>8}  {'y':>8}  {'f(x,y)':>10}  {'|grad|':>8}")
    print("-" * 50)
    for step in range(11):
        g = grad_f(pos[0], pos[1])
        val = f(pos[0], pos[1])
        print(f"{step:4d}  {pos[0]:8.4f}  {pos[1]:8.4f}  {val:10.4f}  {np.linalg.norm(g):8.4f}")
        pos = pos - eta * g  # x_{t+1} = x_t - eta * grad f
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 4. The Chain Rule for Multivariable Functions

    ### Why This Section Matters Most

    If you absorb one thing from this entire lecture, make it this: **backpropagation is the chain rule**. Every time a neural network is trained, the chain rule is applied thousands of times per forward pass, across every layer, every neuron, every parameter. The algorithm is clever about reusing intermediate results (that's the "back-propagation" part), but the mathematical core is purely the chain rule.

    ### The Multivariable Chain Rule

    Suppose $f$ depends on $\mathbf{u}$, and $\mathbf{u}$ depends on $\mathbf{x}$. Then:

    $$\frac{\partial f}{\partial x_j} = \sum_{i} \frac{\partial f}{\partial u_i} \frac{\partial u_i}{\partial x_j}$$

    In words: to see how $f$ changes with $x_j$, sum up all the pathways through which $x_j$ influences $f$, multiplying derivatives along each path.

    ### The Computational Graph Perspective

    A neural network can be drawn as a directed acyclic graph (a computational graph). Each node is an intermediate computation. The chain rule says: to compute $\frac{\partial \text{loss}}{\partial \text{parameter}}$, multiply derivatives along every path from the parameter to the loss, and sum over all paths.

    **Concrete example.** Consider $f = (x + y) \cdot z$. Let $u = x + y$, so $f = u \cdot z$.

    $$\frac{\partial f}{\partial x} = \frac{\partial f}{\partial u} \cdot \frac{\partial u}{\partial x} = z \cdot 1 = z$$

    Now consider a two-layer composition, like a small neural network:

    $$z = \sigma(w_2 \cdot \sigma(w_1 \cdot x + b_1) + b_2)$$

    To find $\frac{\partial z}{\partial w_1}$, you unwind from the outside:

    $$\frac{\partial z}{\partial w_1} = \sigma'(\cdot) \cdot w_2 \cdot \sigma'(\cdot) \cdot x$$

    Each $\sigma'(\cdot)$ is the derivative of the activation function evaluated at that layer's pre-activation value. Each step multiplies by the next layer's weight. This is exactly what backpropagation computes.
    """)
    return


@app.cell
def _(mo):
    mo.image(src="../animations/rendered/ChainRuleComputationGraph.gif")
    return


@app.cell
def _():
    import numpy as np

    # Chain rule through a 2-layer network:
    # z = sigma(w2 * sigma(w1 * x + b1) + b2)
    sigmoid = lambda z: 1 / (1 + np.exp(-z))
    sigmoid_deriv = lambda z: sigmoid(z) * (1 - sigmoid(z))

    x = 1.5
    w1, b1 = 0.8, -0.2
    w2, b2 = 1.2, 0.3

    # Forward pass (save intermediates for backprop)
    z1 = w1 * x + b1          # pre-activation layer 1
    a1 = sigmoid(z1)           # activation layer 1
    z2 = w2 * a1 + b2          # pre-activation layer 2
    output = sigmoid(z2)        # final output

    # Analytic gradient via chain rule: dout/dw1 = sig'(z2) * w2 * sig'(z1) * x
    dout_dw1 = sigmoid_deriv(z2) * w2 * sigmoid_deriv(z1) * x

    # Numerical check
    eps = 1e-7
    z1_p = (w1 + eps) * x + b1
    a1_p = sigmoid(z1_p)
    z2_p = w2 * a1_p + b2
    out_p = sigmoid(z2_p)
    dout_dw1_num = (out_p - output) / eps

    print("2-layer network: z = sigma(w2 * sigma(w1*x + b1) + b2)")
    print(f"  Forward: z1={z1:.4f}, a1={a1:.4f}, z2={z2:.4f}, out={output:.4f}")
    print(f"\n  dout/dw1 (chain rule): {dout_dw1:.8f}")
    print(f"  dout/dw1 (numeric):    {dout_dw1_num:.8f}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### The Jacobian

    When both the input and output are vectors, the chain rule generalizes via the **Jacobian matrix**. If $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian is:

    $$\mathbf{J} = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n} \end{bmatrix}$$

    The Jacobian is an $m \times n$ matrix -- each row is the gradient of one output component with respect to all inputs. The chain rule for vector-valued compositions becomes **matrix multiplication of Jacobians**:

    $$\mathbf{J}_{f \circ g} = \mathbf{J}_f \cdot \mathbf{J}_g$$

    This is elegant and powerful. Backpropagation through a layer is a Jacobian-vector product. In practice, frameworks like PyTorch never form the full Jacobian (too expensive) -- they compute these products efficiently. But understanding that the Jacobian exists and what it represents is essential.

    See [MML Section 5.3: Gradients of Vector-Valued Functions](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) and [MML Section 5.4: Gradients of Matrices](file:///C:/Users/landa/ml-course/textbooks/MML.pdf).
    """)
    return


@app.cell
def _():
    import numpy as np

    # Jacobian of a vector-valued function f: R^2 -> R^3
    # f(x1, x2) = [x1^2 + x2, sin(x1*x2), x1*x2^2]
    def f_vec(x):
        x1, x2 = x
        return np.array([x1**2 + x2, np.sin(x1 * x2), x1 * x2**2])

    # Analytic Jacobian: J[i,j] = df_i/dx_j
    def jacobian_analytic(x):
        x1, x2 = x
        return np.array([
            [2 * x1,              1],                # d(x1^2+x2)/dx
            [x2 * np.cos(x1*x2), x1 * np.cos(x1*x2)],  # d(sin(x1*x2))/dx
            [x2**2,              2 * x1 * x2],       # d(x1*x2^2)/dx
        ])

    # Numerical Jacobian: perturb each input, observe all outputs
    def jacobian_numerical(f, x, h=1e-7):
        n = len(x)
        f0 = f(x)
        m = len(f0)
        J = np.zeros((m, n))
        for j in range(n):
            x_plus = x.copy()
            x_plus[j] += h
            J[:, j] = (f(x_plus) - f0) / h
        return J

    x0 = np.array([1.0, 2.0])
    J_a = jacobian_analytic(x0)
    J_n = jacobian_numerical(f_vec, x0)

    print(f"Jacobian at x = {x0}")
    print(f"\nAnalytic:\n{J_a}")
    print(f"\nNumerical:\n{np.round(J_n, 6)}")
    print(f"\nMax difference: {np.max(np.abs(J_a - J_n)):.2e}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 5. The Hessian

    ### Second-Order Information

    The gradient tells you the slope. The **Hessian** tells you the curvature -- how the slope itself is changing. The Hessian is the matrix of all second partial derivatives:

    $$\mathbf{H} = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots \\ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots \\ \vdots & \vdots & \ddots \end{bmatrix}$$

    For sufficiently smooth functions (which covers almost everything in ML), the Hessian is symmetric: $\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i}$.

    ### Convexity and Positive Semi-Definiteness

    A function is **convex** if its Hessian is positive semi-definite (PSD) everywhere -- meaning $\mathbf{v}^T \mathbf{H} \mathbf{v} \geq 0$ for all vectors $\mathbf{v}$. Equivalently, all eigenvalues of $\mathbf{H}$ are non-negative.

    Why care? Convex functions have a single global minimum. No local minima traps. Linear regression loss is convex (which is why it has a closed-form solution). Logistic regression loss is convex too. Neural network losses are emphatically **not** convex -- which is why training them is so much harder and why we need tricks like momentum, learning rate warmup, and batch normalization.

    ### Why Second-Order Methods Are Expensive but Informative

    **Newton's method** uses the Hessian to take smarter steps:

    $$\mathbf{x}_{t+1} = \mathbf{x}_t - \mathbf{H}^{-1} \nabla f(\mathbf{x}_t)$$

    Instead of taking a fixed-size step in the gradient direction, Newton's method accounts for curvature. In directions where the loss curves gently, it takes large steps; where it curves sharply, it takes small steps. This is much faster than gradient descent near the optimum.

    The problem: for $n$ parameters, the Hessian is $n \times n$. A neural network with 10 million parameters would need a $10^7 \times 10^7$ matrix -- roughly $10^{14}$ entries. That's completely infeasible. This is why most deep learning uses first-order methods (gradient only), with approximations like Adam that try to capture some second-order information cheaply.

    See [MML Section 5.3.1: Higher-Order Derivatives](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) and [Boyd Section 9.5: Newton's Method](file:///C:/Users/landa/ml-course/textbooks/Boyd-ConvexOptimization.pdf).
    """)
    return


@app.cell
def _():
    import numpy as np

    # Hessian and convexity check for f(x,y) = x^2 + 10*y^2
    # Gradient: [2x, 20y]
    # Hessian: [[2, 0], [0, 20]] -- constant, eigenvalues 2 and 20
    H = np.array([[2, 0],
                   [0, 20]])
    eigenvalues = np.linalg.eigvalsh(H)
    is_psd = np.all(eigenvalues >= 0)
    condition_number = eigenvalues.max() / eigenvalues.min()

    print(f"f(x,y) = x^2 + 10*y^2")
    print(f"Hessian:\n{H}")
    print(f"Eigenvalues: {eigenvalues}")
    print(f"PSD (convex)? {is_psd}")
    print(f"Condition number: {condition_number}")
    print(f"\nThe 10:1 eigenvalue ratio explains zig-zagging in gradient descent:")
    print(f"  the y-direction curves 10x steeper than x, so a fixed learning rate")
    print(f"  that works for y overshoots in x, and vice versa.")

    # Newton's method converges in 1 step on quadratics
    x0 = np.array([5.0, 3.0])
    grad = np.array([2 * x0[0], 20 * x0[1]])
    x_newton = x0 - np.linalg.solve(H, grad)  # x - H^{-1} * grad
    print(f"\nNewton's method from {x0}: one step -> {x_newton}  (exact minimum!)")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 6. Vector Calculus Essentials

    ### Matrix Calculus Identities

    When you derive ML algorithms on paper (and you will), you'll repeatedly encounter gradients of expressions involving vectors and matrices. Here are the identities that come up constantly:

    **Gradient of a linear form.** If $f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} = \sum_i w_i x_i$, then:

    $$\nabla_{\mathbf{x}} (\mathbf{w}^T \mathbf{x}) = \mathbf{w}$$

    This should feel intuitive: $\mathbf{w}^T \mathbf{x}$ is linear in $\mathbf{x}$, so its "slope" in each direction $x_i$ is simply $w_i$.

    **Gradient of a quadratic form.** If $f(\mathbf{x}) = \mathbf{x}^T \mathbf{A} \mathbf{x}$, then:

    $$\nabla_{\mathbf{x}} (\mathbf{x}^T \mathbf{A} \mathbf{x}) = (\mathbf{A} + \mathbf{A}^T) \mathbf{x}$$

    When $\mathbf{A}$ is symmetric (which it often is -- think covariance matrices), this simplifies to $2\mathbf{A}\mathbf{x}$.

    **Worked example -- deriving the normal equation for linear regression:**

    The loss is $L(\mathbf{w}) = (\mathbf{y} - \mathbf{X}\mathbf{w})^T(\mathbf{y} - \mathbf{X}\mathbf{w})$. Expand this:

    $$L(\mathbf{w}) = \mathbf{y}^T\mathbf{y} - 2\mathbf{y}^T\mathbf{X}\mathbf{w} + \mathbf{w}^T\mathbf{X}^T\mathbf{X}\mathbf{w}$$

    Now take the gradient with respect to $\mathbf{w}$ using our identities:

    $$\nabla_{\mathbf{w}} L = -2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\mathbf{w}$$

    Set this to zero:

    $$\mathbf{X}^T\mathbf{X}\mathbf{w} = \mathbf{X}^T\mathbf{y}$$
    $$\mathbf{w}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

    That's the normal equation -- the closed-form solution to linear regression. You just derived it using two matrix calculus identities and setting a gradient to zero. This is the kind of derivation you should be able to do fluently.

    **Other useful identities:**

    | Expression | Gradient w.r.t. $\mathbf{x}$ |
    |---|---|
    | $\mathbf{a}^T \mathbf{x}$ | $\mathbf{a}$ |
    | $\mathbf{x}^T \mathbf{x}$ | $2\mathbf{x}$ |
    | $\mathbf{x}^T \mathbf{A} \mathbf{x}$ (A symmetric) | $2\mathbf{A}\mathbf{x}$ |
    | $\|\mathbf{A}\mathbf{x} - \mathbf{b}\|^2$ | $2\mathbf{A}^T(\mathbf{A}\mathbf{x} - \mathbf{b})$ |

    See [MML Section 5.4: Gradients of Matrices](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) and [MML Section 5.5: Useful Identities for Computing Gradients](file:///C:/Users/landa/ml-course/textbooks/MML.pdf).
    """)
    return


@app.cell
def _():
    import numpy as np

    # Verify matrix calculus identities numerically

    # Identity 1: grad(w^T x) = w
    w = np.array([3.0, -1.0, 2.0])
    x = np.array([1.0, 4.0, -2.0])
    f_linear = lambda x: w @ x
    # Numerical gradient
    h = 1e-7
    grad_num = np.array([(f_linear(x + h * np.eye(3)[i]) - f_linear(x)) / h for i in range(3)])
    print(f"grad(w^T x) = w? analytic: {w}, numeric: {np.round(grad_num, 4)}")

    # Identity 2: grad(x^T A x) = (A + A^T) x  for general A
    A = np.array([[2, 1], [3, 4]])  # non-symmetric
    x2 = np.array([1.0, 2.0])
    f_quad = lambda x: x @ A @ x
    grad_analytic = (A + A.T) @ x2
    grad_num2 = np.array([(f_quad(x2 + h * np.eye(2)[i]) - f_quad(x2)) / h for i in range(2)])
    print(f"\ngrad(x^T A x), A non-symmetric:")
    print(f"  (A+A^T)x = {grad_analytic}, numeric = {np.round(grad_num2, 4)}")

    # Normal equation: w* = (X^T X)^{-1} X^T y
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 3))
    y = X @ np.array([1.5, -2.0, 0.5]) + 0.1 * rng.standard_normal(20)
    w_star = np.linalg.solve(X.T @ X, X.T @ y)
    print(f"\nNormal equation solution: w* = {np.round(w_star, 4)}")
    print(f"True weights:                   [1.5, -2.0, 0.5]")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Finite Differences: Numerical Derivatives

    Before moving on, let's build some intuition for derivatives by computing them numerically. The finite difference approximation is:

    $$f'(x) \approx \frac{f(x + \varepsilon) - f(x)}{\varepsilon}$$

    Use the slider below to see how the approximation improves as $\varepsilon \to 0$ (but eventually gets worse due to floating-point precision).
    """)
    return


@app.cell
def _(mo):
    epsilon_slider = mo.ui.slider(start=-5, stop=-1, step=0.5, value=-3, label=r"log₁₀(ε)")
    epsilon_slider
    return (epsilon_slider,)


@app.cell
def _(epsilon_slider, mo):
    import numpy as np
    import matplotlib.pyplot as plt

    # f(x) = x^2, true derivative f'(x) = 2x
    # Evaluate at x = 2, so true derivative = 4
    x0 = 2.0
    true_deriv = 2 * x0  # = 4.0

    eps = 10 ** epsilon_slider.value

    # Finite difference approximation
    f = lambda x: x**2
    approx_deriv = (f(x0 + eps) - f(x0)) / eps
    error = abs(approx_deriv - true_deriv)

    # Plot approximation across a range of epsilon values
    log_eps_range = np.linspace(-15, 0, 200)
    eps_range = 10.0 ** log_eps_range
    approx_range = np.array([(f(x0 + e) - f(x0)) / e for e in eps_range])
    errors = np.abs(approx_range - true_deriv)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: tangent line approximation
    x_plot = np.linspace(0, 4, 200)
    ax1.plot(x_plot, f(x_plot), 'b-', linewidth=2, label=r'$f(x) = x^2$')
    # Secant line
    ax1.plot([x0, x0 + eps], [f(x0), f(x0 + eps)], 'r-o', linewidth=2, markersize=6,
             label=f'Secant (slope={approx_deriv:.6f})')
    # True tangent
    tangent_y = true_deriv * (x_plot - x0) + f(x0)
    ax1.plot(x_plot, tangent_y, 'g--', linewidth=1.5, alpha=0.7,
             label=f'True tangent (slope={true_deriv:.1f})')
    ax1.set_xlim(0, 4)
    ax1.set_ylim(-1, 16)
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title(f'Finite Difference at x={x0}, ε={eps:.1e}')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right: error vs epsilon
    ax2.loglog(eps_range, errors, 'b-', linewidth=2)
    ax2.axvline(eps, color='red', linestyle='--', linewidth=2, label=f'Current ε={eps:.1e}')
    ax2.set_xlabel('ε')
    ax2.set_ylabel('|Approximation error|')
    ax2.set_title('Error vs ε (log-log scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    mo.md(f"""
    **Current settings:** ε = {eps:.1e}

    - True derivative f'(2) = **{true_deriv:.1f}**
    - Finite difference approximation = **{approx_deriv:.10f}**
    - Absolute error = **{error:.2e}**

    Notice: as ε decreases, the error first drops (better approximation) then rises again (floating-point cancellation). The sweet spot is around ε ≈ 10⁻⁸ for 64-bit floats.
    """)
    fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 7. Integration Review

    You need integration less often than differentiation in ML, but it's essential for probability theory. Here's what matters.

    ### Definite Integrals as Areas

    The integral $\int_a^b f(x)\, dx$ gives the signed area under $f$ between $a$ and $b$. In probability, if $f(x) = p(x)$ is a probability density function, then:

    $$P(a \leq X \leq b) = \int_a^b p(x)\, dx$$

    And the total probability axiom requires $\int_{-\infty}^{\infty} p(x)\, dx = 1$.

    **Expectation** is an integral:

    $$\mathbb{E}[X] = \int_{-\infty}^{\infty} x \, p(x)\, dx$$

    **Variance** is too:

    $$\text{Var}(X) = \int_{-\infty}^{\infty} (x - \mu)^2 \, p(x)\, dx$$

    Every time you see $\mathbb{E}[\cdot]$, think "integral weighted by density."

    ### Double Integrals for Joint Distributions

    When you have two random variables with joint density $p(x, y)$, you integrate over both:

    $$P(X \in A, Y \in B) = \int_B \int_A p(x, y)\, dx\, dy$$

    **Marginalization** -- one of the most important operations in probabilistic ML -- is a single integral over the joint:

    $$p(x) = \int_{-\infty}^{\infty} p(x, y)\, dy$$

    This "integrates out" $y$, leaving the marginal distribution of $x$. Bayesian inference uses this constantly: to get $p(\text{prediction})$, you integrate out the model parameters $p(\text{prediction}, \theta)$ over all possible $\theta$.

    ### Change of Variables

    If you transform a random variable $X$ via $Y = g(X)$, the density transforms as:

    $$p_Y(y) = p_X(g^{-1}(y)) \left| \frac{d}{dy} g^{-1}(y) \right|$$

    The absolute value of the derivative of the inverse -- the **Jacobian determinant** -- is a correction factor that accounts for how the transformation stretches or compresses space. In higher dimensions, this generalizes to:

    $$p_{\mathbf{y}}(\mathbf{y}) = p_{\mathbf{x}}(\mathbf{g}^{-1}(\mathbf{y})) \left| \det \mathbf{J}_{g^{-1}} \right|$$

    This shows up in normalizing flows (a type of generative model), in deriving the distribution of transformed random variables, and in understanding why the log-normal distribution has the form it does.

    See [Chan: Integration and Probability](file:///C:/Users/landa/ml-course/textbooks/Chan-Probability.pdf) and [MML Section 6.2: Continuous Probability Distributions](file:///C:/Users/landa/ml-course/textbooks/MML.pdf).
    """)
    return


@app.cell
def _():
    import numpy as np

    # Numerical integration: expectation and variance of a Gaussian
    mu, sigma = 2.0, 1.5

    # Approximate integrals via dense grid (quadrature)
    x = np.linspace(mu - 6*sigma, mu + 6*sigma, 10000)
    dx = x[1] - x[0]
    pdf = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma)**2)

    # Total probability: should be 1
    total_prob = np.sum(pdf) * dx

    # E[X] = integral of x * p(x) dx
    mean_numerical = np.sum(x * pdf) * dx

    # Var(X) = integral of (x - mu)^2 * p(x) dx
    var_numerical = np.sum((x - mu)**2 * pdf) * dx

    print(f"Gaussian(mu={mu}, sigma={sigma})")
    print(f"  Total probability: {total_prob:.6f}  (should be 1)")
    print(f"  E[X]:  numerical = {mean_numerical:.6f}, exact = {mu}")
    print(f"  Var(X): numerical = {var_numerical:.6f}, exact = {sigma**2}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 8. Tying It All Together

    Let me show you how all of these pieces connect in a single ML example.

    **Training logistic regression with gradient descent:**

    1. **The model:** $p(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x})$ where $\sigma(z) = \frac{1}{1+e^{-z}}$ (exponential and logarithm show up).
    2. **The loss:** Negative log-likelihood over $n$ data points (integration's discrete cousin -- summation):
    $$L(\mathbf{w}) = -\sum_{i=1}^n \left[ y_i \ln \sigma(\mathbf{w}^T \mathbf{x}_i) + (1-y_i) \ln(1 - \sigma(\mathbf{w}^T \mathbf{x}_i)) \right]$$
    3. **The gradient** (partial derivatives, chain rule, matrix calculus):
    $$\nabla_{\mathbf{w}} L = -\sum_{i=1}^n (y_i - \sigma(\mathbf{w}^T \mathbf{x}_i)) \mathbf{x}_i = -\mathbf{X}^T(\mathbf{y} - \hat{\mathbf{y}})$$
    4. **The update** (gradient descent -- go against the gradient):
    $$\mathbf{w}_{t+1} = \mathbf{w}_t + \eta \mathbf{X}^T(\mathbf{y} - \hat{\mathbf{y}})$$
    5. **Convergence analysis** (Hessian -- the loss is convex, so we're guaranteed to find the global minimum).

    Every section of this lecture showed up in that one algorithm. That's not a coincidence -- it's why we covered exactly these topics.

    See [ISLR Section 4.3: Logistic Regression](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf) and [Bishop Section 4.3.2: Logistic Regression](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf).
    """)
    return


@app.cell
def _():
    import numpy as np

    # Full logistic regression training from scratch -- tying it all together
    sigmoid = lambda z: 1 / (1 + np.exp(-z))

    # Generate linearly separable 2D data
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 2))
    true_w = np.array([2.0, -1.5])
    y = (X @ true_w > 0).astype(float)

    # Gradient descent on negative log-likelihood
    w = np.zeros(2)
    lr = 0.1
    losses = []

    for step in range(50):
        # Forward: predictions
        y_hat = sigmoid(X @ w)
        # Loss: -sum[y*log(yhat) + (1-y)*log(1-yhat)]
        loss = -np.mean(y * np.log(y_hat + 1e-12) + (1 - y) * np.log(1 - y_hat + 1e-12))
        losses.append(loss)
        # Gradient: -X^T (y - y_hat) / n
        grad = -X.T @ (y - y_hat) / n
        # Update: w <- w - lr * grad
        w = w - lr * grad

    accuracy = np.mean((sigmoid(X @ w) > 0.5) == y)
    print(f"Logistic regression via gradient descent (50 steps, lr={lr})")
    print(f"  Learned w: [{w[0]:.3f}, {w[1]:.3f}]  (true: [2.0, -1.5])")
    print(f"  Final loss: {losses[-1]:.4f}  (started at {losses[0]:.4f})")
    print(f"  Accuracy: {accuracy:.1%}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Practice Exercises

    ### Fundamentals

    1. Compute the derivative of $f(x) = \ln(1 + e^x)$ (this is the softplus function, a smooth approximation to ReLU). Show that $f'(x) = \sigma(x)$.

    2. For $f(x, y) = x^2 e^{-y} + \sin(xy)$, compute $\frac{\partial f}{\partial x}$ and $\frac{\partial f}{\partial y}$.

    3. Write the gradient $\nabla f$ for $f(x_1, x_2, x_3) = x_1^2 + x_1 x_2 + x_3^2 - 2x_3$.

    ### Gradient Descent

    4. Consider $f(x, y) = (x - 3)^2 + (y + 1)^2$. Compute the gradient. Starting from $(0, 0)$ with learning rate $\eta = 0.1$, manually perform 3 steps of gradient descent. Verify you're approaching $(3, -1)$.

    5. For $f(x, y) = x^2 + 10y^2$, compute the gradient and Hessian. Why does gradient descent on this function zig-zag instead of going straight to the minimum? (Hint: look at the eigenvalues of the Hessian -- they differ by a factor of 10.)

    ### Chain Rule and Backpropagation

    6. Let $f(x) = (3x + 1)^4$. Compute $f'(x)$ using the chain rule. Identify the "inner" and "outer" functions.

    7. Consider the computation $L = (y - \sigma(wx + b))^2$ where $\sigma(z) = 1/(1+e^{-z})$. Draw the computational graph and compute $\frac{\partial L}{\partial w}$ by applying the chain rule step by step.

    ### Matrix Calculus

    8. Verify the identity $\nabla_{\mathbf{x}} (\mathbf{x}^T \mathbf{A} \mathbf{x}) = (\mathbf{A} + \mathbf{A}^T)\mathbf{x}$ for the specific case $\mathbf{A} = \begin{bmatrix} 2 & 1 \\ 3 & 4 \end{bmatrix}$ and $\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$. Expand $\mathbf{x}^T \mathbf{A} \mathbf{x}$ as a scalar, take partial derivatives, and check that the result matches.

    9. Derive the gradient of Ridge regression loss $L(\mathbf{w}) = \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2 + \lambda \|\mathbf{w}\|^2$ and find the closed-form solution by setting $\nabla L = 0$. You should get $\mathbf{w}^* = (\mathbf{X}^T\mathbf{X} + \lambda \mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$.

    ### Integration and Probability

    10. The Gaussian density is $p(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$. Show that $\mathbb{E}[X] = \mu$ by computing $\int_{-\infty}^{\infty} x \, p(x)\, dx$. (Hint: use the substitution $u = x - \mu$ and exploit symmetry.)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Code It: Implementation Exercises

    Put the math into practice. Each exercise gives you a skeleton -- fill in the missing pieces.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 1: Gradient Checker

    Write a function that takes any scalar function $f$ and a point $x$, computes the numerical gradient via central differences, and compares it to an analytic gradient you provide. Central differences are more accurate than forward differences:

    $$f'(x) \approx \frac{f(x + h) - f(x - h)}{2h}$$

    Test it on $f(x) = x^3$ at $x = 2$ (analytic derivative: $3x^2 = 12$).
    """)
    return


@app.cell
def _():
    import numpy as np

    def gradient_checker(f, analytic_grad_fn, x, h=1e-5):
        """Compare analytic gradient to central-difference numerical gradient.

        Args:
            f: scalar function f(x) -> float
            analytic_grad_fn: function returning analytic derivative at x
            x: point (float or numpy array) to check at
            h: step size for finite differences

        Returns:
            (analytic_grad, numerical_grad, relative_error)
        """
        analytic = analytic_grad_fn(x)

        # TODO: compute numerical gradient using central differences
        # Hint: (f(x+h) - f(x-h)) / (2*h)
        numerical = ...  # <-- fill this in

        # TODO: compute relative error
        # Hint: |analytic - numerical| / max(|analytic|, |numerical|, 1e-8)
        rel_error = ...  # <-- fill this in

        return analytic, numerical, rel_error

    # Test on f(x) = x^3, f'(x) = 3x^2
    # f = lambda x: x**3
    # f_prime = lambda x: 3 * x**2
    # a, n, err = gradient_checker(f, f_prime, 2.0)
    # print(f"Analytic: {a}, Numerical: {n}, Relative error: {err:.2e}")
    return (gradient_checker,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 2: Gradient Descent with Momentum

    Standard gradient descent can be slow on ill-conditioned problems (like $f(x,y) = x^2 + 50y^2$). Momentum adds a "velocity" term that smooths out oscillations:

    $$\mathbf{v}_{t+1} = \beta \mathbf{v}_t + \nabla f(\mathbf{x}_t)$$
    $$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \, \mathbf{v}_{t+1}$$

    Implement this and compare convergence (number of steps to reach $f < 0.01$) with and without momentum on the ill-conditioned quadratic.
    """)
    return


@app.cell
def _():
    import numpy as np

    def gradient_descent_momentum(grad_fn, x0, lr=0.01, beta=0.9, n_steps=200):
        """Gradient descent with momentum.

        Args:
            grad_fn: function returning gradient at x
            x0: starting point (numpy array)
            lr: learning rate (eta)
            beta: momentum coefficient (0 = no momentum)
            n_steps: number of iterations

        Returns:
            trajectory: list of (x, f_value) at each step
        """
        x = x0.copy()
        v = np.zeros_like(x)  # initialize velocity to zero
        trajectory = []

        for _ in range(n_steps):
            g = grad_fn(x)

            # TODO: update velocity with momentum
            # v = beta * v + g
            v = ...  # <-- fill this in

            # TODO: update position
            # x = x - lr * v
            x = ...  # <-- fill this in

            trajectory.append(x.copy())

        return trajectory

    # Test problem: f(x,y) = x^2 + 50*y^2  (minimum at origin)
    # grad_f = [2x, 100y]
    # f_val = lambda x: x[0]**2 + 50*x[1]**2
    # grad_f = lambda x: np.array([2*x[0], 100*x[1]])
    # start = np.array([5.0, 5.0])
    #
    # traj_no_mom = gradient_descent_momentum(grad_f, start, lr=0.01, beta=0.0)
    # traj_with_mom = gradient_descent_momentum(grad_f, start, lr=0.01, beta=0.9)
    #
    # # Compare: how many steps to reach f < 0.01?
    # for name, traj in [("No momentum", traj_no_mom), ("Momentum=0.9", traj_with_mom)]:
    #     vals = [f_val(t) for t in traj]
    #     reached = next((i for i, v in enumerate(vals) if v < 0.01), len(vals))
    #     print(f"{name}: reached f<0.01 at step {reached}, final f={vals[-1]:.6f}")
    return (gradient_descent_momentum,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Exercise 3: Backpropagation by Hand (in Code)

    Implement forward and backward passes for a tiny network: one hidden layer with sigmoid activation, MSE loss.

    Network: $\hat{y} = w_2 \cdot \sigma(w_1 \cdot x + b_1) + b_2$

    Loss: $L = (\hat{y} - y)^2$

    Compute all gradients ($\partial L / \partial w_1$, $\partial L / \partial b_1$, $\partial L / \partial w_2$, $\partial L / \partial b_2$) analytically via the chain rule, then verify with finite differences.
    """)
    return


@app.cell
def _():
    import numpy as np

    def tiny_network_forward_backward(x, y, w1, b1, w2, b2):
        """Forward + backward pass for: y_hat = w2 * sigmoid(w1*x + b1) + b2

        Returns: (loss, grads_dict) where grads_dict has keys 'w1','b1','w2','b2'
        """
        sigmoid = lambda z: 1 / (1 + np.exp(-z))

        # --- Forward pass (save intermediates) ---
        z1 = w1 * x + b1           # pre-activation
        a1 = sigmoid(z1)            # activation
        y_hat = w2 * a1 + b2        # output
        loss = (y_hat - y) ** 2     # MSE loss

        # --- Backward pass (chain rule, outermost first) ---
        # TODO: compute dL/dy_hat
        dL_dyhat = ...  # <-- 2 * (y_hat - y)

        # TODO: compute dL/dw2, dL/db2
        dL_dw2 = ...    # <-- dL_dyhat * a1
        dL_db2 = ...    # <-- dL_dyhat * 1

        # TODO: compute dL/da1, then dL/dz1 (sigmoid derivative)
        dL_da1 = ...    # <-- dL_dyhat * w2
        dL_dz1 = ...    # <-- dL_da1 * sigmoid(z1) * (1 - sigmoid(z1))

        # TODO: compute dL/dw1, dL/db1
        dL_dw1 = ...    # <-- dL_dz1 * x
        dL_db1 = ...    # <-- dL_dz1 * 1

        grads = {'w1': dL_dw1, 'b1': dL_db1, 'w2': dL_dw2, 'b2': dL_db2}
        return loss, grads

    # Test values
    # x, y = 1.5, 0.8
    # w1, b1, w2, b2 = 0.5, -0.1, 0.8, 0.2
    # loss, grads = tiny_network_forward_backward(x, y, w1, b1, w2, b2)
    # print(f"Loss: {loss:.6f}")
    #
    # # Numerical verification
    # h = 1e-7
    # for name, val in [('w1',w1),('b1',b1),('w2',w2),('b2',b2)]:
    #     args = {'x':x,'y':y,'w1':w1,'b1':b1,'w2':w2,'b2':b2}
    #     args[name] = val + h
    #     loss_plus, _ = tiny_network_forward_backward(**args)
    #     num_grad = (loss_plus - loss) / h
    #     print(f"  d/d{name}: analytic={grads[name]:.6f}, numeric={num_grad:.6f}")
    return (tiny_network_forward_backward,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Exercise 4: Ridge Regression -- Normal Equation vs Gradient Descent

    Implement both the closed-form solution and iterative gradient descent for Ridge regression:

    $$L(\mathbf{w}) = \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2 + \lambda\|\mathbf{w}\|^2$$

    Compare the solutions -- they should match.
    """)
    return


@app.cell
def _():
    import numpy as np

    # Generate synthetic data
    rng = np.random.default_rng(7)
    n, d = 50, 4
    X = rng.standard_normal((n, d))
    w_true = np.array([3.0, -1.0, 0.5, 2.0])
    y = X @ w_true + 0.5 * rng.standard_normal(n)
    lam = 1.0  # regularization strength

    # TODO: Closed-form Ridge solution
    # w* = (X^T X + lambda * I)^{-1} X^T y
    w_closed = ...  # <-- fill this in

    # TODO: Gradient descent for Ridge
    # grad L = 2 * X^T(Xw - y) + 2*lambda*w  (don't forget the regularization term!)
    w_gd = np.zeros(d)
    lr = 0.01
    for step in range(1000):
        # grad = ...  # <-- compute the gradient
        # w_gd = w_gd - lr * grad
        pass

    # print(f"Closed-form: {np.round(w_closed, 4)}")
    # print(f"Grad descent: {np.round(w_gd, 4)}")
    # print(f"True weights: {w_true}")
    # print(f"Max difference (closed vs GD): {np.max(np.abs(w_closed - w_gd)):.6f}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Further Reading

    - [MML Chapter 5: Vector Calculus](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) -- The primary reference for this module. Read Sections 5.1 through 5.5 carefully.
    - [MML Section 5.5: Useful Identities for Computing Gradients](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) -- A compact reference for matrix calculus identities.
    - [MML Section 5.6: Backpropagation and Automatic Differentiation](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) -- Connects the chain rule to neural network training.
    - [Boyd Chapter 9: Unconstrained Minimization](file:///C:/Users/landa/ml-course/textbooks/Boyd-ConvexOptimization.pdf) -- Gradient descent and Newton's method, rigorous treatment.
    - [Bishop Appendix D: Matrix Calculus](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) -- Concise matrix calculus reference.
    - [Murphy Section 7.4: Gradient Descent](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf) -- Modern treatment with practical considerations.
    """)
    return


if __name__ == "__main__":
    app.run()
