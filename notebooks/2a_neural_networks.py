import marimo

app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    return (np,)


@app.cell
def _(mo):
    mo.md(r"""
# Module 2A: Neural Networks & Backpropagation

Welcome to deep learning. Everything you have built so far — linear regression, logistic regression, gradient descent, matrix calculus — was preparation for this moment. You already know the pieces. Now we assemble them into something far more powerful.

This module is long and dense. That is because backpropagation is the single most important algorithm in modern machine learning. If you understand it deeply — not just "the chain rule applied repeatedly" but *why* it works, *how* it flows through a computational graph, and *what* it computes at each step — then every model you encounter from here on will make sense. Transformers, CNNs, GANs, diffusion models: they are all just architectures. Backprop trains all of them.

Take your time with this one.
""")
    return


@app.cell
def _(mo):
    mo.md(r"""
---

## 1. From Linear Models to Neural Networks

You know linear regression: $\hat{y} = \mathbf{w}^\top \mathbf{x} + b$. You know logistic regression: $\hat{y} = \sigma(\mathbf{w}^\top \mathbf{x} + b)$. These are powerful, interpretable, and well-understood. But they have a fundamental limitation: **they can only learn linear decision boundaries**.

Consider the XOR problem. You have four points:

| $x_1$ | $x_2$ | $y$ |
|-------|-------|-----|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

No single line can separate the 0s from the 1s. Logistic regression fails here. Not because of bad optimization or insufficient data — it fails because the function it represents *cannot express* XOR. The hypothesis class is too restricted. See [DLBook 6.1](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) for the full XOR example worked out with a neural network solution.

So what do we do? The key insight: **stack linear transformations with nonlinearities between them**. A single linear layer computes $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$. That is still linear. But if we pass the output through a nonlinear function $g$, then feed the result into *another* linear layer, the composition is no longer linear. It can carve curved, complex decision boundaries.

$$\mathbf{h} = g(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1), \quad \hat{y} = \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2$$

This is a neural network. Two linear transformations with a nonlinearity in between. Simple idea, extraordinary consequences.
""")
    return


@app.cell
def _(np):
    def _run():
        # XOR: a linear model cannot solve this, but a 2-layer network can
        X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y_xor = np.array([0, 1, 1, 0])

        # Hand-picked weights that solve XOR:
        # h = ReLU(W1 @ x + b1), y_hat = w2 @ h + b2
        W1_xor = np.array([[1, 1], [1, 1]])    # both neurons sum x1+x2
        b1_xor = np.array([-0.5, -1.5])         # thresholds at 0.5 and 1.5
        w2_xor = np.array([1, -1])              # subtract: "at least 1" minus "both"
        b2_xor = 0.0

        for i in range(4):
            z = W1_xor @ X_xor[i] + b1_xor     # pre-activation
            h = np.maximum(0, z)                 # ReLU
            y_hat = w2_xor @ h + b2_xor         # output
            print(f"x={X_xor[i]}  h={h}  y_hat={y_hat:.1f}  target={y_xor[i]}")


    _run()
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
---

## 2. The Neuron

A single neuron computes:

$$a = g\!\left(\sum_{i=1}^{d} w_i x_i + b\right) = g(\mathbf{w}^\top \mathbf{x} + b)$$

It takes a weighted sum of its inputs, adds a bias, and passes the result through an **activation function** $g$. That is it. You have already seen this: logistic regression is a single neuron with the sigmoid activation.

**The biological metaphor.** The name "neural network" comes from a loose analogy to biological neurons, which receive electrical signals through dendrites, integrate them in the cell body, and fire (or not) through the axon. McCulloch and Pitts proposed the first mathematical neuron in 1943 based on this idea.

But I want to be direct: **the biological metaphor is mostly misleading at this point**. Real neurons communicate with spikes (timing matters), have complex dendritic computations, exhibit plasticity mechanisms nothing like gradient descent, and are embedded in a physical, chemical, three-dimensional structure. The artificial neuron is a crude cartoon. It is useful as a historical motivation and nothing more. Do not let biological intuition guide your understanding of how these models work — the math will do that.
""")
    return


@app.cell
def _(np):
    # A single neuron: z = w^T x + b, then activation g(z)
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    w_neuron = np.array([0.5, -0.3, 0.8])   # 3 input weights
    b_neuron = 0.1                            # bias
    x_neuron = np.array([1.0, 2.0, 0.5])     # input

    z_neuron = w_neuron @ x_neuron + b_neuron  # weighted sum + bias
    a_neuron = sigmoid(z_neuron)               # activation

    print(f"Pre-activation z = w^T x + b = {z_neuron:.3f}")
    print(f"Activation a = sigmoid(z)    = {a_neuron:.3f}")
    return (sigmoid,)


@app.cell
def _(mo):
    mo.md(r"""
---

## 3. Activation Functions

The activation function $g$ is what makes neural networks nonlinear. Without it, stacking layers would just produce another linear transformation ($\mathbf{W}_2 \mathbf{W}_1 \mathbf{x}$ is still a linear function of $\mathbf{x}$). The choice of activation function has a profound impact on training dynamics.

### Sigmoid

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

You know this from logistic regression. It squashes any real number into $(0, 1)$. Its derivative is $\sigma'(z) = \sigma(z)(1 - \sigma(z))$.

**Problems:** When $|z|$ is large, $\sigma(z)$ saturates — it is nearly flat, so $\sigma'(z) \approx 0$. During backpropagation, gradients pass through these derivatives. If many layers have near-zero derivatives, the gradient signal *vanishes* exponentially as it propagates backward. This is the **vanishing gradient problem**, and it made deep networks extremely difficult to train for decades. Additionally, sigmoid outputs are always positive, which introduces systematic bias in gradient updates.

### Tanh

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = 2\sigma(2z) - 1$$

This is a rescaled sigmoid that maps to $(-1, 1)$ instead of $(0, 1)$. It is **zero-centered**, which is better for gradient flow. But it still saturates for large $|z|$, so the vanishing gradient problem persists.

### ReLU (Rectified Linear Unit)

$$\text{ReLU}(z) = \max(0, z)$$

ReLU changed everything. Introduced to deep learning practice around 2010-2012 (Nair & Hinton, Glorot et al.), it has three properties that make it vastly superior to sigmoid/tanh for deep networks:

1. **No saturation for positive inputs.** The derivative is exactly 1 for $z > 0$, so gradients flow without shrinking.
2. **Sparsity.** For any input, roughly half the neurons output exactly zero. Sparse representations are computationally efficient and can be more interpretable.
3. **Computational simplicity.** It is just a max operation — far cheaper than computing exponentials.

**The catch:** ReLU is not differentiable at $z = 0$ (in practice we just pick 0 or 1 as the subgradient; it does not matter). More importantly, ReLU neurons can "die" — if a neuron's pre-activation is always negative (due to a large negative bias or an unlucky weight update), its output is always zero, its gradient is always zero, and it never updates again. This is the **dying ReLU problem**.

### Modern Variants

- **Leaky ReLU:** $f(z) = \max(\alpha z, z)$ with small $\alpha$ like 0.01. Gives a small gradient for negative inputs, preventing dead neurons.
- **ELU (Exponential Linear Unit):** $f(z) = z$ if $z > 0$, $\alpha(e^z - 1)$ if $z \leq 0$. Smooth, with negative outputs that push mean activations toward zero.
- **GELU (Gaussian Error Linear Unit):** $f(z) = z \cdot \Phi(z)$ where $\Phi$ is the standard Gaussian CDF. Used in Transformers (BERT, GPT). Smooth approximation to ReLU with a probabilistic interpretation.
- **Swish / SiLU:** $f(z) = z \cdot \sigma(z)$. Smooth, non-monotonic. Often slightly outperforms ReLU in deep networks.

**Which should you use?** Start with ReLU. It works well in most settings. If you observe dying neurons, try Leaky ReLU or GELU. For transformer architectures, GELU is standard. Do not overthink this — activation function choice is rarely the bottleneck.

See [DLBook 6.3](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) for an extended discussion of hidden unit types.
""")
    return


@app.cell
def _(np):
    import matplotlib.pyplot as plt

    # Implement activation functions and their derivatives
    def relu(z):
        return np.maximum(0, z)

    def relu_deriv(z):
        return (z > 0).astype(float)

    def sigmoid_fn(z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_deriv(z):
        s = sigmoid_fn(z)
        return s * (1 - s)           # sigma'(z) = sigma(z)(1 - sigma(z))

    def leaky_relu(z, alpha=0.01):
        return np.where(z > 0, z, alpha * z)

    z_vals = np.linspace(-4, 4, 200)

    fig_act, axes_act = plt.subplots(1, 3, figsize=(12, 3.5))
    # Sigmoid and derivative
    axes_act[0].plot(z_vals, sigmoid_fn(z_vals), label="sigmoid")
    axes_act[0].plot(z_vals, sigmoid_deriv(z_vals), "--", label="sigmoid'")
    axes_act[0].set_title("Sigmoid"); axes_act[0].legend(); axes_act[0].grid(True)
    # ReLU and derivative
    axes_act[1].plot(z_vals, relu(z_vals), label="ReLU")
    axes_act[1].plot(z_vals, relu_deriv(z_vals), "--", label="ReLU'")
    axes_act[1].set_title("ReLU"); axes_act[1].legend(); axes_act[1].grid(True)
    # Leaky ReLU
    axes_act[2].plot(z_vals, leaky_relu(z_vals), label="Leaky ReLU")
    axes_act[2].plot(z_vals, np.tanh(z_vals), label="tanh")
    axes_act[2].set_title("Leaky ReLU & Tanh"); axes_act[2].legend(); axes_act[2].grid(True)
    plt.tight_layout()
    fig_act
    return (relu, relu_deriv, sigmoid_fn, sigmoid_deriv,)


@app.cell
def _(mo):
    mo.md(r"""
---

## 4. Feedforward Networks (MLPs)

A **feedforward neural network**, also called a **multilayer perceptron (MLP)**, chains together multiple layers. The standard architecture:

$$\mathbf{h}^{(1)} = g^{(1)}(\mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)})$$
$$\mathbf{h}^{(2)} = g^{(2)}(\mathbf{W}^{(2)} \mathbf{h}^{(1)} + \mathbf{b}^{(2)})$$
$$\vdots$$
$$\hat{\mathbf{y}} = f_{\text{out}}(\mathbf{W}^{(L)} \mathbf{h}^{(L-1)} + \mathbf{b}^{(L)})$$

Each layer applies an affine transformation $\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}$ followed by a nonlinear activation $\mathbf{h}^{(l)} = g^{(l)}(\mathbf{z}^{(l)})$. The final output layer typically uses a task-specific activation: identity for regression, sigmoid for binary classification, softmax for multiclass classification.

**Notation conventions.** $\mathbf{W}^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}$ is the weight matrix for layer $l$, where $n_l$ is the number of neurons in layer $l$ and $n_{l-1}$ is the number of neurons in the previous layer. $\mathbf{b}^{(l)} \in \mathbb{R}^{n_l}$ is the bias vector. The entire network is a function composition:

$$f(\mathbf{x}; \boldsymbol{\theta}) = f_L \circ f_{L-1} \circ \cdots \circ f_1(\mathbf{x})$$

where $\boldsymbol{\theta} = \{\mathbf{W}^{(1)}, \mathbf{b}^{(1)}, \ldots, \mathbf{W}^{(L)}, \mathbf{b}^{(L)}\}$ is the set of all parameters. See [DLBook 6.0](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) for the full treatment of feedforward network architecture, and [Bishop 5.1](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) for an equivalent development.

### Width vs Depth

**Width** (number of neurons per layer) controls the expressiveness of each individual layer — how many features it can compute simultaneously. **Depth** (number of layers) controls the compositionality of the learned representation — how many levels of abstraction the network can build.

In practice, deeper networks tend to be more parameter-efficient than wider ones. A deep network can represent certain functions with exponentially fewer parameters than a shallow one. But deeper networks are harder to train (vanishing/exploding gradients, optimization difficulties). Much of deep learning research from 2012 onward has been about finding ways to train deeper networks reliably.

### The Universal Approximation Theorem

**Theorem (Cybenko 1989, Hornik 1991):** A feedforward network with a single hidden layer containing a finite number of neurons can approximate any continuous function on a compact subset of $\mathbb{R}^n$ to arbitrary accuracy, given a sufficiently wide hidden layer and appropriate activation functions.

This is a beautiful and frequently misunderstood result. Here is what it does and does not tell you:

- **It does tell you:** MLPs are, in principle, expressive enough to represent any continuous function. The model class is not the bottleneck.
- **It does not tell you:** how many neurons you need (could be astronomically many), whether gradient descent will find good weights (the theorem is existential, not constructive), or whether your finite dataset is sufficient to learn the function.

The theorem is a reassurance about representational power, not a practical guarantee. See [DLBook 6.4.1](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) and [MML 5.4](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) for further discussion.
""")
    return


@app.cell
def _(np, relu):
    def _run():
        # A 3-layer MLP forward pass in numpy
        # Architecture: 3 inputs -> 4 hidden -> 4 hidden -> 1 output
        rng = np.random.default_rng(0)
        dims = [3, 4, 4, 1]  # layer sizes

        # Kaiming initialization for ReLU layers
        weights_mlp = []
        biases_mlp = []
        for l in range(len(dims) - 1):
            scale = np.sqrt(2.0 / dims[l])              # He init: sqrt(2/n_in)
            W = rng.standard_normal((dims[l+1], dims[l])) * scale
            b = np.zeros(dims[l+1])
            weights_mlp.append(W)
            biases_mlp.append(b)

        # Forward pass through all layers
        x_mlp = np.array([1.0, -0.5, 0.3])
        h = x_mlp
        for l in range(len(weights_mlp)):
            z = weights_mlp[l] @ h + biases_mlp[l]      # affine: z = Wh + b
            h = relu(z) if l < len(weights_mlp) - 1 else z  # ReLU hidden, identity output
            print(f"Layer {l+1}: z shape={z.shape}, output={np.round(h, 3)}")

        print(f"\nNetwork output: {h[0]:.4f}")


    _run()
    return

@app.cell
def _(mo):
    mo.md(r"""
---

## 5. Loss Functions for Neural Networks

You have already seen these. Neural networks do not introduce new loss functions — they use the same ones from linear/logistic regression, just applied to more complex model outputs.

### MSE for Regression

$$\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

Nothing new here. The network outputs a real number $\hat{y}_i = f(\mathbf{x}_i; \boldsymbol{\theta})$, and we penalize squared deviations from the target.

### Cross-Entropy for Classification

For binary classification with sigmoid output $\hat{p} = \sigma(\mathbf{w}^\top \mathbf{h} + b)$:

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \left[ y_i \log \hat{p}_i + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

For multiclass classification with softmax output $\hat{p}_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$:

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \sum_{k=1}^{K} y_{ik} \log \hat{p}_{ik}$$

This is identical to what you derived for logistic regression via MLE. The network just replaces the simple linear function with a deep composition.

### Why Cross-Entropy Beats MSE for Classification

If you use MSE with a sigmoid output, the gradient $\frac{\partial \mathcal{L}}{\partial z}$ includes a factor of $\sigma'(z)$, which vanishes when the sigmoid saturates. So when the network is *confidently wrong* (sigmoid near 0 or 1, but on the wrong side), the gradient is tiny and learning stalls.

Cross-entropy cancels the sigmoid's derivative. The gradient simplifies to $\frac{\partial \mathcal{L}}{\partial z} = \hat{p} - y$, which is large precisely when the network is wrong. This is not a coincidence — it is a consequence of maximum likelihood estimation with the appropriate exponential family distribution. See [DLBook 6.5.5](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) for the full argument.
""")
    return


@app.cell
def _(np, sigmoid_fn):
    # Loss functions: MSE and cross-entropy
    def mse_loss(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def binary_cross_entropy(y_true, y_pred):
        eps = 1e-12  # numerical stability
        return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))

    def softmax(z):
        e = np.exp(z - np.max(z))             # subtract max for numerical stability
        return e / e.sum()

    # Demo: softmax converts raw logits to probabilities
    logits = np.array([2.0, 1.0, 0.1])
    probs = softmax(logits)
    print(f"Logits:       {logits}")
    print(f"Softmax:      {np.round(probs, 4)}  (sum={probs.sum():.4f})")

    # Compare MSE vs cross-entropy gradient when confidently wrong
    z_wrong = 5.0                              # large positive logit
    p_hat = sigmoid_fn(z_wrong)                # ~0.993, but true label is 0
    grad_mse = 2 * (p_hat - 0) * p_hat * (1 - p_hat)    # includes sigma'
    grad_ce = p_hat - 0                                    # clean gradient
    print(f"\nConfidently wrong (z=5, y=0): p_hat={p_hat:.4f}")
    print(f"  MSE gradient wrt z:   {grad_mse:.6f}  (vanishing!)")
    print(f"  CE  gradient wrt z:   {grad_ce:.6f}  (strong signal)")
    return (mse_loss, softmax,)


@app.cell
def _(mo):
    mo.md(r"""
---

## 6. Forward Pass

The forward pass is simple: plug in numbers, compute layer by layer. Let me walk through a concrete example so there is no ambiguity.

**Example:** A network with 2 inputs, one hidden layer of 2 neurons (ReLU), and 1 output (identity, for regression).

Parameters:
$$\mathbf{W}^{(1)} = \begin{bmatrix} 0.5 & -0.3 \\ 0.8 & 0.2 \end{bmatrix}, \quad \mathbf{b}^{(1)} = \begin{bmatrix} 0.1 \\ -0.1 \end{bmatrix}$$

$$\mathbf{w}^{(2)} = \begin{bmatrix} 0.6 \\ -0.4 \end{bmatrix}, \quad b^{(2)} = 0.05$$

Input: $\mathbf{x} = \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix}$, Target: $y = 1.0$

**Step 1: Hidden layer pre-activation**
$$\mathbf{z}^{(1)} = \mathbf{W}^{(1)}\mathbf{x} + \mathbf{b}^{(1)} = \begin{bmatrix} 0.5(1) + (-0.3)(2) + 0.1 \\ 0.8(1) + 0.2(2) + (-0.1) \end{bmatrix} = \begin{bmatrix} 0.0 \\ 1.1 \end{bmatrix}$$

**Step 2: Hidden layer activation (ReLU)**
$$\mathbf{h}^{(1)} = \text{ReLU}(\mathbf{z}^{(1)}) = \begin{bmatrix} \max(0, 0.0) \\ \max(0, 1.1) \end{bmatrix} = \begin{bmatrix} 0.0 \\ 1.1 \end{bmatrix}$$

**Step 3: Output layer**
$$\hat{y} = \mathbf{w}^{(2)\top}\mathbf{h}^{(1)} + b^{(2)} = 0.6(0.0) + (-0.4)(1.1) + 0.05 = -0.39$$

**Step 4: Loss (MSE for single example)**
$$\mathcal{L} = (y - \hat{y})^2 = (1.0 - (-0.39))^2 = (1.39)^2 = 1.9321$$

That is the forward pass. We computed the prediction and the loss. Now: how do we update the weights to reduce this loss? We need the gradients.
""")
    return


@app.cell
def _(np):
    # Forward pass — exact reproduction of the worked example above
    x_fp = np.array([1.0, 2.0])
    y_fp = 1.0

    W1_fp = np.array([[0.5, -0.3],
                       [0.8,  0.2]])
    b1_fp = np.array([0.1, -0.1])
    w2_fp = np.array([0.6, -0.4])
    b2_fp = 0.05

    # Step 1: pre-activation
    z1_fp = W1_fp @ x_fp + b1_fp               # z = Wx + b
    print(f"z1 = {z1_fp}")

    # Step 2: ReLU activation
    h1_fp = np.maximum(0, z1_fp)                # h = max(0, z)
    print(f"h1 = {h1_fp}")

    # Step 3: output layer
    y_hat_fp = w2_fp @ h1_fp + b2_fp            # y_hat = w2^T h + b2
    print(f"y_hat = {y_hat_fp}")

    # Step 4: MSE loss
    loss_fp = (y_fp - y_hat_fp) ** 2            # L = (y - y_hat)^2
    print(f"loss = {loss_fp:.4f}")
    return (W1_fp, b1_fp, b2_fp, h1_fp, loss_fp, w2_fp, x_fp, y_fp, y_hat_fp, z1_fp,)


@app.cell
def _(mo):
    mo.md(r"""
---

## 7. Backpropagation

This is the heart of deep learning. Read this section carefully, work through the math yourself, and make sure you can reproduce it on paper. Refer to [DLBook 6.5](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) for the general framework, [MML 5.2-5.4](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) for the chain rule foundations, and [Bishop 5.3](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) for a complementary derivation.

### The Problem

We have a loss $\mathcal{L}$ that depends on the network output $\hat{y}$, which depends on all the weights and biases through a long chain of compositions. We need $\frac{\partial \mathcal{L}}{\partial w}$ for *every* weight $w$ in the network. A network might have millions or billions of weights. Computing each gradient independently by perturbing each weight and measuring the change in loss would require one forward pass per weight — completely intractable.

### The Chain Rule, Applied Systematically

Backpropagation is not a new mathematical idea. It is the chain rule of calculus, applied in a specific order (output to input) on a computational graph, with intermediate results cached from the forward pass. That is all.

Recall the chain rule: if $y = f(g(x))$, then $\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}$. In a network with $L$ layers, the loss depends on the output, which depends on layer $L$'s activations, which depend on layer $L$'s weights and on layer $L-1$'s activations, and so on. The chain rule lets us decompose the full gradient into a product of local derivatives, one per layer.

### Computational Graphs

Think of the network as a **directed acyclic graph (DAG)** where:
- **Nodes** represent operations (matrix multiply, add bias, apply activation, compute loss)
- **Edges** represent data flow (tensors flowing between operations)

The forward pass computes values at each node from inputs to output. The backward pass computes gradients at each node from output to inputs. Each node only needs to know its own local derivative — how its output changes with respect to its inputs — and the gradient flowing in from above (how the loss changes with respect to its output). Multiply these together: that is the chain rule.

See [DLBook 6.5.1](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) for a thorough treatment of computational graphs in the context of backpropagation.
""")
    return


@app.cell
def _(mo):
    mo.md(r"""
### Deriving Backprop for Our 2-Layer Network

Let me derive the gradients for the exact network from the forward pass example. We have:

$$\mathbf{z}^{(1)} = \mathbf{W}^{(1)}\mathbf{x} + \mathbf{b}^{(1)}$$
$$\mathbf{h}^{(1)} = \text{ReLU}(\mathbf{z}^{(1)})$$
$$\hat{y} = \mathbf{w}^{(2)\top}\mathbf{h}^{(1)} + b^{(2)}$$
$$\mathcal{L} = (y - \hat{y})^2$$

**Step 1: Gradient of loss with respect to output**

$$\frac{\partial \mathcal{L}}{\partial \hat{y}} = -2(y - \hat{y}) = -2(1.0 - (-0.39)) = -2.78$$

**Step 2: Gradients for output layer parameters**

Since $\hat{y} = \mathbf{w}^{(2)\top}\mathbf{h}^{(1)} + b^{(2)}$:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}^{(2)}} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \mathbf{h}^{(1)} = -2.78 \cdot \begin{bmatrix} 0.0 \\ 1.1 \end{bmatrix} = \begin{bmatrix} 0.0 \\ -3.058 \end{bmatrix}$$

$$\frac{\partial \mathcal{L}}{\partial b^{(2)}} = \frac{\partial \mathcal{L}}{\partial \hat{y}} = -2.78$$

**Step 3: Gradient flowing back to hidden layer**

$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(1)}} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \mathbf{w}^{(2)} = -2.78 \cdot \begin{bmatrix} 0.6 \\ -0.4 \end{bmatrix} = \begin{bmatrix} -1.668 \\ 1.112 \end{bmatrix}$$

**Step 4: Gradient through ReLU**

ReLU's derivative: $\text{ReLU}'(z) = \mathbb{1}[z > 0]$ (1 if $z > 0$, 0 otherwise).

$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(1)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(1)}} \odot \text{ReLU}'(\mathbf{z}^{(1)}) = \begin{bmatrix} -1.668 \\ 1.112 \end{bmatrix} \odot \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 0.0 \\ 1.112 \end{bmatrix}$$

Here $\odot$ is element-wise multiplication. Notice: the first neuron had $z_1^{(1)} = 0.0$, and ReLU killed the gradient. That neuron's incoming weights will receive zero gradient update.

**Step 5: Gradients for hidden layer parameters**

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(1)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(1)}} \cdot \mathbf{x}^\top = \begin{bmatrix} 0.0 \\ 1.112 \end{bmatrix} \begin{bmatrix} 1.0 & 2.0 \end{bmatrix} = \begin{bmatrix} 0.0 & 0.0 \\ 1.112 & 2.224 \end{bmatrix}$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(1)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(1)}} = \begin{bmatrix} 0.0 \\ 1.112 \end{bmatrix}$$

That is backpropagation. We started at the loss and worked backward, computing each gradient using only (a) the gradient from the layer above and (b) locally cached values from the forward pass. Every gradient computation was a multiplication — the chain rule in action.
""")
    return


@app.cell
def _(W1_fp, b1_fp, b2_fp, h1_fp, np, w2_fp, x_fp, y_fp, y_hat_fp, z1_fp):
    def _run():
        # Backpropagation — exact reproduction of the math above
        # Step 1: dL/dy_hat = -2(y - y_hat)
        dL_dy_hat = -2 * (y_fp - y_hat_fp)
        print(f"dL/dy_hat = {dL_dy_hat:.4f}")

        # Step 2: gradients for output layer
        dL_dw2 = dL_dy_hat * h1_fp                  # dL/dw2 = dL/dy_hat * h
        dL_db2 = dL_dy_hat                           # dL/db2 = dL/dy_hat
        print(f"dL/dw2 = {dL_dw2}")
        print(f"dL/db2 = {dL_db2:.4f}")

        # Step 3: gradient flowing back to hidden layer
        dL_dh1 = dL_dy_hat * w2_fp                   # dL/dh = dL/dy_hat * w2
        print(f"dL/dh1 = {dL_dh1}")

        # Step 4: gradient through ReLU (element-wise)
        relu_mask = (z1_fp > 0).astype(float)        # ReLU'(z) = 1[z > 0]
        dL_dz1 = dL_dh1 * relu_mask                  # element-wise: dL/dz = dL/dh * ReLU'(z)
        print(f"dL/dz1 = {dL_dz1}")

        # Step 5: gradients for hidden layer parameters
        dL_dW1 = dL_dz1.reshape(-1, 1) @ x_fp.reshape(1, -1)  # outer product: delta @ x^T
        dL_db1 = dL_dz1                              # dL/db = delta
        print(f"dL/dW1 =\n{dL_dW1}")
        print(f"dL/db1 = {dL_db1}")


    _run()
    return

@app.cell
def _(mo):
    mo.md(r"""
### The General Pattern

For any layer $l$, define $\boldsymbol{\delta}^{(l)} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(l)}}$ (the "error signal" at layer $l$). Then:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} = \boldsymbol{\delta}^{(l)} \cdot \mathbf{h}^{(l-1)\top}$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}} = \boldsymbol{\delta}^{(l)}$$
$$\boldsymbol{\delta}^{(l-1)} = \left(\mathbf{W}^{(l)\top} \boldsymbol{\delta}^{(l)}\right) \odot g'(\mathbf{z}^{(l-1)})$$

The last equation is the key recursion: the error signal at layer $l-1$ is the error signal at layer $l$, projected backward through the weight matrix, and scaled by the activation function's derivative. This recursion runs from the output back to the input — hence "back-propagation."

### The Jacobian Perspective

For those who want the more rigorous view: each layer $f_l$ maps $\mathbf{h}^{(l-1)} \mapsto \mathbf{h}^{(l)}$. The Jacobian $\mathbf{J}_l = \frac{\partial \mathbf{h}^{(l)}}{\partial \mathbf{h}^{(l-1)}}$ captures how layer $l$'s output changes with its input. The chain rule gives:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(l-1)}} = \mathbf{J}_l^\top \frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(l)}}$$

This is a **vector-Jacobian product (VJP)**. Backpropagation computes a sequence of VJPs from the output layer to the input. The critical insight: we never form the full Jacobian matrix explicitly. We only compute the product of the Jacobian transpose with a vector, which is far cheaper.

See [MML 5.3](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) for the chain rule in the multivariate setting and Jacobian calculations.

### Why Backprop Is Efficient

**Naive approach:** For each of the $P$ parameters, perturb it by $\epsilon$, run a full forward pass, compute the finite-difference gradient. Cost: $O(P)$ forward passes.

**Backpropagation:** One forward pass + one backward pass. The backward pass has roughly the same cost as the forward pass (each operation's gradient costs about as much as the operation itself). Total cost: $O(1)$ forward passes, regardless of how many parameters you have.

For a network with 1 million parameters, backprop is a million times faster than finite differences. This is not an exaggeration — it is the reason deep learning is possible at all.
""")
    return


@app.cell
def _(np):
    # Verify backprop gradients using finite differences
    # This is how you sanity-check any gradient implementation
    eps = 1e-5

    # Re-define the full forward pass as a function of all params
    def forward_loss(W1, b1, w2, b2, x, y):
        z1 = W1 @ x + b1
        h1 = np.maximum(0, z1)
        y_hat = w2 @ h1 + b2
        return (y - y_hat) ** 2

    _W1 = np.array([[0.5, -0.3], [0.8, 0.2]])
    _b1 = np.array([0.1, -0.1])
    _w2 = np.array([0.6, -0.4])
    _b2 = 0.05
    _x = np.array([1.0, 2.0])
    _y = 1.0

    # Numerical gradient for W1[1,0] (should be ~1.112)
    W1_plus = _W1.copy(); W1_plus[1, 0] += eps
    W1_minus = _W1.copy(); W1_minus[1, 0] -= eps
    num_grad = (forward_loss(W1_plus, _b1, _w2, _b2, _x, _y)
                - forward_loss(W1_minus, _b1, _w2, _b2, _x, _y)) / (2 * eps)
    print(f"dL/dW1[1,0]: backprop=1.1120, finite-diff={num_grad:.4f}")

    # Numerical gradient for w2[1] (should be ~-3.058)
    w2_plus = _w2.copy(); w2_plus[1] += eps
    w2_minus = _w2.copy(); w2_minus[1] -= eps
    num_grad_w2 = (forward_loss(_W1, _b1, w2_plus, _b2, _x, _y)
                   - forward_loss(_W1, _b1, w2_minus, _b2, _x, _y)) / (2 * eps)
    print(f"dL/dw2[1]:   backprop=-3.0580, finite-diff={num_grad_w2:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
---

## Backpropagation Flow Animation

The following animation illustrates how gradients flow backward through the network during backpropagation:
""")
    return


@app.cell
def _(mo):
    mo.image(src="../animations/rendered/BackpropFlow.gif")
    return


@app.cell
def _(mo):
    mo.md(r"""
---

## 8. Automatic Differentiation

Backpropagation is a specific instance of a broader technique called **automatic differentiation (AD)**. Understanding AD clarifies what deep learning frameworks like PyTorch actually do under the hood.

### Forward Mode vs Reverse Mode

There are two ways to propagate derivatives through a computational graph:

- **Forward mode AD:** Compute $\frac{\partial}{\partial x}$ of every intermediate variable as you go forward. One pass gives you the derivative with respect to *one* input variable. If you have $P$ inputs (parameters), you need $P$ passes. Cost: $O(P) \times$ forward pass.

- **Reverse mode AD:** Compute $\frac{\partial \mathcal{L}}{\partial}$ every intermediate variable as you go backward from a scalar output. One pass gives you the derivative with respect to *all* inputs. Cost: $O(1) \times$ forward pass.

**Reverse mode AD is backpropagation.** Since neural networks have a scalar loss (one output) and many parameters (many inputs), reverse mode is the right choice. This is why all deep learning frameworks implement reverse mode AD.

See [DLBook 6.5.7-6.5.10](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) for the general treatment of forward and reverse mode AD on computational graphs.

### How PyTorch Implements This

PyTorch builds a **dynamic computational graph** as you execute operations on tensors. Each operation records itself and its inputs. When you call `.backward()` on the loss, PyTorch traverses this graph in reverse topological order, applying the chain rule at each node.
""")
    return


@app.cell
def _():
    def _run():
        import torch

        # Create tensors with gradient tracking
        x = torch.tensor([1.0, 2.0])
        W1 = torch.tensor([[0.5, -0.3], [0.8, 0.2]], requires_grad=True)
        b1 = torch.tensor([0.1, -0.1], requires_grad=True)
        w2 = torch.tensor([0.6, -0.4], requires_grad=True)
        b2 = torch.tensor([0.05], requires_grad=True)

        # Forward pass — PyTorch records the graph
        z1 = W1 @ x + b1
        h1 = torch.relu(z1)
        y_hat = w2 @ h1 + b2
        loss = (1.0 - y_hat) ** 2

        # Backward pass — PyTorch traverses the graph in reverse
        loss.backward()

        # Gradients are now stored on each parameter tensor
        print("dL/dW1:", W1.grad)
        print("dL/db1:", b1.grad)
        print("dL/dw2:", w2.grad)
        print("dL/db2:", b2.grad)


    _run()
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
Every tensor with `requires_grad=True` accumulates its gradient. The entire system is built on the same chain rule computations we derived by hand above. There is no magic.

### Karpathy's micrograd

Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) is a roughly 100-line implementation of a scalar-valued autograd engine. It builds a computational graph of `Value` objects, each storing a single number and its gradient. Calling `.backward()` on the output traverses the graph and populates gradients everywhere. If you want to truly understand how autograd works at the implementation level, reading and reimplementing micrograd is one of the best exercises you can do. It strips away all the tensor/GPU complexity and exposes the core idea.
""")
    return


@app.cell
def _(mo):
    mo.md(r"""
---

## 9. Training Neural Networks — The Basics

You know gradient descent from Module 0F. Training a neural network is the same loop — but there are important practical details that are specific to neural networks.

### Weight Initialization

**Never initialize all weights to zero.** If every neuron in a layer has the same weights, they all compute the same output, receive the same gradient, and update identically. They remain identical forever. This is the **symmetry breaking problem** — the network has many neurons but effectively acts as if it has one.

The solution: initialize weights randomly. But the *scale* of the initialization matters enormously.

**Xavier/Glorot initialization (2010):** For a layer with $n_{\text{in}}$ inputs and $n_{\text{out}}$ outputs, draw weights from:

$$W_{ij} \sim \mathcal{N}\!\left(0,\; \frac{2}{n_{\text{in}} + n_{\text{out}}}\right) \quad \text{or} \quad W_{ij} \sim U\!\left[-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}},\; \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right]$$

This keeps the variance of activations roughly constant across layers, preventing them from exploding or vanishing during the forward pass. Designed for sigmoid/tanh activations.

**Kaiming/He initialization (2015):** For ReLU activations, which zero out half the inputs, the variance needs to be doubled:

$$W_{ij} \sim \mathcal{N}\!\left(0,\; \frac{2}{n_{\text{in}}}\right)$$

This is the default initialization in PyTorch for layers followed by ReLU. Use it unless you have a specific reason not to. See [DLBook 8.4](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) for a deeper discussion of initialization strategies.

### Mini-Batch Stochastic Gradient Descent

You saw this in Module 0F. In practice, we almost never use full-batch gradient descent (computing the gradient over the entire dataset) or pure SGD (one example at a time). We use **mini-batch SGD**: compute the gradient over a random subset (batch) of $B$ examples, take a step, repeat.

$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta \cdot \frac{1}{B} \sum_{i \in \text{batch}} \nabla_{\boldsymbol{\theta}} \mathcal{L}(\mathbf{x}_i, y_i; \boldsymbol{\theta})$$

Typical batch sizes: 32, 64, 128, 256. Larger batches give more stable gradient estimates but fewer parameter updates per epoch.

**Terminology:**
- **Epoch:** one complete pass through the entire training set.
- **Iteration (step):** one parameter update (one batch).
- If you have $N$ training examples and batch size $B$, one epoch consists of $\lceil N/B \rceil$ iterations.
""")
    return


@app.cell
def _(np):
    def _run():
        # Weight initialization: observe how scale affects forward pass
        rng = np.random.default_rng(42)
        n_in, n_out = 256, 256
        x_init = rng.standard_normal(n_in)

        # Bad: too large
        W_large = rng.standard_normal((n_out, n_in)) * 1.0
        h_large = np.maximum(0, W_large @ x_init)
        print(f"Std=1.0 init:  activation mean={h_large.mean():.2f}, std={h_large.std():.2f}")

        # Bad: too small
        W_small = rng.standard_normal((n_out, n_in)) * 0.001
        h_small = np.maximum(0, W_small @ x_init)
        print(f"Std=0.001 init: activation mean={h_small.mean():.4f}, std={h_small.std():.4f}")

        # Good: Kaiming/He initialization for ReLU — sqrt(2/n_in)
        W_he = rng.standard_normal((n_out, n_in)) * np.sqrt(2.0 / n_in)
        h_he = np.maximum(0, W_he @ x_init)
        print(f"He init:        activation mean={h_he.mean():.2f}, std={h_he.std():.2f}")

        # Show variance preservation across 10 layers
        h = rng.standard_normal(256)
        for layer in range(10):
            W = rng.standard_normal((256, 256)) * np.sqrt(2.0 / 256)
            h = np.maximum(0, W @ h)
        print(f"\nAfter 10 ReLU layers (He init): mean={h.mean():.2f}, std={h.std():.2f}")


    _run()
    return

@app.cell
def _(mo):
    mo.md(r"""
### The Training Loop in PyTorch

This is the standard pattern you will use for every neural network you train:
""")
    return


@app.cell
def _():
    def _run():
        import torch as _torch
        import torch.nn as nn
        import torch.optim as optim

        # Define model
        model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # Training loop (sketch — would need a dataloader in practice)
        # for epoch in range(num_epochs):
        #     for batch_x, batch_y in dataloader:
        #         # Forward pass
        #         predictions = model(batch_x)
        #         loss = criterion(predictions, batch_y)
        #
        #         # Backward pass
        #         optimizer.zero_grad()   # Clear gradients from previous step
        #         loss.backward()         # Compute gradients via backprop
        #         optimizer.step()        # Update parameters: theta <- theta - eta * grad_L

        print("Model architecture:")
        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nTotal parameters: {total_params:,}")


    _run()
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
Three lines do all the work: `zero_grad()`, `backward()`, `step()`. Everything you learned in this module — the forward pass, the chain rule, the computational graph, the VJPs — happens inside `loss.backward()`. The rest is bookkeeping.

Note `optimizer.zero_grad()`: PyTorch *accumulates* gradients by default (it adds to `.grad` instead of replacing it). You must zero them out before each backward pass, or you get the sum of gradients from multiple batches — not what you want (usually).
""")
    return


@app.cell
def _(mo):
    mo.md(r"""
---

## 10. Why Deep Learning Works (and When It Doesn't)

You now know the mechanics. But *why* do neural networks work so well on so many problems? And when should you not use them?

### Hierarchical Feature Learning

In classical ML, you engineer features by hand: you decide that pixel intensity histograms, edge counts, or word frequencies are the right representation for your data. The model then learns a mapping from your features to outputs.

Neural networks learn their own features. A deep image classifier might learn:
- Layer 1: Edge detectors (horizontal, vertical, diagonal edges)
- Layer 2: Texture and corner detectors (combinations of edges)
- Layer 3: Object part detectors (eyes, wheels, windows)
- Layer 4+: Object and scene detectors

Each layer builds on the representations of the previous layer, creating a **hierarchy of increasingly abstract features**. This is what depth gives you — not just more parameters, but structured, compositional representations. See [DLBook 1.2](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) for an introduction to representation learning and the depth hierarchy.

### Representation Learning

This is the deeper point: the hidden layers of a neural network define a learned, nonlinear **feature transformation**. The output layer then does something simple (linear classification or regression) on top of these learned features. The network simultaneously learns *what to compute* (the representation) and *how to use it* (the output mapping).

This is why neural networks can succeed where feature engineering fails — they discover structure in data that humans might not think to look for.

### When to Use Deep Learning vs Classical ML

Deep learning is not always the right tool. Use it when:
- You have **large amounts of data** (thousands to millions of examples)
- The input is **high-dimensional and unstructured** (images, text, audio)
- **Feature engineering is hard** — you don't know what the right features are
- You have access to **sufficient compute** (GPUs/TPUs)

Use classical ML when:
- Your dataset is small (hundreds of examples)
- Your features are well-understood and tabular
- Interpretability is critical
- You need fast training and inference on limited hardware
- Gradient-boosted trees (XGBoost, LightGBM) often outperform neural networks on tabular data

### The Bitter Lesson

Rich Sutton's 2019 essay ["The Bitter Lesson"](http://www.incompleteideas.net/IncsightNets/BitterLesson.html) makes a provocative claim: over the long run, general methods that leverage computation (search and learning) outperform methods that leverage human domain knowledge. Chess engines, speech recognition, computer vision, and now language models — in each case, hand-engineered approaches were eventually surpassed by simpler methods with more data and compute.

The "bitter" part: researchers who spent years encoding domain knowledge into their systems watched those systems be beaten by brute-force learning. The lesson is not that domain knowledge is useless — it is that **scalable methods win in the long run**, and we should design systems that can effectively use scale.

This is the philosophical underpinning of modern deep learning: build flexible architectures, give them data and compute, and let them learn.
""")
    return


@app.cell
def _(mo):
    mo.md(r"""
---

## Interactive: Neural Network Decision Boundary

Use the slider below to change the number of hidden neurons and see how it affects the decision boundary on a 2D classification task (moons dataset).
""")
    return


@app.cell
def _(mo):
    hidden_neurons_slider = mo.ui.slider(
        start=1, stop=50, value=10, step=1,
        label="Number of hidden neurons"
    )
    hidden_neurons_slider
    return (hidden_neurons_slider,)


@app.cell
def _(hidden_neurons_slider):
    def _run():
        from sklearn.datasets import make_moons
        import torch as _torch2
        import torch.nn as _nn2
        import torch.optim as _optim2

        n_hidden = hidden_neurons_slider.value

        # Generate data
        rng = np.random.default_rng(42)
        X_data, y_data = make_moons(n_samples=300, noise=0.2, random_state=42)
        X_tensor = _torch2.tensor(X_data, dtype=_torch2.float32)
        y_tensor = _torch2.tensor(y_data, dtype=_torch2.float32).unsqueeze(1)

        # Build and train a small network
        _model = _nn2.Sequential(
            _nn2.Linear(2, n_hidden),
            _nn2.ReLU(),
            _nn2.Linear(n_hidden, 1),
            _nn2.Sigmoid()
        )
        _criterion = _nn2.BCELoss()
        _optimizer = _optim2.Adam(_model.parameters(), lr=0.01)

        for _epoch in range(500):
            _pred = _model(X_tensor)
            _loss = _criterion(_pred, y_tensor)
            _optimizer.zero_grad()
            _loss.backward()
            _optimizer.step()

        # Plot decision boundary
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        xx, yy = np.meshgrid(
            np.linspace(X_data[:, 0].min() - 0.5, X_data[:, 0].max() + 0.5, 200),
            np.linspace(X_data[:, 1].min() - 0.5, X_data[:, 1].max() + 0.5, 200)
        )
        grid = _torch2.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=_torch2.float32)
        with _torch2.no_grad():
            zz = _model(grid).numpy().reshape(xx.shape)

        ax.contourf(xx, yy, zz, levels=50, cmap="RdBu", alpha=0.7)
        ax.contour(xx, yy, zz, levels=[0.5], colors="k", linewidths=2)
        ax.scatter(X_data[y_data == 0, 0], X_data[y_data == 0, 1],
                   c="blue", edgecolors="k", s=30, label="Class 0")
        ax.scatter(X_data[y_data == 1, 0], X_data[y_data == 1, 1],
                   c="red", edgecolors="k", s=30, label="Class 1")
        ax.set_title(f"Decision Boundary with {n_hidden} Hidden Neurons (final loss: {_loss.item():.4f})")
        ax.legend()
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        plt.tight_layout()
        fig


    _run()
    return

@app.cell
def _(mo):
    mo.md(r"""
---

## Summary

| Concept | Key Idea |
|---|---|
| Neural network | Composition of affine transformations + nonlinear activations |
| Activation functions | ReLU for most cases; sigmoid/tanh saturate and cause vanishing gradients |
| Universal approximation | MLPs can represent anything, but that does not mean they can *learn* anything |
| Forward pass | Evaluate the network layer by layer; cache intermediate values |
| Backpropagation | Chain rule applied backward through computational graph; computes all gradients in one pass |
| Automatic differentiation | Reverse mode AD = backprop; PyTorch builds dynamic graphs |
| Initialization | Xavier for sigmoid/tanh, Kaiming for ReLU — scale matters |
| Training loop | `zero_grad()` -> `backward()` -> `step()` |

**Key references for this module:**
- [DLBook Chapter 6: Deep Feedforward Networks](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) — the primary reference
- [Bishop Chapter 5: Neural Networks](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) — alternative derivation of backpropagation
- [MML Chapter 5: Vector Calculus](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) — chain rule and Jacobian foundations
- [Murphy PML1 Chapter 13: Neural Networks](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf) — modern treatment with practical guidance
""")
    return


@app.cell
def _(mo):
    mo.md(r"""
---

## Exercises

### Conceptual

1. **Why can't a network of linear layers (without activation functions) learn XOR?** Prove that any composition of affine transformations is itself an affine transformation.

2. **Vanishing gradients.** Consider a 10-layer network where every activation function is sigmoid, and every sigmoid output is near 0.5 (so $\sigma'(z) \approx 0.25$). How much does the gradient shrink as it propagates from layer 10 to layer 1? What if the network uses ReLU instead?

3. **Universal approximation does not imply learnability.** Give three reasons why the Universal Approximation Theorem does not guarantee that gradient descent will find a good solution.

4. **Symmetry breaking.** Explain what happens during training if you initialize all weights in a hidden layer to the same value. Why is random initialization necessary?

### Computational

5. **Backprop by hand.** Using the network and values from Section 6, verify all the gradients computed in Section 7 by computing finite-difference approximations. For each weight $w$, compute $\frac{\mathcal{L}(w + \epsilon) - \mathcal{L}(w - \epsilon)}{2\epsilon}$ with $\epsilon = 10^{-5}$ and check that it matches the backprop gradient.

6. **Implement a neural network from scratch in NumPy.**
   Build a 2-layer MLP (one hidden layer with ReLU, one output layer with identity) and train it on a simple regression task. Your implementation should include:
   - A `forward()` function that returns the prediction *and* caches intermediate values
   - A `backward()` function that takes the cached values and returns gradients for all parameters
   - A training loop with mini-batch SGD

   Generate synthetic data: $y = \sin(x) + \epsilon$ with $\epsilon \sim \mathcal{N}(0, 0.1)$, $x \in [-2\pi, 2\pi]$. Train your network to fit this function. Plot the learned function against the true sine curve.

   **This is the single most important exercise in this module.** If you can implement forward and backward passes from scratch, you understand backpropagation. If you cannot, go back and re-read Sections 6 and 7.

7. **PyTorch verification.** Implement the same network from Exercise 6 in PyTorch. Compare the gradients from your NumPy implementation to PyTorch's `.grad` values on the same input — they should match to floating-point precision.

8. **Activation function exploration.** Train the same architecture on the same data with sigmoid, tanh, ReLU, and GELU activations. Plot learning curves (loss vs epoch) for each. Which converges fastest? Which achieves the lowest final loss?
""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
---

## Code It: Implementation Exercises

Work through these exercises in order. Each builds on the previous one. By the end, you will have a working neural network trained entirely in numpy.
""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
### Exercise 1: Implement Activation Functions

Implement sigmoid, tanh, and ReLU along with their derivatives. Test that the derivatives are correct using finite differences.
""")
    return


@app.cell
def _(np):
    def _run():
        def ex1_sigmoid(z):
            # TODO: implement sigmoid: 1 / (1 + exp(-z))
            pass

        def ex1_sigmoid_deriv(z):
            # TODO: implement sigmoid derivative: sigmoid(z) * (1 - sigmoid(z))
            pass

        def ex1_relu(z):
            # TODO: implement ReLU: max(0, z)
            pass

        def ex1_relu_deriv(z):
            # TODO: implement ReLU derivative: 1 if z > 0, else 0
            pass

        # Test with finite differences: f'(z) ≈ (f(z+eps) - f(z-eps)) / (2*eps)
        _eps = 1e-5
        _z_test = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        # TODO: for each activation, compare your analytic derivative to numerical
        # Example: num_deriv = (ex1_sigmoid(_z_test + _eps) - ex1_sigmoid(_z_test - _eps)) / (2 * _eps)
        # print(f"Sigmoid deriv (analytic): {ex1_sigmoid_deriv(_z_test)}")
        # print(f"Sigmoid deriv (numeric):  {num_deriv}")


    _run()
    return

@app.cell
def _(mo):
    mo.md(r"""
### Exercise 2: Forward Pass Function

Write a forward pass for a 2-layer MLP (input -> hidden with ReLU -> output with identity). Return the prediction and a cache of all intermediate values needed for backprop.
""")
    return


@app.cell
def _(np):
    def _run():
        def ex2_forward(x, W1, b1, W2, b2):
            """
            Forward pass for a 2-layer MLP.
            Args:
                x: input vector (n_in,)
                W1: hidden weights (n_hidden, n_in)
                b1: hidden biases (n_hidden,)
                W2: output weights (n_out, n_hidden)
                b2: output biases (n_out,)
            Returns:
                y_hat: prediction (n_out,)
                cache: dict with intermediate values for backprop
            """
            # TODO: compute z1 = W1 @ x + b1
            z1 = None
            # TODO: compute h1 = ReLU(z1)
            h1 = None
            # TODO: compute y_hat = W2 @ h1 + b2
            y_hat = None

            cache = {"x": x, "z1": z1, "h1": h1}
            return y_hat, cache

        # Test: should match the forward pass example from Section 6
        # _W1 = np.array([[0.5, -0.3], [0.8, 0.2]])
        # _b1 = np.array([0.1, -0.1])
        # _W2 = np.array([[0.6, -0.4]])
        # _b2 = np.array([0.05])
        # _x = np.array([1.0, 2.0])
        # _y_hat, _cache = ex2_forward(_x, _W1, _b1, _W2, _b2)
        # print(f"y_hat = {_y_hat}  (expected: [-0.39])")


    _run()
    return

@app.cell
def _(mo):
    mo.md(r"""
### Exercise 3: Backward Pass Function

Implement the backward pass. Given the cache from the forward pass and the loss gradient, compute gradients for all parameters.
""")
    return


@app.cell
def _(np):
    def _run():
        def ex3_backward(dL_dy_hat, cache, W2):
            """
            Backward pass for a 2-layer MLP.
            Args:
                dL_dy_hat: gradient of loss w.r.t. prediction (n_out,)
                cache: dict from forward pass with x, z1, h1
                W2: output layer weights (needed to propagate gradient back)
            Returns:
                grads: dict with dW1, db1, dW2, db2
            """
            x, z1, h1 = cache["x"], cache["z1"], cache["h1"]

            # TODO: dL/dW2 = dL/dy_hat @ h1^T  (outer product)
            dW2 = None
            # TODO: dL/db2 = dL/dy_hat
            db2 = None

            # TODO: dL/dh1 = W2^T @ dL/dy_hat
            dL_dh1 = None
            # TODO: dL/dz1 = dL/dh1 * ReLU'(z1)
            dL_dz1 = None

            # TODO: dL/dW1 = dL/dz1 @ x^T  (outer product)
            dW1 = None
            # TODO: dL/db1 = dL/dz1
            db1 = None

            return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

        # TODO: test with the example from Section 7
        # Verify against the finite-difference check or the known values


    _run()
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
### Exercise 4: Train on Sine Wave

Put it all together. Generate $y = \sin(x) + \text{noise}$ data, initialize a network with He initialization, and train with mini-batch SGD. Plot the learned function against the true sine curve.
""")
    return


@app.cell
def _(np):
    def _run():
        # Generate data: y = sin(x) + noise
        _rng = np.random.default_rng(42)
        _N = 200
        _x_train = _rng.uniform(-2 * np.pi, 2 * np.pi, _N)
        _y_train = np.sin(_x_train) + _rng.randn(_N) * 0.1

        # Network: 1 input -> 32 hidden (ReLU) -> 1 output
        n_hidden_ex = 32
        lr_ex = 0.01
        epochs_ex = 500
        batch_size_ex = 32

        # TODO: He initialization
        # W1_ex = _rng.randn(n_hidden_ex, 1) * np.sqrt(2.0 / 1)
        # b1_ex = np.zeros(n_hidden_ex)
        # W2_ex = _rng.randn(1, n_hidden_ex) * np.sqrt(2.0 / n_hidden_ex)
        # b2_ex = np.zeros(1)

        # TODO: training loop
        # for epoch in range(epochs_ex):
        #     # shuffle data
        #     perm = _rng.permutation(_N)
        #     for i in range(0, _N, batch_size_ex):
        #         idx = perm[i:i+batch_size_ex]
        #         # forward pass on batch
        #         # compute MSE loss and its gradient: dL/dy_hat = 2/B * (y_hat - y)
        #         # backward pass
        #         # SGD update: param -= lr * grad

        # TODO: plot results
        # import matplotlib.pyplot as plt
        # _x_plot = np.linspace(-2*np.pi, 2*np.pi, 300)
        # predict each point and plot against np.sin(_x_plot)
        print("Implement training loop and plot results")


    _run()
    return

@app.cell
def _(mo):
    mo.md(r"""
### Exercise 5: Gradient Checking

Implement a gradient checker that compares your backprop gradients to finite-difference approximations. This is essential for debugging -- if they do not match, your backward pass has a bug.
""")
    return


@app.cell
def _(np):
    def _run():
        def ex5_gradient_check(forward_fn, params, x, y, eps=1e-5):
            """
            Compare analytic gradients to numerical gradients.
            Args:
                forward_fn: callable(params, x) -> y_hat
                params: list of np arrays (weights and biases)
                x: input, y: target
                eps: finite difference step size
            Returns:
                max_rel_error across all parameters
            """
            # TODO:
            # 1. Run forward + backward to get analytic gradients
            # 2. For each parameter, for each element:
            #    - perturb +eps, compute loss
            #    - perturb -eps, compute loss
            #    - numerical grad = (loss_plus - loss_minus) / (2 * eps)
            # 3. Compute relative error: |analytic - numerical| / max(|analytic|, |numerical|, 1e-8)
            # 4. Return the max relative error (should be < 1e-5)
            pass

        print("Implement gradient checking — essential debugging tool")


    _run()
    return

@app.cell
def _(mo):
    mo.md(r"""
---

**Next module:** [2B - Optimization for Deep Learning](2b_dl_optimization.py) — momentum, Adam, learning rate schedules, and the landscape of neural network loss functions.
""")
    return


if __name__ == "__main__":
    app.run()
