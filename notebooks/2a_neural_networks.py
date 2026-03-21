import marimo

app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
# Module 2A: Neural Networks & Backpropagation

Welcome to deep learning. Everything you have built so far — linear regression, logistic regression, gradient descent, matrix calculus — was preparation for this moment. You already know the pieces. Now we assemble them into something far more powerful.

This module is long and dense. That is because backpropagation is the single most important algorithm in modern machine learning. If you understand it deeply — not just "the chain rule applied repeatedly" but *why* it works, *how* it flows through a computational graph, and *what* it computes at each step — then every model you encounter from here on will make sense. Transformers, CNNs, GANs, diffusion models: they are all just architectures. Backprop trains all of them.

Take your time with this one.
""")
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
Every tensor with `requires_grad=True` accumulates its gradient. The entire system is built on the same chain rule computations we derived by hand above. There is no magic.

### Karpathy's micrograd

Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) is a roughly 100-line implementation of a scalar-valued autograd engine. It builds a computational graph of `Value` objects, each storing a single number and its gradient. Calling `.backward()` on the output traverses the graph and populates gradients everywhere. If you want to truly understand how autograd works at the implementation level, reading and reimplementing micrograd is one of the best exercises you can do. It strips away all the tensor/GPU complexity and exposes the core idea.
""")
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
### The Training Loop in PyTorch

This is the standard pattern you will use for every neural network you train:
""")
    return


@app.cell
def _():
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
Three lines do all the work: `zero_grad()`, `backward()`, `step()`. Everything you learned in this module — the forward pass, the chain rule, the computational graph, the VJPs — happens inside `loss.backward()`. The rest is bookkeeping.

Note `optimizer.zero_grad()`: PyTorch *accumulates* gradients by default (it adds to `.grad` instead of replacing it). You must zero them out before each backward pass, or you get the sum of gradients from multiple batches — not what you want (usually).
""")
    return


@app.cell(hide_code=True)
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
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_moons
    import torch as _torch2
    import torch.nn as _nn2
    import torch.optim as _optim2

    n_hidden = hidden_neurons_slider.value

    # Generate data
    np.random.seed(42)
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
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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

---

**Next module:** [2B - Optimization for Deep Learning](2b_dl_optimization.py) — momentum, Adam, learning rate schedules, and the landscape of neural network loss functions.
""")
    return


if __name__ == "__main__":
    app.run()
