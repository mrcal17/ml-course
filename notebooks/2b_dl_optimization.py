import marimo

app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
# 2B — Optimization for Deep Learning

You know how to compute gradients through a neural network. You know SGD, momentum, and Adam at a mechanical level. Now we need to talk about why training deep networks is genuinely hard, why the tools from convex optimization mostly don't apply, and yet why — somewhat miraculously — it all works anyway.

This module is where the "engineering art" of deep learning lives. The difference between a model that trains beautifully and one that flatlines at high loss is almost always in this material.

Primary reference: [DLBook Ch 8: Optimization for Training Deep Models](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf).
""")
    return


@app.cell
def _(mo):
    mo.md(r"""
---

## 1. How DL Optimization Differs from Convex Optimization

In Module 0F, we optimized convex objectives — least squares, logistic regression. Those problems have a single global minimum, and gradient descent provably converges to it. Life was good.

Neural network loss functions are **non-convex**. The moment you introduce a hidden layer with nonlinear activations, convexity is gone. This means:

- **No guarantee of finding the global minimum.** There may be many local minima, and gradient descent can get stuck in any of them.
- **Saddle points exist.** Points where the gradient is zero but which are neither minima nor maxima.
- **The loss surface is astronomically high-dimensional.** A modest network has millions of parameters — you're navigating a landscape in million-dimensional space.

And yet deep learning works spectacularly well in practice. The gap between theory (which says "this is hopeless") and practice (which says "this trains fine") is one of the most interesting tensions in modern ML. The rest of this module explains why the situation is not as dire as it sounds.

See [DLBook §8.2.1](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) for the discussion of how neural network optimization differs from pure optimization, and [Boyd Ch 4](file:///C:/Users/landa/ml-course/textbooks/Boyd-ConvexOptimization.pdf) if you want a refresher on what convexity actually guarantees.
""")
    return


@app.cell
def _(mo):
    mo.md(r"""
---

## 2. The Loss Landscape of Neural Networks

### Local Minima Are Rarely the Problem

The classical fear was: "gradient descent will get trapped in a bad local minimum." This turns out to be mostly wrong for large networks. Empirically, the local minima that SGD finds tend to have loss values very close to the global minimum. The intuition is that in very high-dimensional spaces, for a point to be a local minimum, the curvature must be positive in *every* direction simultaneously. That's extremely unlikely — it's far more probable that at least one direction curves downward, making the point a saddle point instead.

This was formalized by Dauphin et al. (2014) and connects to results from random matrix theory and statistical physics. The key insight: **the number of local minima grows exponentially with dimension, but almost all of them have similar loss values.**

### Saddle Points — The Real Obstacle

In high dimensions, saddle points vastly outnumber local minima. A saddle point has zero gradient but positive curvature in some directions and negative curvature in others. Gradient descent slows down enormously near saddle points because the gradient magnitude shrinks, but the optimizer hasn't actually converged.

This is one reason momentum and Adam tend to outperform vanilla gradient descent — they carry enough velocity to blow past saddle points rather than stalling at them. See [DLBook §8.2.3](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) for a thorough treatment.

### Flat vs Sharp Minima

Not all minima are created equal. A **sharp minimum** sits in a narrow valley — the loss rises steeply in all directions. A **flat minimum** sits in a broad basin. The generalization argument goes: flat minima are more robust to perturbations in the weights, so models that converge to flat minima should generalize better.

This connects to the batch size debate we'll discuss shortly — large-batch training tends to find sharp minima, while small-batch SGD tends to find flat ones. The story is more nuanced than this (Dinh et al. showed that sharpness measures can be manipulated by reparameterization), but the flat-vs-sharp intuition remains useful.

### Mode Connectivity

A surprising recent finding: different solutions found by independent training runs are often connected by simple low-loss paths through parameter space. You can find a curve (not just a straight line, but a low-dimensional curve) connecting two apparently different minima, along which the loss stays low. This suggests the loss landscape is less rugged than we feared — it's more like a connected valley system than a field of isolated pits.

### Visualizing Loss Surfaces

You can't visualize a million-dimensional surface directly, but you can project it onto 2D. The standard approach: pick two random directions in parameter space, evaluate the loss along a grid, and plot a contour map. Li et al. (2018) introduced **filter normalization** to make these visualizations more meaningful — without it, the scale differences between layers distort the picture. These plots have been invaluable for understanding why, for example, residual connections dramatically smooth the loss landscape.
""")
    return


@app.cell
def _(mo):
    mo.md(r"""
---

## 3. Mini-Batch SGD Revisited

You already know the mechanics from 0F. Now let's understand the subtleties.

### Gradient Noise as Implicit Regularization

Mini-batch SGD doesn't compute the true gradient — it computes a noisy estimate from a random subset of data. This noise is not a bug; it's a feature. The stochasticity helps SGD **escape sharp minima** and settle into flatter regions, which tend to generalize better. This is a form of implicit regularization — you get regularization "for free" just from using mini-batches.

[DLBook §8.1.3](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) covers stochastic gradient estimation.

### Batch Size Tradeoffs

The batch size debate has gone back and forth:

- **Small batches** (32–256): more noise, better generalization (often), but underutilize GPU parallelism.
- **Large batches** (4K–64K): less noise, faster wall-clock time per epoch on modern hardware, but can converge to sharp minima and generalize worse.

The landmark Keskar et al. (2017) paper argued that large batches find sharper minima. Subsequent work (Goyal et al., Hoffer et al.) showed this can be mitigated with proper learning rate scaling and warmup.

### Critical Batch Size

There's a concept called the **critical batch size** — the batch size beyond which doubling the batch size no longer halves the number of optimization steps needed. Below this threshold, larger batches give you near-linear speedup. Above it, you're wasting compute. The critical batch size depends on the task and model, and estimating it can guide efficient training.

### Linear Scaling Rule

When you increase the batch size by a factor of $k$, you should increase the learning rate by the same factor $k$. The intuition: a larger batch gives you a more accurate gradient estimate, so you can afford to take a larger step. This rule works well up to a point (roughly the critical batch size), after which it breaks down and you need warmup to stabilize training.
""")
    return


@app.cell
def _(mo):
    mo.md(r"""
---

## 4. Learning Rate — The Most Important Hyperparameter

If you tune only one thing, tune the learning rate. A learning rate that's too high causes divergence. Too low, and training is painfully slow or gets stuck. The optimal learning rate also changes over the course of training — you want to move fast early and slow down later.

[DLBook §8.5.1](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) discusses learning rate as a hyperparameter; [Murphy PML1 §8.4](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf) gives an additional perspective.

### Learning Rate Warmup

Start training with a very small learning rate and ramp it up linearly over the first few hundred or thousand steps. Why? At initialization, the network weights are random, and the gradients can be unreliable — large steps based on bad gradient estimates can send parameters into a bad region from which recovery is slow. Warmup is especially critical for large batch training and when using Adam (which has noisy second-moment estimates early on due to bias correction).

### Step Decay and Exponential Decay

**Step decay**: multiply the learning rate by a factor (e.g., 0.1) at predetermined epochs. Classic approach — drop the LR at epochs 30, 60, 90 for a 100-epoch schedule.

**Exponential decay**: $\eta_t = \eta_0 \cdot \gamma^t$ for some decay factor $\gamma < 1$. Smoother than step decay but requires choosing $\gamma$ carefully.

Both work but have been largely supplanted by cosine annealing.

### Cosine Annealing — The Modern Default

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$$

The learning rate follows a half-cosine from $\eta_{\max}$ down to $\eta_{\min}$ over $T$ steps. This has become the default schedule in most modern training recipes because it requires minimal tuning — just set the max LR, min LR (often 0), and total steps. The smooth decay avoids the abrupt transitions of step decay.

### Cyclical Learning Rates

Instead of monotonically decreasing the learning rate, oscillate it between a minimum and maximum value. The idea (Leslie Smith, 2017) is that periodically increasing the LR helps escape saddle points and local minima, while the decreasing phases allow convergence.

### Learning Rate Range Test (Smith's Method)

A practical technique: start with a tiny learning rate and increase it exponentially over one epoch (or a few hundred steps), recording the loss at each step. Plot loss vs learning rate. The optimal learning rate is typically somewhere in the region where the loss is decreasing most steeply — before it starts diverging. This takes minutes and saves hours of hyperparameter search.

### One-Cycle Policy

Smith's full recipe: use a single cycle of learning rate that goes from low -> high -> low over the entire training run, combined with an inverse cycle for momentum (high -> low -> high). The rising phase acts as warmup and exploration; the falling phase acts as annealing. The final phase often uses a learning rate much lower than the initial one. This policy consistently produces strong results with minimal tuning.
""")
    return


@app.cell
def _(mo):
    mo.md(r"""
---

## 5. Optimizers in Depth

Building on what you learned in 0F. Reference: [DLBook §8.5: Algorithms with Adaptive Learning Rates](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf).

### SGD + Momentum (Brief Recap)

Momentum maintains an exponentially decaying moving average of past gradients:

$$v_t = \beta v_{t-1} + \nabla L(\theta_t), \quad \theta_{t+1} = \theta_t - \eta \, v_t$$

This accelerates movement along consistent gradient directions and dampens oscillations. Typical $\beta = 0.9$. Nesterov momentum (looking ahead before computing the gradient) gives a modest improvement. SGD with momentum remains a very competitive optimizer, especially for vision tasks.

### Adam: Understanding the Internals

Adam tracks two running averages:
- **First moment** $m_t$ (mean of gradients) — controlled by $\beta_1$ (default 0.9)
- **Second moment** $v_t$ (mean of squared gradients) — controlled by $\beta_2$ (default 0.999)

The $\epsilon$ term (default $10^{-8}$) prevents division by zero when $v_t$ is small. **Bias correction** divides by $(1 - \beta^t)$ because the running averages are initialized to zero and are biased toward zero in early steps. Without correction, the first few updates would be too small.

The key insight of Adam: it adapts the learning rate per parameter. Parameters with historically large gradients get smaller effective learning rates, and vice versa. This is why Adam often converges faster than SGD in the early stages of training.

[DLBook §8.5.3 (Adam)](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) covers the algorithm. Also see [Murphy PML1 §8.4.6](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf).

### AdamW — Decoupled Weight Decay

This is subtle and important. In SGD, L2 regularization and weight decay are mathematically equivalent. **In Adam, they are not.** Standard L2 regularization adds $\lambda \theta$ to the gradient, which then gets scaled by Adam's adaptive learning rate — so heavily updated parameters get *less* regularization, which is the opposite of what you want.

**AdamW** fixes this by applying weight decay directly to the parameter update, *outside* the adaptive scaling:

$$\theta_{t+1} = (1 - \lambda)\theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Loshchilov & Hutter (2019) showed this makes a real difference. **AdamW is the default optimizer in most modern deep learning.** If you're using Adam with weight decay, you should almost certainly be using AdamW instead.

### When SGD Beats Adam

There's a persistent observation that SGD (with properly tuned momentum and learning rate schedule) sometimes achieves better test accuracy than Adam, despite Adam converging faster in training loss. This "generalization gap" has been debated extensively. Possible explanations include Adam's adaptive learning rates allowing it to settle into sharper minima, and the implicit regularization of SGD's more uniform gradient noise.

In practice: **Adam/AdamW converges faster and is easier to tune. SGD sometimes wins on final accuracy but requires more careful tuning.** Vision tasks (especially ImageNet) have traditionally favored SGD; NLP and transformer models almost universally use AdamW.

### LAMB and LARS (Brief)

For truly large-batch training (batch sizes in the thousands or tens of thousands), standard optimizers struggle even with linear scaling. **LARS** (Layer-wise Adaptive Rate Scaling) and **LAMB** (Layer-wise Adaptive Moments for Batch training) adjust the learning rate per-layer based on the ratio of weight norm to gradient norm. LAMB is essentially AdamW with layer-wise scaling. These are specialized tools for distributed training at scale — you probably won't need them unless you're training on hundreds of GPUs.
""")
    return


@app.cell
def _(mo):
    mo.md(r"""
---

## 6. Batch Normalization

Batch normalization (Ioffe & Szegedy, 2015) is one of the most practically important techniques in deep learning. It transformed what architectures are trainable and how fast they converge.

[DLBook §8.7.1 (Batch Normalization)](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf).

### Internal Covariate Shift — The Original Motivation

The original paper argued that training is hard because each layer's input distribution shifts as the preceding layers' weights change — the "internal covariate shift" (ICS) problem. Batch normalization fixes this by normalizing each layer's inputs to zero mean and unit variance.

This motivation is now considered **debatable at best**. Santurkar et al. (2018) showed that batch normalization doesn't actually reduce internal covariate shift, and can even increase it. But it still helps dramatically. The real reasons are likely different (see below).

### The Algorithm

For a mini-batch $\{x_1, \ldots, x_m\}$ at a given layer:

1. Compute mini-batch mean: $\mu_B = \frac{1}{m}\sum x_i$
2. Compute mini-batch variance: $\sigma_B^2 = \frac{1}{m}\sum (x_i - \mu_B)^2$
3. Normalize: $\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$
4. Scale and shift: $y_i = \gamma \hat{x}_i + \beta$

The $\gamma$ and $\beta$ are **learnable parameters** — they allow the network to undo the normalization if that's what's optimal. This is important: without them, you'd be constraining the network's representational power.

### Training vs Inference

During training, you use the mini-batch statistics. During inference, you use **running averages** of mean and variance accumulated during training (exponential moving averages). This is a common source of bugs — if your model behaves differently at train and test time, check that batch norm is in the correct mode.

### Why It Actually Helps

The more convincing explanations:

- **Smoother loss landscape.** Santurkar et al. showed that batch normalization makes the loss surface significantly smoother (smaller Lipschitz constant for the loss and gradients). This allows larger learning rates and more stable training.
- **Enables higher learning rates.** By preventing activations from blowing up, you can use much more aggressive learning rates.
- **Mild regularization.** The noise from using mini-batch statistics acts as a regularizer, similar to dropout.

### Layer Normalization

Batch norm normalizes across the batch dimension. **Layer normalization** normalizes across the feature dimension within each single example. This means it doesn't depend on batch size and works identically at training and inference time. Layer norm is the standard for transformers and sequence models, where batch statistics are less meaningful.

See [Murphy PML2 §16.2](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML2.pdf) for comparisons between normalization variants.

### Group Normalization

A middle ground: divide channels into groups and normalize within each group. Group norm is independent of batch size (like layer norm) but preserves some of the channel structure (like batch norm). It's the go-to when batch size is very small (e.g., in object detection or segmentation where large images force small batches).
""")
    return


@app.cell
def _(mo):
    mo.md(r"""
---

## 7. Gradient Flow Problems

The fundamental challenge of training deep networks: gradients must flow backward through many layers, and things can go wrong.

[DLBook §8.2.5: Cliffs and Exploding Gradients](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf).

### Vanishing Gradients

When gradients are multiplied through many layers, if each multiplication shrinks the gradient (e.g., sigmoid saturation, where the derivative approaches zero), the gradient reaching the early layers becomes negligibly small. Those layers effectively stop learning. This is why deep sigmoid/tanh networks are hard to train, and why ReLU activations were so important — ReLU has a derivative of exactly 1 for positive inputs, so gradients flow through unchanged.

See [DLBook §8.2.5](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) and [Bishop §5.3.4](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) for background on gradient computation through deep networks.

### Exploding Gradients

The opposite problem: if the effective per-layer Jacobian has eigenvalues greater than 1, gradients grow exponentially with depth. You see NaN losses and parameter values shooting to infinity.

The fix is **gradient clipping**: cap the gradient norm at some maximum value.

- **Norm clipping**: if $\|\nabla L\| > c$, rescale to $\frac{c}{\|\nabla L\|} \nabla L$. This preserves gradient direction.
- **Value clipping**: clip each gradient component independently to $[-c, c]$. Simpler but changes direction.

Norm clipping is generally preferred. A common threshold is 1.0 for transformers.

### Skip / Residual Connections

The most important architectural fix for gradient flow. Instead of computing $y = F(x)$, compute:

$$y = F(x) + x$$

The $+x$ creates a "gradient highway" — during backpropagation, the gradient flows directly through the identity shortcut, bypassing the potentially gradient-destroying layers. This is the core idea of ResNets (He et al., 2016), and it's why we can train networks hundreds of layers deep. We'll cover ResNets in detail in [2D - Convolutional Neural Networks](2d_cnn.py).

### Gradient Checkpointing

A practical memory optimization. Normally, during the forward pass, you store all intermediate activations (needed for backpropagation). For very deep networks, this consumes enormous memory.

**Gradient checkpointing** saves memory by only storing activations at certain "checkpoint" layers. During backpropagation, when you need an intermediate activation that wasn't stored, you recompute it from the nearest checkpoint. This trades **compute for memory** — roughly $O(\sqrt{n})$ memory instead of $O(n)$ for $n$ layers, at the cost of one additional forward pass.
""")
    return


@app.cell
def _(mo):
    mo.md(r"""
---

## 8. Practical Training Recipes

Bringing it all together. Here's what a modern training pipeline actually looks like.

### Weight Initialization (Revisited)

You covered this in 2A, but the connection to optimization is worth restating. Xavier/Glorot initialization (for tanh/sigmoid) and He/Kaiming initialization (for ReLU) are designed to keep activation variances stable across layers, preventing the forward-pass analog of vanishing/exploding gradients. Bad initialization can make even a good optimizer fail.

[DLBook §8.4: Parameter Initialization Strategies](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf).

### Mixed-Precision Training (FP16 / BF16)

Modern GPUs have specialized hardware (Tensor Cores) that compute much faster in 16-bit floating point than in 32-bit. **Mixed-precision training** keeps a master copy of weights in FP32 but performs forward and backward passes in FP16 (or BF16). This roughly halves memory usage and can double throughput.

The key challenge is **loss scaling**: FP16 has a small dynamic range, so gradients can underflow to zero. The fix is to multiply the loss by a large scaling factor before backpropagation, then divide the gradients by the same factor before the optimizer step. Modern frameworks (PyTorch's `torch.cuda.amp`, TensorFlow's mixed precision API) handle this automatically.

**BF16** (bfloat16) has the same exponent range as FP32 but fewer mantissa bits, so it rarely needs loss scaling. It's become the default on newer hardware (A100, H100, TPUs).

### Gradient Accumulation

If you want a logical batch size of 4096 but can only fit 256 samples in GPU memory, you can accumulate gradients over 16 forward-backward passes before taking an optimizer step. This is mathematically equivalent to using a batch size of 4096 (assuming no batch normalization complications).

The implementation is simple: skip `optimizer.step()` and `optimizer.zero_grad()` for $k-1$ steps, then do both on the $k$-th step. Just remember to scale the loss by $1/k$ (or equivalently, the gradients).

### The Typical Training Pipeline

Putting it all together, here's a recipe that works well across most deep learning tasks:

1. **Choose architecture** and initialize with He or Xavier (depending on activation function).
2. **Use AdamW** with weight decay 0.01–0.1.
3. **Learning rate finder**: run Smith's range test to find the right ballpark.
4. **Warmup** (linear, over 1–5% of total training steps).
5. **Train** at the peak learning rate.
6. **Anneal** with cosine decay to ~1% of peak LR (or to zero).
7. **Gradient clipping** at norm 1.0 (especially for transformers and RNNs).
8. **Batch normalization** for CNNs, **layer normalization** for transformers.
9. **Mixed precision** if hardware supports it.
10. **Monitor**: track training loss, validation loss, gradient norms, and learning rate. Divergence in gradient norms is an early warning sign.

This is not the only recipe, but it's a solid default. The specific choices (optimizer, schedule, normalization) interact with each other, so changing one may require adjusting others.
""")
    return


@app.cell
def _(mo):
    mo.md(r"""
---

## Summary

| Concept | Key Takeaway |
|---|---|
| Non-convex optimization | No guarantees, but empirically fine — most local minima are good enough |
| Saddle points | The real enemy in high dimensions; momentum/Adam help escape them |
| Batch size | Smaller batches -> implicit regularization; larger -> hardware efficiency |
| Learning rate schedule | Cosine annealing with warmup is the modern default |
| AdamW | The default optimizer — properly decouples weight decay from adaptive LR |
| Batch normalization | Smooths the loss landscape; use layer norm for transformers |
| Gradient clipping | Essential for preventing exploding gradients in deep/recurrent networks |
| Mixed precision | Nearly free 2x speedup and memory savings on modern hardware |
""")
    return


@app.cell
def _(mo):
    mo.md(r"""
---

## Practice Exercises

1. **Learning Rate Finder**: Implement Smith's learning rate range test in PyTorch. Start with LR = $10^{-7}$ and increase exponentially to $10$. Plot loss vs LR on a log scale. Identify the steepest descent region and the divergence point.

2. **Batch Size Experiment**: Train a small CNN on CIFAR-10 with batch sizes 32, 128, 512, and 2048. For each, try (a) the same learning rate and (b) linearly scaled learning rate. Compare training curves and final test accuracy. At what batch size does linear scaling break down?

3. **Optimizer Comparison**: Train the same network with SGD+momentum, Adam, and AdamW (all with cosine annealing). Compare convergence speed and final test accuracy. Does the "generalization gap" between SGD and Adam appear?

4. **Batch Norm vs Layer Norm**: Take a simple CNN and replace all batch norm layers with layer norm (and vice versa). How does training speed and final accuracy change? Now try with batch size 4 — does the comparison shift?

5. **Gradient Clipping**: Train an RNN (or deep MLP with 20+ layers) without gradient clipping. Monitor gradient norms — do you observe explosions? Add norm clipping and compare stability.

6. **Cosine vs Step Decay**: Train the same model with (a) step decay (drop LR by 10x at epochs 30, 60, 90), (b) cosine annealing, and (c) one-cycle policy. Compare training curves. Which converges faster? Which gives the best final accuracy?

7. **Sharp vs Flat Minima** (Conceptual): Explain in your own words why a flat minimum might generalize better than a sharp one. Then describe a scenario where this intuition could be misleading (hint: reparameterization). Reference the Dinh et al. critique.

---

**Next**: [2C - Regularization for Deep Learning](2c_regularization.py) — dropout, data augmentation, early stopping, and other techniques that prevent overfitting in deep networks.
""")
    return


if __name__ == "__main__":
    app.run()
