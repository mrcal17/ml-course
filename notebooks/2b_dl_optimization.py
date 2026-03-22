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
    return np, plt


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
def _(np, plt):
    def _run():
        # Convex vs non-convex: a 1D illustration
        x_landscape = np.linspace(-3, 3, 500)
        convex_f = x_landscape ** 2                          # single global min
        nonconvex_f = x_landscape ** 4 - 3 * x_landscape ** 2 + 1  # multiple local minima

        fig_cvx, axes_cvx = plt.subplots(1, 2, figsize=(9, 3))
        axes_cvx[0].plot(x_landscape, convex_f); axes_cvx[0].set_title("Convex loss")
        axes_cvx[1].plot(x_landscape, nonconvex_f); axes_cvx[1].set_title("Non-convex loss (neural-net-like)")
        for ax in axes_cvx: ax.set_xlabel("θ"); ax.set_ylabel("L(θ)")
        fig_cvx.tight_layout()
        fig_cvx


    _run()
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
def _(np, plt):
    def _run():
        # Saddle point vs local minimum in 2D
        u = np.linspace(-2, 2, 200)
        v = np.linspace(-2, 2, 200)
        U, V = np.meshgrid(u, v)

        # Saddle: f = x^2 - y^2 (positive curvature in x, negative in y)
        saddle = U ** 2 - V ** 2
        # Local min: f = x^2 + y^2
        bowl = U ** 2 + V ** 2

        fig_sp, axes_sp = plt.subplots(1, 2, figsize=(9, 3.5))
        axes_sp[0].contourf(U, V, bowl, levels=30, cmap="viridis")
        axes_sp[0].set_title("Local minimum (all curvatures +)")
        axes_sp[1].contourf(U, V, saddle, levels=30, cmap="RdBu_r")
        axes_sp[1].set_title("Saddle point (mixed curvatures)")
        for ax in axes_sp: ax.set_xlabel("θ₁"); ax.set_ylabel("θ₂")
        fig_sp.tight_layout()
        fig_sp


    _run()
    return

@app.cell
def _(np, plt):
    def _run():
        # Flat vs sharp minima: same loss value, different curvature
        x_fs = np.linspace(-3, 3, 500)
        sharp_min = 10 * x_fs ** 2             # high curvature → sharp
        flat_min = 0.3 * x_fs ** 2             # low curvature  → flat

        fig_fs, ax_fs = plt.subplots(figsize=(6, 3))
        ax_fs.plot(x_fs, sharp_min, label="Sharp minimum (high curvature)")
        ax_fs.plot(x_fs, flat_min, label="Flat minimum (low curvature)")
        ax_fs.set_ylim(-0.5, 5); ax_fs.set_xlabel("θ"); ax_fs.set_ylabel("L(θ)")
        ax_fs.legend(); ax_fs.set_title("Flat minima → more robust to weight perturbations")
        fig_fs.tight_layout()
        fig_fs


    _run()
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
def _(np, plt):
    def _run():
        # Mini-batch gradient noise: full-batch vs mini-batch gradient estimates
        rng = np.random.default_rng(0)
        N_data = 1000
        # Synthetic: true gradient is 2*theta for L = theta^2
        theta_val = 3.0
        true_grad = 2 * theta_val  # = 6.0

        # Simulate per-sample gradients with noise
        per_sample_grads = true_grad + rng.standard_normal(N_data) * 4.0

        batch_sizes_demo = [1, 8, 32, 128, 512]
        fig_mb, ax_mb = plt.subplots(figsize=(7, 3))
        for bs in batch_sizes_demo:
            estimates = [per_sample_grads[i:i+bs].mean() for i in range(0, N_data - bs, bs)]
            ax_mb.hist(estimates, bins=20, alpha=0.5, label=f"BS={bs}", density=True)
        ax_mb.axvline(true_grad, color="k", ls="--", label="True gradient")
        ax_mb.set_xlabel("Gradient estimate"); ax_mb.legend(fontsize=8)
        ax_mb.set_title("Larger batches → less noisy gradient estimates")
        fig_mb.tight_layout()
        fig_mb


    _run()
    return

@app.cell
def _(np):
    def _run():
        # Linear scaling rule: scale LR proportional to batch size
        base_lr = 0.01
        base_bs = 32
        batch_sizes_lr = np.array([32, 64, 128, 256, 512, 1024])
        # Scaled LR = base_lr * (bs / base_bs)
        scaled_lrs = base_lr * (batch_sizes_lr / base_bs)
        for bs, lr in zip(batch_sizes_lr, scaled_lrs):
            print(f"  Batch size {bs:>5d} → LR = {lr:.4f}")


    _run()
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
def _(np, plt):
    def _run():
        # Learning rate schedules: warmup + cosine, step decay, exponential decay
        T_total = 1000
        warmup_steps = 100
        eta_max, eta_min = 1e-3, 1e-6
        t_steps = np.arange(T_total)

        # Cosine annealing with linear warmup
        cosine_lr = np.where(
            t_steps < warmup_steps,
            eta_max * t_steps / warmup_steps,                              # warmup phase
            eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * (t_steps - warmup_steps) / (T_total - warmup_steps)))
        )

        # Step decay: drop by 0.1 at 30%, 60%, 90% of training
        step_lr = np.full(T_total, eta_max)
        for frac in [0.3, 0.6, 0.9]:
            step_lr[int(frac * T_total):] *= 0.1

        # Exponential decay: gamma^t
        gamma_exp = 0.995
        exp_lr = eta_max * gamma_exp ** t_steps

        fig_lr, ax_lr = plt.subplots(figsize=(8, 3.5))
        ax_lr.plot(t_steps, cosine_lr, label="Cosine + warmup")
        ax_lr.plot(t_steps, step_lr, label="Step decay")
        ax_lr.plot(t_steps, exp_lr, label="Exponential decay", alpha=0.7)
        ax_lr.set_xlabel("Step"); ax_lr.set_ylabel("Learning rate")
        ax_lr.set_yscale("log"); ax_lr.legend(); ax_lr.set_title("Common LR Schedules")
        fig_lr.tight_layout()
        fig_lr


    _run()
    return

@app.cell
def _(np, plt):
    def _run():
        # One-cycle policy: LR goes low → high → very low
        T_onecycle = 1000
        peak_frac = 0.3  # peak at 30% of training
        t_oc = np.arange(T_onecycle)
        lr_max_oc, lr_init_oc, lr_final_oc = 1e-3, 1e-4, 1e-6
        peak_step = int(peak_frac * T_onecycle)

        oc_lr = np.where(
            t_oc < peak_step,
            lr_init_oc + (lr_max_oc - lr_init_oc) * t_oc / peak_step,               # ramp up
            lr_max_oc - (lr_max_oc - lr_final_oc) * (t_oc - peak_step) / (T_onecycle - peak_step)  # anneal down
        )

        fig_oc, ax_oc = plt.subplots(figsize=(7, 3))
        ax_oc.plot(t_oc, oc_lr)
        ax_oc.set_xlabel("Step"); ax_oc.set_ylabel("LR")
        ax_oc.set_title("One-Cycle Policy: low → high → very low")
        fig_oc.tight_layout()
        fig_oc


    _run()
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
def _(np, plt):
    def _run():
        # SGD vs Momentum vs Adam on a toy 2D surface: f(x, y) = x^2 + 10*y^2
        # Elongated bowl — momentum helps traverse the narrow valley faster
        def toy_loss(p):
            return p[0]**2 + 10 * p[1]**2
        def toy_grad(p):
            return np.array([2*p[0], 20*p[1]])

        def run_sgd(lr, steps, beta=0.0):
            """SGD with optional momentum (beta=0 → vanilla SGD)."""
            pos = np.array([-4.0, 3.0])
            vel = np.zeros(2)
            traj = [pos.copy()]
            for _ in range(steps):
                g = toy_grad(pos)
                vel = beta * vel + g        # v_t = β v_{t-1} + ∇L
                pos = pos - lr * vel        # θ_{t+1} = θ_t - η v_t
                traj.append(pos.copy())
            return np.array(traj)

        def run_adam(lr, steps, b1=0.9, b2=0.999, eps=1e-8):
            pos = np.array([-4.0, 3.0])
            m, v_ad = np.zeros(2), np.zeros(2)
            traj = [pos.copy()]
            for t_ad in range(1, steps + 1):
                g = toy_grad(pos)
                m = b1 * m + (1 - b1) * g           # first moment
                v_ad = b2 * v_ad + (1 - b2) * g**2  # second moment
                m_hat = m / (1 - b1**t_ad)           # bias correction
                v_hat = v_ad / (1 - b2**t_ad)
                pos = pos - lr * m_hat / (np.sqrt(v_hat) + eps)
                traj.append(pos.copy())
            return np.array(traj)

        n_steps = 50
        traj_sgd = run_sgd(lr=0.02, steps=n_steps, beta=0.0)
        traj_mom = run_sgd(lr=0.02, steps=n_steps, beta=0.9)
        traj_adam = run_adam(lr=0.3, steps=n_steps)

        # Plot trajectories over contour
        xs_opt = np.linspace(-5, 5, 200)
        ys_opt = np.linspace(-4, 4, 200)
        Xo, Yo = np.meshgrid(xs_opt, ys_opt)
        Zo = Xo**2 + 10*Yo**2

        fig_opt, ax_opt = plt.subplots(figsize=(7, 4.5))
        ax_opt.contour(Xo, Yo, Zo, levels=30, cmap="Blues", alpha=0.6)
        ax_opt.plot(*traj_sgd.T, "o-", ms=2, label="Vanilla SGD", alpha=0.8)
        ax_opt.plot(*traj_mom.T, "s-", ms=2, label="SGD + Momentum (β=0.9)", alpha=0.8)
        ax_opt.plot(*traj_adam.T, "^-", ms=2, label="Adam", alpha=0.8)
        ax_opt.set_title("Optimizer trajectories on f(x,y) = x² + 10y²")
        ax_opt.legend(fontsize=8); ax_opt.set_xlabel("θ₁"); ax_opt.set_ylabel("θ₂")
        fig_opt.tight_layout()
        fig_opt


    _run()
    return

@app.cell
def _(np):
    def _run():
        # Adam vs AdamW: the weight decay difference
        # Adam with L2: gradient includes λθ, which gets scaled by adaptive LR
        # AdamW: weight decay applied directly to θ, outside adaptive scaling

        theta_aw = 2.0
        grad_aw = 0.5
        lr_aw = 0.01
        lam_aw = 0.1
        # Assume m_hat=grad, v_hat=grad^2 (steady state, one param)

        # Adam + L2 reg: gradient becomes (grad + λ*θ), then divided by sqrt(v)
        g_l2 = grad_aw + lam_aw * theta_aw  # 0.5 + 0.2 = 0.7
        # Effective update ∝ g_l2 / sqrt(g_l2^2) = sign(g_l2) — decay gets swallowed!
        print("Adam + L2: effective gradient =", g_l2)
        print("  → adaptive scaling makes weight decay strength depend on gradient magnitude")

        # AdamW: adaptive step on grad only, then subtract λ*θ separately
        adam_step = grad_aw / abs(grad_aw)  # sign
        adamw_update = lr_aw * adam_step + lam_aw * theta_aw  # adaptive + explicit decay
        print(f"\nAdamW: adaptive step = {lr_aw * adam_step:.4f}, weight decay = {lam_aw * theta_aw:.4f}")
        print("  → weight decay applied uniformly, independent of gradient history")


    _run()
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
def _(np):
    def _run():
        # Batch Normalization from scratch (forward pass)
        rng = np.random.default_rng(42)
        # Simulated mini-batch: 4 samples, 3 features
        X_bn = rng.standard_normal((4, 3)) * 5 + 10  # mean ≈ 10, std ≈ 5
        eps_bn = 1e-5

        # Step 1-2: batch mean and variance
        mu_B = X_bn.mean(axis=0)                             # (3,)
        var_B = X_bn.var(axis=0)                              # (3,)

        # Step 3: normalize
        X_hat = (X_bn - mu_B) / np.sqrt(var_B + eps_bn)      # zero mean, unit var

        # Step 4: scale and shift (learnable γ, β — initialized to 1 and 0)
        gamma_bn, beta_bn = np.ones(3), np.zeros(3)
        Y_bn = gamma_bn * X_hat + beta_bn                    # γ * x̂ + β

        print("Input X (before BN):")
        print(f"  mean = {X_bn.mean(axis=0).round(2)},  std = {X_bn.std(axis=0).round(2)}")
        print("Output Y (after BN):")
        print(f"  mean = {Y_bn.mean(axis=0).round(4)},  std = {Y_bn.std(axis=0).round(4)}")


    _run()
    return

@app.cell
def _(np):
    def _run():
        # Layer Norm vs Batch Norm: which axis do we normalize over?
        rng = np.random.default_rng(7)
        X_norm = rng.standard_normal((4, 5)) * 3 + 2  # (batch=4, features=5)
        eps_ln = 1e-5

        # Batch norm: normalize across batch dim (axis=0), per feature
        bn_mean = X_norm.mean(axis=0)   # shape (5,)
        bn_var = X_norm.var(axis=0)
        X_batchnorm = (X_norm - bn_mean) / np.sqrt(bn_var + eps_ln)

        # Layer norm: normalize across feature dim (axis=1), per sample
        ln_mean = X_norm.mean(axis=1, keepdims=True)  # shape (4, 1)
        ln_var = X_norm.var(axis=1, keepdims=True)
        X_layernorm = (X_norm - ln_mean) / np.sqrt(ln_var + eps_ln)

        print("Batch Norm — each feature has mean≈0 across batch:")
        print(f"  Column means: {X_batchnorm.mean(axis=0).round(6)}")
        print("Layer Norm — each sample has mean≈0 across features:")
        print(f"  Row means:    {X_layernorm.mean(axis=1).round(6)}")


    _run()
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
def _(np, plt):
    def _run():
        # Vanishing gradients: multiply through many layers
        # Gradient at layer k = product of per-layer derivatives
        n_layers = 30

        # Sigmoid derivative max is 0.25; ReLU derivative is 1 (for positive inputs)
        sigmoid_factor = 0.25   # worst-case sigmoid derivative
        relu_factor = 1.0       # ReLU derivative (positive region)
        tanh_factor = 0.65      # typical tanh derivative

        layers = np.arange(1, n_layers + 1)
        grad_sigmoid = sigmoid_factor ** layers
        grad_relu = relu_factor ** layers
        grad_tanh = tanh_factor ** layers

        fig_vg, ax_vg = plt.subplots(figsize=(7, 3.5))
        ax_vg.plot(layers, grad_sigmoid, label=f"Sigmoid (factor={sigmoid_factor})")
        ax_vg.plot(layers, grad_tanh, label=f"Tanh (factor={tanh_factor})")
        ax_vg.plot(layers, grad_relu, label=f"ReLU (factor={relu_factor})", ls="--")
        ax_vg.set_xlabel("Depth (layers)"); ax_vg.set_ylabel("Gradient magnitude")
        ax_vg.set_yscale("log"); ax_vg.legend()
        ax_vg.set_title("Gradient magnitude vs depth — sigmoid/tanh vanish, ReLU preserves")
        fig_vg.tight_layout()
        fig_vg


    _run()
    return

@app.cell
def _(np):
    def _run():
        # Gradient clipping: norm clipping preserves direction
        gradient = np.array([3.0, 4.0])  # norm = 5.0
        max_norm = 2.0

        grad_norm = np.linalg.norm(gradient)
        if grad_norm > max_norm:
            clipped = gradient * (max_norm / grad_norm)  # rescale to max_norm
        else:
            clipped = gradient

        print(f"Original gradient: {gradient},  norm = {grad_norm:.1f}")
        print(f"Clipped gradient:  {clipped.round(4)},  norm = {np.linalg.norm(clipped):.1f}")
        print(f"Direction preserved: {(gradient / grad_norm).round(4)} == {(clipped / np.linalg.norm(clipped)).round(4)}")


    _run()
    return

@app.cell
def _(np):
    def _run():
        # Residual connection: y = F(x) + x
        # During backprop: dy/dx = dF/dx + 1 (the +1 ensures gradient flows!)
        rng = np.random.default_rng(3)
        x_res = rng.standard_normal(4)

        # Simulate a "layer" F that shrinks its input (bad for gradient flow)
        W_res = rng.standard_normal((4, 4)) * 0.3  # small weights → small Jacobian
        F_x = np.tanh(W_res @ x_res)

        # Without residual: output = F(x)
        y_no_skip = F_x
        # With residual: output = F(x) + x
        y_skip = F_x + x_res

        # Jacobian norms (approximate gradient magnitude)
        J_F = np.diag(1 - np.tanh(W_res @ x_res)**2) @ W_res  # dF/dx
        J_no_skip = np.linalg.norm(J_F)
        J_skip = np.linalg.norm(J_F + np.eye(4))  # dF/dx + I

        print(f"Jacobian norm without skip: {J_no_skip:.4f}")
        print(f"Jacobian norm with skip:    {J_skip:.4f}")
        print("→ Skip connection keeps gradient magnitude closer to 1")


    _run()
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
def _(np):
    def _run():
        # Weight initialization: Xavier vs He — variance preservation across layers
        rng = np.random.default_rng(0)
        n_in, n_out = 256, 256
        n_forward_layers = 10
        x_init = rng.standard_normal((1, n_in))  # single sample

        # Xavier init: Var(W) = 2 / (n_in + n_out) — designed for tanh
        print("=== Xavier init + tanh ===")
        h = x_init.copy()
        for _ in range(n_forward_layers):
            W = rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / (n_in + n_out))
            h = np.tanh(h @ W)
        print(f"  Activation std after {n_forward_layers} layers: {h.std():.6f}")

        # He init: Var(W) = 2 / n_in — designed for ReLU
        print("=== He init + ReLU ===")
        h = x_init.copy()
        for _ in range(n_forward_layers):
            W = rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)
            h = np.maximum(0, h @ W)  # ReLU
        print(f"  Activation std after {n_forward_layers} layers: {h.std():.6f}")

        # Bad init (too large): activations explode
        print("=== Bad init (std=1.0) + ReLU ===")
        h = x_init.copy()
        for _ in range(n_forward_layers):
            W = rng.standard_normal((n_in, n_out)) * 1.0
            h = np.maximum(0, h @ W)
        print(f"  Activation std after {n_forward_layers} layers: {h.std():.2e}  (exploded!)")


    _run()
    return

@app.cell
def _(np):
    def _run():
        # Gradient accumulation pseudocode in numpy
        # Simulate: logical batch=256, physical batch=64, accumulation steps=4
        logical_bs = 256
        physical_bs = 64
        accum_steps = logical_bs // physical_bs  # = 4

        rng = np.random.default_rng(1)
        # Fake per-sample gradients for a single parameter
        all_grads = rng.standard_normal(logical_bs)

        # Method 1: single big batch
        full_batch_grad = all_grads.mean()

        # Method 2: accumulate over mini-batches (divide each by accum_steps)
        accumulated_grad = 0.0
        for step in range(accum_steps):
            mini_batch = all_grads[step * physical_bs : (step + 1) * physical_bs]
            accumulated_grad += mini_batch.mean() / accum_steps  # scale by 1/k

        print(f"Full-batch gradient:    {full_batch_grad:.6f}")
        print(f"Accumulated gradient:   {accumulated_grad:.6f}")
        print(f"Match: {np.isclose(full_batch_grad, accumulated_grad)}")


    _run()
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


@app.cell(hide_code=True)
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
""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
---

## Code It: Implementation Exercises

Work through these exercises to build the optimization tools from scratch in numpy. Each exercise gives you a skeleton — fill in the `TODO` sections.
""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
### Exercise 1: Implement Adam from scratch

Implement the full Adam optimizer update step with bias correction. Use it to minimize $f(\theta) = \theta_1^2 + 10\,\theta_2^2$ starting from $(-4, 3)$.
""")
    return


@app.cell
def _(np):
    def adam_from_scratch(grad_fn, theta_init, lr_adam=0.01, beta1=0.9, beta2=0.999,
                          epsilon=1e-8, n_steps_adam=300):
        """Full Adam optimizer. Returns trajectory of theta values."""
        theta = np.array(theta_init, dtype=float)
        m_ex1 = np.zeros_like(theta)  # first moment
        v_ex1 = np.zeros_like(theta)  # second moment
        trajectory_ex1 = [theta.copy()]

        for t_ex1 in range(1, n_steps_adam + 1):
            g = grad_fn(theta)

            # TODO: update biased first moment estimate m
            # m = ...

            # TODO: update biased second moment estimate v
            # v = ...

            # TODO: compute bias-corrected estimates m_hat and v_hat
            # m_hat = ...
            # v_hat = ...

            # TODO: update theta using Adam rule
            # theta = theta - lr * m_hat / (sqrt(v_hat) + epsilon)

            trajectory_ex1.append(theta.copy())

        return np.array(trajectory_ex1)

    # Test: minimize f(θ) = θ₁² + 10θ₂²
    grad_fn_test = lambda th: np.array([2 * th[0], 20 * th[1]])
    # traj = adam_from_scratch(grad_fn_test, [-4.0, 3.0])
    # print(f"Final θ = {traj[-1].round(6)}")  # should be near [0, 0]
    return (adam_from_scratch,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
### Exercise 2: Implement cosine annealing with warmup

Write a function that returns the learning rate at each step, using linear warmup followed by cosine decay.
""")
    return


@app.cell
def _(np):
    def cosine_schedule_with_warmup(total_steps, warmup_steps_ex, lr_max_ex, lr_min_ex=0.0):
        """Return an array of learning rates: linear warmup then cosine decay."""
        schedule = np.zeros(total_steps)

        for t_ex2 in range(total_steps):
            if t_ex2 < warmup_steps_ex:
                # TODO: linear warmup from 0 to lr_max
                # schedule[t] = ...
                pass
            else:
                # TODO: cosine decay from lr_max to lr_min
                # η_t = η_min + 0.5*(η_max - η_min)*(1 + cos(π * progress))
                # where progress = (t - warmup_steps) / (total_steps - warmup_steps)
                # schedule[t] = ...
                pass

        return schedule

    # Test it:
    # sched = cosine_schedule_with_warmup(1000, 100, lr_max=1e-3, lr_min=1e-6)
    # print(f"LR at step 0: {sched[0]:.6f}")    # should be ~0
    # print(f"LR at step 100: {sched[100]:.6f}") # should be ~lr_max
    # print(f"LR at step 999: {sched[999]:.6f}") # should be ~lr_min
    return (cosine_schedule_with_warmup,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
### Exercise 3: Implement Batch Normalization (forward + running stats)

Write a batch norm layer that handles both training mode (use batch stats) and eval mode (use running averages).
""")
    return


@app.cell
def _(np):
    class BatchNorm1D:
        def __init__(self, n_features, momentum_bn=0.1, eps_ex3=1e-5):
            self.gamma_ex3 = np.ones(n_features)    # learnable scale
            self.beta_ex3 = np.zeros(n_features)     # learnable shift
            self.running_mean = np.zeros(n_features)
            self.running_var = np.ones(n_features)
            self.momentum_bn = momentum_bn
            self.eps_ex3 = eps_ex3

        def forward(self, X_ex3, training=True):
            """
            X: shape (batch_size, n_features)
            Returns normalized, scaled, shifted output of same shape.
            """
            if training:
                # TODO: compute batch mean and variance (axis=0)
                # mu = ...
                # var = ...
                mu = X_ex3.mean(axis=0)   # placeholder — replace with your code
                var = X_ex3.var(axis=0)

                # TODO: update running stats with exponential moving average
                # self.running_mean = (1 - momentum) * running_mean + momentum * mu
                # self.running_var  = (1 - momentum) * running_var  + momentum * var

            else:
                # TODO: use running stats at eval time
                # mu = self.running_mean
                # var = self.running_var
                mu = self.running_mean
                var = self.running_var

            # TODO: normalize, scale, and shift
            # x_hat = (X - mu) / sqrt(var + eps)
            # out = gamma * x_hat + beta
            x_hat = (X_ex3 - mu) / np.sqrt(var + self.eps_ex3)
            out = self.gamma_ex3 * x_hat + self.beta_ex3
            return out

    # Test:
    # bn = BatchNorm1D(3)
    # X_test = rng.standard_normal((8, 3)) * 5 + 10
    # Y_train = bn.forward(X_test, training=True)
    # print(f"Train output mean: {Y_train.mean(axis=0).round(4)}")  # should be ~0
    # Y_eval = bn.forward(X_test, training=False)  # uses running stats
    return (BatchNorm1D,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
### Exercise 4: Gradient clipping (norm clipping)

Implement gradient norm clipping. Given a list of gradient arrays (one per parameter), clip the *global* norm.
""")
    return


@app.cell
def _(np):
    def clip_grad_norm(grads, max_norm_ex4):
        """
        Clip gradients by global norm.
        grads: list of numpy arrays (one per parameter)
        max_norm: maximum allowed global norm
        Returns: clipped gradients (same structure), original norm
        """
        # TODO: compute the global norm = sqrt(sum of squared norms of each grad)
        # global_norm = ...
        global_norm = 0.0  # placeholder

        # TODO: if global_norm > max_norm, scale all grads by (max_norm / global_norm)
        # clipped = ...
        clipped = grads  # placeholder

        return clipped, global_norm

    # Test:
    # g1 = np.array([3.0, 4.0])   # norm = 5
    # g2 = np.array([12.0])       # norm = 12
    # global_norm = sqrt(25 + 144) = 13
    # clipped, gn = clip_grad_norm([g1, g2], max_norm=5.0)
    # print(f"Global norm: {gn:.1f} → clipped to 5.0")
    # print(f"Clipped g1: {clipped[0].round(4)}")
    return (clip_grad_norm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
### Exercise 5: Training loop with everything combined

Put it all together: train a 2-layer MLP on synthetic data using your Adam, cosine schedule, batch norm, and gradient clipping implementations.
""")
    return


@app.cell
def _(np):
    def full_training_loop():
        """
        Skeleton for a complete training loop combining:
        - He initialization
        - Adam optimizer
        - Cosine LR schedule with warmup
        - Batch normalization
        - Gradient norm clipping

        TODO: Fill in each section using the functions you built above.
        """
        rng = np.random.default_rng(42)
        n_samples, n_features_ex5, n_hidden, n_classes = 500, 10, 32, 3

        # Synthetic data
        X_train_ex5 = rng.standard_normal((n_samples, n_features_ex5))
        y_train_ex5 = rng.integers(0, n_classes, n_samples)

        # TODO: He initialization for weights
        # W1 = rng.standard_normal((n_features, n_hidden)) * sqrt(2 / n_features)
        # W2 = rng.standard_normal((n_hidden, n_classes)) * sqrt(2 / n_hidden)

        # TODO: Initialize batch norm for hidden layer

        # TODO: Set up cosine schedule with warmup
        # total_steps = ...
        # schedule = cosine_schedule_with_warmup(...)

        # TODO: Training loop
        # for step in range(total_steps):
        #     1. Sample mini-batch
        #     2. Forward pass: X → Linear → BatchNorm → ReLU → Linear → softmax
        #     3. Compute loss (cross-entropy)
        #     4. Backward pass (compute gradients)
        #     5. Clip gradients
        #     6. Adam update with scheduled LR
        #     7. Log loss every N steps

        return "TODO: implement and return loss history"

    print("Exercise 5: Combine all components into a training loop.")
    print("Use your adam_from_scratch, cosine_schedule_with_warmup,")
    print("BatchNorm1D, and clip_grad_norm implementations.")
    return (full_training_loop,)


@app.cell
def _(mo):
    mo.md(r"""
---

**Next**: [2C - Regularization for Deep Learning](2c_regularization.py) — dropout, data augmentation, early stopping, and other techniques that prevent overfitting in deep networks.
""")
    return


if __name__ == "__main__":
    app.run()
