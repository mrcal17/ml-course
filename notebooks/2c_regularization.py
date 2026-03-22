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
# Regularization for Deep Learning

## 1. The Regularization Puzzle

Here is a fact that should deeply bother you. A modern neural network — say, ResNet-50 — has roughly 25 million parameters. It is commonly trained on ImageNet, which has about 1.2 million images. The model has more than 20 times as many parameters as training examples. Classical statistical learning theory has an unambiguous prediction for this regime: catastrophic overfitting. The model should memorize the training set and generalize terribly.

And yet, these networks achieve remarkable test accuracy. Something is wrong with the classical story.

The puzzle deepens. Zhang et al. (2017) demonstrated that these same networks can memorize a dataset with *completely random labels* — reaching 100% training accuracy on pure noise. This proves the model *has the capacity* to overfit. But when trained on real data with real labels, it does not. The difference cannot be in the architecture. It must be in the learning procedure itself, the data, or the explicit regularization techniques we apply.

This module covers the regularization toolkit for deep learning. Some of these techniques — weight decay, early stopping — are direct analogs of ideas you already know from linear models. Others — dropout, data augmentation, batch normalization — are specific to deep networks and have no clean classical analog. And some of the most important regularization in deep learning is *implicit*: it emerges from the optimizer, the architecture, or the data itself without anyone explicitly asking for it.

The goal is not just to catalog these techniques, but to understand *why* they work — and when they do not.

> **Reading**: [DLBook §7.1](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) for the general framing of regularization in deep learning, [DLBook Ch. 7 introduction](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) for the distinction between explicit and implicit regularization.
""")
    return


@app.cell
def _(mo):
    mo.md(r"""
---

## 2. Weight Decay (L2 Regularization) in Deep Learning

You already know the idea from Ridge regression: add a penalty term $\lambda \|\mathbf{w}\|^2$ to the loss. In deep learning, the regularized objective is:

$$\tilde{\mathcal{L}}(\theta) = \mathcal{L}(\theta) + \frac{\lambda}{2}\|\theta\|^2$$

where $\theta$ represents all the network weights (biases are typically *not* regularized — they control the output offset and penalizing them adds bias without reducing variance).

### Effect on the Loss Landscape

The penalty term adds a bowl-shaped quadratic surface to the loss landscape. This has two effects:

1. **It shrinks weights toward zero.** The gradient of the penalty term is $\lambda \theta$, so at each step, weights are multiplied by a factor of $(1 - \eta\lambda)$ before the gradient update is applied. This is why it is called "weight decay."

2. **It smooths the loss surface.** The Hessian of the regularized loss is $H + \lambda I$ — the regularization term adds $\lambda$ to every eigenvalue of the Hessian. This makes the curvature more uniform, which improves conditioning and makes optimization easier.

In the spectral view: directions in parameter space where the data provides little information (small eigenvalues of $H$) get dominated by the regularization term and shrunk heavily. Directions where the data is informative (large eigenvalues) are barely affected. Weight decay preferentially eliminates the parameters the model is least certain about — exactly the ones most likely to encode noise.

### AdamW vs. L2 Regularization in Adam

This is a critical subtlety, and it connects directly to what you saw in Module 2B. When you add L2 regularization to the loss function and then optimize with Adam, the adaptive learning rate *modifies the effective regularization strength per parameter*. Parameters with large historical gradients (large $v_t$) get smaller effective learning rates, which means the L2 penalty also gets scaled down for those parameters. The regularization is no longer uniform.

AdamW fixes this by *decoupling* weight decay from the gradient-based update. Instead of adding $\lambda\theta$ to the gradient (which then gets divided by $\sqrt{v_t}$), AdamW applies weight decay directly to the parameters after the Adam step:

$$\theta_{t+1} = (1 - \eta\lambda)\theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

The result: every parameter decays at the same rate $\eta\lambda$, regardless of gradient history. Empirically, AdamW generalizes substantially better than Adam with L2. If you are using Adam-family optimizers, use AdamW.

> **Reading**: [DLBook §7.1.1](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) for the spectral analysis of weight decay, [Murphy PML1 §13.4](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf) for weight decay in modern optimizers.
""")
    return


@app.cell
def _(np):
    # Weight decay: the "decay" step in code
    # theta_{t+1} = (1 - lr * lam) * theta_t  -  lr * grad
    theta = rng.standard_normal(5)          # some weights
    grad = rng.standard_normal(5)           # gradient of the data loss
    lr, lam = 0.01, 1e-2

    # Standard L2: add penalty gradient to data gradient, then step
    theta_l2 = theta - lr * (grad + lam * theta)

    # Weight decay (AdamW-style): decay first, then step with data gradient only
    theta_wd = (1 - lr * lam) * theta - lr * grad

    # For vanilla SGD these are identical:
    print("Max difference (SGD):", np.max(np.abs(theta_l2 - theta_wd)))

    # Hessian conditioning: H + lambda*I lifts every eigenvalue
    H = np.array([[0.01, 0], [0, 5.0]])  # ill-conditioned
    H_reg = H + lam * np.eye(2)
    print(f"Eigenvalues before: {np.linalg.eigvalsh(H)}")
    print(f"Eigenvalues after:  {np.linalg.eigvalsh(H_reg)}")
    print(f"Condition number: {np.linalg.cond(H):.0f} -> {np.linalg.cond(H_reg):.0f}")
    return


@app.cell
def _(mo):
    mo.md(r"""
---

## 3. Dropout

Dropout is one of the most influential regularization techniques in deep learning history, introduced by Srivastava et al. (2014) based on earlier work by Hinton.

### The Algorithm

During training, at each forward pass, each neuron in a dropout layer is independently set to zero with probability $p$ (the dropout rate). The remaining neurons have their outputs scaled by $\frac{1}{1-p}$ to maintain the expected value of the layer output. At test time, all neurons are active and no scaling is applied — this is called **inverted dropout** and is the standard implementation.

The alternative — training without scaling and multiplying all weights by $(1-p)$ at test time — is mathematically equivalent but less convenient in practice, because it requires modifying the model at inference time.

### Intuition 1: Ensemble of Sub-networks

A network with $n$ neurons and dropout rate 0.5 can be seen as sampling from $2^n$ possible sub-networks, each defined by a different binary mask. Each forward pass trains a different sub-network. At test time, the full network (with scaled weights) approximates the average prediction of all these sub-networks.

This is an exponentially large ensemble trained with extreme weight sharing — every sub-network shares parameters with every other. Ensemble methods reduce variance, and this is exactly what dropout achieves.

### Intuition 2: Preventing Co-adaptation

Without dropout, neurons can "co-adapt": neuron A learns to fix the mistakes of neuron B, creating a fragile dependency. If neuron B changes slightly during later training, neuron A's correction becomes counterproductive. Dropout forces each neuron to be individually useful, because it cannot rely on any specific partner being present. This produces more robust, distributed representations.

### The Bayesian Connection

There is a deeper reason dropout works, and it connects to the Bayesian model averaging you may encounter later. Consider a network with weights $\theta$ and binary dropout masks $\mathbf{z} \in \{0, 1\}^d$, where each $z_i \sim \text{Bernoulli}(1-p)$. The prediction at test time is:

$$\hat{y} = \mathbb{E}_{\mathbf{z}}[f(\mathbf{x}; \theta \odot \mathbf{z})]$$

This is a Monte Carlo approximation to a Bayesian model average, where the "posterior" over models is induced by the dropout distribution. Gal and Ghahramani (2016) formalized this: dropout training is approximately equivalent to variational inference in a deep Gaussian process. The dropout rate corresponds to the prior precision, and the weight decay coefficient interacts with it to define the prior variance.

The practical implication: you can use dropout at test time (called **MC Dropout**) to get uncertainty estimates by running multiple forward passes and measuring the variance of predictions.

### Practical Guidelines

- **Hidden layers**: dropout rate of 0.5 is the standard starting point.
- **Input layer**: dropout rate of 0.1–0.2 (you want to corrupt inputs less aggressively).
- **CNNs**: dropout is less effective in convolutional layers because the spatial structure already provides strong regularization through weight sharing. Spatial dropout (dropping entire feature maps instead of individual pixels) works somewhat better.
- **Modern architectures**: in transformers, dropout is applied to attention weights and feed-forward layers, typically at rates of 0.1–0.3. Many state-of-the-art models (especially large language models) use very little or no dropout, relying instead on data scale and other regularization methods.

> **Reading**: [DLBook §7.12](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) for the full treatment of dropout, [Bishop §10.2](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) for the Bayesian perspective on model averaging.
""")
    return


@app.cell
def _(np):
    # Inverted dropout from scratch
    # During training: mask and scale by 1/(1-p)
    # During test: use activations unchanged

    p_drop = 0.5                             # dropout rate
    h = np.array([1.0, 2.0, 3.0, 4.0, 5.0]) # hidden activations

    rng = np.random.default_rng(0)
    mask = (rng.random(h.shape) > p_drop).astype(float)  # Bernoulli(1-p)
    h_train = h * mask / (1 - p_drop)        # inverted dropout

    # Key property: E[h_train] == h  (unbiased)
    n_trials = 100_000
    samples = np.array([
        h * (rng.random(h.shape) > p_drop).astype(float) / (1 - p_drop)
        for _ in range(n_trials)
    ])
    print(f"Original h:      {h}")
    print(f"One masked pass:  {h_train}")
    print(f"Mean over {n_trials} passes: {samples.mean(axis=0).round(3)}")
    return


@app.cell
def _(np):
    # MC Dropout: run T forward passes at test time, measure prediction variance
    # Simulated single-neuron example: y = w^T (h * mask / (1-p))
    rng_mc = np.random.default_rng(42)
    w = np.array([0.5, -1.0, 0.8, 0.3, -0.6])
    h_test = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    p_mc = 0.5
    T = 500  # number of MC samples

    preds = np.array([
        w @ (h_test * (rng_mc.random(h_test.shape) > p_mc) / (1 - p_mc))
        for _ in range(T)
    ])
    print(f"MC Dropout mean: {preds.mean():.3f}")
    print(f"MC Dropout std:  {preds.std():.3f}  (uncertainty estimate)")
    print(f"Deterministic:   {w @ h_test:.3f}")
    return


@app.cell
def _(mo):
    mo.md(r"""
---

## 4. Early Stopping

The simplest and often most effective regularizer: just stop training before the model overfits.

### The Procedure

1. Split your data into training and validation sets.
2. After each epoch, evaluate the loss on the validation set.
3. Track the best validation loss seen so far and save a checkpoint of the model at that point.
4. If the validation loss has not improved for $k$ consecutive epochs (the **patience**), stop training.
5. Return the saved checkpoint, *not* the model from the final epoch.

The last point is critical and commonly botched. The model at the final epoch is the *most overfit* model from the entire run. You want the model from the epoch where validation performance peaked.

### Why Early Stopping is Equivalent to L2 Regularization

This is a beautiful result due to Bishop (1995) and elaborated by Goodfellow et al. Consider gradient descent on a quadratic loss surface (which is exact for linear models and locally approximate for neural networks). The weight trajectory under gradient descent with learning rate $\eta$ and $\tau$ steps is:

$$w_i(\tau) = w_i^* \left(1 - (1 - \eta\lambda_i)^\tau\right)$$

where $\lambda_i$ is the $i$-th eigenvalue of the Hessian and $w_i^*$ is the corresponding component of the optimal weight. Compare this to the L2-regularized solution:

$$\tilde{w}_i = w_i^* \cdot \frac{\lambda_i}{\lambda_i + \alpha}$$

Both expressions suppress components corresponding to small eigenvalues (directions of low curvature, where the model is uncertain). The number of training steps $\tau$ plays the role of the inverse regularization strength $1/\alpha$: more steps $=$ less regularization. This is why early stopping works — limiting training time is mathematically similar to adding a weight penalty.

The correspondence is $\tau \approx \frac{1}{\eta\alpha}$, so smaller learning rate or fewer steps both increase the effective regularization.

> **Reading**: [DLBook §7.8](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) for the full derivation and geometric interpretation of early stopping as regularization.
""")
    return


@app.cell
def _(np):
    # Early stopping <-> L2 equivalence on a quadratic loss
    # w_i(tau) = w_i* (1 - (1 - eta*lam_i)^tau)
    # w_i_ridge = w_i* * lam_i / (lam_i + alpha)

    eta_es = 0.01
    eigenvalues = np.array([0.01, 0.1, 1.0, 10.0])  # Hessian eigenvalues
    w_star = np.ones(4)                                # optimal weights

    # GD after tau steps
    tau = 50
    w_gd = w_star * (1 - (1 - eta_es * eigenvalues) ** tau)

    # L2 regularization with equivalent alpha ~ 1/(eta*tau)
    alpha = 1.0 / (eta_es * tau)
    w_ridge = w_star * eigenvalues / (eigenvalues + alpha)

    print("Eigenvalue | w(GD)  | w(Ridge) | Shrinkage match?")
    for i in range(4):
        print(f"  {eigenvalues[i]:6.2f}   | {w_gd[i]:.4f} | {w_ridge[i]:.4f}  | "
              f"{'close' if abs(w_gd[i]-w_ridge[i])<0.1 else 'diverges'}")
    # Small eigenvalues (uncertain directions) are shrunk most by both methods
    return


@app.cell
def _(np):
    # Early stopping logic (pseudo-training loop)
    # Simulated val losses that first decrease then increase
    rng_es = np.random.default_rng(7)
    val_losses = 1.0 - 0.5 * np.exp(-np.arange(50) / 10) + 0.005 * np.arange(50)
    val_losses += rng_es.normal(0, 0.01, 50)

    best_loss, best_epoch, patience, wait = np.inf, 0, 5, 0
    for epoch, vl in enumerate(val_losses):
        if vl < best_loss:
            best_loss, best_epoch, wait = vl, epoch, 0
            # checkpoint = model.state_dict()  # save best
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}, "
                      f"best was epoch {best_epoch} (loss={best_loss:.4f})")
                break
    return


@app.cell
def _(mo):
    mo.md(r"""
---

## 5. Data Augmentation

If you had to pick a single regularization technique, data augmentation would often be the strongest choice. It directly addresses the root cause of overfitting: insufficient data.

### The Principle

Create new training examples by applying transformations to existing data that *change the input but preserve the correct label*. This is a way of injecting prior knowledge about invariances into the model. If you know that a photo of a cat is still a photo of a cat when flipped horizontally, you can tell the model this by training on both the original and the flipped version.

### Image Augmentation

The standard augmentation pipeline for image classification:

- **Geometric**: random horizontal flips, rotations (small angles, e.g. $\pm 15°$), random crops (train on 224$\times$224 crops of 256$\times$256 images), affine transformations.
- **Photometric**: color jitter (random adjustments to brightness, contrast, saturation), grayscale conversion (with small probability).
- **Erasing methods**: Cutout (mask a random rectangular region), Random Erasing (replace a patch with noise).
- **Mixing methods**: Mixup (blend two images and their labels: $\tilde{x} = \alpha x_i + (1-\alpha)x_j$, $\tilde{y} = \alpha y_i + (1-\alpha)y_j$ with $\alpha \sim \text{Beta}(0.2, 0.2)$), CutMix (paste a patch from one image onto another, blending labels proportionally).

Mixup and CutMix deserve special attention because they also smooth the label space, which connects to label smoothing (Section 7).

### Text Augmentation

Text is harder to augment because most transformations destroy semantic content. Common approaches:

- **Synonym replacement**: randomly replace words with synonyms (via WordNet or embedding neighbors).
- **Back-translation**: translate to another language and back, producing paraphrases.
- **Random insertion/deletion/swap**: small perturbations that preserve meaning at the sentence level.

### The Key Principle

An augmentation is valid if and only if the transformation preserves the label. Flipping a cat image horizontally gives another cat — valid. Flipping the digit "6" vertically gives something closer to "9" — invalid. Rotating a satellite image 180 degrees is fine; rotating handwritten text 180 degrees changes the content. You must understand the domain to design augmentations correctly.

Data augmentation is so effective because it *multiplies the effective dataset size* while encoding genuine prior knowledge. A model trained with aggressive augmentation sees a different version of each training example at every epoch, making memorization far more difficult.

> **Reading**: [DLBook §7.4](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) for the general theory of data augmentation as regularization.
""")
    return


@app.cell
def _(np):
    # Mixup augmentation: blend two examples and their labels
    # x_mix = alpha * x_i + (1 - alpha) * x_j
    # y_mix = alpha * y_i + (1 - alpha) * y_j

    rng_mix = np.random.default_rng(42)

    # Two "images" (flattened 2x2 for demo) and one-hot labels (3 classes)
    x_i = np.array([0.8, 0.2, 0.1, 0.9])  # image i
    y_i = np.array([1.0, 0.0, 0.0])        # class 0
    x_j = np.array([0.1, 0.7, 0.6, 0.3])  # image j
    y_j = np.array([0.0, 0.0, 1.0])        # class 2

    alpha_mix = rng_mix.beta(0.2, 0.2)     # Beta(0.2,0.2) -> bimodal near 0 or 1
    x_mix = alpha_mix * x_i + (1 - alpha_mix) * x_j
    y_mix = alpha_mix * y_i + (1 - alpha_mix) * y_j

    print(f"alpha = {alpha_mix:.3f}")
    print(f"Mixed input:  {x_mix.round(3)}")
    print(f"Mixed label:  {y_mix.round(3)}  (soft target across classes)")
    return


@app.cell
def _(np):
    import matplotlib.pyplot as _plt

    # Visualize simple augmentations on a tiny "image"
    rng_aug = np.random.default_rng(0)
    img = rng_aug.random((8, 8))  # grayscale 8x8

    fig_aug, axes_aug = _plt.subplots(1, 4, figsize=(10, 2.5))

    axes_aug[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes_aug[0].set_title("Original")

    axes_aug[1].imshow(img[:, ::-1], cmap="gray", vmin=0, vmax=1)
    axes_aug[1].set_title("H-Flip")

    # Random crop (6x6 from 8x8)
    r, c = rng_aug.integers(0, 3, size=2)
    axes_aug[2].imshow(img[r:r+6, c:c+6], cmap="gray", vmin=0, vmax=1)
    axes_aug[2].set_title(f"Crop [{r}:{r+6},{c}:{c+6}]")

    # Brightness jitter
    jitter = np.clip(img + rng_aug.uniform(-0.3, 0.3), 0, 1)
    axes_aug[3].imshow(jitter, cmap="gray", vmin=0, vmax=1)
    axes_aug[3].set_title("Brightness jitter")

    for ax in axes_aug:
        ax.axis("off")
    fig_aug.suptitle("Common image augmentations", y=1.02)
    fig_aug.tight_layout()
    fig_aug
    return


@app.cell
def _(mo):
    mo.md(r"""
---

## 6. Batch Normalization as Regularization

You encountered batch normalization in Module 2B as an optimization technique. But it also acts as a regularizer, and understanding why helps explain some otherwise puzzling empirical results.

### The Mechanism

During training, BN normalizes each activation using the mean and variance computed from the *current mini-batch*. These statistics are noisy estimates of the true population statistics — the noise depends on which examples happen to be in the batch. This injected noise acts similarly to dropout: it prevents the network from relying too precisely on any single activation value.

The regularization effect is a *side effect* of the mini-batch computation, not the primary design goal. But it is real and measurable:

- Larger batch sizes $\rightarrow$ less noise in BN statistics $\rightarrow$ less regularization $\rightarrow$ more overfitting. This is one reason very large batch training often requires additional explicit regularization.
- BN reduces the need for dropout. In many modern architectures, using both BN and dropout simultaneously is counterproductive — the noise sources can interfere, and removing dropout after adding BN often improves performance.

### A Caveat

At test time, BN uses running statistics (population estimates accumulated during training), so the regularizing noise disappears entirely. The model is deterministic. This means BN's regularization only helps *during training*, which is exactly where you need it — but it also means the test-time model is a different function from the training-time model, which can occasionally cause subtle bugs.

> **Reading**: [DLBook §8.7.1](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) for batch normalization mechanics.
""")
    return


@app.cell
def _(np):
    # Batch normalization: noise depends on batch size
    # Smaller batch -> noisier mean/var estimates -> more regularization

    rng_bn = np.random.default_rng(0)
    population = rng_bn.normal(loc=3.0, scale=2.0, size=10000)
    pop_mean, pop_std = population.mean(), population.std()

    for batch_size in [4, 16, 64, 256]:
        # Simulate many mini-batch BN normalizations
        errors = []
        for _ in range(1000):
            batch = rng_bn.choice(population, size=batch_size, replace=False)
            bn_mean, bn_std = batch.mean(), batch.std()
            errors.append(abs(bn_mean - pop_mean))
        print(f"Batch {batch_size:>3d}: mean estimation error = "
              f"{np.mean(errors):.4f}  (noise = regularization)")
    return


@app.cell
def _(mo):
    mo.md(r"""
---

## 7. Label Smoothing

Standard classification training uses hard targets: the true class gets probability 1, all others get 0. The cross-entropy loss pushes the model to produce increasingly extreme logits to approximate these targets — driving the predicted probability ever closer to 1.0, which requires weights to grow without bound.

Label smoothing replaces the hard target distribution with:

$$y_{\text{smooth}} = (1 - \epsilon)\, y_{\text{hard}} + \frac{\epsilon}{K}$$

where $K$ is the number of classes and $\epsilon$ is typically 0.1. For a 10-class problem with true class $c$, the smoothed target for class $c$ is $0.9 + 0.01 = 0.91$, and the target for each other class is $0.01$.

### Why It Works

The cross-entropy loss with label smoothing is:

$$\mathcal{L}_{\text{smooth}} = (1 - \epsilon)\, H(y_{\text{hard}}, p) + \epsilon\, H(u, p)$$

where $u$ is the uniform distribution and $H(\cdot, \cdot)$ is cross-entropy. The second term is equivalent to maximizing the entropy of the predicted distribution $p$. This is a direct connection to the information-theoretic concepts from Module 0D: the model is encouraged to maintain some uncertainty, avoiding the degenerate solution of placing all probability mass on a single class.

The practical benefits:

1. **Better calibration.** The model's confidence levels more accurately reflect its true accuracy.
2. **Improved generalization.** Preventing extreme logits implicitly constraints the weight magnitudes, acting as a form of regularization.
3. **Robustness to label noise.** When some training labels are wrong (which is common in practice), hard targets force the model to memorize mistakes. Soft targets are more forgiving.

Label smoothing is used in virtually all modern image classifiers and machine translation systems. It costs nothing computationally and consistently helps.

> **Reading**: [DLBook §7.3.1](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) for noise robustness in regularization, [Murphy PML1 §10.4](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf) for cross-entropy and calibration.
""")
    return


@app.cell
def _(np):
    # Label smoothing implementation
    # y_smooth = (1 - eps) * y_hard + eps / K

    def smooth_labels(y_hard, K, eps=0.1):
        """Convert hard one-hot labels to smoothed targets."""
        return (1 - eps) * y_hard + eps / K

    K = 10   # number of classes
    eps = 0.1

    y_hard = np.zeros(K)
    y_hard[3] = 1.0   # true class = 3
    y_smooth = smooth_labels(y_hard, K, eps)

    print(f"Hard labels:     {y_hard}")
    print(f"Smoothed labels: {y_smooth}")
    print(f"Sum = {y_smooth.sum():.2f}  (still a valid distribution)")

    # Cross-entropy comparison: smooth targets penalize over-confident logits
    def cross_entropy(target, pred):
        return -np.sum(target * np.log(pred + 1e-12))

    # Softmax from logits
    def softmax(z):
        e = np.exp(z - z.max())
        return e / e.sum()

    # Confident vs. moderate prediction
    logits_confident = np.zeros(K); logits_confident[3] = 10.0
    logits_moderate  = np.zeros(K); logits_moderate[3] = 3.0

    p_conf = softmax(logits_confident)
    p_mod  = softmax(logits_moderate)

    print(f"\nCE(hard, confident):   {cross_entropy(y_hard, p_conf):.4f}")
    print(f"CE(hard, moderate):    {cross_entropy(y_hard, p_mod):.4f}")
    print(f"CE(smooth, confident): {cross_entropy(y_smooth, p_conf):.4f}")
    print(f"CE(smooth, moderate):  {cross_entropy(y_smooth, p_mod):.4f}")
    print("^ Smooth labels penalize extreme confidence less aggressively")
    return


@app.cell
def _(mo):
    mo.md(r"""
---

## 8. The Double Descent Phenomenon

Everything we have discussed so far assumes the classical bias-variance tradeoff: as model complexity increases, training error decreases monotonically, and test error follows a U-shape — first decreasing (bias reduction) then increasing (variance increase). The optimal model sits at the bottom of the U.

This picture is wrong for modern neural networks. Or rather, it is incomplete.

### The Modern Discovery

Belkin et al. (2019) and subsequent work demonstrated a striking phenomenon. As model complexity increases past the **interpolation threshold** — the point where the model has just enough parameters to perfectly fit (memorize) the training data — test error spikes dramatically, as classical theory predicts. But if you keep increasing complexity *beyond* this threshold, test error *decreases again*, often reaching levels below the original U-curve minimum.

The resulting curve has two descents: the classical one before the interpolation threshold, and a second descent in the **overparameterized regime** (where parameters far outnumber data points). This is the regime where modern deep networks operate.

### Epoch-wise Double Descent

The phenomenon also appears over training time, not just model size. For a fixed (large) model, the test error may first decrease, then increase, then decrease again as training progresses. This is "epoch-wise double descent" and it means that the common practice of early stopping could actually be *suboptimal* — if you had trained longer, the model might have recovered and achieved better test performance.

### Why This Happens

The intuition, still being refined by theorists, goes roughly like this:

- At the interpolation threshold, the model is forced to find a solution that exactly fits the training data, but it has *no freedom left* — there is essentially one interpolating solution, and it tends to be highly irregular (large weights, sharp decision boundaries).
- In the overparameterized regime, there are *many* interpolating solutions. The optimizer (SGD) selects among them, and it tends to pick smooth, low-norm solutions — ones that generalize well. The extra parameters provide room for the model to interpolate the data *gently*.

This connects directly to implicit regularization (Section 9): the optimizer's inductive bias toward simple solutions is what makes the overparameterized regime work.

### Practical Implications

1. **Bigger models can be better**, even when they memorize training data. Do not be afraid of overparameterization.
2. **Early stopping should be done carefully.** Monitor for long enough to see if you are in a double descent regime.
3. **Classical model selection metrics** (AIC, BIC, cross-validation with small models) may select *worse* models than simply making the model as large as computationally feasible and training with proper regularization.

> **Reading**: [DLBook Ch. 7 introduction](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) for capacity and generalization, [ESL §7.1-7.3](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf) for the classical bias-variance framework that double descent challenges.
""")
    return


@app.cell
def _(np):
    import matplotlib.pyplot as _plt2

    # Double descent illustration with polynomial regression
    # Vary polynomial degree from 1 to > n_samples
    rng_dd = np.random.default_rng(42)
    n_dd = 15
    x_dd = rng_dd.uniform(-1, 1, n_dd)
    y_dd = np.sin(3 * x_dd) + rng_dd.normal(0, 0.3, n_dd)
    x_test = np.linspace(-1, 1, 200)
    y_test_true = np.sin(3 * x_test)

    degrees = list(range(1, 25))
    train_err, test_err = [], []

    for d in degrees:
        # Fit polynomial of degree d (with small ridge for numerical stability)
        V_train = np.vander(x_dd, d + 1, increasing=True)
        V_test  = np.vander(x_test, d + 1, increasing=True)
        # Minimum norm solution via pseudoinverse
        w_dd = np.linalg.lstsq(V_train, y_dd, rcond=None)[0]
        train_pred = V_train @ w_dd
        test_pred  = V_test @ w_dd
        train_err.append(np.mean((train_pred - y_dd) ** 2))
        test_err.append(np.mean((test_pred - y_test_true) ** 2))

    fig_dd, ax_dd = _plt2.subplots(figsize=(8, 4))
    ax_dd.semilogy(degrees, train_err, "o-", label="Train MSE")
    ax_dd.semilogy(degrees, test_err, "s-", label="Test MSE")
    ax_dd.axvline(n_dd, color="gray", ls="--", label=f"n_samples={n_dd} (interpolation threshold)")
    ax_dd.set_xlabel("Polynomial degree (model complexity)")
    ax_dd.set_ylabel("MSE (log scale)")
    ax_dd.set_title("Double Descent in Polynomial Regression")
    ax_dd.legend()
    ax_dd.set_ylim(1e-4, 1e3)
    fig_dd.tight_layout()
    fig_dd
    return


@app.cell
def _(mo):
    mo.md(r"""
---

## 9. Implicit Regularization

Perhaps the most surprising discovery of the deep learning era is how much regularization happens *without anyone asking for it*.

### SGD as a Regularizer

SGD does not just find *any* minimum of the loss — it preferentially finds *flat* minima (regions where the loss does not change much under small perturbations of the parameters). Flat minima are associated with better generalization because they are robust to the distribution shift between training and test data.

The mechanism is the gradient noise inherent in mini-batch SGD. Sharp minima have narrow basins of attraction, and the stochastic noise in the gradient estimate kicks the parameters out of these basins. Flat minima have wide basins that can absorb the noise. The effective regularization strength is proportional to the ratio $\eta / B$ (learning rate over batch size), which explains why this ratio matters more for generalization than either quantity alone.

This is why large-batch training tends to generalize worse: the gradient estimates are more accurate, so the implicit regularization from noise is weaker.

### Architecture as Regularization

Some architectural choices are powerful regularizers:

- **Convolutional layers** enforce translation equivariance and local connectivity. A CNN cannot learn one set of features for the top-left corner and a completely different set for the bottom-right — the same filters are applied everywhere. This weight sharing is a massive reduction in effective parameters.
- **Residual connections** bias the network toward learning small perturbations of the identity function, which is a form of prior toward simple functions.
- **Attention mechanisms** force the model to attend selectively to relevant parts of the input, rather than treating all input positions as equally important.

Each of these choices bakes in prior knowledge about the problem structure, reducing the effective hypothesis space without any explicit penalty term.

### The Lottery Ticket Hypothesis

Frankle and Carlin (2019) proposed a compelling conjecture: within a randomly initialized dense network, there exists a sparse subnetwork (a "winning ticket") that, when trained in isolation from the same initialization, can match the performance of the full network. The remaining parameters are essentially irrelevant.

The implication is that overparameterization may be necessary not because the final solution needs all those parameters, but because having many parameters makes it more likely that a good subnetwork exists and that SGD can find it. The extra parameters are scaffolding that aids optimization, not capacity that encodes information. After training, the network can be pruned dramatically — often removing 90%+ of weights — with minimal loss in accuracy.

This connects to double descent: the overparameterized regime works not because the model *uses* all its parameters, but because having extra parameters creates room for the optimizer to find simple solutions.

> **Reading**: [DLBook §7.1](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) for parameter norms and implicit regularization, [DLBook §7.5](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) for noise robustness, [ESL §11.5](file:///C:/Users/landa/ml-course/textbooks/ESL.pdf) for capacity control in neural networks.
""")
    return


@app.cell
def _(np):
    # SGD gradient noise: smaller batch -> noisier gradient -> more regularization
    # Effective noise ~ eta / B

    rng_sgd = np.random.default_rng(0)
    n_data = 1000
    X_sgd = rng_sgd.normal(0, 1, (n_data, 5))
    w_true = np.array([1.0, -0.5, 0.3, 0.0, 0.0])
    y_sgd = X_sgd @ w_true + rng_sgd.normal(0, 0.1, n_data)

    def gradient_full(X, y, w):
        return -2 / len(X) * X.T @ (y - X @ w)

    def gradient_sgd(X, y, w, batch_size, rng):
        idx = rng.choice(len(X), batch_size, replace=False)
        return -2 / batch_size * X[idx].T @ (y[idx] - X[idx] @ w)

    w_test_pt = np.zeros(5)
    g_full = gradient_full(X_sgd, y_sgd, w_test_pt)

    for B in [8, 32, 128, 512]:
        noises = [np.linalg.norm(gradient_sgd(X_sgd, y_sgd, w_test_pt, B, rng_sgd) - g_full)
                  for _ in range(200)]
        print(f"Batch {B:>3d}: gradient noise ||g_sgd - g_full|| = {np.mean(noises):.4f}")
    print("Smaller batch = more noise = stronger implicit regularization")
    return


@app.cell
def _(mo):
    mo.md(r"""
---

## 10. Putting It All Together: A Regularization Strategy

In practice, you rarely use just one regularization technique. A typical modern training pipeline includes:

1. **Data augmentation** — always. This is the highest-impact regularizer in most settings.
2. **Weight decay** (via AdamW) — standard. Values of $10^{-2}$ to $10^{-4}$ are common.
3. **Dropout** — use in fully connected layers, less in convolutional layers. Start with $p = 0.5$ for hidden layers, reduce if the model underfits.
4. **Batch normalization** or **layer normalization** — primarily for optimization, but the regularization effect is a bonus.
5. **Label smoothing** — $\epsilon = 0.1$ is nearly universal in classification.
6. **Early stopping** — always. Save the best checkpoint by validation metric.

The relative importance shifts with scale. For small datasets, data augmentation and dropout dominate. For very large models trained on massive datasets (GPT-scale), explicit regularization matters less — the sheer volume of data acts as a regularizer, and the implicit biases of the architecture and optimizer do the rest.

> **Reading**: [DLBook Ch. 7](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) in its entirety provides the most comprehensive treatment of regularization in deep learning. [Murphy PML1 §14.5](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf) covers practical regularization strategies.
""")
    return


@app.cell
def _(mo):
    mo.md(r"""
---

## Interactive: Dropout Effect on Decision Boundary

Use the slider below to vary the dropout rate and observe how it affects the decision boundary of a small neural network on a 2D classification task.
""")
    return


@app.cell
def _(mo):
    dropout_slider = mo.ui.slider(
        start=0.0, stop=0.9, value=0.0, step=0.1,
        label="Dropout rate"
    )
    dropout_slider
    return (dropout_slider,)


@app.cell
def _(dropout_slider):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_moons
    import torch
    import torch.nn as nn
    import torch.optim as optim

    dropout_rate = dropout_slider.value

    # Generate data
    rng = np.random.default_rng(42)
    X_data, y_data = make_moons(n_samples=200, noise=0.2, random_state=42)
    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.float32).unsqueeze(1)

    # Build network with dropout
    _model = nn.Sequential(
        nn.Linear(2, 32),
        nn.ReLU(),
        nn.Dropout(p=dropout_rate),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Dropout(p=dropout_rate),
        nn.Linear(32, 1),
        nn.Sigmoid()
    )
    _criterion = nn.BCELoss()
    _optimizer = optim.Adam(_model.parameters(), lr=0.01)

    # Train
    _model.train()
    for _epoch in range(800):
        _pred = _model(X_tensor)
        _loss = _criterion(_pred, y_tensor)
        _optimizer.zero_grad()
        _loss.backward()
        _optimizer.step()

    # Evaluate decision boundary
    _model.eval()
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    xx, yy = np.meshgrid(
        np.linspace(X_data[:, 0].min() - 0.5, X_data[:, 0].max() + 0.5, 200),
        np.linspace(X_data[:, 1].min() - 0.5, X_data[:, 1].max() + 0.5, 200)
    )
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        zz = _model(grid).numpy().reshape(xx.shape)

    ax.contourf(xx, yy, zz, levels=50, cmap="RdBu", alpha=0.7)
    ax.contour(xx, yy, zz, levels=[0.5], colors="k", linewidths=2)
    ax.scatter(X_data[y_data == 0, 0], X_data[y_data == 0, 1],
               c="blue", edgecolors="k", s=30, label="Class 0")
    ax.scatter(X_data[y_data == 1, 0], X_data[y_data == 1, 1],
               c="red", edgecolors="k", s=30, label="Class 1")
    ax.set_title(f"Decision Boundary with Dropout Rate = {dropout_rate:.1f} (final loss: {_loss.item():.4f})")
    ax.legend()
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    plt.tight_layout()
    fig
    return


@app.cell
def _(mo):
    mo.md(r"""
---

## Exercises

1. **Weight decay decomposition.** Consider a two-layer linear network $f(x) = W_2 W_1 x$ with weight decay applied to both $W_1$ and $W_2$. Show that this is *not* equivalent to L2 regularization on the effective weight matrix $W = W_2 W_1$. What does weight decay actually penalize in this case? (Hint: think about the nuclear norm.)

2. **Dropout as noise injection.** For a single neuron $y = w^\top x + b$ with dropout applied to the inputs (rate $p$), derive the expected output $\mathbb{E}[y]$ and the variance $\text{Var}(y)$ as a function of $p$, $w$, and the input statistics. Verify that inverted dropout preserves the expected output.

3. **Early stopping experiment.** Train a neural network on a small dataset (e.g., 1000 examples from CIFAR-10) and plot both training loss and validation loss as a function of epoch. Identify the early stopping point. Then continue training for 5x as many epochs — do you observe epoch-wise double descent?

4. **Augmentation design.** You are building a classifier for chest X-rays (detecting pneumonia). Which of the following augmentations are valid and which would corrupt the label? Justify each: (a) horizontal flip, (b) vertical flip, (c) rotation $\pm 10°$, (d) random brightness, (e) adding Gaussian noise, (f) Mixup with another X-ray.

5. **Label smoothing and temperature.** Show that label smoothing with parameter $\epsilon$ and cross-entropy loss can be decomposed into two terms: the standard cross-entropy with hard targets and a KL divergence term. What is the target distribution of the KL term?

6. **Comparing regularizers.** Take a fully connected network on MNIST. Train four versions: (a) no regularization, (b) dropout only ($p=0.5$), (c) weight decay only ($\lambda = 10^{-3}$), (d) both. Plot test accuracy and training accuracy for each. Which combination gives the best generalization gap?

7. **Implicit regularization.** Train the same architecture with SGD (batch size 32) and full-batch gradient descent on a small dataset. Compare the test accuracies. Measure the L2 norm of the final weights in each case. Does SGD find a lower-norm solution?
""")
    return


@app.cell
def _(mo):
    mo.md(r"""
---

## Code It: Implementation Exercises

Work through these exercises to build each regularization technique from scratch. Each exercise provides a skeleton with `TODO` markers for you to fill in.
""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
### Exercise 1: Manual Weight Decay SGD

Implement a single SGD step with decoupled weight decay (the AdamW approach). Given parameter `w`, gradient `g`, learning rate `lr`, and decay factor `lam`, compute the updated weight. Do NOT add the penalty to the gradient -- apply decay directly to the parameter.
""")
    return


@app.cell
def _(np):
    def weight_decay_step(w, g, lr, lam):
        """One step of SGD with decoupled weight decay.

        Args:
            w: parameter vector (numpy array)
            g: gradient of the data loss w.r.t. w
            lr: learning rate
            lam: weight decay coefficient

        Returns:
            Updated w after one step.
        """
        # TODO: apply decoupled weight decay (2 lines)
        # Step 1: decay the weights by factor (1 - lr * lam)
        # Step 2: subtract lr * gradient
        w_new = None  # TODO: replace this
        return w_new

    # -- Test your implementation --
    _w = np.array([1.0, -2.0, 3.0])
    _g = np.array([0.1, -0.1, 0.2])
    _result = weight_decay_step(_w, _g, lr=0.01, lam=0.1)
    # Expected: (1 - 0.001)*[1,-2,3] - 0.01*[0.1,-0.1,0.2]
    # = [0.999, -1.998, 2.997] - [0.001, -0.001, 0.002]
    # = [0.998, -1.997, 2.995]
    print(f"Result: {_result}")
    print(f"Expected: [0.998, -1.997, 2.995]")
    return


@app.cell
def _(mo):
    mo.md(r"""
### Exercise 2: Inverted Dropout Forward Pass

Implement inverted dropout for a hidden layer activation. During training, zero out each neuron independently with probability `p` and scale survivors by `1/(1-p)`. During evaluation, return activations unchanged.
""")
    return


@app.cell
def _(np):
    def inverted_dropout(h, p, training=True, rng=None):
        """Apply inverted dropout to hidden activations.

        Args:
            h: activation vector (numpy array)
            p: dropout probability (fraction to DROP)
            training: if False, return h unchanged
            rng: numpy random Generator

        Returns:
            Dropped-out activations (same shape as h).
        """
        if not training or p == 0:
            return h
        if rng is None:
            rng = np.random.default_rng()

        # TODO: create binary mask where each entry is 1 with prob (1-p)
        mask = None  # TODO
        # TODO: apply mask and scale by 1/(1-p)
        h_dropped = None  # TODO
        return h_dropped

    # -- Test --
    _rng = np.random.default_rng(42)
    _h = np.ones(10000)
    _out = inverted_dropout(_h, p=0.5, training=True, rng=_rng)
    print(f"Fraction kept: {(_out > 0).mean():.3f}  (expect ~0.5)")
    print(f"Mean output:   {_out.mean():.3f}  (expect ~1.0, proving unbiased)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
### Exercise 3: Early Stopping Tracker

Implement an `EarlyStoppingTracker` class that tracks validation loss, keeps the best score, and signals when to stop after `patience` epochs of no improvement.
""")
    return


@app.cell
def _():
    class EarlyStoppingTracker:
        def __init__(self, patience=5):
            self.patience = patience
            self.best_loss = float("inf")
            self.best_epoch = 0
            self.wait = 0

        def step(self, epoch, val_loss):
            """Record a new validation loss.

            Returns:
                True if training should STOP, False to continue.
            """
            # TODO: if val_loss improved, update best_loss, best_epoch, reset wait
            # TODO: else increment wait
            # TODO: return True if wait >= patience, else False
            pass  # TODO: replace with your logic

    # -- Test --
    _tracker = EarlyStoppingTracker(patience=3)
    _losses = [1.0, 0.8, 0.7, 0.75, 0.76, 0.78, 0.6, 0.65, 0.66, 0.67]
    for _ep, _vl in enumerate(_losses):
        _stop = _tracker.step(_ep, _vl)
        print(f"Epoch {_ep}: val_loss={_vl:.2f}, "
              f"best={_tracker.best_loss:.2f}@ep{_tracker.best_epoch}, "
              f"stop={_stop}")
        if _stop:
            break
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
### Exercise 4: Label Smoothing Cross-Entropy

Implement cross-entropy loss with label smoothing. Given a hard class index, number of classes K, and smoothing factor epsilon, compute the smoothed target distribution and return the cross-entropy loss against predicted logits.
""")
    return


@app.cell
def _(np):
    def label_smoothing_ce(logits, true_class, K, eps=0.1):
        """Cross-entropy loss with label smoothing.

        Args:
            logits: raw scores, shape (K,)
            true_class: integer class label
            K: number of classes
            eps: smoothing parameter

        Returns:
            Scalar loss value.
        """
        # TODO: build smoothed target vector y_smooth of shape (K,)
        # true class gets (1 - eps) + eps/K, others get eps/K
        y_smooth = None  # TODO

        # TODO: compute softmax probabilities from logits
        # (subtract max for numerical stability)
        probs = None  # TODO

        # TODO: compute cross-entropy = -sum(y_smooth * log(probs))
        loss = None  # TODO
        return loss

    # -- Test --
    _logits = np.array([2.0, 1.0, 0.1, -1.0, 0.5])
    _loss_smooth = label_smoothing_ce(_logits, true_class=0, K=5, eps=0.1)
    _loss_hard   = label_smoothing_ce(_logits, true_class=0, K=5, eps=0.0)
    print(f"Loss (hard targets):   {_loss_hard}")
    print(f"Loss (smooth, eps=0.1): {_loss_smooth}")
    print("Smooth loss should be slightly higher (penalizes overconfidence)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
### Exercise 5: Mixup Data Augmentation

Implement the Mixup augmentation scheme. Given two batches of data `(x1, y1)` and `(x2, y2)` with one-hot labels, sample `alpha ~ Beta(a, a)` and return the mixed inputs and labels.
""")
    return


@app.cell
def _(np):
    def mixup(x1, y1, x2, y2, beta_param=0.2, rng=None):
        """Mixup augmentation for one pair of examples.

        Args:
            x1, x2: input vectors (numpy arrays, same shape)
            y1, y2: one-hot label vectors (numpy arrays, same shape)
            beta_param: parameter for Beta(a, a) distribution
            rng: numpy random Generator

        Returns:
            (x_mix, y_mix) tuple of mixed input and label.
        """
        if rng is None:
            rng = np.random.default_rng()

        # TODO: sample alpha from Beta(beta_param, beta_param)
        alpha = None  # TODO

        # TODO: compute mixed input and label
        x_mix = None  # TODO
        y_mix = None  # TODO
        return x_mix, y_mix

    # -- Test --
    _rng_mx = np.random.default_rng(0)
    _x1 = np.array([1.0, 0.0, 0.0])
    _y1 = np.array([1.0, 0.0])       # class 0
    _x2 = np.array([0.0, 0.0, 1.0])
    _y2 = np.array([0.0, 1.0])       # class 1

    _xm, _ym = mixup(_x1, _y1, _x2, _y2, beta_param=0.2, rng=_rng_mx)
    print(f"Mixed input: {_xm}")
    print(f"Mixed label: {_ym}  (should be a soft distribution)")
    return


if __name__ == "__main__":
    app.run()
