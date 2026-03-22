import marimo

app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    np.random.seed(42)
    return (np,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Module 3B: Generative Models

    Everything we have built so far --- classifiers, regressors, even transformers --- has been *discriminative*. We learned boundaries, we learned mappings from inputs to outputs, we learned conditional distributions $p(y|x)$. That is powerful, but it sidesteps a deeper question: **what does the data itself look like?**

    Generative modeling is the study of learning $p(x)$ --- the full probability distribution over data. Not "is this a cat or a dog?" but "what does *any plausible image* look like?" This is a fundamentally harder problem. You are no longer drawing a decision boundary through data; you are trying to capture the entire structure of the data manifold. But the payoff is enormous: once you have $p(x)$, you can *sample* from it to create new data, *evaluate* how likely a given datapoint is, *interpolate* between examples, and *condition* generation on auxiliary information.

    This module covers the four major families of generative models --- autoencoders and VAEs, GANs, diffusion models, and normalizing flows --- along with the mathematical machinery that makes each one work. You already have the prerequisites: probability and MLE from Part 1, neural network architectures from Part 2, and the optimization intuitions from everything in between. Now we put it all together.

    > **Core references:** [Goodfellow et al. Ch 20: Deep Generative Models](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) for the foundational theory, and [Murphy PML2 Ch 20: Dimensionality Reduction](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML2.pdf) through [Ch 22: Generative Models Revisited](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML2.pdf) for the modern landscape.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 1. The Generative Modeling Problem

    Let me be precise about what we are trying to do. We have a dataset $\{x^{(1)}, x^{(2)}, \dots, x^{(N)}\}$ drawn from some unknown distribution $p_{\text{data}}(x)$. We want to learn a model $p_\theta(x)$ that approximates $p_{\text{data}}(x)$ as closely as possible.

    If $x$ is an image of size $256 \times 256 \times 3$, then $p(x)$ is a distribution over a space with nearly 200,000 dimensions. The overwhelming majority of points in that space look like static noise. Natural images occupy a tiny, thin, twisted manifold embedded in that high-dimensional space. Learning $p(x)$ means learning where that manifold is and what it looks like.

    **Discriminative vs. generative** --- the distinction is clean. A discriminative model learns $p(y|x)$: given an input, what is the output? A generative model learns $p(x)$, or equivalently $p(x, y) = p(x|y)p(y)$, which lets you generate new data and also do classification via Bayes' rule. See [Bishop PRML Section 1.5.4](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) for the classical treatment of this distinction.

    There are two things you might want from a generative model:
    1. **Density estimation**: evaluate $p_\theta(x)$ for any $x$ (useful for anomaly detection, compression, model comparison)
    2. **Sampling**: draw $x \sim p_\theta(x)$ to create new data (useful for generation, augmentation, creative applications)

    Different model families give you different subsets of these capabilities, as we will see.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2. Autoencoders: The Precursor

    Before we get to proper generative models, we need autoencoders --- both because they are historically important and because VAEs extend them directly.

    An autoencoder is a neural network trained to reconstruct its own input. It consists of:
    - An **encoder** $f_\phi: \mathcal{X} \to \mathcal{Z}$ that maps input $x$ to a latent code $z = f_\phi(x)$
    - A **decoder** $g_\theta: \mathcal{Z} \to \mathcal{X}$ that maps the code back: $\hat{x} = g_\theta(z)$

    The loss is reconstruction error: $\mathcal{L} = \|x - g_\theta(f_\phi(x))\|^2$.

    The key is the **bottleneck**: $\dim(\mathcal{Z}) \ll \dim(\mathcal{X})$. If the latent space is smaller than the input space, the network cannot simply memorize an identity mapping. It is forced to learn a compressed representation --- extracting the most salient features and discarding noise.

    **Connection to PCA.** If you use linear activations and a squared-error loss, the autoencoder learns exactly the same subspace as PCA. The encoder projects onto the top-$k$ principal components, and the decoder reconstructs from them. This is proven in [MML Section 10.2](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) and [Goodfellow et al. Section 14.1](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf). A nonlinear autoencoder is therefore a nonlinear generalization of PCA --- it can capture curved manifolds that PCA cannot.

    **Why autoencoders are not generative models.** You can encode and decode existing data, but you cannot *sample new data*. Why not? Because the latent space has no particular structure. There is no guarantee that a random point $z$ in the latent space will decode to something meaningful. The encoder maps training data to scattered islands in $\mathcal{Z}$; the vast space between those islands is uncharted territory. If you sample from the gaps, you get garbage.

    This is the exact problem that VAEs solve.
    """)
    return


@app.cell
def _(np):
    # Linear autoencoder = PCA: encode to 2D, decode back to 5D
    # Generate data on a 2D subspace embedded in 5D
    true_basis = np.random.randn(2, 5)  # 2D subspace in 5D
    X_ae = np.random.randn(200, 2) @ true_basis  # data lives on 2D manifold

    # "Encoder" via PCA: project onto top-2 principal components
    U, S, Vt = np.linalg.svd(X_ae, full_matrices=False)
    W_enc = Vt[:2].T         # encoder weights (5x2)
    z_ae = X_ae @ W_enc      # encode: 5D -> 2D

    # "Decoder": project back to 5D
    X_recon = z_ae @ W_enc.T  # decode: 2D -> 5D

    recon_error = np.mean((X_ae - X_recon) ** 2)
    print(f"Linear AE reconstruction error: {recon_error:.6f}")
    print(f"(Near zero because data truly lives on a 2D subspace)")
    return (W_enc, z_ae)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3. Variational Autoencoders (VAEs)

    VAEs are one of the most elegant constructions in modern machine learning. They take the autoencoder architecture and put it on a rigorous probabilistic foundation.

    ### 3.1 The Generative Story

    A VAE defines a generative process:
    1. Sample a latent variable: $z \sim p(z) = \mathcal{N}(0, I)$
    2. Generate data from the latent: $x \sim p_\theta(x|z)$

    The decoder network parameterizes $p_\theta(x|z)$. The prior $p(z)$ is a simple, fixed distribution --- standard Gaussian. The idea is that the decoder learns to map *any* point in latent space to a plausible data point. This is what was missing from vanilla autoencoders: structure in the latent space.

    ### 3.2 The Intractability Problem

    We want to maximize the log-likelihood $\log p_\theta(x)$ for each training example. But:

    $$p_\theta(x) = \int p_\theta(x|z) \, p(z) \, dz$$

    This integral is intractable. The latent space is high-dimensional, and $p_\theta(x|z)$ is a complex nonlinear function parameterized by a neural network. We cannot evaluate or optimize this directly.

    We also cannot compute the true posterior $p_\theta(z|x) = p_\theta(x|z)p(z) / p_\theta(x)$, because the denominator is exactly the intractable integral above.

    ### 3.3 Variational Inference and the ELBO

    The solution is **variational inference**: instead of computing the true posterior, we *approximate* it. We introduce a recognition network (encoder) $q_\phi(z|x)$ --- typically a Gaussian whose mean and variance are output by a neural network:

    $$q_\phi(z|x) = \mathcal{N}(z; \mu_\phi(x), \sigma^2_\phi(x) I)$$

    Now we derive the **Evidence Lower Bound (ELBO)**. Starting from the log-likelihood and applying Jensen's inequality (or equivalently, writing out the KL divergence between $q$ and the true posterior):

    $$\log p_\theta(x) \geq \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{reconstruction term}} - \underbrace{D_{\text{KL}}(q_\phi(z|x) \| p(z))}_{\text{regularization term}}$$

    This is the ELBO, and maximizing it is the entire training objective for VAEs. Let me explain each term:

    - **Reconstruction term** $\mathbb{E}_{q}[\log p_\theta(x|z)]$: This says "encode $x$ into $z$ using the encoder, then decode $z$ back --- how well do you reconstruct $x$?" For Gaussian $p_\theta(x|z)$, this reduces to negative MSE. This term alone would give you a vanilla autoencoder.

    - **KL regularization** $D_{\text{KL}}(q_\phi(z|x) \| p(z))$: This penalizes the encoder for producing a posterior that deviates from the prior $\mathcal{N}(0, I)$. It forces the latent space to be *organized* --- every region of the standard Gaussian should map to something reasonable. This is what makes sampling possible.

    The tension between these two terms is the core of VAE training. Reconstruction wants each input to map to a precise, unique point in latent space. Regularization wants all inputs to map to overlapping, spread-out distributions near the origin. The balance between them determines the quality of both reconstruction and generation.

    For the full derivation, see [Bishop PRML Section 10.1](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) for variational inference foundations, and [Murphy PML2 Section 21.2](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML2.pdf) for the VAE-specific treatment.
    """)
    return


@app.cell
def _(np):
    # ELBO in code: KL(q || p) for diagonal Gaussian q vs standard normal prior
    # q = N(mu, diag(sigma^2)), p = N(0, I)
    # KL = 0.5 * sum(mu^2 + sigma^2 - log(sigma^2) - 1)

    d_latent = 4  # latent dimension
    # Encoder outputs for one data point
    mu_q = np.array([0.8, -0.3, 1.2, -0.5])       # encoder mean
    log_var_q = np.array([-0.5, -1.0, 0.2, -0.3])  # encoder log-variance
    sigma2_q = np.exp(log_var_q)

    # Closed-form KL divergence
    kl_div = 0.5 * np.sum(mu_q**2 + sigma2_q - log_var_q - 1)
    print(f"KL(q || p) = {kl_div:.4f}")
    print(f"Per-dimension: {(mu_q**2 + sigma2_q - log_var_q - 1) / 2}")
    print(f"(Each term is 0 only when mu=0, sigma=1)")
    return (mu_q, log_var_q, sigma2_q, kl_div)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.4 The Reparameterization Trick

    There is a practical problem. The ELBO involves an expectation over $q_\phi(z|x)$, which means we need to *sample* $z$ from the encoder. But sampling is not differentiable --- you cannot backpropagate through a random number generator.

    The reparameterization trick resolves this beautifully. Instead of sampling $z \sim \mathcal{N}(\mu, \sigma^2 I)$ directly, we write:

    $$z = \mu_\phi(x) + \sigma_\phi(x) \odot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$

    Now the randomness is in $\varepsilon$, which does not depend on the parameters. The dependence on $\phi$ flows through the deterministic operations $\mu_\phi$ and $\sigma_\phi$, and standard backpropagation works. This is a simple change of variables, but it was the key insight that made VAE training practical. See [Goodfellow et al. Section 20.9](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) and [Murphy PML2 Section 21.2.2](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML2.pdf).
    """)
    return


@app.cell
def _(np):
    # Reparameterization trick: z = mu + sigma * epsilon
    # Gradients flow through mu and sigma, not through the sampling

    mu_reparam = np.array([1.0, -0.5])         # encoder output: mean
    sigma_reparam = np.array([0.3, 0.8])        # encoder output: std dev

    # Sample epsilon from standard normal (fixed randomness)
    epsilon = np.random.randn(5, 2)             # 5 samples, 2D latent

    # Reparameterized samples: deterministic function of (mu, sigma, epsilon)
    z_samples = mu_reparam + sigma_reparam * epsilon  # shape: (5, 2)

    print("epsilon (random, parameter-free):")
    print(epsilon.round(3))
    print(f"\nz = mu + sigma * epsilon (differentiable w.r.t. mu, sigma):")
    print(z_samples.round(3))
    print(f"\nSample mean:  {z_samples.mean(axis=0).round(3)}  (should be near mu={mu_reparam})")
    return (z_samples,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.5 Why VAEs Produce Blurry Outputs

    VAEs are known for generating blurry images, and the reason is fundamental, not incidental.

    First, the reconstruction term with Gaussian likelihood is equivalent to MSE, which penalizes pixel-level deviations equally. When there is uncertainty --- when multiple outputs are plausible for a given $z$ --- MSE produces the *mean* of those outputs, which is blurry.

    Second, the ELBO is a *lower bound* on the log-likelihood. We are optimizing a surrogate, not the actual objective. The gap between the ELBO and $\log p(x)$ equals $D_{\text{KL}}(q_\phi(z|x) \| p_\theta(z|x))$, which is always non-negative. This means VAEs exhibit **mode-covering** behavior: the approximate posterior $q$ tries to cover all the modes of the true posterior, spreading probability mass broadly rather than concentrating it at the sharpest peaks.
    """)
    return


@app.cell
def _(np):
    # Demonstrating why MSE produces blurry outputs
    # When multiple outputs are plausible, MSE averages them

    # Suppose two equally likely "sharp" images for a given z
    sharp_img_1 = np.array([1.0, 0.0, 0.0, 1.0])  # e.g., shifted left
    sharp_img_2 = np.array([0.0, 1.0, 1.0, 0.0])  # e.g., shifted right

    # MSE-optimal prediction = mean of plausible outputs = blurry
    mse_optimal = 0.5 * sharp_img_1 + 0.5 * sharp_img_2
    print(f"Sharp image 1:      {sharp_img_1}")
    print(f"Sharp image 2:      {sharp_img_2}")
    print(f"MSE-optimal output: {mse_optimal}  <-- blurry average!")
    print(f"\nMSE to img 1: {np.mean((mse_optimal - sharp_img_1)**2):.3f}")
    print(f"MSE to img 2: {np.mean((mse_optimal - sharp_img_2)**2):.3f}")
    print(f"MSE of img1 to img2: {np.mean((sharp_img_1 - sharp_img_2)**2):.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4. Generative Adversarial Networks (GANs)

    GANs take a completely different approach to generation. Where VAEs are grounded in probabilistic inference, GANs are grounded in game theory. The result: sharper outputs, but at the cost of training stability and density estimation.

    ### 4.1 The Adversarial Game

    A GAN consists of two networks:
    - **Generator** $G_\theta$: takes random noise $z \sim p(z)$ and produces a fake sample $G_\theta(z)$
    - **Discriminator** $D_\phi$: takes a sample (real or fake) and outputs the probability that it is real

    They play a minimax game:

    $$\min_G \max_D \; \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p(z)}[\log(1 - D(G(z)))]$$

    The discriminator tries to maximize this: correctly classify real data as real and fake data as fake. The generator tries to minimize it: fool the discriminator into thinking fake data is real.

    At the Nash equilibrium of this game (if it exists), the generator produces samples indistinguishable from real data, and the discriminator outputs $1/2$ for everything --- it literally cannot tell the difference. Goodfellow proved that the global optimum of this objective is achieved when $p_G = p_{\text{data}}$. See the original theoretical analysis in [Goodfellow et al. Section 20.10.4](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf).
    """)
    return


@app.cell
def _(np):
    # GAN objective in numpy: the minimax value function
    # V(D, G) = E[log D(x)] + E[log(1 - D(G(z)))]

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    # Simulated discriminator scores (logits) on real and fake data
    real_logits = np.array([2.0, 1.5, 1.8, 2.2])   # D thinks these are real
    fake_logits = np.array([-1.5, -0.8, -1.2, 0.3]) # D thinks most are fake

    D_real = sigmoid(real_logits)
    D_fake = sigmoid(fake_logits)

    # Discriminator wants to MAXIMIZE this
    V_D = np.mean(np.log(D_real + 1e-8)) + np.mean(np.log(1 - D_fake + 1e-8))
    # Generator wants to MINIMIZE this (or equivalently maximize log D(G(z)))
    G_loss = -np.mean(np.log(D_fake + 1e-8))  # non-saturating variant

    print(f"D(real): {D_real.round(3)}  (should be ~1)")
    print(f"D(fake): {D_fake.round(3)}  (D wants ~0, G wants ~1)")
    print(f"V(D,G) = {V_D:.4f}  (D maximizes, G minimizes)")
    print(f"G loss (non-saturating) = {G_loss:.4f}")
    return (sigmoid,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.2 Training Dynamics

    In practice, we alternate:
    1. **Train discriminator**: freeze $G$, take a batch of real data and a batch of generated data, update $D$ to better distinguish them
    2. **Train generator**: freeze $D$, generate a batch, update $G$ to better fool $D$

    This alternating optimization is where the trouble begins.

    ### 4.3 Why GANs Are Hard to Train

    **Mode collapse.** The generator discovers that a small number of outputs reliably fool the discriminator and stops exploring. Instead of learning the full data distribution, it produces a handful of "greatest hits" on repeat. The discriminator eventually catches on, but the generator just finds new modes to collapse onto, and the cycle continues.

    **Training instability.** The minimax game does not always converge. In theory, simultaneous gradient descent on a minimax objective can oscillate or diverge. You need careful hyperparameter tuning, architecture choices, and often heuristic tricks (label smoothing, spectral normalization, progressive growing) to get stable training.

    **Vanishing gradients for $G$.** When the discriminator becomes too good --- confidently outputting 0 for all generated samples --- the gradient signal to the generator vanishes. The generator is told "you're terrible" but gets no useful information about *how* to improve. This is why the original paper suggests optimizing $\max_G \mathbb{E}[\log D(G(z))]$ instead of $\min_G \mathbb{E}[\log(1 - D(G(z)))]$ in early training --- the gradient is stronger when the generator is bad.
    """)
    return


@app.cell
def _(np):
    # Optimal discriminator: D*(x) = p_data(x) / (p_data(x) + p_G(x))
    # When p_G = p_data, D*(x) = 0.5 everywhere

    x_grid = np.linspace(-4, 4, 200)

    # Two 1D Gaussians as p_data and p_G
    p_data = np.exp(-0.5 * (x_grid - 1)**2) / np.sqrt(2 * np.pi)
    p_gen  = np.exp(-0.5 * (x_grid + 0.5)**2 / 1.5**2) / (np.sqrt(2 * np.pi) * 1.5)

    # Optimal discriminator
    D_star = p_data / (p_data + p_gen + 1e-10)

    print(f"At x=1.0 (high p_data): D* = {D_star[np.argmin(np.abs(x_grid-1))]:.3f}")
    print(f"At x=-0.5 (high p_gen): D* = {D_star[np.argmin(np.abs(x_grid+0.5))]:.3f}")
    print(f"When p_data = p_gen, D* -> 0.5 (equilibrium)")
    return (D_star,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.4 Wasserstein GAN (WGAN)

    The Wasserstein GAN addresses the vanishing gradient problem by changing the divergence measure. Instead of the Jensen-Shannon divergence (implicit in the original GAN objective), WGAN uses the **Earth Mover's distance** (Wasserstein-1 distance):

    $$W(p_{\text{data}}, p_G) = \inf_{\gamma \in \Pi(p_{\text{data}}, p_G)} \mathbb{E}_{(x,y) \sim \gamma}[\|x - y\|]$$

    Intuitively, this measures the minimum "work" needed to reshape one distribution into another. The crucial property is that $W$ provides useful gradients even when the two distributions have non-overlapping support --- exactly the situation where JS divergence saturates.

    In practice, WGAN replaces the discriminator with a **critic** (no sigmoid, outputs an unbounded score) and enforces a Lipschitz constraint via weight clipping or gradient penalty. The training is measurably more stable, and the critic loss actually correlates with sample quality --- a rare luxury in GAN training.

    ### 4.5 StyleGAN and the State of the Art

    StyleGAN (and its successors StyleGAN2, StyleGAN3) represents the pinnacle of GAN-based image generation. Key innovations include:
    - A **mapping network** that transforms the noise vector $z$ into an intermediate latent space $w$, which is then injected into the generator at multiple scales via adaptive instance normalization
    - **Progressive growing**: training at low resolution first, then gradually adding layers for higher resolution
    - **Style mixing**: injecting different latent codes at different layers, giving control over coarse vs. fine features

    The images StyleGAN produces are remarkably sharp and realistic. But GANs fundamentally do not give you $p(x)$ --- you cannot evaluate the likelihood of a given image. You have a sampler without a density. For an extended discussion, see [Murphy PML2 Section 22.4](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML2.pdf).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 5. Diffusion Models: The Current Frontier

    Diffusion models have overtaken both VAEs and GANs as the dominant paradigm for high-quality generation. They combine the training stability of VAEs with image quality that surpasses GANs, and they provide tractable likelihoods as a bonus. This is the engine behind DALL-E 2, Stable Diffusion, Midjourney, and most modern image generation systems.

    ### 5.1 The Core Idea

    The insight is disarmingly simple: **destruction is easy, creation is hard --- so learn to reverse destruction.**

    The **forward process** gradually corrupts data by adding Gaussian noise over $T$ timesteps until it becomes pure noise. The **reverse process** learns to undo this corruption step by step. If you can learn to denoise at every noise level, you can start from pure Gaussian noise and iteratively denoise to produce a clean sample.
    """)
    return


@app.cell
def _(mo):
    mo.image(src="../animations/rendered/DiffusionProcess.gif")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.2 The Forward Process

    Given a data point $x_0$, the forward process produces a sequence $x_1, x_2, \dots, x_T$ by adding noise at each step:

    $$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} \, x_{t-1}, \, \beta_t I)$$

    Here $\beta_t \in (0, 1)$ is a noise schedule (increasing over time --- more noise at later steps). The signal is scaled down by $\sqrt{1-\beta_t}$ and noise of variance $\beta_t$ is added. After enough steps, $x_T \approx \mathcal{N}(0, I)$ regardless of $x_0$.

    A critical convenience: you can jump directly to any timestep without iterating. Define $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$. Then:

    $$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} \, x_0, \, (1 - \bar{\alpha}_t) I)$$

    This means: $x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \varepsilon$, where $\varepsilon \sim \mathcal{N}(0, I)$. We can generate any noisy version of $x_0$ in a single step. This is what makes training efficient.
    """)
    return


@app.cell
def _(np):
    # Forward diffusion process: noise schedule and closed-form q(x_t | x_0)
    T_diff = 50
    # Linear beta schedule from 1e-4 to 0.02
    betas = np.linspace(1e-4, 0.02, T_diff)
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas)  # cumulative product: alpha_bar_t

    # Start from a clean 1D "data point"
    x0 = np.array([3.0])
    eps = np.random.randn(T_diff, 1)  # pre-sample noise for each step

    # Closed-form: x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
    x_t_vals = np.sqrt(alpha_bar)[:, None] * x0 + np.sqrt(1 - alpha_bar)[:, None] * eps

    print(f"x_0 = {x0[0]:.2f}")
    print(f"x_10 (alpha_bar={alpha_bar[9]:.3f}): {x_t_vals[9, 0]:.3f}")
    print(f"x_25 (alpha_bar={alpha_bar[24]:.3f}): {x_t_vals[24, 0]:.3f}")
    print(f"x_50 (alpha_bar={alpha_bar[49]:.3f}): {x_t_vals[49, 0]:.3f}")
    print(f"\nAs t->T, alpha_bar->0, so x_t -> pure noise N(0,1)")
    return (alpha_bar, betas)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.3 The Reverse Process

    The reverse process learns to undo each corruption step:

    $$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \, \Sigma_\theta(x_t, t))$$

    A neural network (typically a U-Net with attention layers) takes the noisy input $x_t$ and the timestep $t$, and predicts the parameters of the denoising distribution. In the seminal DDPM paper (Ho et al., 2020), the variance is fixed and the network only predicts the mean.

    ### 5.4 Training: Noise Prediction

    Here is where the simplification happens. Instead of predicting the mean directly, the network predicts the *noise that was added*. The training objective becomes:

    $$\mathcal{L} = \mathbb{E}_{t, x_0, \varepsilon} \left[ \| \varepsilon - \varepsilon_\theta(x_t, t) \|^2 \right]$$

    where $x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \varepsilon$ and $t$ is sampled uniformly from $\{1, \dots, T\}$.

    This is just MSE between the true noise and the predicted noise. No adversarial training, no minimax games, no posterior approximation gaps. A single, stable, well-understood loss function. Train a denoiser, and generation falls out as a consequence.
    """)
    return


@app.cell
def _(alpha_bar, np):
    # DDPM training step: one forward pass of the loss function
    # L = E[||eps - eps_theta(x_t, t)||^2]

    d_data = 8  # data dimension
    x0_train = np.random.randn(d_data) * 0.5 + 2.0  # one training sample

    # 1. Sample random timestep
    t_step = np.random.randint(0, len(alpha_bar))
    ab_t = alpha_bar[t_step]

    # 2. Sample noise
    eps_true = np.random.randn(d_data)

    # 3. Create noisy version (closed-form)
    x_t_noisy = np.sqrt(ab_t) * x0_train + np.sqrt(1 - ab_t) * eps_true

    # 4. "Network prediction" (placeholder -- in practice a U-Net)
    eps_pred = eps_true + 0.1 * np.random.randn(d_data)  # imperfect prediction

    # 5. Loss = MSE between true and predicted noise
    loss = np.mean((eps_true - eps_pred) ** 2)
    print(f"Timestep t={t_step}, alpha_bar_t={ab_t:.4f}")
    print(f"Diffusion loss (noise prediction MSE): {loss:.6f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.5 Sampling

    To generate a new sample:
    1. Draw $x_T \sim \mathcal{N}(0, I)$
    2. For $t = T, T-1, \dots, 1$:
       - Predict the noise: $\hat{\varepsilon} = \varepsilon_\theta(x_t, t)$
       - Compute the denoised estimate and take one reverse step to get $x_{t-1}$
    3. Return $x_0$

    This iterative process is slow (typically $T = 1000$ steps), which is the main practical drawback of diffusion models. Considerable research has gone into reducing the number of steps required.
    """)
    return


@app.cell
def _(alpha_bar, betas, np):
    # Simplified DDPM sampling loop (1D toy example)
    # Assume a perfect denoiser that knows the true x0

    T_samp = len(betas)
    true_x0 = 3.0  # ground truth we want to recover

    # Start from pure noise
    x_t_samp = np.random.randn()

    for t in reversed(range(T_samp)):
        # In reality, a network predicts the noise. Here we simulate a
        # "perfect" noise predictor by computing what the noise would be.
        ab_t = alpha_bar[t]
        pred_noise = (x_t_samp - np.sqrt(ab_t) * true_x0) / np.sqrt(1 - ab_t + 1e-8)

        # DDPM reverse step: compute mu_t, then sample x_{t-1}
        beta_t = betas[t]
        mu_t = (1 / np.sqrt(1 - beta_t)) * (x_t_samp - beta_t / np.sqrt(1 - ab_t + 1e-8) * pred_noise)

        if t > 0:
            x_t_samp = mu_t + np.sqrt(beta_t) * np.random.randn()
        else:
            x_t_samp = mu_t  # final step: no noise

    print(f"True x_0: {true_x0}")
    print(f"Recovered x_0 after {T_samp} reverse steps: {x_t_samp:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.6 The Score Matching Perspective

    There is an equivalent and illuminating way to view diffusion models. The **score function** of a distribution is $\nabla_x \log p(x)$ --- the gradient of the log-density with respect to the data. It points in the direction of increasing probability density.

    If you know the score function, you can generate samples via **Langevin dynamics**: start from noise and iteratively follow the score, adding a small amount of noise at each step:

    $$x_{t+1} = x_t + \frac{\eta}{2} \nabla_x \log p(x_t) + \sqrt{\eta} \, \varepsilon_t$$

    The noise prediction network $\varepsilon_\theta$ is directly related to the score function. Specifically, for a noisy distribution at noise level $\sigma_t$:

    $$\nabla_{x_t} \log q(x_t) \approx -\frac{\varepsilon_\theta(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}}$$

    So training a noise predictor is equivalent to learning the score function at every noise level. This connection, formalized by Song and Ermon (2019), unifies denoising diffusion and score matching into a single framework.
    """)
    return


@app.cell
def _(np):
    # Score function and Langevin dynamics for a 1D Gaussian mixture
    # Score = d/dx log p(x) -- points toward higher density

    def gmm_score(x, means, stds, weights):
        """Score function for a Gaussian mixture (1D)."""
        # p(x) = sum_k w_k * N(x; mu_k, sigma_k^2)
        components = np.array([w * np.exp(-0.5*((x - m)/s)**2) / s
                               for m, s, w in zip(means, stds, weights)])
        p_x = components.sum(axis=0) + 1e-10
        # d/dx log p(x) = (1/p(x)) * sum_k w_k * N(x;mu_k,sig_k) * (-(x-mu_k)/sig_k^2)
        grad_components = np.array([w * np.exp(-0.5*((x - m)/s)**2) / s * (-(x - m)/s**2)
                                    for m, s, w in zip(means, stds, weights)])
        return grad_components.sum(axis=0) / p_x

    # Langevin dynamics: sample from mixture of two Gaussians
    means_ld, stds_ld, weights_ld = [-2, 2], [0.7, 0.7], [0.5, 0.5]
    eta = 0.05  # step size
    x_lang = np.random.randn() * 3  # start from noise
    for _ in range(200):
        x_lang = x_lang + (eta / 2) * gmm_score(x_lang, means_ld, stds_ld, weights_ld) + np.sqrt(eta) * np.random.randn()

    print(f"Langevin sample after 200 steps: {x_lang:.3f}")
    print(f"(Target modes at x=-2 and x=2)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.7 Connection to SDEs

    The discrete forward process can be generalized to a continuous-time **Stochastic Differential Equation**:

    $$dx = f(x, t) \, dt + g(t) \, dw$$

    where $w$ is a Wiener process. The reverse process is also an SDE, and solving it requires knowing the score $\nabla_x \log p_t(x)$. This continuous-time perspective, developed by Song et al. (2021), provides a unified framework where DDPM, score matching, and other diffusion variants are special cases of different discretizations of the same SDE. I mention this primarily so you know the connection exists --- the details are beyond our scope here, but see [Murphy PML2 Section 22.6](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML2.pdf) for a thorough treatment.

    ### 5.8 Why Diffusion Models Won

    The comparison with previous paradigms is striking:
    - **vs. VAEs**: No ELBO gap. The variational bound in diffusion models is much tighter because the forward process is fixed and simple. No blurriness from mode-covering.
    - **vs. GANs**: No adversarial training. No mode collapse. No discriminator to balance. Just minimize MSE on noise prediction. The training is as stable as training a regular supervised model.
    - **Quality**: Diffusion models achieve state-of-the-art FID scores (the standard metric for image generation quality) on essentially all benchmarks.

    The tradeoff is sampling speed --- diffusion models require many forward passes to generate a single sample, whereas GANs need only one. This has spurred research into distillation, consistency models, and few-step samplers.

    ### 5.9 Key Variants

    - **DDPM** (Denoising Diffusion Probabilistic Models): The foundational paper. Fixed variance schedule, $\varepsilon$-prediction, 1000 steps.
    - **DDIM** (Denoising Diffusion Implicit Models): A deterministic sampling procedure that allows fewer steps with minimal quality loss. Treats the reverse process as an ODE rather than an SDE.
    - **Classifier-free guidance**: A technique for conditional generation. Train the model both with and without conditioning information (e.g., text). At sampling time, extrapolate *away* from the unconditional prediction toward the conditional one. This sharpens outputs and improves adherence to the condition. It is the secret sauce behind the vivid, prompt-following behavior of modern text-to-image systems.

    For the mathematical foundations of all of the above, see [Goodfellow et al. Ch 20](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) for the classical deep generative framework and [Murphy PML2 Section 22.5-22.6](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML2.pdf) for the modern diffusion model treatment.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 6. Normalizing Flows

    Normalizing flows take a different route to tractable generative modeling: **exact likelihood through invertible transformations**.

    The idea: start with a simple distribution $z \sim p(z) = \mathcal{N}(0, I)$ and transform it through a sequence of bijective (invertible) functions $f_1, f_2, \dots, f_K$ to get $x = f_K \circ \cdots \circ f_1(z)$.

    Since each $f_k$ is invertible, we can compute the exact density of $x$ via the change of variables formula:

    $$\log p(x) = \log p(z) - \sum_{k=1}^{K} \log \left| \det \frac{\partial f_k}{\partial f_{k-1}} \right|$$

    The log-determinant of the Jacobian accounts for how the transformation stretches or compresses volume in the data space. This gives us *exact* log-likelihoods --- no bounds, no approximations.

    The catch: computing the Jacobian determinant is $O(d^3)$ in general, which is prohibitive for high-dimensional data. So flow architectures must be carefully designed so that the Jacobian is triangular (making the determinant just the product of diagonal entries). This leads to architectural constraints --- coupling layers (RealNVP), autoregressive transforms (MAF, IAF), and residual flows --- that limit expressiveness compared to unconstrained architectures.

    Flows produce good but not state-of-the-art samples. Their strength is exact density evaluation, which is valuable for tasks like anomaly detection and model comparison. See [Murphy PML2 Section 22.2](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML2.pdf) for a comprehensive treatment.
    """)
    return


@app.cell
def _(np):
    # Normalizing flow: 1D change of variables
    # z ~ N(0,1), apply invertible f(z) = z + a*tanh(b*z), compute exact p(x)

    a_flow, b_flow = 1.5, 0.8

    def flow_forward(z_in):
        """Invertible transformation: x = z + a*tanh(b*z)."""
        return z_in + a_flow * np.tanh(b_flow * z_in)

    def flow_log_det_jacobian(z_in):
        """log |dx/dz| = log |1 + a*b*(1 - tanh^2(b*z))|."""
        return np.log(np.abs(1 + a_flow * b_flow * (1 - np.tanh(b_flow * z_in)**2)))

    # Sample from base distribution and transform
    z_flow = np.random.randn(5000)
    x_flow = flow_forward(z_flow)

    # Exact log-likelihood for a specific point (using numerical inverse)
    x_test = 1.5
    from scipy.optimize import brentq
    z_test = brentq(lambda z: flow_forward(z) - x_test, -10, 10)
    log_pz = -0.5 * z_test**2 - 0.5 * np.log(2 * np.pi)  # log N(0,1)
    log_px = log_pz - flow_log_det_jacobian(z_test)         # change of variables

    print(f"x = {x_test}, inverse z = {z_test:.4f}")
    print(f"log p(z) = {log_pz:.4f}")
    print(f"log |det J| = {flow_log_det_jacobian(z_test):.4f}")
    print(f"log p(x) = {log_px:.4f}  (exact -- no bounds!)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 7. Comparing Generative Model Families

    Here is the landscape at a glance:

    | Property | VAE | GAN | Diffusion | Flow |
    |---|---|---|---|---|
    | **Likelihood** | Lower bound (ELBO) | None | Tractable (via VLB) | Exact |
    | **Sample quality** | Blurry | Sharp | Excellent | Good |
    | **Training stability** | Stable | Unstable | Stable | Stable |
    | **Mode coverage** | Good | Poor (mode collapse) | Excellent | Good |
    | **Sampling speed** | Fast (single pass) | Fast (single pass) | Slow (iterative) | Moderate |
    | **Latent space** | Structured, smooth | Unstructured | Implicit | Structured |
    | **Architecture constraints** | Minimal | Minimal | Minimal | Invertibility required |

    **VAEs** are your go-to when you need a structured latent space for downstream tasks (interpolation, attribute manipulation, representation learning) and can tolerate some blur.

    **GANs** shine when you need the sharpest possible outputs for a specific domain and have the engineering budget to handle training instability. Domain-specific applications (face synthesis, super-resolution) still sometimes use GANs.

    **Diffusion models** are the default choice for high-quality generation in 2024-2026. They combine the best properties of VAEs and GANs with none of the major drawbacks (except sampling speed, which is being actively mitigated).

    **Flows** occupy a niche: use them when exact likelihoods matter more than sample quality, or as components within larger systems (e.g., the prior in a VAE).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 8. Applications

    The applications of generative models have exploded:

    **Image generation and editing.** Text-to-image systems like DALL-E, Stable Diffusion, and Midjourney use diffusion models conditioned on text embeddings (from CLIP or T5). Inpainting, outpainting, style transfer, and image-to-image translation are all variants of conditional generation.

    **Text-to-image pipeline** (simplified): A text encoder (transformer) maps the prompt to an embedding. A diffusion model generates an image in a compressed latent space, conditioned on that embedding via cross-attention. A decoder (often a VAE decoder) upsamples the latent to pixel space. This is the architecture of Stable Diffusion.

    **Drug discovery and protein design.** Diffusion models over 3D molecular structures and protein conformations are enabling computational drug design. The idea is the same: learn the distribution of valid molecular structures, then sample novel molecules with desired properties.

    **Data augmentation.** When labeled data is scarce, generative models can synthesize additional training examples. This is especially valuable in medical imaging, where data collection is expensive and privacy-constrained.

    **Audio and music generation.** Diffusion models and autoregressive models generate high-fidelity speech (e.g., WaveNet descendants) and music. The same principles apply: learn the distribution over audio spectrograms or waveforms, then sample.

    **Video generation.** The frontier is extending diffusion models to the temporal dimension --- generating coherent video sequences. This requires modeling temporal consistency on top of spatial quality, and is an active area of research.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 9. The Bigger Picture

    Take a step back and notice the trajectory. Autoencoders gave us compression. VAEs gave us principled latent spaces and generation through variational inference. GANs gave us sharp samples through adversarial training. Diffusion models gave us everything --- quality, stability, mode coverage --- through the simple idea of learning to reverse noise.

    Each generation of models solved the previous generation's key weakness. And the math underlying each approach --- variational inference for VAEs, game theory for GANs, stochastic processes for diffusion --- draws from fundamentally different branches of mathematics. Generative modeling is a field where knowing more math unlocks more powerful architectures.

    What comes next? The current frontier includes consistency models (one-step diffusion), flow matching (a continuous-time generalization that subsumes both flows and diffusion), and scaling laws for generative models. The field is moving fast, but the foundations in this module will not change.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Practice

    ### Conceptual Questions

    1. **ELBO decomposition.** Write out the ELBO for a VAE. What happens if you set the KL weight to zero? What happens if you set it very high? Describe the behavior of the model in each extreme.

    2. **Mode collapse vs. mode covering.** Explain why GANs tend toward mode collapse while VAEs tend toward mode covering. Which property of the training objective causes each behavior? (Hint: think about the direction of the KL divergence.)

    3. **Reparameterization.** Why can we not simply backpropagate through $z \sim q_\phi(z|x)$? Write out the reparameterization trick and explain why it solves the problem.

    4. **Diffusion training.** A friend claims that diffusion models "just learn to predict noise, which is trivial." Explain why this is wrong --- what makes the noise prediction task hard and informative?

    5. **Comparison.** You need to build three systems: (a) an anomaly detector for manufacturing defects, (b) a photorealistic face generator, (c) a system that interpolates smoothly between two images. Which generative model family would you choose for each, and why?

    ### Mathematical Exercises

    6. **KL divergence in VAEs.** Derive the closed-form expression for $D_{\text{KL}}(\mathcal{N}(\mu, \sigma^2 I) \| \mathcal{N}(0, I))$. Show that it equals $\frac{1}{2} \sum_{j=1}^{d} (\mu_j^2 + \sigma_j^2 - \ln \sigma_j^2 - 1)$. Reference: [Bishop PRML Section 1.6.1](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) for KL fundamentals.

    7. **Optimal discriminator.** For a fixed generator $G$, show that the optimal discriminator is $D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_G(x)}$. Then substitute this back into the GAN objective and show that it reduces to $2 \, D_{\text{JS}}(p_{\text{data}} \| p_G) - 2\log 2$, where $D_{\text{JS}}$ is the Jensen-Shannon divergence.

    8. **Forward diffusion.** Starting from $q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} \, x_{t-1}, \beta_t I)$, derive the marginal $q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} \, x_0, (1-\bar{\alpha}_t)I)$ by induction. Verify that as $t \to \infty$, $x_t \to \mathcal{N}(0, I)$.

    9. **Change of variables.** For a normalizing flow with transformation $x = f(z)$, derive the change of variables formula $\log p(x) = \log p(f^{-1}(x)) + \log |\det J_{f^{-1}}(x)|$ from the multivariate change of variables theorem. Reference: [MML Section 6.7](file:///C:/Users/landa/ml-course/textbooks/MML.pdf).

    ### Implementation Challenges

    10. **VAE on MNIST.** Implement a VAE with a 2D latent space on MNIST. Train it and visualize: (a) the latent space colored by digit class, (b) a grid of samples from the latent space, (c) interpolations between two digits in latent space. Observe the smooth structure that the KL term creates.

    11. **Diffusion from scratch.** Implement a minimal DDPM on a 2D toy dataset (e.g., a mixture of Gaussians or a Swiss roll). Implement the forward noising process, the noise prediction network (a small MLP with timestep conditioning), and the sampling loop. Visualize the progressive denoising from noise to data.

    > **Next module:** 3C --- Reinforcement Learning Foundations --- where we move from learning distributions over data to learning policies that maximize reward over time. The shift from "what does the data look like?" to "what should I do?" introduces an entirely new set of challenges.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Code It: Implementation Exercises

    The exercises below ask you to implement core generative model components from scratch in numpy. Each exercise gives you a skeleton with `TODO` markers. Fill in the missing code.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 1: VAE Forward Pass

    Implement a single forward pass of a VAE on 1D data. You are given encoder and decoder weight matrices. Compute:
    1. The encoder output (mu and log_var)
    2. Reparameterized z samples
    3. The decoder reconstruction
    4. The full ELBO loss (reconstruction + KL)
    """)
    return


@app.cell
def _(np):
    def vae_forward(x_batch, W_enc_mu, W_enc_logvar, W_dec):
        """
        Single VAE forward pass on a batch of 1D data.

        Args:
            x_batch: (N, D) input data
            W_enc_mu: (D, L) encoder weights for mean
            W_enc_logvar: (D, L) encoder weights for log-variance
            W_dec: (L, D) decoder weights
        Returns:
            elbo: scalar, the ELBO (higher is better)
            x_recon: (N, D) reconstructed data
            z: (N, L) latent samples
        """
        # TODO: Encode -- compute mu and log_var
        mu = None       # shape (N, L)
        log_var = None  # shape (N, L)

        # TODO: Reparameterize -- sample z = mu + exp(0.5*log_var) * epsilon
        epsilon = np.random.randn(*mu.shape)
        z = None  # shape (N, L)

        # TODO: Decode -- reconstruct x
        x_recon = None  # shape (N, D)

        # TODO: Reconstruction loss (negative MSE, averaged over batch)
        recon_loss = None  # scalar

        # TODO: KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl_loss = None  # scalar

        # ELBO = reconstruction - KL
        elbo = recon_loss - kl_loss
        return elbo, x_recon, z

    # Test dimensions (uncomment after implementing):
    # N, D, L = 32, 10, 3
    # x_test = np.random.randn(N, D)
    # elbo, x_r, z_v = vae_forward(x_test, np.random.randn(D,L)*0.1,
    #                                np.random.randn(D,L)*0.1, np.random.randn(L,D)*0.1)
    # print(f"ELBO: {elbo:.4f}, recon shape: {x_r.shape}, z shape: {z_v.shape}")
    return (vae_forward,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 2: GAN Training Step

    Implement one training step for a GAN with simple linear generator and discriminator on 1D data. You need to compute:
    1. Discriminator loss on real and fake data
    2. Generator loss (non-saturating)
    3. Gradient updates for both (use finite differences for the gradient)
    """)
    return


@app.cell
def _(np):
    def gan_training_step(real_data, G_weights, D_weights, z_dim, lr=0.01):
        """
        One GAN training step with linear G and D.

        Args:
            real_data: (N, D) real samples
            G_weights: (z_dim, D) generator weight matrix
            D_weights: (D, 1) discriminator weight matrix
            z_dim: int, dimension of noise input
            lr: learning rate
        Returns:
            D_loss: scalar discriminator loss
            G_loss: scalar generator loss
            G_weights: updated generator weights
            D_weights: updated discriminator weights
        """
        N = real_data.shape[0]
        _sigmoid = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

        # TODO: Sample noise and generate fake data
        z = np.random.randn(N, z_dim)
        fake_data = None  # shape (N, D)

        # TODO: Discriminator scores (apply sigmoid to D_weights^T x)
        D_real = None  # shape (N, 1), probability real data is real
        D_fake = None  # shape (N, 1), probability fake data is real

        # TODO: Discriminator loss: -mean(log(D_real) + log(1-D_fake))
        D_loss = None

        # TODO: Generator loss (non-saturating): -mean(log(D_fake))
        G_loss = None

        # Gradient updates (provided -- just need losses above to be correct)
        # D gradient: push D_real toward 1, D_fake toward 0
        dD = -(real_data.T @ (1 - D_real) - fake_data.T @ D_fake) / N
        D_weights = D_weights - lr * dD

        # G gradient: push D_fake toward 1
        dG = -z.T @ ((1 - D_fake) * D_weights.T) / N
        G_weights = G_weights - lr * dG

        return D_loss, G_loss, G_weights, D_weights

    # Test (uncomment after implementing):
    # z_dim_test, D_dim = 2, 5
    # dl, gl, _, _ = gan_training_step(np.random.randn(64, D_dim),
    #     np.random.randn(z_dim_test, D_dim)*0.1, np.random.randn(D_dim, 1)*0.1, z_dim_test)
    # print(f"D_loss: {dl:.4f}, G_loss: {gl:.4f}")
    return (gan_training_step,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 3: Diffusion Forward Process and Loss

    Implement the full DDPM forward process and training loss:
    1. Build the noise schedule (linear beta schedule, compute alpha_bar)
    2. Given x_0, sample a random timestep, noise x_0 to get x_t
    3. Compute the training loss assuming a (placeholder) noise prediction
    """)
    return


@app.cell
def _(np):
    def diffusion_training_loss(x0_batch, T_steps=100, beta_min=1e-4, beta_max=0.02):
        """
        Compute DDPM training loss for a batch of data.

        Args:
            x0_batch: (N, D) clean data
            T_steps: number of diffusion timesteps
            beta_min, beta_max: noise schedule endpoints
        Returns:
            loss: scalar MSE between true noise and "predicted" noise
            alpha_bar_sched: (T,) cumulative product schedule
        """
        N, D = x0_batch.shape

        # TODO: Build linear beta schedule and compute alpha_bar
        betas_ex = None           # shape (T_steps,)
        alphas_ex = None          # shape (T_steps,)
        alpha_bar_sched = None    # shape (T_steps,) -- cumulative product

        # TODO: Sample random timesteps for each example in batch
        t_batch = None  # shape (N,), integers in [0, T_steps)

        # TODO: Sample noise epsilon ~ N(0, I)
        eps_ex = None  # shape (N, D)

        # TODO: Compute x_t = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * eps
        ab_t = alpha_bar_sched[t_batch]  # shape (N,)
        x_t = None  # shape (N, D)

        # Placeholder "network prediction" (in practice this is a neural net)
        eps_pred_ex = eps_ex + 0.05 * np.random.randn(N, D)

        # TODO: Loss = mean squared error between eps and eps_pred
        loss = None

        return loss, alpha_bar_sched

    # Test (uncomment after implementing):
    # loss_test, ab_test = diffusion_training_loss(np.random.randn(64, 8))
    # print(f"Diffusion loss: {loss_test:.6f}")
    # print(f"alpha_bar at t=0: {ab_test[0]:.4f}, at t=T: {ab_test[-1]:.4f}")
    return (diffusion_training_loss,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 4: Normalizing Flow (Affine Coupling Layer)

    Implement an affine coupling layer -- the building block of RealNVP flows. The idea:
    - Split input x into two halves: x1, x2
    - x1 passes through unchanged
    - x2 is transformed: y2 = x2 * exp(s(x1)) + t(x1), where s and t are learned functions
    - The Jacobian is triangular, so log|det J| = sum(s(x1))

    Implement both the forward pass (for density evaluation) and the inverse (for sampling).
    """)
    return


@app.cell
def _(np):
    def affine_coupling_forward(x, W_s, b_s, W_t, b_t):
        """
        Forward pass of affine coupling layer.

        Args:
            x: (N, D) input (D must be even)
            W_s, b_s: weights/bias for scale network (D//2, D//2) and (D//2,)
            W_t, b_t: weights/bias for translation network
        Returns:
            y: (N, D) transformed output
            log_det: (N,) log |det Jacobian| for each sample
        """
        D = x.shape[1]
        half = D // 2

        # Split
        x1, x2 = x[:, :half], x[:, half:]

        # TODO: Compute scale s and translation t from x1
        s = None  # shape (N, half) -- use tanh to keep scale bounded
        t = None  # shape (N, half)

        # TODO: Transform x2
        y2 = None  # shape (N, half)

        # TODO: Concatenate [x1, y2]
        y = None  # shape (N, D)

        # TODO: Log determinant = sum of s values per sample
        log_det = None  # shape (N,)

        return y, log_det

    def affine_coupling_inverse(y, W_s, b_s, W_t, b_t):
        """
        Inverse of affine coupling layer (for sampling).

        Args: same weight matrices, y is the transformed data
        Returns:
            x: (N, D) original data
        """
        D = y.shape[1]
        half = D // 2

        y1, y2 = y[:, :half], y[:, half:]
        x1 = y1  # x1 passes through unchanged

        # TODO: Compute s, t from x1 (same as forward)
        s = None
        t = None

        # TODO: Invert the transformation: x2 = (y2 - t) * exp(-s)
        x2 = None

        x = np.concatenate([x1, x2], axis=1)
        return x

    # Test (uncomment after implementing):
    # D_flow = 4
    # x_test_flow = np.random.randn(10, D_flow)
    # Ws, bs = np.random.randn(D_flow//2, D_flow//2)*0.1, np.zeros(D_flow//2)
    # Wt, bt = np.random.randn(D_flow//2, D_flow//2)*0.1, np.zeros(D_flow//2)
    # y_flow, ld = affine_coupling_forward(x_test_flow, Ws, bs, Wt, bt)
    # x_rec = affine_coupling_inverse(y_flow, Ws, bs, Wt, bt)
    # print(f"Reconstruction error: {np.max(np.abs(x_test_flow - x_rec)):.2e}")
    return (affine_coupling_forward, affine_coupling_inverse)


if __name__ == "__main__":
    app.run()
