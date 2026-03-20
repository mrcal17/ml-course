import marimo

app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Module 3B: Generative Models

    Everything we have built so far --- classifiers, regressors, even transformers --- has been *discriminative*. We learned boundaries, we learned mappings from inputs to outputs, we learned conditional distributions $p(y|x)$. That is powerful, but it sidesteps a deeper question: **what does the data itself look like?**

    Generative modeling is the study of learning $p(x)$ --- the full probability distribution over data. Not "is this a cat or a dog?" but "what does *any plausible image* look like?" This is a fundamentally harder problem. You are no longer drawing a decision boundary through data; you are trying to capture the entire structure of the data manifold. But the payoff is enormous: once you have $p(x)$, you can *sample* from it to create new data, *evaluate* how likely a given datapoint is, *interpolate* between examples, and *condition* generation on auxiliary information.

    This module covers the four major families of generative models --- autoencoders and VAEs, GANs, diffusion models, and normalizing flows --- along with the mathematical machinery that makes each one work. You already have the prerequisites: probability and MLE from Part 1, neural network architectures from Part 2, and the optimization intuitions from everything in between. Now we put it all together.

    > **Core references:** [Goodfellow et al. Ch 20: Deep Generative Models](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) for the foundational theory, and [Murphy PML2 Ch 20: Dimensionality Reduction](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML2.pdf) through [Ch 22: Generative Models Revisited](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML2.pdf) for the modern landscape.
    """)
    return


@app.cell
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


@app.cell
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

    ### 3.4 The Reparameterization Trick

    There is a practical problem. The ELBO involves an expectation over $q_\phi(z|x)$, which means we need to *sample* $z$ from the encoder. But sampling is not differentiable --- you cannot backpropagate through a random number generator.

    The reparameterization trick resolves this beautifully. Instead of sampling $z \sim \mathcal{N}(\mu, \sigma^2 I)$ directly, we write:

    $$z = \mu_\phi(x) + \sigma_\phi(x) \odot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$

    Now the randomness is in $\varepsilon$, which does not depend on the parameters. The dependence on $\phi$ flows through the deterministic operations $\mu_\phi$ and $\sigma_\phi$, and standard backpropagation works. This is a simple change of variables, but it was the key insight that made VAE training practical. See [Goodfellow et al. Section 20.9](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) and [Murphy PML2 Section 21.2.2](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML2.pdf).

    ### 3.5 Why VAEs Produce Blurry Outputs

    VAEs are known for generating blurry images, and the reason is fundamental, not incidental.

    First, the reconstruction term with Gaussian likelihood is equivalent to MSE, which penalizes pixel-level deviations equally. When there is uncertainty --- when multiple outputs are plausible for a given $z$ --- MSE produces the *mean* of those outputs, which is blurry.

    Second, the ELBO is a *lower bound* on the log-likelihood. We are optimizing a surrogate, not the actual objective. The gap between the ELBO and $\log p(x)$ equals $D_{\text{KL}}(q_\phi(z|x) \| p_\theta(z|x))$, which is always non-negative. This means VAEs exhibit **mode-covering** behavior: the approximate posterior $q$ tries to cover all the modes of the true posterior, spreading probability mass broadly rather than concentrating it at the sharpest peaks.
    """)
    return


@app.cell
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

    ### 4.2 Training Dynamics

    In practice, we alternate:
    1. **Train discriminator**: freeze $G$, take a batch of real data and a batch of generated data, update $D$ to better distinguish them
    2. **Train generator**: freeze $D$, generate a batch, update $G$ to better fool $D$

    This alternating optimization is where the trouble begins.

    ### 4.3 Why GANs Are Hard to Train

    **Mode collapse.** The generator discovers that a small number of outputs reliably fool the discriminator and stops exploring. Instead of learning the full data distribution, it produces a handful of "greatest hits" on repeat. The discriminator eventually catches on, but the generator just finds new modes to collapse onto, and the cycle continues.

    **Training instability.** The minimax game does not always converge. In theory, simultaneous gradient descent on a minimax objective can oscillate or diverge. You need careful hyperparameter tuning, architecture choices, and often heuristic tricks (label smoothing, spectral normalization, progressive growing) to get stable training.

    **Vanishing gradients for $G$.** When the discriminator becomes too good --- confidently outputting 0 for all generated samples --- the gradient signal to the generator vanishes. The generator is told "you're terrible" but gets no useful information about *how* to improve. This is why the original paper suggests optimizing $\max_G \mathbb{E}[\log D(G(z))]$ instead of $\min_G \mathbb{E}[\log(1 - D(G(z)))]$ in early training --- the gradient is stronger when the generator is bad.

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


@app.cell
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


@app.cell
def _(mo):
    mo.md(r"""
    ### 5.2 The Forward Process

    Given a data point $x_0$, the forward process produces a sequence $x_1, x_2, \dots, x_T$ by adding noise at each step:

    $$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} \, x_{t-1}, \, \beta_t I)$$

    Here $\beta_t \in (0, 1)$ is a noise schedule (increasing over time --- more noise at later steps). The signal is scaled down by $\sqrt{1-\beta_t}$ and noise of variance $\beta_t$ is added. After enough steps, $x_T \approx \mathcal{N}(0, I)$ regardless of $x_0$.

    A critical convenience: you can jump directly to any timestep without iterating. Define $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$. Then:

    $$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} \, x_0, \, (1 - \bar{\alpha}_t) I)$$

    This means: $x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \varepsilon$, where $\varepsilon \sim \mathcal{N}(0, I)$. We can generate any noisy version of $x_0$ in a single step. This is what makes training efficient.

    ### 5.3 The Reverse Process

    The reverse process learns to undo each corruption step:

    $$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \, \Sigma_\theta(x_t, t))$$

    A neural network (typically a U-Net with attention layers) takes the noisy input $x_t$ and the timestep $t$, and predicts the parameters of the denoising distribution. In the seminal DDPM paper (Ho et al., 2020), the variance is fixed and the network only predicts the mean.

    ### 5.4 Training: Noise Prediction

    Here is where the simplification happens. Instead of predicting the mean directly, the network predicts the *noise that was added*. The training objective becomes:

    $$\mathcal{L} = \mathbb{E}_{t, x_0, \varepsilon} \left[ \| \varepsilon - \varepsilon_\theta(x_t, t) \|^2 \right]$$

    where $x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \varepsilon$ and $t$ is sampled uniformly from $\{1, \dots, T\}$.

    This is just MSE between the true noise and the predicted noise. No adversarial training, no minimax games, no posterior approximation gaps. A single, stable, well-understood loss function. Train a denoiser, and generation falls out as a consequence.

    ### 5.5 Sampling

    To generate a new sample:
    1. Draw $x_T \sim \mathcal{N}(0, I)$
    2. For $t = T, T-1, \dots, 1$:
       - Predict the noise: $\hat{\varepsilon} = \varepsilon_\theta(x_t, t)$
       - Compute the denoised estimate and take one reverse step to get $x_{t-1}$
    3. Return $x_0$

    This iterative process is slow (typically $T = 1000$ steps), which is the main practical drawback of diffusion models. Considerable research has gone into reducing the number of steps required.

    ### 5.6 The Score Matching Perspective

    There is an equivalent and illuminating way to view diffusion models. The **score function** of a distribution is $\nabla_x \log p(x)$ --- the gradient of the log-density with respect to the data. It points in the direction of increasing probability density.

    If you know the score function, you can generate samples via **Langevin dynamics**: start from noise and iteratively follow the score, adding a small amount of noise at each step:

    $$x_{t+1} = x_t + \frac{\eta}{2} \nabla_x \log p(x_t) + \sqrt{\eta} \, \varepsilon_t$$

    The noise prediction network $\varepsilon_\theta$ is directly related to the score function. Specifically, for a noisy distribution at noise level $\sigma_t$:

    $$\nabla_{x_t} \log q(x_t) \approx -\frac{\varepsilon_\theta(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}}$$

    So training a noise predictor is equivalent to learning the score function at every noise level. This connection, formalized by Song and Ermon (2019), unifies denoising diffusion and score matching into a single framework.

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


@app.cell
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


@app.cell
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


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 9. The Bigger Picture

    Take a step back and notice the trajectory. Autoencoders gave us compression. VAEs gave us principled latent spaces and generation through variational inference. GANs gave us sharp samples through adversarial training. Diffusion models gave us everything --- quality, stability, mode coverage --- through the simple idea of learning to reverse noise.

    Each generation of models solved the previous generation's key weakness. And the math underlying each approach --- variational inference for VAEs, game theory for GANs, stochastic processes for diffusion --- draws from fundamentally different branches of mathematics. Generative modeling is a field where knowing more math unlocks more powerful architectures.

    What comes next? The current frontier includes consistency models (one-step diffusion), flow matching (a continuous-time generalization that subsumes both flows and diffusion), and scaling laws for generative models. The field is moving fast, but the foundations in this module will not change.
    """)
    return


@app.cell
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


if __name__ == "__main__":
    app.run()
