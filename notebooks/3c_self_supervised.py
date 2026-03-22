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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Self-Supervised Learning

    You know how to train a transformer. You know how attention works, how gradients flow through deep networks, and how generative models learn distributions over data. Now we confront the question that has reshaped the entire field over the last five years: *where does the training signal come from?*

    Supervised learning requires labels. Labels require humans. Humans are slow and expensive. Self-supervised learning sidesteps this bottleneck entirely --- it manufactures supervision from the structure of the data itself. This single idea is the foundation of every large language model, every modern vision backbone, and every multimodal system you interact with today.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 1. The Labeled Data Bottleneck

    The fundamental tension in machine learning has always been this: models are hungry for data, but *labeled* data is scarce.

    Consider ImageNet --- the dataset that launched the deep learning revolution. It contains about 14 million images with human-assigned labels. Building it required years of effort and tens of thousands of annotators through Amazon Mechanical Turk. And 14 million is *tiny* compared to the billions of images on the internet. Medical imaging is worse: you need board-certified radiologists to label each scan. Legal document classification needs lawyers. Protein structure annotation needs biochemists.

    Meanwhile, the internet generates roughly 500 million tweets per day, 720,000 hours of YouTube video, and billions of web pages --- all unlabeled, all free. The question becomes: can we learn from this ocean of unlabeled data?

    Self-supervised learning says yes. The trick is to create a *pretext task* --- a supervised learning objective derived entirely from the data's own structure. You don't need external labels because the data labels itself.

    This isn't a hack or a workaround. It turns out that models trained this way learn *better* representations than models trained on curated labeled datasets. The reason is simple: self-supervised objectives force the model to understand the deep structure of data --- grammar, semantics, spatial relationships, physical intuitions --- rather than memorizing surface-level label correlations ([Murphy PML2, Ch. 34](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML2.pdf)).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2. Pretext Tasks --- The General Idea

    A pretext task is any task you can construct from unlabeled data alone by hiding part of the input and asking the model to predict it. The critical insight: **the task itself doesn't matter**. What matters is that solving the task forces the model to learn useful internal representations.

    Here are classic examples from computer vision:

    - **Predicting rotations.** Rotate an image by 0, 90, 180, or 270 degrees. Train a network to predict which rotation was applied. To solve this, the model must understand object orientation, gravity, and scene layout.
    - **Jigsaw puzzles.** Divide an image into a 3x3 grid, shuffle the patches, and train a model to predict the correct arrangement. This requires understanding spatial relationships between object parts.
    - **Colorization.** Convert a color image to grayscale and train a model to predict the original colors. The model must learn that grass is green, sky is blue, and skin has certain tones --- which requires semantic understanding.

    None of these tasks are intrinsically useful. Nobody needs a rotation-prediction model in production. But the representations learned in the process --- the features in the intermediate layers --- transfer powerfully to downstream tasks like object detection, segmentation, and classification.

    This is the self-supervised recipe: (1) design a pretext task from unlabeled data, (2) train a large model on it, (3) transfer the learned representations to your actual task of interest.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's see a pretext task in action --- the rotation prediction task. We simulate a tiny "image" (a 4x4 patch) and apply rotations. The labels come entirely from which rotation we applied.
    """)
    return


@app.cell
def _(np):
    # Pretext task demo: rotation prediction
    # The image "labels itself" — no human annotation needed
    image = np.array([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]])  # top-heavy pattern

    rotations = {0: image,
                 90: np.rot90(image, k=1),
                 180: np.rot90(image, k=2),
                 270: np.rot90(image, k=3)}

    # Each rotation becomes a training pair: (rotated_image, rotation_label)
    for angle, rotated in rotations.items():
        print(f"Rotation {angle:>3}°  →  label={angle // 90}")
        print(rotated, "\n")
    return (rotations,)


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 3. Language Self-Supervision

    Language turned out to be the perfect domain for self-supervised learning. Text has rich sequential structure, and predicting the next word (or a missing word) requires deep understanding of syntax, semantics, world knowledge, and reasoning. Let's walk through the three major paradigms.

    ### 3.1 Autoregressive Language Modeling (GPT-style)

    The simplest and most powerful idea: given a sequence of tokens, predict the next one.

    $$P(x_t \mid x_1, x_2, \ldots, x_{t-1})$$

    The model reads tokens left-to-right and at each position outputs a probability distribution over the vocabulary for the next token. Training maximizes the log-likelihood of the actual next token across the entire corpus. This is just maximum likelihood estimation --- nothing exotic.

    The transformer processes this with **causal (masked) self-attention**: position $t$ can attend to positions $1, \ldots, t-1$ but not to positions $t+1, \ldots, T$. This ensures the model can only use left context, which is exactly what you need for generation --- you can't peek at the future when generating text sequentially.

    This is the training objective behind GPT, GPT-2, GPT-3, GPT-4, Claude, LLaMA, and essentially every modern large language model. Its power comes from two properties:

    1. **Simplicity.** Next-token prediction is trivially parallelizable across all positions in a sequence during training (using the causal mask). No complex data augmentation or negative sampling needed.
    2. **Scalability.** Every sentence in every book, article, and webpage on the internet provides training signal. There is effectively unlimited data.

    The limitation is that the representation at position $t$ only encodes left context. This is **unidirectional** --- fine for generation, but potentially suboptimal for tasks where you want to understand a full sentence (e.g., classifying sentiment requires reading the whole thing) ([Murphy PML2, Ch. 34.2](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML2.pdf)).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here is autoregressive next-token prediction in numpy. We build the causal mask and compute the loss at each position.
    """)
    return


@app.cell
def _(np):
    # Autoregressive language modeling: causal mask + next-token prediction
    T = 5  # sequence length
    vocab_size = 8
    tokens = np.array([2, 5, 1, 7, 3])  # a toy sequence

    # Causal mask: position t can only attend to positions <= t
    causal_mask = np.tril(np.ones((T, T)))  # lower-triangular
    print("Causal mask (1 = can attend):\n", causal_mask)

    # Simulate model logits at each position, compute cross-entropy loss
    rng = np.random.default_rng(42)
    logits = rng.standard_normal((T, vocab_size))  # (T, vocab_size)
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)  # softmax

    # Loss: -log P(next token | previous tokens), for positions 1..T-1
    targets = tokens[1:]  # next tokens to predict
    loss_per_pos = -np.log(probs[:-1, targets][np.diag_indices(T - 1)])
    print(f"\nPer-position loss: {loss_per_pos.round(3)}")
    print(f"Average autoregressive loss: {loss_per_pos.mean():.3f}")
    return (causal_mask,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.2 Masked Language Modeling (BERT-style)

    BERT flipped the script: instead of predicting the next token, mask out random tokens and predict them from *all* surrounding context.

    The procedure: take a sentence, randomly select 15% of the tokens, and replace them. Of those selected tokens, 80% become `[MASK]`, 10% become a random token, and 10% stay unchanged. The model sees the corrupted input and must predict the original tokens at the masked positions.

    This is **bidirectional** --- every position can attend to every other position (no causal mask). The model sees both left and right context when predicting a masked token, which gives it a richer understanding of the full input.

    The 80/10/10 replacement strategy addresses a subtle problem: the `[MASK]` token never appears during fine-tuning or inference. If the model only ever saw `[MASK]` during pretraining, there would be a **distribution mismatch**. By sometimes showing the original token or a random token, the model learns to be robust and not depend on the presence of `[MASK]`.

    BERT-style models excel at *understanding* tasks: classification, named entity recognition, question answering, semantic similarity. But they can't generate text naturally because they weren't trained to produce sequences autoregressively ([Murphy PML2, Ch. 34.3](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML2.pdf)).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's implement BERT's 80/10/10 masking strategy and see which tokens get corrupted.
    """)
    return


@app.cell
def _(np):
    # BERT-style masking: 15% of tokens selected, then 80/10/10 split
    rng = np.random.default_rng(7)
    sentence = ["The", "cat", "sat", "on", "the", "mat", "and", "purred", "loudly", "."]
    MASK_TOKEN, vocab_sz = "[MASK]", 1000

    mask_prob = 0.15
    selected = rng.random((len(sentence))) < mask_prob  # which tokens to corrupt

    corrupted = list(sentence)
    for i in np.where(selected)[0]:
        r = rng.random()
        if r < 0.8:      # 80% → replace with [MASK]
            corrupted[i] = MASK_TOKEN
        elif r < 0.9:     # 10% → replace with random token
            corrupted[i] = f"<rand_{rng.integers(vocab_sz)}>"
        # else: 10% → keep original (do nothing)

    print(f"Original:  {' '.join(sentence)}")
    print(f"Corrupted: {' '.join(corrupted)}")
    print(f"Targets:   predict original tokens at positions {list(np.where(selected)[0])}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 3.3 Span Corruption (T5-style)

    T5 unifies everything into a text-to-text framework. Instead of masking individual tokens, it masks *contiguous spans* and replaces each span with a single sentinel token (like `<extra_id_0>`). The model then generates the missing spans in sequence.

    For example:
    - **Input:** "The `<extra_id_0>` walks across the `<extra_id_1>` in the morning."
    - **Target:** "`<extra_id_0>` cat `<extra_id_1>` street"

    This is more efficient than BERT's token-level masking because multiple tokens are compressed into a single sentinel, reducing the target sequence length. It also naturally frames the problem as sequence-to-sequence, making T5 flexible for both understanding and generation tasks.

    ### 3.4 GPT vs. BERT --- The Resolution

    For years there was a genuine debate: autoregressive (GPT) or masked (BERT) --- which is better? The answer turned out to depend on scale.

    At moderate scale (~300M parameters), BERT-style models outperform GPT-style models on understanding benchmarks. The bidirectional context genuinely helps.

    At very large scale (tens of billions of parameters and beyond), autoregressive models converge to strong performance on *everything* --- understanding included. GPT-4, Claude, and similar models handle classification, reasoning, and analysis tasks that BERT was designed for, despite being trained with a unidirectional objective. The sheer volume of training data and model capacity compensates for the architectural constraint.

    The practical consequence: most frontier models today are autoregressive. The simplicity and scalability of next-token prediction won.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 4. Vision Self-Supervision

    Language self-supervision succeeded because text has discrete tokens and clear sequential structure. Images are continuous, high-dimensional, and spatially structured --- which required different approaches.

    ### 4.1 Contrastive Learning (SimCLR, MoCo)

    Contrastive learning is built on a beautifully simple principle: **similar things should be close in embedding space, dissimilar things should be far apart.**

    **SimCLR** (Simple Contrastive Learning of Representations) works as follows:

    1. Take an image $x$. Apply two different random augmentations (crop, flip, color jitter, blur) to produce two views $\tilde{x}_i$ and $\tilde{x}_j$. These are a **positive pair** --- two views of the same underlying image.
    2. Pass both through an encoder $f(\cdot)$ (a ResNet, typically) followed by a small projection head $g(\cdot)$ to get embeddings $z_i$ and $z_j$.
    3. In a minibatch of $N$ images, you have $2N$ augmented views. For each positive pair $(z_i, z_j)$, the other $2(N-1)$ views serve as **negatives**.
    4. Minimize the **NT-Xent loss** (Normalized Temperature-scaled Cross-Entropy):

    $$\mathcal{L}_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k) / \tau)}$$

    where $\text{sim}(u, v) = u^\top v / (\|u\| \|v\|)$ is cosine similarity and $\tau$ is a temperature parameter.

    This is essentially a $2N$-way classification problem: given anchor $z_i$, identify which of the $2N - 1$ other embeddings is its positive pair. The temperature $\tau$ controls the sharpness of this distribution --- lower temperature makes the model more discriminative but training less stable.

    The key design decisions that make SimCLR work:

    - **Strong augmentations.** Weak augmentations create a trivially easy task. The model needs to be invariant to cropping, color distortion, and blurring --- which forces it to learn semantic features rather than low-level shortcuts.
    - **Large batch sizes.** More negatives means a harder discrimination task. SimCLR used batches of 4096--8192. This requires significant compute.
    - **The projection head matters.** Representations from before the projection head ($f(x)$) transfer better than those after ($g(f(x))$). The projection head discards information useful for the pretext task but not for downstream tasks.

    **MoCo** (Momentum Contrast) solves the large-batch problem differently. Instead of requiring all negatives in a single batch, it maintains a **queue** of negative embeddings from recent batches. A **momentum-updated encoder** --- a slowly-moving copy of the main encoder --- produces the queue's embeddings, ensuring they remain consistent even though the main encoder updates rapidly.

    The momentum encoder's parameters $\theta_k$ are updated as an exponential moving average of the main encoder's parameters $\theta_q$:

    $$\theta_k \leftarrow m \cdot \theta_k + (1 - m) \cdot \theta_q$$

    with $m = 0.999$ typically. This lets MoCo maintain a large, consistent set of negatives (65,536 in the original paper) without enormous batch sizes ([Murphy PML2, Ch. 34.5](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML2.pdf)).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Below we compute the NT-Xent loss from SimCLR on a small batch. Watch how cosine similarity and temperature interact.
    """)
    return


@app.cell
def _(np):
    # SimCLR NT-Xent loss: sim(u,v) = u^T v / (||u|| ||v||)
    rng = np.random.default_rng(0)
    N_batch = 4  # 4 images → 8 augmented views (2 per image)
    dim = 16

    # Simulate 2N embeddings from the projection head
    z = rng.standard_normal((2 * N_batch, dim))
    z = z / np.linalg.norm(z, axis=1, keepdims=True)  # L2 normalize

    # Cosine similarity matrix: sim(z_i, z_k) for all pairs
    sim_matrix = z @ z.T  # (2N, 2N), already normalized so dot = cosine

    # NT-Xent loss for anchor i=0, positive j=1 (two views of image 0)
    tau = 0.5  # temperature
    i, j = 0, 1
    numerator = np.exp(sim_matrix[i, j] / tau)
    # Denominator: sum over all k != i
    mask_self = np.ones(2 * N_batch, dtype=bool)
    mask_self[i] = False
    denominator = np.exp(sim_matrix[i, mask_self] / tau).sum()

    loss_ij = -np.log(numerator / denominator)
    print(f"sim(z_0, z_1) = {sim_matrix[i, j]:.3f}  (positive pair)")
    print(f"NT-Xent loss(i=0, j=1, τ={tau}) = {loss_ij:.3f}")
    return (sim_matrix, tau)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now let's see the effect of temperature $\tau$ on the loss landscape.
    """)
    return


@app.cell
def _(np, sim_matrix):
    # Temperature effect: lower τ → sharper distribution, higher τ → flatter
    taus = [0.1, 0.5, 1.0, 5.0]
    i_anchor, j_pos = 0, 1
    mask_k = np.ones(sim_matrix.shape[0], dtype=bool)
    mask_k[i_anchor] = False

    for t in taus:
        logits = sim_matrix[i_anchor, mask_k] / t
        probs = np.exp(logits) / np.exp(logits).sum()  # softmax over negatives+positive
        # j_pos is at index 0 in the masked array (since i=0 was removed)
        prob_positive = probs[0]
        print(f"τ={t:<4}  P(positive) = {prob_positive:.4f}  loss = {-np.log(prob_positive):.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's also implement MoCo's momentum update rule.
    """)
    return


@app.cell
def _(np):
    # MoCo momentum update: θ_k ← m * θ_k + (1 - m) * θ_q
    rng = np.random.default_rng(1)
    d_param = 8
    theta_q = rng.standard_normal(d_param)   # query encoder (updated by gradient)
    theta_k = rng.standard_normal(d_param)   # key encoder (updated by momentum)
    m = 0.999  # momentum coefficient

    print(f"Before: ||θ_q - θ_k|| = {np.linalg.norm(theta_q - theta_k):.4f}")
    for step in range(200):
        theta_q += rng.standard_normal(d_param) * 0.01  # simulate gradient step
        theta_k = m * theta_k + (1 - m) * theta_q    # momentum update
    print(f"After 200 steps: ||θ_q - θ_k|| = {np.linalg.norm(theta_q - theta_k):.4f}")
    print("Key encoder tracks query encoder slowly — keeps negatives consistent")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4.2 Non-Contrastive Methods (BYOL, SimSiam)

    Here's a puzzle. Contrastive methods need negative pairs to prevent **representation collapse** --- the degenerate solution where the encoder maps everything to the same point, achieving zero contrastive loss trivially. What if we could learn without negatives at all?

    **BYOL** (Bootstrap Your Own Latent) does exactly this, and when it was published, nobody fully understood why it worked.

    BYOL has two networks:
    - An **online network** (encoder + projector + predictor) that processes one augmented view
    - A **target network** (encoder + projector, no predictor) that processes the other view

    The online network tries to predict the target network's output. The target network is updated via momentum (like MoCo's momentum encoder), *not* via gradient descent. Critically, gradients do **not** flow through the target network --- there's a **stop-gradient** operation.

    **SimSiam** simplified this further by removing momentum entirely. Both branches use the same encoder, but one branch has an extra predictor MLP and the other has a stop-gradient. The loss is just negative cosine similarity between the prediction and the (detached) target.

    Why doesn't this collapse? The full theoretical explanation is still debated, but the practical answer involves two mechanisms:
    1. The **asymmetric architecture** (predictor on one side, not the other) breaks the symmetry that enables collapse.
    2. The **stop-gradient** prevents the trivial solution where both branches jointly move toward a constant.

    Empirically, these methods match or exceed contrastive approaches while being simpler to implement and tune.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here is a minimal SimSiam-style forward pass with the stop-gradient and predictor asymmetry.
    """)
    return


@app.cell
def _(np):
    # SimSiam: asymmetric architecture with stop-gradient
    rng = np.random.default_rng(2)
    d_emb = 8

    # Simulate two augmented views through encoder
    z1 = rng.standard_normal(d_emb)
    z2 = rng.standard_normal(d_emb)

    # Predictor MLP (simplified as linear): only on one branch
    W_pred = rng.standard_normal((d_emb, d_emb)) * 0.1
    p1 = W_pred @ z1  # online branch: encoder + predictor

    # Loss: negative cosine similarity with stop-gradient on z2
    cos_sim = lambda a, b: a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

    loss_simsiam = -cos_sim(p1, z2)  # z2 is DETACHED (no grad flows back)
    print(f"SimSiam loss = {loss_simsiam:.4f}")
    print(f"Predictor output norm: {np.linalg.norm(p1):.3f}")
    print(f"Target norm: {np.linalg.norm(z2):.3f}")
    print("Asymmetry (predictor) + stop-grad prevent collapse")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's see what collapse looks like and why it fails with contrastive loss.
    """)
    return


@app.cell
def _(np):
    # Representation collapse: all embeddings → same point
    # NT-Xent loss becomes log(2N-1) when all similarities are equal
    N_collapse = 4
    z_collapsed = np.ones((2 * N_collapse, 8))  # everything identical
    z_collapsed = z_collapsed / np.linalg.norm(z_collapsed, axis=1, keepdims=True)

    sim_collapsed = z_collapsed @ z_collapsed.T  # all 1s
    tau_c = 0.5
    # Every entry in denominator has same exp(1/τ), so P(positive) = 1/(2N-1)
    loss_collapsed = -np.log(1.0 / (2 * N_collapse - 1))
    print(f"Collapsed loss = log(2N-1) = log({2*N_collapse-1}) = {loss_collapsed:.3f}")
    print("This is a local optimum the model can get stuck in — contrastive loss")
    print("has no gradient to escape because all similarities are identical.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4.3 Masked Image Modeling (MAE)

    **MAE** (Masked Autoencoder) is BERT for images. Take an image, divide it into patches (just like ViT), randomly mask 75% of the patches, and train the model to reconstruct the masked patches from the visible ones.

    The architecture has an asymmetric encoder-decoder:
    - The **encoder** (a ViT) processes *only* the visible patches --- no mask tokens, no wasted computation on missing positions.
    - A lightweight **decoder** takes the encoder's output plus mask tokens and reconstructs the original pixel values.

    Three things are striking about MAE:

    **Why 75% masking?** Images are far more redundant than text. With natural images, neighboring patches are highly correlated. If you only mask 15% (like BERT), a model can interpolate from neighbors without learning any high-level semantics. Masking 75% forces the model to understand objects, scenes, and context at a semantic level because local interpolation is no longer sufficient.

    **Why not for text?** Text is information-dense. Each token carries significant meaning, and masking 75% would leave too little context for meaningful prediction. The 15% masking rate in BERT is already challenging because language has less spatial redundancy.

    **Efficiency.** Because the encoder only processes 25% of patches, MAE pretraining is 3-4x faster than alternatives that process all patches. This makes it highly practical.

    MAE representations transfer well to downstream tasks and complement the contrastive/non-contrastive approaches nicely ([Murphy PML2, Ch. 34.6](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML2.pdf)).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's simulate MAE's masking on a small image grid and measure the reconstruction loss.
    """)
    return


@app.cell
def _(np):
    # MAE: mask 75% of patches, reconstruct from visible 25%
    rng = np.random.default_rng(3)
    n_patches = 16  # e.g., 4x4 grid of patches
    patch_dim = 4   # pixels per patch (flattened)
    mask_ratio = 0.75

    # Simulate an "image" as n_patches vectors
    patches = rng.standard_normal((n_patches, patch_dim))

    # Random masking
    n_mask = int(n_patches * mask_ratio)
    perm = rng.permutation(n_patches)
    masked_idx = perm[:n_mask]
    visible_idx = perm[n_mask:]

    print(f"Total patches: {n_patches}")
    print(f"Visible: {len(visible_idx)} ({100*(1-mask_ratio):.0f}%)  →  encoder processes only these")
    print(f"Masked:  {len(masked_idx)} ({100*mask_ratio:.0f}%)  →  decoder reconstructs these")

    # Reconstruction loss (MSE) on masked patches only
    fake_reconstruction = patches[masked_idx] + rng.standard_normal((n_mask, patch_dim)) * 0.5
    mse = np.mean((patches[masked_idx] - fake_reconstruction) ** 2)
    print(f"\nReconstruction MSE on masked patches: {mse:.3f}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 5. Multimodal Self-Supervision

    The most exciting development in self-supervised learning extends beyond a single modality. Different modalities --- text, images, audio --- describe the same underlying world, and *their alignment provides supervision*.

    ### 5.1 CLIP: Connecting Text and Images

    **CLIP** (Contrastive Language-Image Pretraining) is deceptively simple. Take 400 million image-text pairs scraped from the internet. Train two encoders --- one for images, one for text --- so that matching pairs have high cosine similarity and non-matching pairs have low similarity.

    In a batch of $N$ image-text pairs, there are $N$ positive pairs (each image with its caption) and $N^2 - N$ negative pairs (each image with every other caption, and vice versa). The loss is symmetric contrastive --- the same NT-Xent structure from SimCLR, but across modalities.

    What makes CLIP revolutionary is what happens at inference. To classify an image into one of $K$ classes:

    1. Create $K$ text prompts: "a photo of a {class name}" for each class.
    2. Encode all prompts and the image.
    3. The predicted class is whichever text embedding has the highest cosine similarity to the image embedding.

    This is **zero-shot classification** --- no labeled training examples for the target task whatsoever. CLIP matched a supervised ResNet-50 on ImageNet zero-shot. On many distribution-shifted benchmarks, it *exceeded* supervised models because its representations were far more robust.

    CLIP changed computer vision because it decoupled visual representations from fixed label sets. A CLIP model can classify images into *any* set of categories described in natural language, including categories it never saw during training. This is a qualitative shift from supervised learning, where the label set is fixed at training time ([Murphy PML2, Ch. 34.7](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML2.pdf)).

    ### 5.2 Beyond CLIP

    The CLIP paradigm has been extended in multiple directions:

    - **ALIGN** (Google) scaled to over 1 billion noisy image-text pairs with minimal data curation, demonstrating that scale compensates for noise.
    - **Florence** (Microsoft) added object detection and dense prediction capabilities to the CLIP framework.
    - **ImageBind** (Meta) extended contrastive alignment to six modalities: images, text, audio, depth, thermal, and IMU data. The insight is that you don't need paired data for all modality combinations --- if images are paired with text *and* images are paired with audio, all three modalities become aligned transitively.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here is CLIP's contrastive loss on a batch of image-text pairs, including the zero-shot classification procedure.
    """)
    return


@app.cell
def _(np):
    # CLIP: contrastive loss across image and text modalities
    rng = np.random.default_rng(4)
    N_clip = 4  # batch of 4 image-text pairs
    d_clip = 16

    # Simulate image and text encoders producing L2-normalized embeddings
    img_emb = rng.standard_normal((N_clip, d_clip))
    txt_emb = rng.standard_normal((N_clip, d_clip))
    img_emb = img_emb / np.linalg.norm(img_emb, axis=1, keepdims=True)
    txt_emb = txt_emb / np.linalg.norm(txt_emb, axis=1, keepdims=True)

    # Similarity matrix: (N, N) — diagonal = matching pairs
    logits_clip = img_emb @ txt_emb.T / 0.07  # temperature = 0.07 (CLIP default)
    labels_clip = np.arange(N_clip)  # diagonal entries are positives

    # Symmetric cross-entropy loss
    def cross_entropy_rows(logits, labels):
        log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        return -log_probs[np.arange(len(labels)), labels].mean()

    loss_i2t = cross_entropy_rows(logits_clip, labels_clip)       # image→text
    loss_t2i = cross_entropy_rows(logits_clip.T, labels_clip)     # text→image
    clip_loss = (loss_i2t + loss_t2i) / 2
    print(f"CLIP loss: {clip_loss:.3f}  (image→text: {loss_i2t:.3f}, text→image: {loss_t2i:.3f})")
    return (img_emb, txt_emb)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And zero-shot classification: match an image embedding against text prompt embeddings.
    """)
    return


@app.cell
def _(np, img_emb, txt_emb):
    # Zero-shot classification: no training on target labels!
    classes = ["dog", "cat", "car", "tree"]
    # txt_emb[k] = encoding of "a photo of a {class}" for each class
    # Classify the first image
    query_img = img_emb[0]
    scores = query_img @ txt_emb.T  # cosine similarity (already normalized)
    predicted = classes[np.argmax(scores)]
    print("Zero-shot classification scores:")
    for cls, s in zip(classes, scores):
        print(f"  '{cls}': {s:.3f}")
    print(f"Prediction: '{predicted}'  (no labeled training data needed)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 6. The Foundation Model Paradigm

    Everything above converges on a paradigm that has restructured the entire field:

    > **Pretrain a large model on massive unlabeled data using self-supervised learning. Then adapt it to specific tasks with minimal labeled data.**

    These pretrained models are called **foundation models** --- they serve as the foundation on which task-specific models are built. GPT-4, Claude, LLaMA, ViT-large, and CLIP are all foundation models.

    ### Why Pretraining Works

    Self-supervised pretraining learns **general-purpose representations** --- features that capture the statistical structure of the data domain. Language models learn syntax, semantics, factual knowledge, and reasoning patterns. Vision models learn edges, textures, object parts, spatial relationships, and scene structure.

    These representations are useful across a vast range of downstream tasks. Fine-tuning merely steers the already-capable representations toward a specific objective. This is why you can fine-tune a language model on 1,000 medical documents and get a competent medical assistant --- the model already knows language, reasoning, and substantial world knowledge. The fine-tuning just activates and directs it ([Murphy PML1, Ch. 19](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf)).

    ### Emergent Abilities at Scale

    Large foundation models exhibit **emergent abilities** --- capabilities that appear suddenly as the model scales up, rather than improving gradually. Models below a certain size fail completely at tasks like multi-step arithmetic, chain-of-thought reasoning, or translating between rare languages. Above a threshold, they succeed reliably. This is not well-understood theoretically, but it's consistently observed empirically.

    ### Few-Shot and Zero-Shot Learning

    Foundation models enable new adaptation paradigms:

    - **Zero-shot:** The model performs a task it was never explicitly trained for, guided only by a natural language instruction. "Translate the following English text to French: ..."
    - **Few-shot:** Provide a handful of input-output examples in the prompt. The model generalizes from these examples to new inputs.
    - **In-context learning:** The model "learns" from examples provided in the prompt without any gradient updates. This is qualitatively different from traditional learning --- the model's parameters don't change, yet its behavior adapts to the demonstrated pattern.

    In-context learning remains somewhat mysterious. The model was never trained to do few-shot learning explicitly --- it emerged as a byproduct of next-token prediction at scale. One hypothesis: during pretraining, the model encounters many implicit few-shot patterns in text (e.g., lists, question-answer pairs, translation examples), and it learns to extrapolate from such patterns ([Murphy PML2, Ch. 35](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML2.pdf)).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 7. Fine-Tuning Strategies

    Given a pretrained foundation model and a downstream task, how do you adapt it? The answer depends on your dataset size, compute budget, and how far your task is from the pretraining domain.

    ### 7.1 Full Fine-Tuning

    Update all model parameters on your labeled dataset. This gives the model maximum flexibility to adapt but carries risks:

    - **Catastrophic forgetting:** The model may lose useful pretrained knowledge.
    - **Overfitting:** If your dataset is small relative to the model size, you'll overfit quickly.
    - **Compute cost:** For models with billions of parameters, full fine-tuning is expensive.

    Full fine-tuning works best when you have a large labeled dataset (tens of thousands of examples or more) and significant compute.

    ### 7.2 Linear Probing

    Freeze the entire pretrained model. Add a single linear layer on top. Train only this linear layer.

    This is the most restrictive form of adaptation --- you're testing whether the pretrained representations are *already* linearly separable for your task. It's fast, cheap, and serves as an excellent diagnostic: if linear probing works well, the pretrained representations are high quality.

    The downside is that some tasks require nonlinear combinations of features, and linear probing will underperform. But it's always worth trying first as a baseline.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's compare linear probing vs. full fine-tuning on frozen representations.
    """)
    return


@app.cell
def _(np):
    # Linear probing: train only a linear head on frozen features
    rng = np.random.default_rng(5)
    n_samples, d_feat, n_classes = 200, 32, 3

    # Simulate frozen pretrained features + labels
    X_feat = rng.standard_normal((n_samples, d_feat))
    y_true = rng.integers(0, n_classes, n_samples)

    # Linear probe: W @ x + b, trained with simple gradient descent
    W_probe = rng.standard_normal((n_classes, d_feat)) * 0.01
    b_probe = np.zeros(n_classes)
    lr = 0.01

    for epoch in range(50):
        logits_lp = X_feat @ W_probe.T + b_probe  # (n, n_classes)
        probs_lp = np.exp(logits_lp) / np.exp(logits_lp).sum(axis=1, keepdims=True)
        # Gradient of cross-entropy w.r.t. logits
        grad = probs_lp.copy()
        grad[np.arange(n_samples), y_true] -= 1  # (n, n_classes)
        W_probe -= lr * (grad.T @ X_feat) / n_samples
        b_probe -= lr * grad.mean(axis=0)

    preds = np.argmax(X_feat @ W_probe.T + b_probe, axis=1)
    acc = (preds == y_true).mean()
    print(f"Linear probe accuracy: {acc:.1%}")
    print(f"Trainable params: {W_probe.size + b_probe.size} (vs {d_feat * 100}+ for full model)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 7.3 LoRA: Low-Rank Adaptation

    **LoRA** (Low-Rank Adaptation) is the most important parameter-efficient fine-tuning method. The idea: instead of updating the full weight matrix $W \in \mathbb{R}^{d \times d}$, learn a low-rank update $\Delta W = BA$ where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times d}$ with $r \ll d$.

    During fine-tuning, the forward pass computes:

    $$h = (W + BA)x = Wx + BAx$$

    $W$ is frozen. Only $A$ and $B$ are trained. With $r = 8$ and $d = 4096$, you're training $2 \times 8 \times 4096 = 65{,}536$ parameters per layer instead of $4096^2 = 16{,}777{,}216$. That's a **256x reduction** in trainable parameters per layer.

    LoRA works because fine-tuning updates tend to be low-rank in practice --- the task-specific adjustment lives in a low-dimensional subspace of the full parameter space. At inference time, you can merge $BA$ into $W$, so there's **no additional latency** ([Murphy PML1, Ch. 19.3](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf)).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    LoRA in numpy: the forward pass with a low-rank update, and parameter count comparison.
    """)
    return


@app.cell
def _(np):
    # LoRA: h = Wx + BAx, where W is frozen, only A and B are trained
    rng = np.random.default_rng(6)
    d_model = 256
    r_lora = 8  # low rank

    W_frozen = rng.standard_normal((d_model, d_model)) * 0.01  # pretrained weight (FROZEN)
    A_lora = rng.standard_normal((r_lora, d_model)) * 0.01      # trainable
    B_lora = np.zeros((d_model, r_lora))                   # init B to zero → ΔW starts at 0

    x_input = rng.standard_normal(d_model)

    # Forward pass: h = Wx + BAx
    h = W_frozen @ x_input + B_lora @ (A_lora @ x_input)

    full_params = d_model * d_model
    lora_params = (d_model * r_lora) * 2  # A + B
    print(f"Full fine-tuning params: {full_params:,}")
    print(f"LoRA params (rank={r_lora}):    {lora_params:,}")
    print(f"Reduction: {full_params / lora_params:.0f}x")

    # At inference: merge W_merged = W + BA (no extra cost)
    W_merged = W_frozen + B_lora @ A_lora
    h_merged = W_merged @ x_input
    print(f"\nMerged output matches: {np.allclose(h, h_merged)}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 7.4 Prompt Tuning and Prefix Tuning

    Instead of modifying model weights at all, learn a set of continuous "virtual tokens" that are prepended to the input:

    - **Prompt tuning:** Learn a sequence of embeddings prepended to the input embedding sequence. The model is completely frozen; only the prompt embeddings are trained.
    - **Prefix tuning:** Similar, but the learned vectors are prepended to the key and value matrices at every attention layer, giving them more influence over the model's computation.

    These methods are extremely parameter-efficient (often < 0.1% of model parameters) and allow serving multiple tasks from a single frozen model --- just swap the learned prefix. The trade-off is that they underperform LoRA and full fine-tuning on harder tasks.

    ### 7.5 When to Use Which

    | Method | Trainable Params | Best When | Risk |
    |---|---|---|---|
    | Full fine-tuning | 100% | Large dataset, sufficient compute, significant domain shift | Overfitting, forgetting |
    | LoRA | ~0.1--1% | Moderate dataset, limited compute, moderate domain shift | Slight underfitting on hard tasks |
    | Linear probing | < 0.01% | Quick evaluation, very small dataset | Underfitting |
    | Prompt/prefix tuning | < 0.1% | Multi-task serving, very limited compute | Underfitting on complex tasks |

    The practical default for most people today is LoRA. It's the sweet spot of efficiency and performance.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 8. Representation Quality --- How to Evaluate

    Self-supervised learning produces *representations*, not predictions. How do you measure whether those representations are good?

    ### 8.1 Linear Probing Accuracy

    Train a linear classifier on frozen representations. If a simple linear model achieves high accuracy, the representations have learned to separate the relevant classes --- the geometric structure of the embedding space is well-organized.

    This is the standard benchmark for comparing self-supervised methods. On ImageNet, top self-supervised methods achieve ~75-80% linear probing accuracy, approaching the ~76% of a supervised ResNet-50.

    ### 8.2 Transfer Performance

    The ultimate test: take the pretrained model, fine-tune it on a different dataset/task, and measure performance. Good representations transfer broadly. Poor representations are task-specific.

    Standard transfer benchmarks include:
    - **Vision:** Fine-tune on CIFAR-10/100, Flowers-102, DTD (textures), or medical imaging datasets
    - **Language:** Fine-tune on GLUE/SuperGLUE benchmarks, SQuAD, or domain-specific tasks

    A representation is good if it transfers well across diverse tasks, not just one ([Murphy PML1, Ch. 19.1](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf)).

    ### 8.3 Representation Similarity Analysis

    Compare learned representations to known reference representations:

    - **CKA (Centered Kernel Alignment):** Measures similarity between two sets of representations. Useful for comparing different layers of a model, or comparing two models trained with different methods.
    - **Representational Similarity Analysis (RSA):** Compute pairwise similarity matrices for both representations and compare them. If two methods produce similar representational geometry, they've learned similar structure.

    ### 8.4 Probing Classifiers

    Go beyond linear probes. Train small MLPs to predict specific properties from frozen representations:

    - Can you predict object position from the representation? Then it encodes spatial information.
    - Can you predict color? Texture? Object count? Scene type?
    - Each probing task reveals what information the representation has captured --- and what it has discarded.

    This gives a fine-grained picture of representation content, beyond aggregate accuracy numbers ([Murphy PML2, Ch. 34.6](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML2.pdf)).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's compute CKA to compare two representation matrices.
    """)
    return


@app.cell
def _(np):
    # CKA: Centered Kernel Alignment — measures representation similarity
    rng = np.random.default_rng(7)
    n_cka = 50  # number of data points

    # Two "representation matrices" from different models
    X_rep = rng.standard_normal((n_cka, 16))  # model A representations
    Y_rep = X_rep @ rng.standard_normal((16, 16)) * 0.5 + rng.standard_normal((n_cka, 16)) * 0.1  # similar

    # Linear CKA: HSIC(K, L) / sqrt(HSIC(K,K) * HSIC(L,L))
    def center(K):
        H = np.eye(len(K)) - 1.0 / len(K)
        return H @ K @ H

    K = center(X_rep @ X_rep.T)
    L = center(Y_rep @ Y_rep.T)
    hsic_kl = np.trace(K @ L) / (n_cka - 1) ** 2
    hsic_kk = np.trace(K @ K) / (n_cka - 1) ** 2
    hsic_ll = np.trace(L @ L) / (n_cka - 1) ** 2
    cka = hsic_kl / np.sqrt(hsic_kk * hsic_ll)
    print(f"CKA similarity: {cka:.3f}  (1.0 = identical geometry, 0.0 = unrelated)")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Summary

    Self-supervised learning is the most consequential idea in modern machine learning. The key insights:

    1. **You don't need labels.** Create supervision from the data's own structure --- predict masked tokens, reconstruct masked patches, align paired modalities.
    2. **Pretext tasks learn representations.** The task is a vehicle; the learned features are the destination.
    3. **Language and vision require different approaches.** Language uses autoregressive or masked prediction on discrete tokens. Vision uses contrastive learning, non-contrastive methods, or masked image modeling on continuous patches.
    4. **Multimodal alignment is powerful.** CLIP-style contrastive learning across modalities enables zero-shot generalization.
    5. **Foundation models change the economics.** Pretrain once at enormous cost, adapt cheaply many times. LoRA makes this practical for everyone.
    6. **Scale changes what's possible.** Emergent abilities, in-context learning, and zero-shot performance all appear at sufficient scale.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Practice

    **Conceptual:**

    1. Explain why BERT's `[MASK]` token creates a distribution mismatch between pretraining and fine-tuning. How does the 80/10/10 masking strategy mitigate this? Could you design a different mitigation strategy?

    2. SimCLR requires large batch sizes (4096+) while MoCo works with small batches (256). Explain the architectural difference that enables this and discuss the trade-offs of each approach.

    3. BYOL and SimSiam learn without negative pairs. Describe two mechanisms that prevent representation collapse. Why is the stop-gradient operation essential --- what happens if you remove it?

    4. MAE masks 75% of image patches, while BERT masks only 15% of text tokens. Explain why this difference exists in terms of the information density and redundancy of images versus text.

    **Analytical:**

    5. A CLIP model is trained on 400M image-text pairs. At inference, you want to classify images into 200 bird species. Write out the zero-shot classification procedure step by step. What determines whether this will work well? When would fine-tuning be necessary?

    6. You have a pretrained ViT-Large (300M parameters) and a dataset of 500 labeled medical X-rays across 5 diagnostic classes. Which fine-tuning strategy would you use and why? What if the dataset grew to 50,000 images? What if it grew to 500,000?

    7. Consider the NT-Xent loss from SimCLR. What is the role of the temperature parameter $\tau$? What happens when $\tau \to 0$? When $\tau \to \infty$? Sketch the gradient behavior in each regime.

    8. You train two self-supervised models: one with SimCLR (contrastive) and one with MAE (reconstructive). Both achieve similar linear probing accuracy on ImageNet. Design an experiment using probing classifiers to determine whether they've learned different types of information.

    **Implementation-oriented:**

    9. Sketch the pseudocode for a LoRA layer. Show how the forward pass combines the frozen weight matrix with the low-rank update. How would you merge the LoRA weights back into the base model at inference time?

    10. Design a self-supervised pretext task for time-series sensor data (e.g., accelerometer readings from a smartwatch). What should you mask or predict? What augmentations make sense? What downstream task would benefit from the learned representations?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Code It

    Put the concepts into practice. Each exercise gives you a skeleton --- fill in the `TODO` sections.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Exercise 1: NT-Xent Loss from Scratch

    Implement the full SimCLR NT-Xent loss for a batch of $N$ images (producing $2N$ embeddings). Given a `(2N, d)` matrix of L2-normalized embeddings where rows `2k` and `2k+1` are positive pairs, compute the average loss across all anchors.

    Recall:
    $$\mathcal{L}_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k \neq i} \exp(\text{sim}(z_i, z_k) / \tau)}$$
    """)
    return


@app.cell
def _(np):
    def nt_xent_loss(z, tau=0.5):
        """
        Compute NT-Xent loss for a batch of 2N normalized embeddings.
        z: (2N, d) array, L2-normalized. Rows 2k and 2k+1 are positive pairs.
        Returns: scalar average loss.
        """
        two_N = z.shape[0]

        # TODO: Compute the (2N, 2N) cosine similarity matrix
        # sim_matrix = ...

        # TODO: Build a (2N,) array of positive-pair indices
        # For anchor 2k, positive is 2k+1 and vice versa
        # pos_indices = ...

        # TODO: For each anchor i, compute the NT-Xent loss:
        #   numerator = exp(sim(z_i, z_{pos}) / tau)
        #   denominator = sum over k != i of exp(sim(z_i, z_k) / tau)
        #   loss_i = -log(numerator / denominator)
        # loss = ...

        # TODO: Return the average loss over all 2N anchors
        pass

    # Test your implementation
    rng = np.random.default_rng(10)
    _z_test = rng.standard_normal((8, 16))  # 4 images, 8 views
    _z_test = _z_test / np.linalg.norm(_z_test, axis=1, keepdims=True)
    # result = nt_xent_loss(_z_test, tau=0.5)
    # print(f"NT-Xent loss: {result:.4f}")  # should be around 1.5-2.5
    return (nt_xent_loss,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 2: BERT Masking Pipeline

    Implement the full BERT masking pipeline. Given a sequence of token IDs, apply the 15% mask rate with the 80/10/10 strategy. Return the corrupted input, the mask of which positions were selected, and the original tokens at those positions (the targets).
    """)
    return


@app.cell
def _(np):
    def bert_mask(token_ids, vocab_size, mask_token_id=0, mask_prob=0.15, seed=None):
        """
        Apply BERT-style masking to a sequence of token IDs.
        Returns: (corrupted_ids, selected_mask, target_ids)
          - corrupted_ids: same shape as token_ids, with masking applied
          - selected_mask: boolean array, True at positions that were selected
          - target_ids: original token IDs at selected positions
        """
        rng = np.random.default_rng(seed)
        T_len = len(token_ids)
        corrupted_ids = token_ids.copy()

        # TODO: Randomly select ~15% of positions
        # selected_mask = ...

        # TODO: For each selected position, apply 80/10/10:
        #   80% chance → replace with mask_token_id
        #   10% chance → replace with random token from [0, vocab_size)
        #   10% chance → keep original

        # TODO: Extract original tokens at selected positions
        # target_ids = ...

        # return corrupted_ids, selected_mask, target_ids
        pass

    # Test your implementation
    # _tokens = np.array([12, 45, 78, 23, 56, 89, 34, 67, 90, 11])
    # corrupted, mask, targets = bert_mask(_tokens, vocab_size=100, seed=42)
    # print(f"Original:  {_tokens}")
    # print(f"Corrupted: {corrupted}")
    # print(f"Mask positions: {np.where(mask)[0]}")
    # print(f"Targets: {targets}")
    return (bert_mask,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 3: LoRA Forward and Merge

    Implement a LoRA layer class that stores the frozen weight $W$, low-rank matrices $A$ and $B$, and supports both the decomposed forward pass ($Wx + BAx$) and merging ($W_\text{merged} = W + BA$).
    """)
    return


@app.cell
def _(np):
    class LoRALayer:
        def __init__(self, d_in, d_out, rank):
            """
            Initialize frozen weight W and LoRA matrices A, B.
            A: (rank, d_in), B: (d_out, rank)
            B should be initialized to zeros so ΔW = BA starts at zero.
            """
            self.d_in = d_in
            self.d_out = d_out
            self.rank = rank

            # TODO: Initialize W (random), A (random small), B (zeros)
            # self.W = ...
            # self.A = ...
            # self.B = ...

        def forward(self, x):
            """Compute h = Wx + BAx (decomposed, for training)."""
            # TODO: Implement the LoRA forward pass
            pass

        def merge(self):
            """Return W_merged = W + BA (for inference, no extra cost)."""
            # TODO: Merge LoRA weights into the base weight
            pass

        def param_count(self):
            """Return (full_params, lora_params) for comparison."""
            # TODO: Count parameters
            pass

    # Test your implementation
    # rng = np.random.default_rng(0)
    # lora = LoRALayer(d_in=128, d_out=128, rank=4)
    # x_test = rng.standard_normal(128)
    # h_decomposed = lora.forward(x_test)
    # W_merged = lora.merge()
    # h_merged = W_merged @ x_test
    # print(f"Outputs match: {np.allclose(h_decomposed, h_merged)}")
    # full_p, lora_p = lora.param_count()
    # print(f"Full params: {full_p:,}, LoRA params: {lora_p:,}, Ratio: {full_p/lora_p:.0f}x")
    return (LoRALayer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 4: MAE Masking and Reconstruction Loss

    Implement the MAE pipeline: given image patches, randomly mask 75%, and compute the MSE reconstruction loss only on masked patches (as MAE does --- visible patches are not penalized).
    """)
    return


@app.cell
def _(np):
    def mae_pipeline(patches, mask_ratio=0.75, seed=None):
        """
        Simulate MAE masking and compute reconstruction loss.
        patches: (n_patches, patch_dim) array of patch embeddings.
        Returns dict with keys:
          'visible_idx', 'masked_idx', 'n_visible', 'n_masked',
          'reconstruction_mse'
        """
        rng = np.random.default_rng(seed)
        n_p = patches.shape[0]

        # TODO: Compute number of patches to mask
        # n_mask = ...

        # TODO: Random permutation to select which patches to mask
        # visible_idx = ...
        # masked_idx = ...

        # TODO: Simulate a (bad) reconstruction by adding noise to masked patches
        # fake_recon = ...

        # TODO: Compute MSE only on masked patches
        # mse = ...

        # return {
        #     'visible_idx': visible_idx,
        #     'masked_idx': masked_idx,
        #     'n_visible': n_p - n_mask,
        #     'n_masked': n_mask,
        #     'reconstruction_mse': mse,
        # }
        pass

    # Test your implementation
    # rng = np.random.default_rng(99)
    # _patches = rng.standard_normal((64, 16))  # 8x8 grid, 16-dim patches
    # result = mae_pipeline(_patches, mask_ratio=0.75, seed=42)
    # print(f"Visible: {result['n_visible']}, Masked: {result['n_masked']}")
    # print(f"Reconstruction MSE: {result['reconstruction_mse']:.4f}")
    return (mae_pipeline,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 5: CLIP-Style Zero-Shot Classifier

    Build a zero-shot classifier. Given a set of class names and pre-computed text/image embeddings, classify each image by finding the text embedding with highest cosine similarity. Then compute accuracy.
    """)
    return


@app.cell
def _(np):
    def clip_zero_shot(image_embeddings, text_embeddings, true_labels, temperature=0.07):
        """
        Zero-shot classification using CLIP-style cosine similarity.
        image_embeddings: (n_images, d) L2-normalized
        text_embeddings:  (n_classes, d) L2-normalized
        true_labels:      (n_images,) integer class indices
        Returns: (predictions, accuracy, similarity_matrix)
        """
        # TODO: Compute cosine similarity between every image and every class text
        # similarity_matrix = ...  # (n_images, n_classes)

        # TODO: Scale by temperature
        # logits = ...

        # TODO: Predict class for each image (argmax over classes)
        # predictions = ...

        # TODO: Compute accuracy
        # accuracy = ...

        # return predictions, accuracy, similarity_matrix
        pass

    # Test your implementation
    # rng = np.random.default_rng(42)
    # n_img, n_cls, d = 100, 5, 64
    # _img_emb = rng.standard_normal((n_img, d))
    # _img_emb /= np.linalg.norm(_img_emb, axis=1, keepdims=True)
    # _txt_emb = rng.standard_normal((n_cls, d))
    # _txt_emb /= np.linalg.norm(_txt_emb, axis=1, keepdims=True)
    # _labels = rng.integers(0, n_cls, n_img)
    # preds, acc, sim = clip_zero_shot(_img_emb, _txt_emb, _labels)
    # print(f"Zero-shot accuracy: {acc:.1%} (random baseline: {1/n_cls:.1%})")
    return (clip_zero_shot,)


if __name__ == "__main__":
    app.run()
