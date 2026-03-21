import marimo

app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Module 3A: Attention & Transformers

    > *"Attention is all you need."* --- Vaswani et al., 2017

    This is, without exaggeration, the single most important lecture in this entire course. Everything that has happened in machine learning since 2017 --- GPT, BERT, ChatGPT, Stable Diffusion, AlphaFold 2 --- traces back to the ideas in this module. If you understand what follows here, you understand the engine that drives modern AI.

    You already know the problem. In the last module on sequence models, we built encoder-decoder architectures where an RNN reads an input sequence and compresses it into a single hidden state vector, then a decoder RNN unrolls that vector back into an output sequence. You felt the tension: how can one fixed-size vector carry the meaning of an entire sentence, an entire paragraph, an entire document? It cannot. That bottleneck is where we begin.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 1. The Attention Mechanism --- The Key Insight

    ### The Bottleneck Problem

    Recall the standard seq2seq model. The encoder processes tokens $x_1, x_2, \ldots, x_T$ and produces hidden states $h_1, h_2, \ldots, h_T$. But the decoder only receives $h_T$ --- the final hidden state. All information about the source sequence must be squeezed through this single vector.

    For short sequences, this works passably. For long sequences, it fails catastrophically. The model "forgets" early tokens. Information gets overwritten. Translation quality degrades sharply with sentence length.

    This is not a training problem --- it is an architectural problem. No amount of data or compute fixes a bottleneck that is structurally too narrow.

    ### The Solution: Let the Decoder Look Back

    Bahdanau et al. (2014) proposed an elegantly simple fix: instead of forcing the decoder to work from only $h_T$, let it look at **all** encoder hidden states $h_1, h_2, \ldots, h_T$ at every decoding step.

    But looking at all of them equally is not useful either. At each decoding step, some source positions are more relevant than others. When translating the third word of a French sentence into English, you probably care most about specific source words, not all of them equally.

    So the mechanism works as follows. At each decoder time step $t$, with decoder hidden state $s_t$:

    1. **Score** each encoder hidden state: $e_{t,i} = \text{score}(s_t, h_i)$ for all encoder positions $i$
    2. **Normalize** the scores into weights: $\alpha_{t,i} = \text{softmax}(e_t)_i$
    3. **Compute a weighted sum** --- the context vector: $c_t = \sum_i \alpha_{t,i} \, h_i$
    4. **Use** $c_t$ alongside $s_t$ to produce the output

    The score function can be a simple dot product, a small neural network, or other variants. The key insight is the softmax: it produces a probability distribution over source positions, so $c_t$ is a soft, differentiable selection of which encoder states to focus on.

    ### Attention as a Soft Dictionary Lookup

    Here is the analogy that makes attention click. Think of a Python dictionary: you provide a **query** (the key you are looking up), and the dictionary checks it against stored **keys**, and returns the associated **value**.

    Attention does the same thing, but *soft*. Instead of an exact match returning one value, the query is compared against all keys, producing a similarity score for each. These scores become weights (via softmax), and the output is a weighted combination of all values.

    | Dictionary Lookup | Attention |
    |---|---|
    | Query (your search key) | Decoder state $s_t$ |
    | Keys (stored keys) | Encoder states $h_i$ |
    | Values (stored values) | Encoder states $h_i$ (same as keys here) |
    | Exact match -> one result | Soft match -> weighted combination |

    This query-key-value framing will become central when we move to self-attention. Keep it in your mind.

    ### Visualizing Attention

    One beautiful property of attention: the weights $\alpha_{t,i}$ are interpretable. You can plot them as a heatmap with source positions on one axis and target positions on the other. You will see near-diagonal patterns for monotonic alignments (like English-French translation) and more complex patterns for languages with different word orders (like English-Japanese).

    This interpretability was a major selling point of attention, though as we will see, interpreting attention weights in deeper models is more nuanced than it first appears.

    See [Murphy PML1 S15.4](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf) for a thorough treatment of attention mechanisms in the encoder-decoder context.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 2. Self-Attention --- Attention Without Encoder/Decoder

    Bahdanau attention operates *between* an encoder and decoder. The breakthrough of the Transformer is applying attention *within* a single sequence. This is **self-attention**.

    ### The Core Idea

    In self-attention, each position in a sequence attends to **every other position in the same sequence**. There is no separate encoder and decoder --- just one sequence talking to itself.

    Why is this powerful? Consider the sentence: *"The animal didn't cross the street because it was too tired."* What does "it" refer to? The animal. To figure that out, the representation of "it" needs to incorporate information from "animal." Self-attention gives every token a direct pathway to every other token in the sequence, regardless of distance. No more forgetting over long distances. No more sequential bottleneck.

    ### Computing Q, K, V from the Same Input

    Given an input matrix $X \in \mathbb{R}^{n \times d}$ (where $n$ is sequence length and $d$ is the embedding dimension), we compute three matrices:

    $$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

    where $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}$ are learned weight matrices.

    Each row of $Q$ is a query vector --- "what am I looking for?" Each row of $K$ is a key vector --- "what do I contain?" Each row of $V$ is a value vector --- "what information do I provide if selected?"

    The same input $X$ generates all three, but through *different* learned projections. This is critical: the model learns to produce different representations for querying, being queried, and providing information.

    ### Scaled Dot-Product Attention

    The full self-attention computation is:

    $$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

    Step by step:
    1. $QK^\top$ computes the dot product of every query with every key, producing an $n \times n$ matrix of raw scores
    2. Divide by $\sqrt{d_k}$ (the scaling factor --- more on this shortly)
    3. Apply softmax row-wise to get attention weights (each row sums to 1)
    4. Multiply by $V$ to get the weighted combination of value vectors

    The output has the same shape as $V$: each position gets a new representation that is an attention-weighted mixture of all value vectors.

    ### Why the Scaling Factor $\sqrt{d_k}$?

    This is a detail that matters more than it looks. Consider two random vectors in $\mathbb{R}^{d_k}$ with entries drawn from $\mathcal{N}(0, 1)$. Their dot product has mean 0 and **variance $d_k$**. As $d_k$ grows, some dot products become very large in magnitude. When you feed large values into softmax, the output becomes extremely peaked --- essentially a hard argmax --- and gradients vanish.

    Dividing by $\sqrt{d_k}$ rescales the variance back to 1, keeping the softmax in a regime where gradients flow well. It is a small thing, but without it, training becomes unstable for large $d_k$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A Concrete Numerical Example

    Let us walk through self-attention with a tiny example. Suppose we have 3 tokens with embedding dimension $d = 4$ and $d_k = 2$.

    $$X = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 1 & 0 & 0 \end{bmatrix}$$

    Suppose (with made-up weights):

    $$W_Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad W_K = \begin{bmatrix} 0 & 1 \\ 1 & 0 \\ 0 & 1 \\ 1 & 0 \end{bmatrix}, \quad W_V = \begin{bmatrix} 1 & 1 \\ 1 & 0 \\ 0 & 1 \\ 0 & 1 \end{bmatrix}$$

    Then:

    $$Q = XW_Q = \begin{bmatrix} 2 & 0 \\ 0 & 2 \\ 1 & 1 \end{bmatrix}, \quad K = XW_K = \begin{bmatrix} 0 & 2 \\ 2 & 0 \\ 1 & 1 \end{bmatrix}, \quad V = XW_V = \begin{bmatrix} 1 & 2 \\ 1 & 1 \\ 2 & 1 \end{bmatrix}$$

    Scores: $QK^\top = \begin{bmatrix} 0 & 4 & 2 \\ 4 & 0 & 2 \\ 2 & 2 & 2 \end{bmatrix}$

    Scaled scores (dividing by $\sqrt{2} \approx 1.414$): $\begin{bmatrix} 0 & 2.83 & 1.41 \\ 2.83 & 0 & 1.41 \\ 1.41 & 1.41 & 1.41 \end{bmatrix}$

    After row-wise softmax (approximate):

    $$\alpha \approx \begin{bmatrix} 0.06 & 0.70 & 0.24 \\ 0.70 & 0.06 & 0.24 \\ 0.33 & 0.33 & 0.33 \end{bmatrix}$$

    Token 1 attends mostly to token 2 (weight 0.70). Token 2 attends mostly to token 1 (weight 0.70). Token 3 attends roughly equally to all three. The output $\alpha V$ gives each token a new representation that is a weighted mix of all value vectors according to these weights.

    This is the heart of the Transformer. Everything else is supporting structure around this operation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3. Multi-Head Attention

    A single attention head computes one set of attention weights. But a token might need to attend to different positions for different reasons: one for syntactic dependency, another for semantic similarity, another for positional proximity.

    ### Parallel Attention Heads

    Multi-head attention runs $h$ attention heads in parallel, each with its own learned projections:

    $$\text{head}_i = \text{Attention}(XW_Q^i, XW_K^i, XW_V^i)$$

    The outputs are concatenated and projected:

    $$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O$$

    where $W_O \in \mathbb{R}^{hd_v \times d}$ is a final linear projection.

    ### Dimension Arithmetic

    If the model dimension is $d = 512$ and we use $h = 8$ heads, each head operates in dimension $d_k = d_v = d / h = 64$. The total computation is roughly the same as a single head with full dimensionality, because we split the representation space rather than duplicating it.

    This is an important design choice: multi-head attention is not more expensive than single-head attention of the same total dimension. It is a better *use* of the same capacity.

    ### What Different Heads Learn

    Empirical studies of trained Transformers reveal that different attention heads specialize. Some heads learn to attend to the previous token. Some learn syntactic relationships (subject-verb agreement across long distances). Some learn positional patterns. Some learn semantic similarity. This diversity of attention patterns, arising from the same training signal, is one reason multi-head attention works so well.

    See [Murphy PML1 S15.4.3](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf) for further discussion of multi-head attention.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4. The Transformer Architecture

    Now we assemble the full architecture described in "Attention Is All You Need" (Vaswani et al., 2017). The original Transformer is an encoder-decoder model, though as we will see, many modern variants use only one half.

    ### The Encoder Block

    Each encoder block consists of two sub-layers:

    1. **Multi-head self-attention**
    2. **Position-wise feed-forward network (FFN)**

    Crucially, each sub-layer is wrapped with a **residual connection** and **layer normalization**:

    $$\text{output} = \text{LayerNorm}(x + \text{SubLayer}(x))$$

    So the full encoder block is:

    $$z = \text{LayerNorm}(x + \text{MultiHeadAttn}(x))$$
    $$\text{output} = \text{LayerNorm}(z + \text{FFN}(z))$$

    ### The Decoder Block

    Each decoder block has three sub-layers:

    1. **Masked multi-head self-attention** (the decoder attends to its own previous outputs)
    2. **Multi-head cross-attention** (the decoder attends to the encoder output --- this is like Bahdanau attention)
    3. **Position-wise FFN**

    Each sub-layer again has residual connections and layer normalization.

    ### Residual Connections

    You saw these in the CNN module when we discussed ResNets. The idea is identical: instead of learning $F(x)$, learn $F(x) + x$. This lets gradients flow directly through the addition, preventing degradation in deep networks. The original Transformer stacks 6 encoder blocks and 6 decoder blocks --- that is 12 blocks deep, and without residual connections, training would collapse.

    ### Layer Normalization

    Why LayerNorm instead of BatchNorm? BatchNorm normalizes across the batch dimension --- it computes statistics over all examples in a mini-batch for each feature. This works well for fixed-size inputs (images), but sequences have variable lengths. Padding and masking make batch statistics unreliable. LayerNorm normalizes across the feature dimension for each individual example, making it independent of batch composition and sequence length.

    In practice: LayerNorm computes the mean and variance across the $d$ features for each token independently, then normalizes. No dependence on other examples in the batch. No dependence on sequence length. Clean and simple.

    ### The Feed-Forward Network

    The FFN is applied identically to each position (hence "position-wise"):

    $$\text{FFN}(x) = W_2 \, \sigma(W_1 x + b_1) + b_2$$

    where $W_1 \in \mathbb{R}^{d \times d_{ff}}$, $W_2 \in \mathbb{R}^{d_{ff} \times d}$, and typically $d_{ff} = 4d$. The activation $\sigma$ was ReLU in the original paper; modern Transformers often use GELU or SwiGLU.

    This FFN is where much of the "knowledge" in a Transformer is believed to reside. The attention layers route information; the FFN layers process it. Some researchers describe attention as the "communication" step and the FFN as the "computation" step.

    ### The Full Stack

    The original Transformer uses $N = 6$ blocks for both encoder and decoder. The encoder processes the full input and produces a set of representations. The decoder generates output tokens one at a time, attending to both its own past outputs (masked self-attention) and the encoder representations (cross-attention).

    See [Goodfellow et al., Deep Learning](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) for background on residual connections and normalization layers in deep networks.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 5. Positional Encoding --- Giving Transformers a Sense of Order

    ### The Problem

    Here is something that should bother you: self-attention computes dot products between all pairs of positions, but nothing in the computation depends on *where* those positions are in the sequence. If you scramble the order of input tokens, the attention weights change (because the inputs change), but the *mechanism* treats every pair of positions identically. Self-attention is **permutation-equivariant**.

    This means that without some additional signal, the Transformer cannot distinguish "The cat sat on the mat" from "mat the on sat cat The." It has no notion of word order. We need to inject positional information explicitly.

    ### Sinusoidal Positional Encoding

    The original Transformer adds a positional encoding vector to each token embedding before feeding it into the model. For position $\text{pos}$ and dimension $i$:

    $$PE_{(\text{pos}, 2i)} = \sin\!\left(\frac{\text{pos}}{10000^{2i/d}}\right)$$
    $$PE_{(\text{pos}, 2i+1)} = \cos\!\left(\frac{\text{pos}}{10000^{2i/d}}\right)$$

    Each dimension of the positional encoding is a sinusoid with a different frequency. Low dimensions oscillate quickly (high frequency); high dimensions oscillate slowly (low frequency). This creates a unique "signature" for each position.

    Why sinusoidal? The authors hypothesized that it would allow the model to learn relative positions easily, because for any fixed offset $k$, $PE_{\text{pos}+k}$ can be written as a linear function of $PE_{\text{pos}}$. The model could, in principle, learn to compute "how far apart are these two tokens" by linear operations on their positional encodings.

    ### Learned Positional Embeddings

    In practice, many models (including BERT and GPT-2) simply learn a positional embedding matrix $P \in \mathbb{R}^{L \times d}$ where $L$ is the maximum sequence length. Position $i$ gets embedding $P_i$, learned through backpropagation like any other parameter.

    This works well but limits the model to sequences of length at most $L$. You cannot extrapolate beyond what you trained on.

    ### Rotary Positional Embeddings (RoPE)

    RoPE, introduced by Su et al. (2021), has become the dominant approach in modern large language models (LLaMA, Mistral, etc.). Instead of adding position information to the embeddings, RoPE encodes position by *rotating* the query and key vectors in 2D subspaces before computing dot products.

    The key property: the dot product between a rotated query at position $m$ and a rotated key at position $n$ depends only on the *relative* distance $m - n$, not on the absolute positions. This gives the model relative position awareness naturally, and it extrapolates to longer sequences better than learned absolute embeddings.

    The details involve applying rotation matrices to pairs of dimensions, where the rotation angle is proportional to position. It is elegant and effective.

    ### Relative vs. Absolute Position

    Absolute encodings assign a fixed representation to each position (0, 1, 2, ...). Relative encodings care only about the distance between positions. Linguistic relationships are mostly relative ("the adjective is two words before the noun"), so relative approaches tend to generalize better. RoPE achieves relative encoding through an absolute mechanism (per-position rotation), giving the best of both worlds.

    See [Murphy PML2 S16.3](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML2.pdf) for coverage of positional encoding variants.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 6. Masked Self-Attention --- Preventing the Future from Leaking

    In autoregressive generation (predicting one token at a time, left to right), the model at position $i$ must only attend to positions $\leq i$. If it could see future tokens, it would simply copy them --- the task becomes trivial and the model learns nothing useful.

    The solution is **causal masking**. Before the softmax, we set all entries above the diagonal of the $QK^\top$ matrix to $-\infty$:

    $$\text{mask}_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

    $$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top + \text{mask}}{\sqrt{d_k}}\right) V$$

    Since $\text{softmax}(-\infty) = 0$, future positions contribute zero weight. The attention becomes a lower-triangular matrix: each token can only attend to itself and everything before it.

    This is used in all decoder models and is what makes autoregressive generation possible during training. During inference, the model generates one token at a time anyway, but during training, causal masking allows us to compute all positions in parallel while still preventing information leakage.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 7. Transformer Variants and the Zoo of Models

    The original Transformer is an encoder-decoder. But the architecture splits naturally into components that can be used independently, giving rise to three major families.

    ### Encoder-Only: BERT

    BERT (Devlin et al., 2018) uses only the Transformer encoder. It processes the entire input bidirectionally --- every token can attend to every other token, with no causal mask. This makes it powerful for *understanding* tasks (classification, named entity recognition, question answering).

    BERT is trained with **masked language modeling**: randomly mask 15% of tokens and train the model to predict them from context. This forces the model to build rich contextual representations.

    ### Decoder-Only: GPT

    GPT (Radford et al., 2018) uses only the Transformer decoder (without cross-attention, since there is no encoder). It is autoregressive: trained to predict the next token given all previous tokens. The causal mask ensures it only looks left.

    GPT-2, GPT-3, GPT-4, LLaMA, Claude --- all decoder-only. This architecture dominates modern language models.

    ### Encoder-Decoder: T5, BART

    T5 and BART retain the full encoder-decoder structure. They excel at tasks with distinct input-output structure: translation, summarization, question answering where you want to condition on a source text and generate a target.

    ### Why Decoder-Only Won

    This is worth reflecting on. BERT-style models were dominant from 2018 to 2020. But decoder-only models won out for several reasons:

    1. **Simplicity.** One architecture, one training objective (next-token prediction). No need for task-specific heads.
    2. **Scaling.** Scaling laws (discussed next) showed that next-token prediction loss decreases smoothly with scale. Autoregressive models benefit from every additional parameter and training token in a predictable way.
    3. **Generality.** Any task can be framed as text generation. Classification, translation, code generation, reasoning --- all become "continue this text." Few-shot learning via prompting works naturally.
    4. **Emergent capabilities.** Large autoregressive models develop abilities (chain-of-thought reasoning, in-context learning) that were not explicitly trained. These emerged most clearly in decoder-only models.

    See [Murphy PML1 S15.5](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf) for a comparison of Transformer architectural variants.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 8. Scaling Laws and Emergent Abilities

    ### Scaling Laws

    Kaplan et al. (2020) at OpenAI discovered that Transformer language model loss follows **power laws** in three variables:

    - **Parameters** ($N$): $L \propto N^{-0.076}$
    - **Dataset size** ($D$): $L \propto D^{-0.095}$
    - **Compute** ($C$): $L \propto C^{-0.050}$

    These are remarkably smooth and predictable. You can estimate the performance of a model 10x larger before training it. This predictability is what gave labs the confidence to train billion- and trillion-parameter models.

    The Chinchilla paper (Hoffmann et al., 2022) refined these laws, showing that many models were *undertrained* --- they had too many parameters for the amount of data they were trained on. The optimal ratio is roughly 20 tokens per parameter. This shifted the field toward training smaller models on more data (Chinchilla-70B outperformed the much larger Gopher-280B).

    ### Emergent Abilities

    Perhaps the most intriguing phenomenon: certain capabilities appear to emerge *suddenly* at scale. A model with 1B parameters cannot do multi-step arithmetic; a model with 100B parameters can. A model with 10B parameters gives random answers to analogy questions; a model with 175B parameters answers them correctly.

    These emergent abilities --- including in-context learning, chain-of-thought reasoning, and instruction following --- were not predicted by scaling laws (which only track overall loss). They suggest that smooth improvements in loss can lead to sudden qualitative changes in capability.

    ### The Foundation Model Paradigm

    The result of all this is a new paradigm: **pretrain** a massive model on a huge corpus of text (or images, or both), then **adapt** it to specific tasks through fine-tuning, prompting, or reinforcement learning from human feedback (RLHF). One model, trained once at enormous cost, serves as the foundation for thousands of downstream applications.

    This is fundamentally different from the classical ML workflow of training a separate model for each task. The Transformer made this possible.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 9. Computational Considerations

    ### The $O(n^2)$ Problem

    Self-attention computes a full $n \times n$ attention matrix, where $n$ is the sequence length. This is $O(n^2)$ in both time and memory. For $n = 1{,}000$, this is manageable. For $n = 100{,}000$, it becomes a serious bottleneck.

    This quadratic cost is the fundamental tension of the Transformer: its greatest strength (every token can attend to every other token) is also its greatest weakness (you have to compute and store all those pairwise interactions).

    ### KV Cache

    During autoregressive generation, at step $t$, you compute $Q$ only for the new token but need $K$ and $V$ for all previous tokens. Rather than recomputing $K$ and $V$ from scratch at each step, you cache them. This is the **KV cache**: store the key and value vectors for all past positions and append to them as you generate.

    Without the KV cache, generation is $O(n^2)$ per token (recomputing everything). With the KV cache, it is $O(n)$ per token (one query against all cached keys). For long sequences, the KV cache can consume significant GPU memory, which is why KV cache compression and quantization are active research areas.

    ### Flash Attention

    Standard attention materializes the full $n \times n$ attention matrix in GPU memory. Flash Attention (Dao et al., 2022) is a **hardware-aware** algorithm that computes exact attention without materializing this matrix, by tiling the computation to exploit the GPU memory hierarchy (SRAM vs. HBM).

    Flash Attention is not an approximation --- it computes the same result as standard attention, just faster and with less memory. It achieves 2-4x speedup on typical workloads and enables much longer sequences. It is now the default in most training frameworks.

    ### Sparse and Linear Attention (Brief)

    Various methods attempt to break the $O(n^2)$ barrier:
    - **Sparse attention** (Longformer, BigBird): attend to a fixed pattern (local window + global tokens) instead of all pairs
    - **Linear attention**: reformulate attention to avoid the explicit $n \times n$ matrix, achieving $O(n)$ complexity

    These have had moderate success but standard dense attention with Flash Attention remains dominant, largely because the constant factors in sparse/linear methods often eat into the asymptotic gains for practical sequence lengths.

    ### Context Length

    The maximum sequence length a model can process --- its context window --- is a critical practical parameter. Early Transformers were limited to 512 or 1,024 tokens. Modern models support 128K tokens or more. Longer contexts enable processing entire documents, codebases, or long conversations, but require careful engineering (RoPE scaling, Flash Attention, KV cache management).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 10. Why Transformers Changed Everything

    Let me be direct about why the Transformer is arguably the most important neural network architecture ever proposed.

    **Parallelism.** RNNs process sequences one step at a time. Transformers process all positions simultaneously (during training). This makes them vastly more efficient on modern GPUs, which are designed for massive parallelism. This single property is what enabled training on internet-scale datasets.

    **Scaling.** Transformers scale smoothly and predictably. Double the parameters, double the data, and you get a predictable improvement. No other architecture has demonstrated such clean scaling behavior over such a wide range.

    **Transfer learning.** A Transformer pretrained on a large corpus develops representations that transfer remarkably well to downstream tasks. This was true of CNNs for images (ImageNet pretraining), but Transformers took it to another level --- a single model can perform translation, summarization, question answering, code generation, and mathematical reasoning.

    **Universality.** The same architecture --- self-attention, FFN, residual connections, layer norm --- works across domains:
    - **Text**: GPT, BERT, LLaMA, Claude
    - **Images**: Vision Transformer (ViT) splits images into patches and treats them as tokens
    - **Audio**: Whisper uses a Transformer encoder-decoder for speech recognition
    - **Video**: VideoMAE, Gemini
    - **Protein structure**: AlphaFold 2 uses a Transformer-based architecture to predict protein folding
    - **Multimodal**: Models that process text, images, and audio simultaneously

    No previous architecture achieved this kind of cross-domain generality. The Transformer is a general-purpose computation engine that learns to route and process information, and it turns out this is useful for almost everything.

    See [Murphy PML1 S15.4.5](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf) for additional perspective on the Transformer's impact.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## PyTorch Implementation: Self-Attention from Scratch

    Understanding the code makes the math concrete. Here is scaled dot-product attention and multi-head attention implemented in PyTorch.
    """)
    return


@app.cell
def _(mo):
    mo.image(src="../animations/rendered/AttentionWeights.gif")
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import math
    return F, math, nn, torch


@app.cell
def _(F, math, nn):
    class ScaledDotProductAttention(nn.Module):
        """Scaled dot-product attention as described in Vaswani et al. (2017)."""

        def forward(self, Q, K, V, mask=None):
            """
            Args:
                Q: queries  (batch, n_heads, seq_len, d_k)
                K: keys     (batch, n_heads, seq_len, d_k)
                V: values   (batch, n_heads, seq_len, d_v)
                mask: optional mask (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)
            Returns:
                output: weighted sum of values (batch, n_heads, seq_len, d_v)
                attention_weights: (batch, n_heads, seq_len, seq_len)
            """
            d_k = Q.size(-1)

            # Step 1: Compute raw attention scores
            scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)

            # Step 2: Apply mask (for causal / padding masking)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))

            # Step 3: Softmax to get attention weights
            attention_weights = F.softmax(scores, dim=-1)

            # Step 4: Weighted sum of values
            output = attention_weights @ V

            return output, attention_weights
    return (ScaledDotProductAttention,)


@app.cell
def _(ScaledDotProductAttention, nn):
    class MultiHeadAttention(nn.Module):
        """Multi-head attention: h parallel attention heads."""

        def __init__(self, d_model, n_heads):
            super().__init__()
            assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads  # dimension per head

            # Learned projection matrices
            self.W_Q = nn.Linear(d_model, d_model)  # projects to all heads at once
            self.W_K = nn.Linear(d_model, d_model)
            self.W_V = nn.Linear(d_model, d_model)
            self.W_O = nn.Linear(d_model, d_model)  # output projection

            self.attention = ScaledDotProductAttention()

        def forward(self, x, mask=None):
            """
            Args:
                x: input tensor (batch, seq_len, d_model)
                mask: optional attention mask
            Returns:
                output: (batch, seq_len, d_model)
            """
            batch_size, seq_len, _ = x.size()

            # Project and reshape: (batch, seq_len, d_model) -> (batch, n_heads, seq_len, d_k)
            Q = self.W_Q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            K = self.W_K(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            V = self.W_V(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

            # Apply attention
            attn_output, attn_weights = self.attention(Q, K, V, mask)

            # Concatenate heads: (batch, n_heads, seq_len, d_k) -> (batch, seq_len, d_model)
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

            # Final linear projection
            output = self.W_O(attn_output)

            return output
    return (MultiHeadAttention,)


@app.cell
def _(MultiHeadAttention, nn):
    class TransformerBlock(nn.Module):
        """A single Transformer encoder block."""

        def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
            super().__init__()
            self.attention = MultiHeadAttention(d_model, n_heads)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model),
            )
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, mask=None):
            # Sub-layer 1: Multi-head self-attention with residual + layer norm
            attn_out = self.attention(x, mask)
            x = self.norm1(x + self.dropout(attn_out))

            # Sub-layer 2: Feed-forward with residual + layer norm
            ffn_out = self.ffn(x)
            x = self.norm2(x + self.dropout(ffn_out))

            return x
    return (TransformerBlock,)


@app.cell
def _(torch):
    def create_causal_mask(seq_len):
        """Lower-triangular mask: position i can attend to positions <= i."""
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        return mask  # shape: (1, 1, seq_len, seq_len)
    return (create_causal_mask,)


@app.cell
def _(TransformerBlock, create_causal_mask, torch):
    # Quick test
    batch_size, seq_len, d_model, n_heads, d_ff = 2, 10, 64, 8, 256

    x = torch.randn(batch_size, seq_len, d_model)
    block = TransformerBlock(d_model, n_heads, d_ff)

    # Without mask (encoder-style, bidirectional)
    out = block(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")

    # With causal mask (decoder-style)
    mask = create_causal_mask(seq_len)
    out_masked = block(x, mask)
    print(f"Masked output shape: {out_masked.shape}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Study this code carefully. Notice how the `view` and `transpose` operations split the single large projection into multiple heads, and how `contiguous().view()` concatenates them back. This reshape-based multi-head implementation is more efficient than literally running separate attention computations per head.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Practice Exercises

    ### Conceptual

    1. **Attention bottleneck.** Explain in your own words why the fixed-size context vector in seq2seq is a bottleneck. What information is lost? Why does this problem get worse with longer sequences?

    2. **Self-attention vs. RNN.** An RNN must propagate information from token 1 to token 100 through 99 sequential steps. How many steps does self-attention need for the same information transfer? What are the tradeoffs?

    3. **Scaling factor.** Suppose $d_k = 10{,}000$. Without scaling, what would happen to the softmax distribution? Why is this a problem for gradient-based learning?

    4. **Positional encoding.** If we removed positional encodings entirely, give a concrete example of two different sentences that would produce identical Transformer outputs.

    5. **Causal masking.** Draw the 5x5 attention mask matrix for a decoder with 5 positions. What value fills the upper triangle, and why that specific value?

    ### Computational

    6. **Attention by hand.** Given $Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$, $K = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$, $V = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$, compute $\text{Attention}(Q, K, V)$ step by step with $d_k = 2$. What do you notice about the result?

    7. **Parameter count.** A Transformer has $d = 512$, $h = 8$ heads, $d_{ff} = 2048$, and 6 encoder blocks. Calculate the total number of parameters in the attention layers and FFN layers (ignore biases and layer norm for simplicity).

    8. **Modify the code.** Add a `CrossAttention` module to the PyTorch code above, where queries come from one sequence and keys/values come from another. This is what the decoder uses to attend to the encoder output.

    9. **Complexity.** For a sequence of length $n = 4096$ and model dimension $d = 1024$, calculate the size (in number of elements) of the attention matrix. Compare this to the size of the weight matrices $W_Q$, $W_K$, $W_V$.

    10. **Attention visualization.** Using the code above, run the `MultiHeadAttention` module on a random input and extract the attention weights from each head. Do different heads produce different attention patterns even before training? Why or why not?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Key References

    - **Vaswani et al. (2017)**, "Attention Is All You Need" --- the foundational paper. Read at least Sections 1-3.
    - **Bahdanau et al. (2014)**, "Neural Machine Translation by Jointly Learning to Align and Translate" --- the original attention mechanism.
    - [Murphy, Probabilistic Machine Learning (Vol. 1), Chapter 15.4](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf) --- comprehensive treatment of attention and Transformers.
    - [Murphy, Probabilistic Machine Learning (Vol. 2), Chapter 16.3](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML2.pdf) --- advanced topics including positional encodings and Transformer variants.
    - [Goodfellow et al., Deep Learning](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) --- background on residual connections, normalization, and deep network training.
    - **Kaplan et al. (2020)**, "Scaling Laws for Neural Language Models" --- the scaling laws paper.
    - **Hoffmann et al. (2022)**, "Training Compute-Optimal Large Language Models" --- the Chinchilla paper.

    ---

    **Next module: 3B --- Training Large Language Models** --- pretraining objectives, tokenization, RLHF, and the engineering of modern foundation models.
    """)
    return


if __name__ == "__main__":
    app.run()
