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
    # Module 3A: Attention & Transformers

    > *"Attention is all you need."* --- Vaswani et al., 2017

    This is, without exaggeration, the single most important lecture in this entire course. Everything that has happened in machine learning since 2017 --- GPT, BERT, ChatGPT, Stable Diffusion, AlphaFold 2 --- traces back to the ideas in this module. If you understand what follows here, you understand the engine that drives modern AI.

    You already know the problem. In the last module on sequence models, we built encoder-decoder architectures where an RNN reads an input sequence and compresses it into a single hidden state vector, then a decoder RNN unrolls that vector back into an output sequence. You felt the tension: how can one fixed-size vector carry the meaning of an entire sentence, an entire paragraph, an entire document? It cannot. That bottleneck is where we begin.
    """)
    return


@app.cell
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Code: Bahdanau-style attention in numpy** --- score, normalize, and compute the context vector.
    """)
    return


@app.cell
def _(np):
    def softmax(x, axis=-1):
        """Numerically stable softmax."""
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / e_x.sum(axis=axis, keepdims=True)

    # Encoder hidden states: 5 source positions, dim=4
    h_enc = rng.standard_normal((5, 4))
    # Decoder state at one time step
    s_dec = rng.standard_normal((1, 4))

    # Score via dot product: e_i = s_dec . h_i
    scores = (s_dec @ h_enc.T).squeeze(0)  # shape (5,)
    # Normalize into attention weights
    alpha = softmax(scores)                # shape (5,), sums to 1
    # Context vector: weighted sum of encoder states
    context = alpha @ h_enc                # shape (4,)

    print("Attention weights:", np.round(alpha, 3))
    print("Context vector:   ", np.round(context, 3))
    return (softmax,)


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
    **Code: Q, K, V projections and scaled dot-product attention in numpy.**
    """)
    return


@app.cell
def _(np, softmax):
    # Input: 3 tokens, embedding dim d=4, projection dim d_k=2
    X_sa = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [1, 1, 0, 0]], dtype=float)

    d_k_sa = 2
    # Random learned projection matrices (d x d_k)
    W_Q_sa = rng.standard_normal((4, d_k_sa)) * 0.5
    W_K_sa = rng.standard_normal((4, d_k_sa)) * 0.5
    W_V_sa = rng.standard_normal((4, d_k_sa)) * 0.5

    # Project: Q = X @ W_Q, etc.
    Q_sa = X_sa @ W_Q_sa   # (3, 2)
    K_sa = X_sa @ W_K_sa   # (3, 2)
    V_sa = X_sa @ W_V_sa   # (3, 2)

    # Scaled dot-product attention
    scores_sa = (Q_sa @ K_sa.T) / np.sqrt(d_k_sa)  # (3, 3) scaled scores
    weights_sa = softmax(scores_sa, axis=-1)         # row-wise softmax
    output_sa = weights_sa @ V_sa                    # (3, 2) attention output

    print("Attention weights (each row sums to 1):")
    print(np.round(weights_sa, 3))
    print("\nOutput (weighted mix of V):")
    print(np.round(output_sa, 3))
    return (output_sa, weights_sa,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Code: Demonstrating why the scaling factor matters.**
    """)
    return


@app.cell
def _(np, softmax):
    # Show softmax saturation without scaling
    d_k_demo = 512
    q_demo = rng.standard_normal((1, d_k_demo))
    K_demo = rng.standard_normal((8, d_k_demo))

    raw_dots = q_demo @ K_demo.T  # variance ~ d_k
    scaled_dots = raw_dots / np.sqrt(d_k_demo)

    print(f"Unscaled dot products — std: {raw_dots.std():.1f}")
    print(f"  softmax: {np.round(softmax(raw_dots), 4)}")
    print(f"\nScaled dot products — std: {scaled_dots.std():.1f}")
    print(f"  softmax: {np.round(softmax(scaled_dots), 4)}")
    print("\nNotice: unscaled softmax is nearly one-hot (peaked).")
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
    **Code: Reproducing the concrete numerical example above, step by step.**
    """)
    return


@app.cell
def _(np, softmax):
    # Exact matrices from the worked example
    X_ex = np.array([[1,0,1,0],[0,1,0,1],[1,1,0,0]], dtype=float)
    W_Q_ex = np.array([[1,0],[0,1],[1,0],[0,1]], dtype=float)
    W_K_ex = np.array([[0,1],[1,0],[0,1],[1,0]], dtype=float)
    W_V_ex = np.array([[1,1],[1,0],[0,1],[0,1]], dtype=float)

    Q_ex = X_ex @ W_Q_ex;  K_ex = X_ex @ W_K_ex;  V_ex = X_ex @ W_V_ex
    scores_ex = Q_ex @ K_ex.T                       # raw scores
    scaled_ex = scores_ex / np.sqrt(2)               # scale by sqrt(d_k)
    alpha_ex = softmax(scaled_ex, axis=-1)           # attention weights
    out_ex = alpha_ex @ V_ex                         # final output

    print("Q:\n", Q_ex)
    print("K:\n", K_ex)
    print("Scores (QK^T):\n", scores_ex)
    print("Attention weights:\n", np.round(alpha_ex, 2))
    print("Output (alpha @ V):\n", np.round(out_ex, 2))
    return


@app.cell
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
    **Code: Multi-head attention in numpy** --- split into heads, attend independently, concatenate.
    """)
    return


@app.cell
def _(np, softmax):
    def multihead_attention(X_mh, n_heads, W_Q_mh, W_K_mh, W_V_mh, W_O_mh):
        """Multi-head attention: X is (n, d), returns (n, d)."""
        n, d = X_mh.shape
        d_k_h = d // n_heads  # per-head dimension

        Q_mh = X_mh @ W_Q_mh  # (n, d)
        K_mh = X_mh @ W_K_mh
        V_mh = X_mh @ W_V_mh

        # Reshape into (n_heads, n, d_k) — split the last dim
        Q_h = Q_mh.reshape(n, n_heads, d_k_h).transpose(1, 0, 2)  # (h, n, d_k)
        K_h = K_mh.reshape(n, n_heads, d_k_h).transpose(1, 0, 2)
        V_h = V_mh.reshape(n, n_heads, d_k_h).transpose(1, 0, 2)

        # Scaled dot-product attention per head
        scores_mh = Q_h @ K_h.transpose(0, 2, 1) / np.sqrt(d_k_h)  # (h, n, n)
        attn_mh = softmax(scores_mh, axis=-1)
        head_outs = attn_mh @ V_h  # (h, n, d_k)

        # Concatenate heads: (n, h*d_k) = (n, d)
        concat = head_outs.transpose(1, 0, 2).reshape(n, d)
        return concat @ W_O_mh  # final projection (n, d)

    # Demo: 4 tokens, d=8, 2 heads
    _d, _h = 8, 2
    _X = rng.standard_normal((4, _d))
    _WQ = rng.standard_normal((_d, _d)) * 0.3
    _WK = rng.standard_normal((_d, _d)) * 0.3
    _WV = rng.standard_normal((_d, _d)) * 0.3
    _WO = rng.standard_normal((_d, _d)) * 0.3

    mha_out = multihead_attention(_X, _h, _WQ, _WK, _WV, _WO)
    print(f"Input shape:  {_X.shape}")
    print(f"Output shape: {mha_out.shape}")  # same (4, 8)
    return (multihead_attention,)


@app.cell
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
    **Code: Layer normalization in numpy** --- normalize each token's features to zero mean, unit variance.
    """)
    return


@app.cell
def _(np):
    def layer_norm(x, gamma=None, beta=None, eps=1e-5):
        """LayerNorm: normalize across the last (feature) dimension."""
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        if gamma is not None:
            x_norm = gamma * x_norm + beta
        return x_norm

    # Demo: 3 tokens, dim=4
    _x_ln = np.array([[2.0, 4.0, 6.0, 8.0],
                       [1.0, 1.0, 1.0, 1.0],
                       [0.0, 10., 0.0, 10.]], dtype=float)
    _normed = layer_norm(_x_ln)
    print("Before LayerNorm:\n", _x_ln)
    print("After LayerNorm:\n", np.round(_normed, 3))
    print("Row means (should be ~0):", np.round(_normed.mean(axis=-1), 6))
    return (layer_norm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Code: Position-wise FFN in numpy** --- expand to 4x width, apply ReLU, project back.
    """)
    return


@app.cell
def _(np):
    def ffn(x, W1, b1, W2, b2):
        """Position-wise FFN: x @ W1 + b1 -> ReLU -> @ W2 + b2."""
        hidden = np.maximum(0, x @ W1 + b1)  # ReLU activation
        return hidden @ W2 + b2

    # Demo: d=4, d_ff=16 (4x expansion)
    _d_ffn, _d_ff = 4, 16
    _W1 = rng.standard_normal((_d_ffn, _d_ff)) * 0.3
    _b1 = np.zeros(_d_ff)
    _W2 = rng.standard_normal((_d_ff, _d_ffn)) * 0.3
    _b2 = np.zeros(_d_ffn)

    _x_ffn = rng.standard_normal((3, _d_ffn))  # 3 tokens
    _out_ffn = ffn(_x_ffn, _W1, _b1, _W2, _b2)
    print(f"FFN input shape:  {_x_ffn.shape}")
    print(f"FFN output shape: {_out_ffn.shape}")  # same (3, 4)
    return (ffn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Code: A full Transformer encoder block in numpy** --- attention + residual + layernorm + FFN + residual + layernorm.
    """)
    return


@app.cell
def _(ffn, layer_norm, multihead_attention, np):
    def transformer_block_np(X_tb, n_heads_tb, params):
        """One encoder block: MHA -> Add&Norm -> FFN -> Add&Norm."""
        # Sub-layer 1: multi-head self-attention + residual + layernorm
        attn_out_tb = multihead_attention(
            X_tb, n_heads_tb,
            params["WQ"], params["WK"], params["WV"], params["WO"]
        )
        z_tb = layer_norm(X_tb + attn_out_tb)  # residual + norm

        # Sub-layer 2: FFN + residual + layernorm
        ffn_out_tb = ffn(z_tb, params["W1"], params["b1"], params["W2"], params["b2"])
        out_tb = layer_norm(z_tb + ffn_out_tb)  # residual + norm
        return out_tb

    # Init random params for d=8, 2 heads, d_ff=32
    _dtb = 8
    _params = {
        "WQ": rng.standard_normal((_dtb, _dtb))*0.2, "WK": rng.standard_normal((_dtb, _dtb))*0.2,
        "WV": rng.standard_normal((_dtb, _dtb))*0.2, "WO": rng.standard_normal((_dtb, _dtb))*0.2,
        "W1": rng.standard_normal((_dtb, 32))*0.2, "b1": np.zeros(32),
        "W2": rng.standard_normal((32, _dtb))*0.2, "b2": np.zeros(_dtb),
    }
    _x_tb = rng.standard_normal((5, _dtb))
    _out_tb = transformer_block_np(_x_tb, 2, _params)
    print(f"Encoder block: {_x_tb.shape} -> {_out_tb.shape}")
    return (transformer_block_np,)


@app.cell
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
    **Code: Sinusoidal positional encoding** --- each position gets a unique fingerprint of sin/cos waves at different frequencies.
    """)
    return


@app.cell
def _(np):
    def sinusoidal_pe(max_len, d_model):
        """PE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1) = cos(...)."""
        pe = np.zeros((max_len, d_model))
        pos = np.arange(max_len)[:, None]             # (max_len, 1)
        div = 10000 ** (np.arange(0, d_model, 2) / d_model)  # denominator
        pe[:, 0::2] = np.sin(pos / div)               # even dims
        pe[:, 1::2] = np.cos(pos / div)               # odd dims
        return pe

    pe_demo = sinusoidal_pe(50, 16)
    print(f"PE shape: {pe_demo.shape}  (50 positions, 16 dims)")
    print(f"PE[0,:4] = {np.round(pe_demo[0, :4], 3)}")
    print(f"PE[1,:4] = {np.round(pe_demo[1, :4], 3)}")
    print(f"PE[49,:4]= {np.round(pe_demo[49, :4], 3)}")

    # Each position has a unique encoding
    # The dot product between nearby positions is higher than distant ones
    sim_1_2 = pe_demo[1] @ pe_demo[2]
    sim_1_40 = pe_demo[1] @ pe_demo[40]
    print(f"\nDot product PE[1]·PE[2]  = {sim_1_2:.2f} (nearby)")
    print(f"Dot product PE[1]·PE[40] = {sim_1_40:.2f} (distant)")
    return (sinusoidal_pe,)


@app.cell
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
    **Code: Causal (look-ahead) mask and masked attention in numpy.**
    """)
    return


@app.cell
def _(np, softmax):
    def masked_attention(Q_m, K_m, V_m, causal=True):
        """Scaled dot-product attention with optional causal mask."""
        n_m, d_k_m = Q_m.shape
        scores_m = (Q_m @ K_m.T) / np.sqrt(d_k_m)

        if causal:
            # Upper triangle = -inf  ->  softmax(-inf) = 0
            mask_m = np.triu(np.full((n_m, n_m), -np.inf), k=1)
            scores_m = scores_m + mask_m

        weights_m = softmax(scores_m, axis=-1)
        return weights_m @ V_m, weights_m

    # 5 tokens, d_k=4
    _Q_m = rng.standard_normal((5, 4))
    _K_m = rng.standard_normal((5, 4))
    _V_m = rng.standard_normal((5, 4))

    _, causal_weights = masked_attention(_Q_m, _K_m, _V_m, causal=True)
    print("Causal attention weights (lower-triangular):")
    print(np.round(causal_weights, 2))
    print("\nUpper triangle is zero — no token sees the future.")
    return (masked_attention,)


@app.cell
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


@app.cell
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


@app.cell
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
    **Code: KV cache simulation** --- generate tokens one at a time, appending to cached keys/values.
    """)
    return


@app.cell
def _(np, softmax):
    # Simulate autoregressive generation with a KV cache
    _d_kv = 4
    _W_Q_kv = rng.standard_normal((_d_kv, _d_kv)) * 0.3
    _W_K_kv = rng.standard_normal((_d_kv, _d_kv)) * 0.3
    _W_V_kv = rng.standard_normal((_d_kv, _d_kv)) * 0.3

    K_cache = np.empty((0, _d_kv))  # start empty
    V_cache = np.empty((0, _d_kv))

    # Pretend we generate 5 tokens
    embeddings = rng.standard_normal((5, _d_kv))
    for t in range(5):
        x_t = embeddings[t:t+1]       # current token embedding (1, d)
        q_t = x_t @ _W_Q_kv           # query for this step only
        k_t = x_t @ _W_K_kv
        v_t = x_t @ _W_V_kv

        # Append to cache
        K_cache = np.vstack([K_cache, k_t])
        V_cache = np.vstack([V_cache, v_t])

        # Attend: q_t against all cached keys
        scores_kv = (q_t @ K_cache.T) / np.sqrt(_d_kv)  # (1, t+1)
        attn_kv = softmax(scores_kv, axis=-1)
        out_kv = attn_kv @ V_cache  # (1, d)

        print(f"Step {t}: attend over {K_cache.shape[0]} cached positions")

    print(f"\nFinal KV cache size: {K_cache.shape}")
    return


@app.cell
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


@app.cell
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

    ## Code It: Implementation Exercises

    Work through these exercises to build Transformer components from scratch in numpy. Each exercise gives you a skeleton with `TODO` placeholders to fill in.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 1: Scaled Dot-Product Attention from Scratch

    Implement the full attention formula: $\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$

    Fill in the four TODOs below. Use only numpy operations.
    """)
    return


@app.cell
def _(np):
    def attention_exercise(Q_ex1, K_ex1, V_ex1):
        """Implement scaled dot-product attention."""
        d_k_ex1 = Q_ex1.shape[-1]

        # TODO 1: Compute raw scores — dot product of Q and K^T
        scores_ex1 = None

        # TODO 2: Scale by sqrt(d_k)
        scaled_ex1 = None

        # TODO 3: Apply softmax row-wise (hint: subtract max for stability, then exp and normalize)
        e_x = np.exp(scaled_ex1 - np.max(scaled_ex1, axis=-1, keepdims=True))
        weights_ex1 = e_x / e_x.sum(axis=-1, keepdims=True)

        # TODO 4: Compute output — weighted sum of V
        output_ex1 = None

        return output_ex1, weights_ex1

    # Test with the identity matrices from Practice Exercise 6:
    _I = np.eye(2)
    # _out, _w = attention_exercise(_I, _I, _I)
    # print("Output:\n", np.round(_out, 3))
    # print("Weights:\n", np.round(_w, 3))
    return (attention_exercise,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 2: Multi-Head Attention

    Implement multi-head attention by splitting Q, K, V into `n_heads` chunks, running attention on each, and concatenating. Fill in the TODOs.
    """)
    return


@app.cell
def _(np):
    def multihead_exercise(X_ex2, n_heads_ex2, W_Q_ex2, W_K_ex2, W_V_ex2, W_O_ex2):
        """Multi-head attention from scratch."""
        n_ex2, d_ex2 = X_ex2.shape
        d_k_ex2 = d_ex2 // n_heads_ex2

        # Project inputs
        Q_ex2 = X_ex2 @ W_Q_ex2
        K_ex2 = X_ex2 @ W_K_ex2
        V_ex2 = X_ex2 @ W_V_ex2

        # TODO 1: Reshape Q into (n_heads, n, d_k)
        # Hint: reshape to (n, n_heads, d_k) then transpose axes
        Q_heads = None

        # TODO 2: Same for K and V
        K_heads = None
        V_heads = None

        # TODO 3: Compute attention for each head
        # scores shape: (n_heads, n, n), then softmax, then @ V_heads
        head_outputs = None  # shape: (n_heads, n, d_k)

        # TODO 4: Concatenate heads back to (n, d) and apply W_O
        # Hint: transpose to (n, n_heads, d_k) then reshape to (n, d)
        concat = None
        output_ex2 = None

        return output_ex2

    # Uncomment to test:
    # _d, _h = 8, 2
    # _X = rng.standard_normal((4, _d))
    # _out = multihead_exercise(_X, _h,
    #     rng.standard_normal((_d,_d))*0.3, rng.standard_normal((_d,_d))*0.3,
    #     rng.standard_normal((_d,_d))*0.3, rng.standard_normal((_d,_d))*0.3)
    # print("Output shape:", _out.shape)  # should be (4, 8)
    return (multihead_exercise,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 3: Sinusoidal Positional Encoding

    Implement the sinusoidal PE formula. Even dimensions get sin, odd dimensions get cos.

    $$PE_{(\text{pos}, 2i)} = \sin\!\left(\frac{\text{pos}}{10000^{2i/d}}\right), \quad PE_{(\text{pos}, 2i+1)} = \cos\!\left(\frac{\text{pos}}{10000^{2i/d}}\right)$$
    """)
    return


@app.cell
def _(np):
    def positional_encoding_exercise(max_len_ex3, d_model_ex3):
        """Build the sinusoidal positional encoding matrix."""
        pe_ex3 = np.zeros((max_len_ex3, d_model_ex3))
        positions = np.arange(max_len_ex3)[:, None]  # (max_len, 1)

        # TODO 1: Compute the denominator: 10000^(2i/d) for i = 0, 1, ..., d/2 - 1
        div_term = None

        # TODO 2: Fill even columns with sin(pos / div_term)
        pe_ex3[:, 0::2] = None

        # TODO 3: Fill odd columns with cos(pos / div_term)
        pe_ex3[:, 1::2] = None

        return pe_ex3

    # Uncomment to test:
    # _pe = positional_encoding_exercise(20, 8)
    # print("PE shape:", _pe.shape)
    # print("PE[0]:", np.round(_pe[0], 3))
    # print("PE[1]:", np.round(_pe[1], 3))
    return (positional_encoding_exercise,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Exercise 4: Causal Masking

    Implement masked self-attention for autoregressive decoding. The key step: add $-\infty$ to scores for positions where $j > i$ before softmax.
    """)
    return


@app.cell
def _(np):
    def causal_attention_exercise(Q_ex4, K_ex4, V_ex4):
        """Scaled dot-product attention with causal mask."""
        n_ex4, d_k_ex4 = Q_ex4.shape

        # Compute scaled scores
        scores_ex4 = (Q_ex4 @ K_ex4.T) / np.sqrt(d_k_ex4)

        # TODO 1: Create causal mask — upper triangle filled with -inf
        # Hint: use np.triu with k=1 to get the strict upper triangle
        causal_mask_ex4 = None

        # TODO 2: Add the mask to scores
        masked_scores_ex4 = None

        # TODO 3: Apply softmax row-wise
        e_x4 = np.exp(masked_scores_ex4 - np.max(masked_scores_ex4, axis=-1, keepdims=True))
        weights_ex4 = e_x4 / e_x4.sum(axis=-1, keepdims=True)

        return weights_ex4 @ V_ex4, weights_ex4

    # Uncomment to test:
    # _Q = rng.standard_normal((4, 3))
    # _K = rng.standard_normal((4, 3))
    # _V = rng.standard_normal((4, 3))
    # _, _w = causal_attention_exercise(_Q, _K, _V)
    # print("Causal weights (upper triangle should be 0):")
    # print(np.round(_w, 3))
    return (causal_attention_exercise,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 5: Full Transformer Encoder Block

    Combine everything: multi-head attention, residual connections, layer normalization, and FFN into one encoder block. This is the core repeating unit of the Transformer.
    """)
    return


@app.cell
def _(np):
    def encoder_block_exercise(X_ex5, n_heads_ex5, W_Q_ex5, W_K_ex5, W_V_ex5, W_O_ex5,
                                W1_ex5, b1_ex5, W2_ex5, b2_ex5):
        """One Transformer encoder block: MHA -> Add&Norm -> FFN -> Add&Norm."""

        def _layer_norm(x_ln):
            mu = x_ln.mean(axis=-1, keepdims=True)
            var = x_ln.var(axis=-1, keepdims=True)
            return (x_ln - mu) / np.sqrt(var + 1e-5)

        def _softmax(x_sm, axis=-1):
            e = np.exp(x_sm - np.max(x_sm, axis=axis, keepdims=True))
            return e / e.sum(axis=axis, keepdims=True)

        # TODO 1: Compute multi-head attention output (you can reuse your multihead logic)
        # For simplicity, you can do single-head attention: Q=X@WQ, K=X@WK, V=X@WV
        # scores -> scale -> softmax -> @ V -> @ W_O
        attn_out = None

        # TODO 2: Residual connection + layer norm
        z_ex5 = None

        # TODO 3: FFN — W1 with ReLU, then W2
        ffn_out = None

        # TODO 4: Second residual connection + layer norm
        out_ex5 = None

        return out_ex5

    # Uncomment to test:
    # _d = 8
    # _x = rng.standard_normal((5, _d))
    # _out = encoder_block_exercise(_x, 1,
    #     rng.standard_normal((_d,_d))*0.2, rng.standard_normal((_d,_d))*0.2,
    #     rng.standard_normal((_d,_d))*0.2, rng.standard_normal((_d,_d))*0.2,
    #     rng.standard_normal((_d,32))*0.2, np.zeros(32),
    #     rng.standard_normal((32,_d))*0.2, np.zeros(_d))
    # print("Output shape:", _out.shape)  # should be (5, 8)
    return (encoder_block_exercise,)


@app.cell
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
