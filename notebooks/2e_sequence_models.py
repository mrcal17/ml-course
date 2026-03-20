import marimo

app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
# Sequence Models

In the last module you learned how CNNs exploit spatial structure — local patterns, translation invariance, hierarchical features. That was the right inductive bias for images. But the world is full of data where the dominant structure is not spatial but *temporal*. Sentences unfold word by word. Stock prices evolve tick by tick. Audio is a pressure wave sampled thousands of times per second. DNA is a sequence of nucleotides whose order encodes the instructions for life.

For all of these, **order matters**. Shuffle the pixels in an image and a CNN can still (with some effort) learn features. Shuffle the words in a sentence and you destroy meaning entirely. "The cat sat on the mat" is not the same as "mat the on sat cat the." Sequential data demands architectures that respect this ordering — architectures with *memory*.

This module builds recurrent neural networks from first principles, diagnoses their fundamental flaw (vanishing gradients), and then shows how gated architectures like LSTMs and GRUs fix it. We will finish with encoder-decoder models and a preview of the attention mechanism that eventually superseded everything here.

> **Reading:** [Goodfellow et al., Ch 10: Sequence Modeling](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) is the primary reference. [Murphy PML2](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML2.pdf) covers modern perspectives on sequence models. [Bishop, PRML](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) provides additional context on graphical model views of sequences.
""")
    return


@app.cell
def _(mo):
    mo.md(r"""
---

## 1. Why Sequences Are Different

Consider a feedforward network that takes a sentence as input. You could one-hot encode each word, concatenate them into one big vector, and feed it through dense layers. This approach has two fatal problems.

**Variable length.** Sentences have different numbers of words. Feedforward networks require fixed-size inputs. You could pad to some maximum length, but that wastes computation on short sequences and truncates long ones.

**No parameter sharing across positions.** If the network learns that "Monday" at position 3 means something, it has no way to transfer that knowledge to "Monday" at position 7. The weights connecting position 3 to the hidden layer are completely separate from those connecting position 7. CNNs solved a version of this for spatial data with weight sharing across spatial locations. We need something analogous for sequential positions.

**No long-range dependency modeling.** In the sentence "The students who aced the exam were thrilled," the verb "were" must agree with "students" — not "exam," which is closer. Capturing such dependencies requires the model to carry information across many time steps.

What we need is a network that:
1. Processes inputs one step at a time
2. Maintains a **hidden state** that summarizes what it has seen so far
3. Shares parameters across all time steps
4. Can, in principle, relate any time step to any other

This is exactly what recurrent neural networks provide. See [DLBook, Section 10.1](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) for the formal setup.
""")
    return


@app.cell
def _(mo):
    mo.md(r"""
---

## 2. Vanilla Recurrent Neural Networks

### 2.1 The Recurrence Relation

An RNN processes a sequence $(x_1, x_2, \ldots, x_T)$ one element at a time, maintaining a hidden state $h_t$ that gets updated at each step:

$$h_t = \tanh(W_{hh}\, h_{t-1} + W_{xh}\, x_t + b_h)$$

$$y_t = W_{hy}\, h_t + b_y$$

That is the entire architecture. Three weight matrices — $W_{xh}$ (input to hidden), $W_{hh}$ (hidden to hidden), and $W_{hy}$ (hidden to output) — shared across *every* time step. The hidden state $h_t$ is a vector that serves as the network's memory: it encodes a compressed summary of everything the network has processed from $x_1$ through $x_t$.

The nonlinearity is typically $\tanh$ (though ReLU variants exist). The initial hidden state $h_0$ is usually set to zeros, though it can also be learned.

Think about what this means. At time step 1, $h_1$ depends on $x_1$. At time step 2, $h_2$ depends on $x_2$ and $h_1$ — which itself depends on $x_1$. By time step $t$, $h_t$ is a function of the entire input history $(x_1, \ldots, x_t)$. The hidden state is a lossy compression of the past. See [DLBook, Section 10.2](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) for the formal treatment.

### 2.2 Unrolling Through Time

Here is the key insight: if you "unroll" the recurrence by writing out each time step explicitly, an RNN is just a very deep feedforward network with **shared weights**. Step 1 feeds into step 2 feeds into step 3, and so on — exactly like layers in a deep network, except every "layer" uses the same $W_{hh}$, $W_{xh}$, and $b_h$.

```
x_1 -> [RNN Cell] -> h_1 -> [RNN Cell] -> h_2 -> [RNN Cell] -> h_3 -> ...
          ^                    ^                    ^
         h_0                  h_1                  h_2
```

This is why we can train RNNs with standard backpropagation — once unrolled, there is nothing special about the computation graph. The only novelty is that weight gradients must be **accumulated** across all time steps (since the same weights appear at every step).

This unrolling view also immediately reveals the fundamental problem, which we will get to shortly. See [DLBook, Section 10.1: Unfolding Computational Graphs](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf).
""")
    return


@app.cell
def _(mo):
    mo.md(r"""
### RNN Unrolling Animation

The following animation illustrates how an RNN is unrolled through time, showing the shared weights at each step:
""")
    return


@app.cell
def _(mo):
    mo.image(src="../animations/rendered/RNNUnrolling.gif")
    return


@app.cell
def _(mo):
    mo.md(r"""
### 2.3 A Concrete Example

Let me walk through a tiny RNN to make this concrete. Suppose we have a hidden size of 2, input size of 1, and the following (made-up) weights:

$$W_{hh} = \begin{bmatrix} 0.5 & 0.1 \\ 0.2 & 0.3 \end{bmatrix}, \quad W_{xh} = \begin{bmatrix} 0.4 \\ 0.7 \end{bmatrix}, \quad b_h = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

Start with $h_0 = [0, 0]^T$ and process the input sequence $x = (1.0, 0.5, -0.3)$.

**Step 1** ($x_1 = 1.0$):

$$h_1 = \tanh\!\left(\begin{bmatrix} 0.5 & 0.1 \\ 0.2 & 0.3 \end{bmatrix}\begin{bmatrix} 0 \\ 0 \end{bmatrix} + \begin{bmatrix} 0.4 \\ 0.7 \end{bmatrix}(1.0)\right) = \tanh\begin{bmatrix} 0.4 \\ 0.7 \end{bmatrix} = \begin{bmatrix} 0.380 \\ 0.604 \end{bmatrix}$$

**Step 2** ($x_2 = 0.5$):

$$h_2 = \tanh\!\left(\begin{bmatrix} 0.5 & 0.1 \\ 0.2 & 0.3 \end{bmatrix}\begin{bmatrix} 0.380 \\ 0.604 \end{bmatrix} + \begin{bmatrix} 0.4 \\ 0.7 \end{bmatrix}(0.5)\right) = \tanh\begin{bmatrix} 0.450 \\ 0.607 \end{bmatrix} = \begin{bmatrix} 0.422 \\ 0.542 \end{bmatrix}$$

**Step 3** ($x_3 = -0.3$): The same process continues. Notice how $h_3$ depends on $x_3$ directly through $W_{xh}$, but also on $x_1$ and $x_2$ indirectly through the accumulated hidden state. The hidden state carries forward a compressed representation of everything the network has seen.
""")
    return


@app.cell
def _(mo):
    mo.md(r"""
### 2.4 Backpropagation Through Time (BPTT)

Training an RNN means computing gradients of the loss with respect to $W_{hh}$, $W_{xh}$, and $W_{hy}$. Since the same weights are used at every time step, the gradient is the sum of contributions from each step.

Suppose we have a loss $L = \sum_{t=1}^{T} L_t$ (a loss at each time step). By the chain rule:

$$\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W_{hh}}$$

For a single time step's contribution:

$$\frac{\partial L_t}{\partial W_{hh}} = \sum_{k=1}^{t} \frac{\partial L_t}{\partial h_t} \left(\prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}}\right) \frac{\partial h_k}{\partial W_{hh}}$$

Look at that product term $\prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}}$. Each factor is the Jacobian of $h_j$ with respect to $h_{j-1}$, which involves $W_{hh}$ and the derivative of the activation function. When $t - k$ is large — when we are trying to propagate gradient information many steps back — we are multiplying many matrices together.

This is the source of everything that follows.
""")
    return


@app.cell
def _(mo):
    mo.md(r"""
---

## 3. The Vanishing and Exploding Gradient Problem

### 3.1 Why It Happens

Each Jacobian $\frac{\partial h_j}{\partial h_{j-1}}$ has the form:

$$\frac{\partial h_j}{\partial h_{j-1}} = \text{diag}(\tanh'(z_j)) \cdot W_{hh}$$

where $z_j$ is the pre-activation and $\tanh'$ outputs values in $(0, 1]$. The product of $t - k$ such terms behaves like $W_{hh}^{t-k}$ (modulated by the activation derivatives).

Now think about eigenvalues. If the largest singular value of $W_{hh}$ is greater than 1, repeated multiplication makes the product grow exponentially — **exploding gradients**. If the largest singular value is less than 1, the product shrinks exponentially — **vanishing gradients**.

This is not a hypothetical concern. For a sequence of length 100, you are multiplying roughly 100 matrices together. Even a singular value of 0.9 gives $0.9^{100} \approx 2.7 \times 10^{-5}$. A singular value of 1.1 gives $1.1^{100} \approx 1.4 \times 10^{4}$. The dynamic range is absurd.

See [DLBook, Section 10.7: The Challenge of Long-Term Dependencies](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) for the rigorous analysis.

### 3.2 Exploding Gradients: The Easy Fix

Exploding gradients are loud — your loss goes to NaN. The standard fix is **gradient clipping**: if the global norm of the gradient exceeds a threshold $\theta$, rescale it:

$$\hat{g} \leftarrow \frac{\theta}{\|g\|} g \quad \text{if } \|g\| > \theta$$

This keeps the gradient direction but limits its magnitude. It works remarkably well in practice. Typical values of $\theta$ are between 1 and 5.

### 3.3 Vanishing Gradients: The Hard Problem

Vanishing gradients are silent. The network trains, the loss decreases — but it only learns short-range dependencies. Information from 20 steps ago simply cannot influence the gradient at the current step, because the gradient signal has decayed to near zero.

This is what limits vanilla RNNs to effective memory of roughly 10-20 time steps. For many real-world problems — language modeling, machine translation, long-horizon time series — this is completely inadequate.

You cannot fix this with gradient clipping. You cannot fix it with better initialization (though orthogonal initialization of $W_{hh}$ helps somewhat). You cannot fix it with ReLU activations (you just trade vanishing for exploding). The fundamental issue is architectural: information is being repeatedly squashed through a nonlinearity and multiplied by the same matrix.

The solution requires rethinking how information flows through time. See [DLBook, Section 10.9: Leaky Units and Other Strategies](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf).
""")
    return


@app.cell
def _(mo):
    mo.md(r"""
---

## 4. Long Short-Term Memory (LSTM)

Hochreiter and Schmidhuber (1997) solved the vanishing gradient problem with an architectural innovation that remains one of the most important ideas in deep learning. The key insight: **let gradients flow through an additive path**.

### 4.1 The Cell State: A Conveyor Belt

An LSTM introduces a **cell state** $c_t$ alongside the hidden state $h_t$. Think of the cell state as a conveyor belt running through time. Information can be placed onto it, read from it, or removed from it — but by default, it just flows forward unchanged. Because the default operation is *addition* (not multiplication by a weight matrix), gradients can flow backward through time without vanishing.

### 4.2 The Three Gates

The LSTM controls information flow with three gates, each of which is itself a small neural network with a sigmoid activation (outputting values between 0 and 1):

**Forget gate** — decides what to remove from the cell state:
$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$$

**Input gate** — decides what new information to add:
$$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$$
$$\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c) \quad \text{(candidate cell state)}$$

**Output gate** — decides what to output as the hidden state:
$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$$

### 4.3 The Update Equations

Putting it together:

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

$$h_t = o_t \odot \tanh(c_t)$$

where $\odot$ denotes element-wise multiplication.

Read those two equations carefully. The cell state update is **additive**: $c_t$ is a weighted combination of the old cell state $c_{t-1}$ and new candidate content $\tilde{c}_t$. The forget gate $f_t$ controls how much of the old state to keep. The input gate $i_t$ controls how much new information to write. If $f_t = 1$ and $i_t = 0$, the cell state passes through unchanged — gradients flow perfectly.

This is why LSTMs solve the vanishing gradient problem. The gradient of $c_t$ with respect to $c_{t-1}$ is:

$$\frac{\partial c_t}{\partial c_{t-1}} = f_t$$

No weight matrix multiplication. No squashing nonlinearity. Just the forget gate, which the network learns to keep close to 1 when long-term memory matters. Compare this to the vanilla RNN where the corresponding term was $\text{diag}(\tanh'(z)) \cdot W_{hh}$ — a product that inevitably vanishes or explodes.

### 4.4 Intuition for Each Gate

Think about processing the sentence: "The movie, despite some awkward scenes in the middle, was absolutely brilliant."

- When processing "The movie," the input gate opens to write "movie" (the subject) into the cell state.
- During "despite some awkward scenes in the middle," the forget gate stays close to 1 — do not erase the subject. The input gate might write some information about the negative clause, but the core subject memory persists.
- When reaching "was," the network can read from the cell state to determine that the subject is "movie" (singular) and predict "was" rather than "were."
- When reaching "brilliant," the output gate selects the sentiment-relevant information.

### 4.5 Peephole Connections

A common variant adds **peephole connections** where the gates also receive the cell state as input:

$$f_t = \sigma(W_f [h_{t-1}, x_t] + w_{f,\text{peep}} \odot c_{t-1} + b_f)$$

This lets the gates make decisions based on what is actually stored in the cell, not just the previous output. In practice, peepholes help occasionally but are not standard. Most implementations omit them.

> **Reading:** [DLBook, Section 10.10: The Long Short-Term Memory and Other Gated RNNs](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) for the complete formal treatment.
""")
    return


@app.cell
def _(mo):
    mo.md(r"""
---

## 5. Gated Recurrent Unit (GRU)

Cho et al. (2014) proposed a simplified gating mechanism that achieves comparable performance with fewer parameters.

### 5.1 The Two Gates

The GRU uses only two gates:

**Update gate** — analogous to combining the LSTM's forget and input gates:
$$z_t = \sigma(W_z [h_{t-1}, x_t] + b_z)$$

**Reset gate** — controls how much of the previous hidden state to expose:
$$r_t = \sigma(W_r [h_{t-1}, x_t] + b_r)$$

### 5.2 The Update

$$\tilde{h}_t = \tanh(W_h [r_t \odot h_{t-1}, x_t] + b_h)$$

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

Notice the elegance. The update gate $z_t$ interpolates between keeping the old state ($z_t = 0$) and replacing it with the candidate ($z_t = 1$). This is a single gate doing the job of both the forget and input gates — with the constraint that they must sum to 1. The reset gate $r_t$ controls how much of $h_{t-1}$ is used to compute the candidate.

There is no separate cell state. The hidden state itself carries long-term information.

### 5.3 GRU vs. LSTM in Practice

The empirical evidence is clear: GRU and LSTM perform comparably on most tasks. Neither consistently dominates the other. GRU has fewer parameters (two gates instead of three, no separate cell state), so it trains faster and is preferable when compute is limited. LSTM has more expressive capacity through the independent forget and input gates.

My practical advice: try both. If your sequences are very long and you need fine-grained memory control, lean toward LSTM. If you want a simpler model and faster iteration, start with GRU.
""")
    return


@app.cell
def _(mo):
    mo.md(r"""
---

## 6. Bidirectional RNNs

A standard RNN processes the sequence left-to-right. At time step $t$, the hidden state $h_t$ encodes information about $x_1, \ldots, x_t$ — everything *before and including* the current step. But it knows nothing about what comes *after*.

For many tasks, the future context matters as much as the past. In the sentence "He went to the ___," knowing that the next word is "bank" and the word after is "to deposit money" (vs. "to fish") completely changes the meaning.

A **bidirectional RNN** runs two independent RNNs over the sequence:
- A forward RNN producing $\overrightarrow{h}_t$ from $(x_1, \ldots, x_t)$
- A backward RNN producing $\overleftarrow{h}_t$ from $(x_T, \ldots, x_t)$

The representations are concatenated: $h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t]$.

This doubles the hidden state size but gives each position access to the full sequence context. Bidirectional LSTMs (BiLSTMs) were the dominant architecture for NLP tasks like named entity recognition and sentiment analysis from roughly 2015-2018.

**When bidirectional works:** Any task where you have the complete input sequence available at once — classification, tagging, encoding for translation.

**When it does not work:** Autoregressive generation (predicting the next word), real-time streaming (processing audio as it arrives), any setting where you cannot look ahead.

> **Reading:** [DLBook, Section 10.3: Bidirectional RNNs](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf).
""")
    return


@app.cell
def _(mo):
    mo.md(r"""
---

## 7. Sequence-to-Sequence Models

### 7.1 The Encoder-Decoder Architecture

Many important problems map a *variable-length input sequence* to a *variable-length output sequence*: machine translation (English sentence to French sentence), text summarization (article to summary), speech recognition (audio to text). The input and output lengths are generally different and unknown in advance.

Sutskever et al. (2014) introduced the **encoder-decoder** (or seq2seq) framework:

1. An **encoder** RNN reads the input sequence $(x_1, \ldots, x_T)$ and produces a final hidden state $h_T$. This vector — the **context vector** $c$ — is a fixed-length summary of the entire input.

2. A **decoder** RNN takes $c$ as its initial hidden state and generates the output sequence $(y_1, y_2, \ldots)$ one token at a time. At each step, it takes the previous output $y_{t-1}$ as input and produces the next output $y_t$.

The entire system is trained end-to-end with teacher forcing (we will discuss this below).

### 7.2 The Bottleneck Problem

There is a fundamental limitation: the entire input sequence must be compressed into a single fixed-dimensional vector $c$. For short sequences this works fine, but for long sequences — a paragraph, a document — too much information is lost in the compression.

Empirically, seq2seq performance degrades noticeably for input sequences beyond about 20-30 tokens. This is a representational bottleneck, not a gradient problem.

The solution — **attention** — lets the decoder look back at *all* encoder hidden states $(h_1, \ldots, h_T)$ rather than relying solely on $c$. This is the subject of the next module (Attention & Transformers) and it fundamentally changed the field. For now, just understand that the bottleneck exists and why it is limiting.

> **Reading:** [DLBook, Section 10.4: Encoder-Decoder Sequence-to-Sequence Architectures](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf).
""")
    return


@app.cell
def _(mo):
    mo.md(r"""
---

## 8. Practical Considerations

### 8.1 Truncated BPTT

For long sequences (hundreds or thousands of steps), full BPTT is computationally prohibitive — you must store all intermediate hidden states, and the backward pass takes as long as the forward pass. **Truncated BPTT** processes the sequence in chunks of length $k$ (typically 20-100 steps), backpropagating only within each chunk.

The hidden state is carried forward between chunks (maintaining the RNN's memory), but gradients are not propagated across chunk boundaries. This introduces bias — the network cannot learn dependencies longer than $k$ steps — but it makes training tractable.

### 8.2 Teacher Forcing

During training of a sequence-to-sequence model, the decoder must generate tokens one at a time. At each step, should its input be the *actual previous ground-truth token* or the *model's own predicted token*?

**Teacher forcing** uses the ground-truth token. This makes training stable and fast: the model always conditions on correct context, so errors do not compound. But it creates a train-test mismatch: at inference time, the model must condition on its own predictions, which may be wrong. This is called **exposure bias**.

Mitigation strategies include **scheduled sampling** (randomly using predictions instead of ground truth during training, with increasing probability over time) and curriculum learning.

### 8.3 Beam Search

At inference time, greedily picking the highest-probability token at each step is suboptimal — the locally best choice may lead to a globally poor sequence. **Beam search** maintains the top $B$ candidate sequences (beams) at each step, expanding each by all possible next tokens and keeping the top $B$ overall.

Typical beam widths are 4-10. Wider beams give better results but are linearly more expensive. Beam search was essential for neural machine translation and remained important until large-scale autoregressive transformers made greedy/sampling-based decoding competitive.

### 8.4 Implementation Details in PyTorch

When processing batches of variable-length sequences, you need to handle **padding** and **packing**. PyTorch provides `torch.nn.utils.rnn.pack_padded_sequence` and `pad_packed_sequence` for this. Packed sequences tell the RNN to skip padding tokens, which is both faster and avoids contaminating the hidden state with padding information.
""")
    return


@app.cell
def _(mo):
    mo.md(r"""
---

## 9. PyTorch Implementation

Let me show you a complete, minimal character-level language model using both vanilla RNN and LSTM. This is the "hello world" of sequence modeling.

### 9.1 Vanilla RNN
""")
    return


@app.cell
def _():
    import torch
    import torch.nn as nn

    class CharRNN(nn.Module):
        def __init__(self, vocab_size, hidden_size, num_layers=1):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, vocab_size)

        def forward(self, x, h=None):
            # x: (batch, seq_len) of token indices
            # h: (num_layers, batch, hidden_size) or None
            if h is None:
                h = torch.zeros(self.num_layers, x.size(0), self.hidden_size,
                              device=x.device)

            emb = self.embedding(x)           # (batch, seq_len, hidden_size)
            out, h_new = self.rnn(emb, h)     # out: (batch, seq_len, hidden_size)
            logits = self.fc(out)             # (batch, seq_len, vocab_size)
            return logits, h_new

    # Usage:
    model_rnn = CharRNN(vocab_size=65, hidden_size=128, num_layers=2)
    x = torch.randint(0, 65, (32, 50))        # batch of 32, sequence length 50
    logits, hidden = model_rnn(x)              # logits: (32, 50, 65)
    print(f"CharRNN output shape: {logits.shape}")
    print(f"Hidden state shape: {hidden.shape}")
    return


@app.cell
def _(mo):
    mo.md(r"""
### 9.2 LSTM Version

Swapping `nn.RNN` for `nn.LSTM` is almost trivial — the only difference is that the hidden state is now a tuple `(h, c)`:
""")
    return


@app.cell
def _():
    import torch as _torch
    import torch.nn as _nn

    class CharLSTM(_nn.Module):
        def __init__(self, vocab_size, hidden_size, num_layers=1):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            self.embedding = _nn.Embedding(vocab_size, hidden_size)
            self.lstm = _nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
            self.fc = _nn.Linear(hidden_size, vocab_size)

        def forward(self, x, state=None):
            if state is None:
                h = _torch.zeros(self.num_layers, x.size(0), self.hidden_size,
                              device=x.device)
                c = _torch.zeros_like(h)
                state = (h, c)

            emb = self.embedding(x)
            out, state_new = self.lstm(emb, state)
            logits = self.fc(out)
            return logits, state_new

    # Usage:
    model_lstm = CharLSTM(vocab_size=65, hidden_size=256, num_layers=2)
    x_test = _torch.randint(0, 65, (32, 50))
    logits_lstm, state_lstm = model_lstm(x_test)
    print(f"CharLSTM output shape: {logits_lstm.shape}")
    print(f"Hidden state shapes: h={state_lstm[0].shape}, c={state_lstm[1].shape}")
    return


@app.cell
def _(mo):
    mo.md(r"""
The training loop for these models would include two critical details:

1. `state = tuple(s.detach() for s in state)` — this implements truncated BPTT by detaching the hidden state from the computation graph between chunks. Without this, PyTorch would try to backpropagate through the *entire* sequence, eventually running out of memory.
2. `clip_grad_norm_` — gradient clipping is not optional for RNN training.
""")
    return


@app.cell
def _():
    import torch as _t2
    import torch.nn as _n2

    # Training loop sketch for CharLSTM:
    vocab_size = 65
    # model = CharLSTM(vocab_size=vocab_size, hidden_size=256, num_layers=2)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # criterion = nn.CrossEntropyLoss()
    #
    # for epoch in range(num_epochs):
    #     state = None
    #     for batch_x, batch_y in dataloader:
    #         logits, state = model(batch_x, state)
    #
    #         # Detach hidden state to implement truncated BPTT
    #         state = tuple(s.detach() for s in state)
    #
    #         loss = criterion(logits.view(-1, vocab_size), batch_y.view(-1))
    #         optimizer.zero_grad()
    #         loss.backward()
    #
    #         # Gradient clipping — essential for RNN training
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    #
    #         optimizer.step()

    print("Training loop key points:")
    print("1. Detach hidden state between chunks (truncated BPTT)")
    print("2. Gradient clipping (clip_grad_norm_) is essential")
    print("3. Loss computed per-token: logits.view(-1, vocab_size)")
    return


@app.cell
def _(mo):
    mo.md(r"""
### 9.3 Bidirectional LSTM
""")
    return


@app.cell
def _():
    import torch as _t3
    import torch.nn as _n3

    class BiLSTMClassifier(_n3.Module):
        def __init__(self, vocab_size, embed_dim, hidden_size, num_classes):
            super().__init__()
            self.embedding = _n3.Embedding(vocab_size, embed_dim)
            self.lstm = _n3.LSTM(embed_dim, hidden_size, num_layers=2,
                               batch_first=True, bidirectional=True)
            # hidden_size * 2 because bidirectional concatenates both directions
            self.fc = _n3.Linear(hidden_size * 2, num_classes)

        def forward(self, x):
            emb = self.embedding(x)
            out, (h, c) = self.lstm(emb)
            # Use the last time step's output (contains both directions)
            logits = self.fc(out[:, -1, :])
            return logits

    model_bi = BiLSTMClassifier(vocab_size=10000, embed_dim=128, hidden_size=256, num_classes=5)
    x_bi = _t3.randint(0, 10000, (16, 100))  # batch of 16, sequence length 100
    out_bi = model_bi(x_bi)
    print(f"BiLSTM classifier output shape: {out_bi.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model_bi.parameters()):,}")
    return


@app.cell
def _(mo):
    mo.md(r"""
Setting `bidirectional=True` is all it takes. The output hidden size doubles because forward and backward states are concatenated.
""")
    return


@app.cell
def _(mo):
    mo.md(r"""
---

## 10. Historical Context and What Came Next

### 10.1 The RNN Era (2014-2018)

RNNs — specifically LSTMs and BiLSTMs — were the dominant architecture for virtually all sequence tasks during this period. The key milestones:

- **2014:** Sutskever et al. demonstrated seq2seq for machine translation. Cho et al. introduced the GRU.
- **2015:** Bahdanau et al. added attention to seq2seq, eliminating the bottleneck problem. BiLSTMs became standard for NLP tasks.
- **2016-2017:** Stacked LSTMs with attention achieved state-of-the-art on machine translation, question answering, and summarization.
- **2017:** "Attention Is All You Need" (Vaswani et al.) introduced the Transformer, which replaced recurrence entirely with self-attention.
- **2018:** BERT and GPT demonstrated that transformers scaled to massive datasets and transfer learning, effectively ending the RNN era for most NLP tasks.

### 10.2 Why Transformers Won

The fundamental weakness of RNNs is **sequential computation**. To process position $t$, you must first process positions $1$ through $t-1$. This means:
- Training cannot be parallelized across time steps (GPUs sit idle)
- Very long sequences are slow regardless of hardware
- The hidden state is a *fixed-size bottleneck* through which all information must flow

Transformers process all positions in parallel using self-attention, with direct connections between any two positions regardless of distance. This eliminated both the parallelization problem and the long-range dependency problem in one stroke.

### 10.3 Where RNNs Still Matter

Despite the transformer revolution, recurrent architectures remain relevant in specific settings:

- **Real-time streaming:** Processing audio or sensor data as it arrives, where you cannot wait for the full sequence. RNNs naturally operate in this online mode.
- **Resource-constrained deployment:** An LSTM is far smaller than a transformer for the same task. On edge devices, this matters.
- **Some time series tasks:** When the sequential inductive bias is genuinely correct and data is limited, RNNs can outperform transformers.
- **State space models:** Mamba (Gu and Dao, 2023) and similar architectures are essentially linear RNNs with carefully structured weight matrices. They achieve transformer-level performance with linear (not quadratic) scaling in sequence length. RWKV is another example — a "linear attention" model that can be computed as an RNN.

These modern recurrent models represent a remarkable full-circle moment: the field abandoned RNNs for transformers, then discovered that carefully designed recurrent architectures can match transformers while being more efficient. Understanding classical RNNs and LSTMs is essential background for appreciating why this line of research is exciting.

> **Reading:** [DLBook, Section 10.11-10.12](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) covers optimization strategies for long-term dependencies and explicit memory architectures. [Murphy PML2](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML2.pdf) provides a more modern perspective that includes attention and transformers.
""")
    return


@app.cell
def _(mo):
    mo.md(r"""
---

## Practice

1. **By hand:** Given $W_{hh} = \begin{bmatrix} 0.8 & 0 \\ 0 & 0.3 \end{bmatrix}$, compute $W_{hh}^{10}$ and $W_{hh}^{50}$. Explain what happens to gradient information flowing through each eigenvalue's direction over 10 and 50 time steps. Why does the second component's gradient effectively vanish while the first does not?

2. **Conceptual:** Write out the LSTM equations for a single time step. Trace what happens when $f_t = 1$, $i_t = 0$ for all $t$. What is $c_T$? What is $\frac{\partial c_T}{\partial c_1}$? Why does this matter for gradient flow?

3. **Conceptual:** Explain why bidirectional RNNs cannot be used for autoregressive language modeling (predicting the next word). What would go wrong?

4. **Implementation:** Modify the CharLSTM code above to use a GRU instead. Train both on a small text corpus (Shakespeare works well — look up `torchtext` datasets or just download a plain text file). Compare training curves and generation quality. Which converges faster?

5. **Implementation:** Build a simple seq2seq model for reversing strings (input: "hello", target: "olleh"). This is a toy task but it forces you to implement the encoder-decoder pattern. Start with a fixed-length context vector, then add attention and compare.

6. **Analysis:** Train a vanilla RNN and an LSTM on sequences of increasing length (20, 50, 100, 200 tokens). Plot the gradient norms flowing to the first time step as a function of sequence length. You should see the vanilla RNN's gradient vanish while the LSTM's remains stable. This is the single most important experiment in this module — it will make the theory visceral.

7. **Reading:** Work through [DLBook, Sections 10.2-10.4](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) carefully. Pay particular attention to the computational graph diagrams — they make the unrolling and gradient flow explicit in a way that equations alone cannot.
""")
    return


if __name__ == "__main__":
    app.run()
