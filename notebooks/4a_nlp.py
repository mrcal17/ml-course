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
    # Path A --- Natural Language Processing

    Natural language processing sits at the intersection of linguistics, computer science, and machine learning. It is arguably the subfield that has seen the most dramatic transformation in the deep learning era --- and the one most responsible for the current wave of public interest in AI. This guide maps out the landscape, highlights the concepts that matter most, and points you toward resources and projects that will build real competency.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 1. The NLP Landscape: A Brief History

    NLP has passed through three distinct eras, each layered on top of the last:

    **Rule-based systems (1950s--1990s).** Hand-written grammars, pattern matching, expert systems. These worked in narrow domains but were brittle. Think of early chatbots like ELIZA (1966) --- impressive as demos, useless in production.

    **Statistical and feature-engineered NLP (1990s--2013).** Bag-of-words, TF-IDF, n-gram language models, Hidden Markov Models for POS tagging, conditional random fields for NER. The shift was from hand-written rules to learning patterns from data, but humans still decided *which* features to extract. See [Bishop Ch. 1](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) for the philosophical shift from rules to probabilistic modeling.

    **Neural NLP (2013--present).** Word2Vec (2013) showed that neural networks could learn useful word representations. Sequence-to-sequence models (2014) enabled machine translation. Attention (2015) removed the information bottleneck. Transformers (2017) changed everything. GPT and BERT (2018) demonstrated that pre-training on massive corpora followed by fine-tuning produced state-of-the-art results on virtually every NLP benchmark. The scaling era (2020--present) showed that simply making models bigger, on more data, produced emergent capabilities nobody predicted.

    Understanding this trajectory is important because older ideas are not dead --- they inform how modern systems work. N-gram statistics still appear inside tokenizers. The distributional hypothesis behind Word2Vec is the same insight driving modern embeddings.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 2. Tokenization: Turning Text into Numbers

    Before a model can process text, you need to convert strings into integer sequences. This is more consequential than it sounds.

    **Word-level tokenization** splits on whitespace and punctuation. Simple, but the vocabulary explodes and rare words get mapped to `<UNK>`. Out-of-vocabulary words are invisible to the model.

    **Character-level tokenization** has a tiny vocabulary but sequences become very long, making it hard for models to learn long-range dependencies.

    **Subword tokenization** is the winning compromise. The key algorithms:

    - **Byte Pair Encoding (BPE).** Start with individual characters. Repeatedly merge the most frequent adjacent pair. After *k* merges, you have a vocabulary of size *k* + base characters. Used by GPT-2, GPT-3, GPT-4, and many others. The original paper is Sennrich et al. (2016).
    - **WordPiece.** Similar to BPE but selects merges based on likelihood of the training corpus rather than raw frequency. Used by BERT.
    - **SentencePiece.** A language-independent implementation that treats the input as a raw byte stream (no pre-tokenization by whitespace). Supports both BPE and unigram language model modes. Used by T5, LLaMA, and many multilingual models.

    *Why subword tokenization won:* It handles rare and unseen words gracefully (they decompose into known subwords), keeps vocabulary sizes manageable (32k--100k tokens), and produces sequences short enough for transformers to process efficiently.

    **Practical note:** Tokenization choices directly affect model performance. A tokenizer optimized for English will waste tokens on Chinese or code. This is why many recent models (LLaMA 2, GPT-4) expanded their tokenizer vocabularies.
    """)
    return


@app.cell
def _(np):
    # --- Tokenization demo: word-level, char-level, and a minimal BPE ---
    sentence = "the cat sat on the mat"

    # Word-level tokenization
    words = sentence.split()
    word_vocab = {w: i for i, w in enumerate(sorted(set(words)))}
    word_ids = [word_vocab[w] for w in words]

    # Character-level tokenization
    char_vocab = {c: i for i, c in enumerate(sorted(set(sentence)))}
    char_ids = [char_vocab[c] for c in sentence]

    print("Word tokens:", words, "->", word_ids)
    print("Char tokens:", list(sentence), "->", char_ids)
    print(f"Word vocab size: {len(word_vocab)}, Char vocab size: {len(char_vocab)}")
    return (word_vocab, char_vocab)


@app.cell
def _():
    # --- Minimal Byte Pair Encoding (BPE) from scratch ---
    def bpe_train(corpus, num_merges):
        """Run BPE: repeatedly merge the most frequent adjacent pair."""
        # Start with character-level tokens (space-separated within words)
        tokens = [list(word) + ["</w>"] for word in corpus.split()]
        merges = []
        for _ in range(num_merges):
            # Count adjacent pairs
            pairs = {}
            for word in tokens:
                for a, b in zip(word, word[1:]):
                    pairs[(a, b)] = pairs.get((a, b), 0) + 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            merges.append(best)
            # Apply merge
            new_tokens = []
            for word in tokens:
                merged, i = [], 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i+1]) == best:
                        merged.append(word[i] + word[i+1])
                        i += 2
                    else:
                        merged.append(word[i])
                        i += 1
                new_tokens.append(merged)
            tokens = new_tokens
        return merges, tokens

    merges, final_tokens = bpe_train("low lower newest widest", num_merges=5)
    print("BPE merges (in order):", merges)
    print("Final subword tokens:", final_tokens)
    return (bpe_train,)


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 3. Word Embeddings: The Ideas That Still Matter

    Before transformers made contextual embeddings the norm, static embeddings were revolutionary.

    **Word2Vec** (Mikolov et al., 2013): Two architectures --- CBOW (predict center word from context) and Skip-gram (predict context from center word). The key insight: words appearing in similar contexts get similar vectors. This produces the famous analogy structure: *king - man + woman = queen*.

    **GloVe** (Pennington et al., 2014): Factorizes the word co-occurrence matrix. Combines the global statistics of matrix factorization with the local context window approach of Word2Vec.

    These embeddings are *static* --- "bank" gets the same vector whether it means a river bank or a financial bank. Modern transformer models produce *contextual* embeddings where each token's representation depends on its full context. But the distributional hypothesis underlying all of these is the same: meaning comes from usage patterns. See [Goodfellow Ch. 12](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) for neural approaches to sequence modeling.
    """)
    return


@app.cell
def _(np):
    # --- Word embeddings: co-occurrence matrix and cosine similarity ---
    corpus = ["the cat sat on the mat", "the dog sat on the rug", "the cat chased the dog"]
    vocab = sorted(set(w for s in corpus for w in s.split()))
    w2i = {w: i for i, w in enumerate(vocab)}

    # Build co-occurrence matrix (window size = 1)
    cooccur = np.zeros((len(vocab), len(vocab)))
    for sent in corpus:
        tokens = sent.split()
        for i, tok in enumerate(tokens):
            for j in range(max(0, i-1), min(len(tokens), i+2)):
                if i != j:
                    cooccur[w2i[tok], w2i[tokens[j]]] += 1

    # Cosine similarity: sim(a,b) = a·b / (||a|| ||b||)
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    # "cat" and "dog" should be similar (appear in similar contexts)
    sim_cat_dog = cosine_sim(cooccur[w2i["cat"]], cooccur[w2i["dog"]])
    sim_cat_mat = cosine_sim(cooccur[w2i["cat"]], cooccur[w2i["mat"]])
    print(f"Cosine similarity — cat vs dog: {sim_cat_dog:.3f}")
    print(f"Cosine similarity — cat vs mat: {sim_cat_mat:.3f}")
    print("(cat and dog are more similar — they share distributional context)")
    return (cosine_sim, cooccur, w2i, vocab)


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 4. Language Modeling in Depth

    A language model assigns probabilities to sequences of tokens. Formally: $P(x_1, x_2, \ldots, x_n)$, typically factored autoregressively as $\prod P(x_t | x_{<t})$.

    **Perplexity** is the standard intrinsic evaluation metric. It is the exponentiated average negative log-likelihood per token: $\text{PPL} = \exp\left(-\frac{1}{N}\sum \log P(x_t | x_{<t})\right)$. Lower is better. Intuitively, perplexity of *k* means the model is "as confused as if it had to choose uniformly among *k* options at each step." GPT-4-class models achieve perplexities in the low single digits on standard benchmarks.

    **Sampling strategies** control text generation:

    - **Greedy decoding:** Always pick the highest-probability token. Produces repetitive, boring text.
    - **Temperature:** Divide logits by temperature *T* before softmax. *T* < 1 sharpens the distribution (more deterministic), *T* > 1 flattens it (more random). *T* = 0 is equivalent to greedy.
    - **Top-k sampling:** Only sample from the *k* most probable tokens. Eliminates the long tail of unlikely tokens.
    - **Top-p (nucleus) sampling:** Sample from the smallest set of tokens whose cumulative probability exceeds *p*. Adapts the number of candidates dynamically --- when the model is confident, few tokens are considered; when uncertain, more are.
    - In practice, most systems combine temperature with top-p. The exact settings are task-dependent and often tuned empirically.

    See [Murphy PML2](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML2.pdf) for a rigorous probabilistic treatment of sequence models.
    """)
    return


@app.cell
def _(np):
    # --- Perplexity calculation from token probabilities ---
    # Suppose a toy LM assigns these probs to each token in a sequence
    token_probs = np.array([0.4, 0.6, 0.1, 0.8, 0.3])  # P(x_t | x_{<t})

    # PPL = exp( -1/N * sum(log P(x_t | x_{<t})) )
    N = len(token_probs)
    ppl = np.exp(-np.sum(np.log(token_probs)) / N)
    print(f"Token probs: {token_probs}")
    print(f"Perplexity:  {ppl:.2f}")
    print("Lower PPL = model is less 'surprised' by the sequence")
    return (ppl,)


@app.cell
def _(np):
    # --- Temperature scaling and top-k / top-p sampling ---
    logits = np.array([2.0, 1.0, 0.5, -1.0, -2.0])
    tokens = ["the", "cat", "dog", "moon", "xylophone"]

    def softmax(x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    # Temperature: divide logits by T before softmax
    for T in [0.5, 1.0, 2.0]:
        probs = softmax(logits / T)
        print(f"T={T}: {dict(zip(tokens, probs.round(3)))}")

    # Top-k: zero out all but top k before sampling
    k = 3
    top_k_idx = np.argsort(logits)[-k:]
    masked = np.full_like(logits, -1e9)
    masked[top_k_idx] = logits[top_k_idx]
    print(f"\nTop-{k} probs: {dict(zip(tokens, softmax(masked).round(3)))}")

    # Top-p (nucleus): keep smallest set with cumulative prob >= p
    p = 0.9
    probs = softmax(logits)
    sorted_idx = np.argsort(probs)[::-1]
    cumsum = np.cumsum(probs[sorted_idx])
    cutoff = np.searchsorted(cumsum, p) + 1
    nucleus_idx = sorted_idx[:cutoff]
    print(f"Top-p={p} keeps: {[tokens[i] for i in nucleus_idx]}")
    return (softmax,)


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 4.5. TF-IDF: Classic Feature Engineering for Text

    Before neural embeddings, **TF-IDF** (Term Frequency--Inverse Document Frequency) was the dominant way to represent documents as vectors. It remains useful as a sparse retrieval baseline and inside hybrid search systems.

    For term $t$ in document $d$ from corpus $D$:

    $$\text{TF}(t, d) = \frac{\text{count}(t, d)}{\text{total tokens in } d}, \qquad \text{IDF}(t, D) = \log \frac{|D|}{|\{d \in D : t \in d\}|}$$

    $$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

    Words that are frequent in a document but rare across the corpus get high TF-IDF scores --- these are the most "informative" words for that document.
    """)
    return


@app.cell
def _(np):
    # --- TF-IDF from scratch ---
    docs = [
        "the cat sat on the mat",
        "the dog chased the cat",
        "the bird flew over the mat",
    ]
    all_words = sorted(set(w for d in docs for w in d.split()))

    # TF(t, d) = count(t, d) / len(d)
    tf = np.zeros((len(docs), len(all_words)))
    for i, doc in enumerate(docs):
        words = doc.split()
        for j, w in enumerate(all_words):
            tf[i, j] = words.count(w) / len(words)

    # IDF(t) = log(|D| / |{d : t in d}|)
    idf = np.zeros(len(all_words))
    for j, w in enumerate(all_words):
        doc_freq = sum(1 for d in docs for _ in [] if w in d.split()) or sum(w in d.split() for d in docs)
        idf[j] = np.log(len(docs) / doc_freq)

    tfidf = tf * idf  # broadcast: each row scaled by idf
    print("Vocabulary:", all_words)
    print("\nTF-IDF matrix (rows=docs, cols=words):")
    print(np.round(tfidf, 3))
    print("\nHighest TF-IDF per doc:")
    for i, doc in enumerate(docs):
        best = all_words[np.argmax(tfidf[i])]
        print(f"  Doc {i} ('{doc}'): most distinctive word = '{best}'")
    return (tfidf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4.6 Attention in NLP Context

    The core idea of **attention** (Bahdanau et al., 2015) is to let a model dynamically focus on relevant parts of the input when producing each output element. In the NLP context, given a query $\mathbf{q}$ and a set of key-value pairs $(\mathbf{K}, \mathbf{V})$, scaled dot-product attention computes:

    $$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

    This mechanism is the core of the Transformer and powers every modern NLP model. The $\sqrt{d_k}$ scaling prevents dot products from growing too large and pushing softmax into saturated regions.
    """)
    return


@app.cell
def _(np, softmax):
    # --- Scaled dot-product attention (single head) ---
    rng = np.random.default_rng(42)
    seq_len, d_k = 4, 8  # 4 tokens, embedding dim 8
    Q = rng.standard_normal((seq_len, d_k))  # queries
    K = rng.standard_normal((seq_len, d_k))  # keys
    V = rng.standard_normal((seq_len, d_k))  # values

    # scores = Q K^T / sqrt(d_k)
    scores = Q @ K.T / np.sqrt(d_k)

    # attention weights (softmax over keys for each query)
    attn_weights = np.array([softmax(row) for row in scores])

    # output = weighted sum of values
    attn_output = attn_weights @ V

    print("Attention weights (each row sums to 1):")
    print(np.round(attn_weights, 3))
    print(f"\nOutput shape: {attn_output.shape} (seq_len x d_k)")
    return (attn_weights,)


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 5. The BERT Family: Bidirectional Encoders

    BERT (Devlin et al., 2019) was the first model to convincingly demonstrate that pre-training a bidirectional transformer encoder on masked language modeling (MLM) plus next sentence prediction (NSP) could be fine-tuned to achieve state-of-the-art on a wide range of NLP tasks.

    The family of variants addressed its limitations:

    - **RoBERTa** (Liu et al., 2019): Trained longer, on more data, with larger batches, and dropped NSP. Showed BERT was significantly under-trained.
    - **ALBERT** (Lan et al., 2020): Parameter sharing across layers + factorized embedding matrices. Much smaller model with competitive performance.
    - **DistilBERT** (Sanh et al., 2019): Knowledge distillation to compress BERT to 60% of its size while retaining 97% of performance. Great for production.
    - **DeBERTa** (He et al., 2021): Disentangled attention --- separate content and position embeddings that interact through disentangled matrices. Consistently outperforms RoBERTa.

    BERT-family models are *encoders* --- they produce rich contextual representations and are ideal for classification, NER, and other understanding tasks. They do not generate text autoregressively.
    """)
    return


@app.cell
def _(np):
    def _run():
        # --- Masked Language Modeling (MLM): the BERT pre-training objective ---
        sentence_tokens = ["the", "cat", "sat", "on", "the", "mat"]
        mask_prob = 0.15  # BERT masks ~15% of tokens

        rng = np.random.default_rng(7)
        masked = sentence_tokens.copy()
        masked_positions = []
        for i in range(len(masked)):
            if rng.random() < mask_prob or i == 2:  # force at least one mask
                masked[i] = "[MASK]"
                masked_positions.append(i)

        # Simulate model output: probability distribution over small vocab
        small_vocab = ["the", "cat", "sat", "on", "mat", "dog", "ran"]
        fake_logits = rng.standard_normal((len(masked_positions)), len(small_vocab))
        # Make correct answer have highest logit (pretend model is good)
        for idx, pos in enumerate(masked_positions):
            correct_word = sentence_tokens[pos]
            fake_logits[idx, small_vocab.index(correct_word)] += 3.0

        probs = np.exp(fake_logits) / np.exp(fake_logits).sum(axis=1, keepdims=True)
        print(f"Original:  {sentence_tokens}")
        print(f"Masked:    {masked}")
        for idx, pos in enumerate(masked_positions):
            pred = small_vocab[np.argmax(probs[idx])]
            print(f"  Position {pos}: predicted '{pred}' (correct: '{sentence_tokens[pos]}')")


    _run()
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 6. The GPT Lineage: Autoregressive Decoders

    The GPT line takes the opposite approach: train a decoder-only transformer to predict the next token.

    - **GPT-1** (Radford et al., 2018): 117M parameters. Demonstrated that generative pre-training + discriminative fine-tuning works.
    - **GPT-2** (Radford et al., 2019): 1.5B parameters. Showed that scaling up produces zero-shot capabilities. "Too dangerous to release" (they released it).
    - **GPT-3** (Brown et al., 2020): 175B parameters. Demonstrated in-context learning --- the ability to perform tasks from a few examples in the prompt, with no gradient updates. This was the paper that launched the "scaling laws" paradigm.
    - **GPT-4** (OpenAI, 2023): Multimodal (text + vision). Dramatic capability improvements. Architecture details undisclosed.
    - **The scaling thesis:** Performance scales predictably with compute, data, and parameters (Kaplan et al., 2020; Hoffmann et al., 2022 "Chinchilla"). This insight drove the race to larger models.

    The key conceptual shift: GPT-3 showed that a sufficiently large language model can be *prompted* rather than *fine-tuned*. This opened the door to the current era of general-purpose AI assistants.
    """)
    return


@app.cell
def _(np, softmax):
    # --- Causal (autoregressive) attention: the GPT masking pattern ---
    rng = np.random.default_rng(0)
    seq_len_ar = 5
    d = 4
    Q_ar = rng.standard_normal((seq_len_ar, d))
    K_ar = rng.standard_normal((seq_len_ar, d))
    V_ar = rng.standard_normal((seq_len_ar, d))

    scores_ar = Q_ar @ K_ar.T / np.sqrt(d)

    # Causal mask: token i can only attend to tokens 0..i
    causal_mask = np.triu(np.ones((seq_len_ar, seq_len_ar)), k=1)  # upper triangle = 1
    scores_ar = np.where(causal_mask == 1, -1e9, scores_ar)

    attn_causal = np.array([softmax(row) for row in scores_ar])
    print("Causal attention weights (lower-triangular pattern):")
    print(np.round(attn_causal, 3))
    print("\nEach token can only attend to itself and earlier tokens.")
    return (attn_causal,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 7. Instruction Tuning and Alignment

    Raw pre-trained models are not helpful assistants --- they are next-token predictors that may produce toxic, incorrect, or unhelpful text. Alignment bridges this gap (see also 3D --- Reinforcement Learning if you covered it):

    - **Supervised Fine-Tuning (SFT):** Train on curated (instruction, response) pairs. The model learns to follow instructions rather than just complete text.
    - **Reinforcement Learning from Human Feedback (RLHF):** Train a reward model on human preference comparisons, then optimize the policy (language model) against this reward using PPO. This is how InstructGPT and ChatGPT were trained.
    - **Direct Preference Optimization (DPO):** Eliminates the separate reward model by reparameterizing the RLHF objective. The policy is trained directly on preference pairs. Simpler, more stable, increasingly preferred.

    The alignment problem is far from solved --- reward hacking, sycophancy, and specification gaming remain active research areas.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 8. Key NLP Tasks

    With modern models, these tasks are often approached via fine-tuning or prompting rather than bespoke architectures:

    - **Text classification:** Sentiment analysis, spam detection, topic categorization. Fine-tune a BERT-family encoder or prompt an LLM.
    - **Named Entity Recognition (NER):** Identify and classify entities (people, organizations, locations). Token-level classification with a BERT encoder is still the go-to.
    - **Question Answering:** Extractive (highlight a span in a passage) or generative (produce an answer in free text). SQuAD is the classic benchmark.
    - **Summarization:** Abstractive (generate new text) vs. extractive (select key sentences). Models like BART and T5 are encoder-decoder architectures designed for this.
    - **Machine Translation:** The original killer app of sequence-to-sequence models. Modern systems use transformer-based encoder-decoders (NLLB, MarianMT) or prompt multilingual LLMs.

    See [Murphy PML1](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf) for probabilistic foundations underlying classification and sequence modeling tasks.
    """)
    return


@app.cell
def _(np):
    def _run():
        # --- Bag-of-words text classification (sentiment) ---
        # Positive/negative word lists as a simple baseline
        positive_words = {"good", "great", "love", "excellent", "wonderful", "best", "amazing"}
        negative_words = {"bad", "terrible", "hate", "awful", "worst", "boring", "poor"}

        reviews = [
            "this movie was great and wonderful",
            "terrible film awful acting worst ever",
            "good story but bad ending",
        ]

        for review in reviews:
            words = set(review.lower().split())
            pos_score = len(words & positive_words)
            neg_score = len(words & negative_words)
            label = "POSITIVE" if pos_score > neg_score else ("NEGATIVE" if neg_score > pos_score else "MIXED")
            print(f"'{review}' -> +{pos_score}/-{neg_score} -> {label}")


    _run()
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 9. Retrieval-Augmented Generation (RAG)

    RAG is one of the most practically important patterns in modern NLP. It addresses LLM hallucination and knowledge staleness by grounding generation in retrieved documents.

    **Architecture:** Query -> Retrieve relevant documents -> Augment the prompt with retrieved context -> Generate a response.

    **Key components:**

    - **Chunking:** Split documents into passages (fixed-size, sentence-based, or semantic). Chunk size is a critical hyperparameter --- too small loses context, too large dilutes relevance.
    - **Embedding:** Encode chunks into dense vectors using models like `text-embedding-3-small`, `BGE`, `E5`, or `GTE`.
    - **Vector databases:** Store and search embeddings efficiently. Options include Pinecone, Weaviate, Chroma, Qdrant, pgvector, and FAISS.
    - **Retrieval:** Given a query embedding, find the top-*k* most similar chunks via approximate nearest neighbor search (HNSW, IVF).
    - **Reranking:** A cross-encoder (e.g., Cohere Rerank, `bge-reranker`) scores each (query, document) pair more accurately than embedding similarity alone. Computationally expensive but dramatically improves relevance.
    - **Generation:** Feed retrieved context + query to an LLM. Prompt engineering matters --- instruct the model to answer based on the provided context and say "I don't know" when the context is insufficient.

    **Advanced patterns:** Hybrid search (combining dense retrieval with BM25 sparse retrieval), query decomposition, iterative retrieval, agentic RAG (the model decides when and what to retrieve).
    """)
    return


@app.cell
def _(np):
    def _run():
        # --- Minimal RAG: embed, index, retrieve with cosine similarity ---
        # Fake embeddings (in practice you'd use a real embedding model)
        rng = np.random.default_rng(99)
        knowledge_base = [
            "Python was created by Guido van Rossum in 1991.",
            "The Eiffel Tower is 330 meters tall.",
            "Transformers were introduced in the 2017 paper Attention Is All You Need.",
            "Water boils at 100 degrees Celsius at sea level.",
        ]
        # Simulate embeddings (dim=16)
        doc_embeddings = rng.standard_normal((len(knowledge_base)), 16)
        # Normalize for cosine similarity
        doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)

        # Query embedding (simulated — biased toward doc 2)
        query = "What paper introduced transformers?"
        query_emb = doc_embeddings[2] + rng.standard_normal(16) * 0.3
        query_emb = query_emb / np.linalg.norm(query_emb)

        # Retrieve: cosine similarity = dot product (since normalized)
        similarities = doc_embeddings @ query_emb
        top_k = 2
        top_indices = np.argsort(similarities)[::-1][:top_k]

        print(f"Query: '{query}'")
        print(f"\nTop-{top_k} retrieved documents:")
        for idx in top_indices:
            print(f"  [{similarities[idx]:.3f}] {knowledge_base[idx]}")


    _run()
    return

@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 10. Evaluation

    Evaluating NLP systems is notoriously difficult because language quality is subjective.

    - **BLEU** (Papineni et al., 2002): N-gram precision between generated and reference text. Standard for machine translation. Flawed --- high BLEU does not guarantee good translations, and vice versa.
    - **ROUGE** (Lin, 2004): N-gram recall-oriented metric. Standard for summarization. ROUGE-1 (unigrams), ROUGE-2 (bigrams), ROUGE-L (longest common subsequence).
    - **BERTScore** (Zhang et al., 2020): Computes similarity between generated and reference tokens using BERT embeddings. Correlates better with human judgment than BLEU/ROUGE.
    - **Human evaluation:** The gold standard but expensive and slow. Inter-annotator agreement is often low.
    - **LLM-as-judge:** Use a strong LLM (e.g., GPT-4) to evaluate outputs. Surprisingly effective. Concerns: bias toward verbose responses, positional bias, self-preference. Mitigations: structured rubrics, pairwise comparison, position swapping.

    For classification tasks, standard metrics apply: accuracy, precision, recall, F1, and AUC. See [ISLR Ch. 4](file:///C:/Users/landa/ml-course/textbooks/ISLR.pdf) for classification fundamentals.
    """)
    return


@app.cell
def _(np):
    # --- BLEU and ROUGE-1 from scratch ---
    from collections import Counter

    def bleu_1(reference, hypothesis):
        """Unigram BLEU (precision of hypothesis n-grams in reference)."""
        ref_counts = Counter(reference.split())
        hyp_counts = Counter(hypothesis.split())
        clipped = sum(min(hyp_counts[w], ref_counts.get(w, 0)) for w in hyp_counts)
        return clipped / max(sum(hyp_counts.values()), 1)

    def rouge_1(reference, hypothesis):
        """ROUGE-1: unigram recall (how many ref unigrams appear in hyp)."""
        ref_tokens = set(reference.split())
        hyp_tokens = set(hypothesis.split())
        overlap = len(ref_tokens & hyp_tokens)
        return overlap / max(len(ref_tokens), 1)

    ref = "the cat sat on the mat"
    hyp1 = "the cat sat on the mat"      # perfect
    hyp2 = "a dog sat on a rug"           # partial overlap
    hyp3 = "completely unrelated sentence" # no overlap

    for hyp in [hyp1, hyp2, hyp3]:
        print(f"Hyp: '{hyp}'")
        print(f"  BLEU-1:  {bleu_1(ref, hyp):.3f}   ROUGE-1: {rouge_1(ref, hyp):.3f}")
    return (bleu_1, rouge_1)


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 11. Recommended Reading and Resources

    **Textbooks:**
    - **Jurafsky & Martin, *Speech and Language Processing* (3rd edition draft):** Free online at [web.stanford.edu/~jurafsky/slp3](https://web.stanford.edu/~jurafsky/slp3/). The most comprehensive NLP textbook. Covers everything from regex to transformers.
    - [Goodfellow et al., Deep Learning --- Ch. 12: Applications (NLP sections)](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf)
    - [Murphy, Probabilistic Machine Learning: Advanced Topics](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML2.pdf) --- sequence models, attention, language models

    **Courses:**
    - **Stanford CS224N: NLP with Deep Learning.** Lectures freely available on YouTube. The standard graduate NLP course.
    - **Hugging Face NLP Course:** [huggingface.co/learn/nlp-course](https://huggingface.co/learn/nlp-course). Hands-on, code-first, free.

    **Key Papers (read in this order):**

    1. Mikolov et al., "Efficient Estimation of Word Representations in Vector Space" (Word2Vec), 2013. [arXiv:1301.3781](https://arxiv.org/abs/1301.3781)
    2. Vaswani et al., "Attention Is All You Need" (Transformer), 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
    3. Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers," 2019. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
    4. Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2), 2019. [OpenAI blog](https://openai.com/research/better-language-models)
    5. Brown et al., "Language Models are Few-Shot Learners" (GPT-3), 2020. [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
    6. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," 2020. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
    7. Ouyang et al., "Training language models to follow instructions with human feedback" (InstructGPT / RLHF), 2022. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
    8. Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (DPO), 2023. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
    9. Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla), 2022. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
    10. Kaplan et al., "Scaling Laws for Neural Language Models," 2020. [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 12. Project Ideas

    **Beginner--Intermediate:**
    - **Sentiment analyzer.** Fine-tune DistilBERT on a movie review or product review dataset. Deploy as an API. Evaluate with precision/recall and compare to a simple bag-of-words baseline.
    - **Named Entity Recognition system.** Fine-tune a BERT model on CoNLL-2003. Visualize predictions using spaCy's displacy.

    **Intermediate:**
    - **Build a RAG system.** Index a corpus of documents (e.g., Wikipedia articles, company docs, or a textbook) into a vector database. Build a retrieval pipeline with reranking. Connect to an LLM for generation. Evaluate retrieval quality (recall@k) and generation quality (BERTScore, human eval).
    - **Domain-specific fine-tuning.** Take a small open-weight model (e.g., Mistral 7B, LLaMA 3 8B) and fine-tune it on domain-specific data (legal, medical, scientific). Use LoRA/QLoRA for efficiency. Compare to the base model and to prompting a larger model.

    **Advanced:**
    - **Build an LLM evaluation harness.** Implement multiple evaluation strategies (automated metrics, LLM-as-judge with structured rubrics, human evaluation interface). Compare how well each correlates with human preference.
    - **Multi-hop RAG with agentic retrieval.** Build a system that decomposes complex questions, retrieves evidence iteratively, and synthesizes answers. Evaluate on multi-hop QA benchmarks like HotpotQA.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 13. Code It: Implementation Exercises

    The exercises below ask you to implement core NLP concepts from scratch using only Python and NumPy. Fill in the `TODO` sections.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 1: N-gram Language Model

    Build a bigram language model from a corpus. Compute the probability of a sentence and its perplexity.

    Given a bigram model: $P(w_t | w_{t-1}) = \frac{\text{count}(w_{t-1}, w_t)}{\text{count}(w_{t-1})}$
    """)
    return


@app.cell
def _(np):
    # Exercise 1: Bigram language model
    ex1_corpus = [
        "the cat sat on the mat",
        "the cat ate the fish",
        "the dog sat on the rug",
        "the dog ate the bone",
    ]

    def build_bigram_model(corpus):
        """Build bigram counts from a corpus of sentences."""
        bigram_counts = {}  # (w1, w2) -> count
        unigram_counts = {}  # w1 -> count (as bigram prefix)
        for sentence in corpus:
            tokens = ["<s>"] + sentence.split() + ["</s>"]
            for i in range(len(tokens) - 1):
                # TODO: count bigrams and unigrams
                pass
        return bigram_counts, unigram_counts

    def bigram_probability(sentence, bigram_counts, unigram_counts):
        """Compute P(sentence) = product of bigram probabilities."""
        tokens = ["<s>"] + sentence.split() + ["</s>"]
        log_prob = 0.0
        for i in range(len(tokens) - 1):
            # TODO: compute log P(tokens[i+1] | tokens[i])
            # Use add-1 (Laplace) smoothing to handle unseen bigrams
            pass
        return log_prob

    def perplexity_from_logprob(log_prob, num_tokens):
        """PPL = exp(-log_prob / N)"""
        # TODO: compute and return perplexity
        pass

    # Test your implementation:
    # bi_counts, uni_counts = build_bigram_model(ex1_corpus)
    # test_sent = "the cat sat on the rug"
    # lp = bigram_probability(test_sent, bi_counts, uni_counts)
    # ppl_val = perplexity_from_logprob(lp, len(test_sent.split()) + 1)
    # print(f"Log prob: {lp:.3f}, Perplexity: {ppl_val:.2f}")
    return (build_bigram_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 2: TF-IDF Retrieval System

    Implement a TF-IDF based document retrieval system. Given a query, rank documents by cosine similarity of their TF-IDF vectors.
    """)
    return


@app.cell
def _(np):
    # Exercise 2: TF-IDF retrieval
    ex2_docs = [
        "machine learning is a subset of artificial intelligence",
        "deep learning uses neural networks with many layers",
        "natural language processing deals with text and speech",
        "computer vision analyzes images and video data",
        "reinforcement learning trains agents through reward signals",
    ]
    ex2_query = "neural networks for language understanding"

    def build_tfidf_matrix(documents):
        """Build TF-IDF matrix from a list of documents."""
        # TODO: 1) build vocabulary from all documents
        # TODO: 2) compute TF matrix (docs x vocab)
        # TODO: 3) compute IDF vector
        # TODO: 4) return tfidf matrix and vocabulary list
        vocab = sorted(set(w for d in documents for w in d.split()))
        tfidf_mat = np.zeros((len(documents), len(vocab)))
        # ... fill in tfidf_mat ...
        return tfidf_mat, vocab

    def tfidf_retrieve(query, tfidf_mat, vocab, documents, top_k=3):
        """Retrieve top-k documents by cosine similarity to query."""
        # TODO: 1) compute query TF-IDF vector
        # TODO: 2) compute cosine similarity with each document
        # TODO: 3) return top-k (document, score) pairs
        return []

    # Test your implementation:
    # mat, v = build_tfidf_matrix(ex2_docs)
    # results = tfidf_retrieve(ex2_query, mat, v, ex2_docs)
    # for doc, score in results:
    #     print(f"  [{score:.3f}] {doc}")
    return (build_tfidf_matrix,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 3: Multi-Head Attention

    Implement multi-head attention from scratch. Split Q, K, V into `num_heads` heads, apply scaled dot-product attention to each, and concatenate.

    $$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

    where $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$
    """)
    return


@app.cell
def _(np):
    # Exercise 3: Multi-head attention
    def multi_head_attention(Q, K, V, num_heads):
        """
        Q, K, V: (seq_len, d_model)
        Returns: (seq_len, d_model)
        """
        seq_len, d_model = Q.shape
        assert d_model % num_heads == 0
        d_k_head = d_model // num_heads

        # TODO: 1) Initialize random projection matrices W_Q, W_K, W_V, W_O
        #          Each W_Q_i, W_K_i, W_V_i is (d_model, d_k_head)
        #          W_O is (d_model, d_model)

        # TODO: 2) For each head:
        #   a) Project: Q_i = Q @ W_Q_i, K_i = K @ W_K_i, V_i = V @ W_V_i
        #   b) Compute scaled dot-product attention
        #   c) Collect head outputs

        # TODO: 3) Concatenate all heads -> (seq_len, d_model)

        # TODO: 4) Apply output projection W_O

        output = np.zeros((seq_len, d_model))  # placeholder
        return output

    # Test:
    # rng = np.random.default_rng(42)
    # seq, dm = 6, 16
    # Q_test = rng.standard_normal((seq, dm))
    # K_test = rng.standard_normal((seq, dm))
    # V_test = rng.standard_normal((seq, dm))
    # out = multi_head_attention(Q_test, K_test, V_test, num_heads=4)
    # print(f"Output shape: {out.shape}")  # should be (6, 16)
    return (multi_head_attention,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 4: BM25 Retrieval

    Implement BM25, the classic sparse retrieval algorithm still used in hybrid search systems. BM25 scores a document $d$ against a query $q$:

    $$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

    where $f(t,d)$ is term frequency, $k_1=1.5$, $b=0.75$.
    """)
    return


@app.cell
def _(np):
    # Exercise 4: BM25 retrieval
    ex4_docs = [
        "the transformer architecture revolutionized natural language processing",
        "attention mechanisms allow models to focus on relevant input parts",
        "recurrent neural networks process sequences step by step",
        "convolutional networks are primarily used for image processing",
        "language models predict the next token in a sequence",
    ]

    def bm25_score(query, documents, k1=1.5, b=0.75):
        """Score each document against the query using BM25."""
        scores = np.zeros(len(documents))
        doc_lens = [len(d.split()) for d in documents]
        avgdl = np.mean(doc_lens)
        N = len(documents)

        query_terms = query.lower().split()
        for term in query_terms:
            # TODO: 1) compute IDF(term) = log((N - df + 0.5) / (df + 0.5))
            #          where df = number of docs containing term

            # TODO: 2) for each document, compute BM25 contribution of this term

            pass

        return scores

    # Test:
    # ex4_query = "attention in language models"
    # scores = bm25_score(ex4_query, ex4_docs)
    # ranked = sorted(enumerate(scores), key=lambda x: -x[1])
    # print(f"Query: '{ex4_query}'")
    # for idx, s in ranked:
    #     print(f"  [{s:.3f}] {ex4_docs[idx]}")
    return (bm25_score,)


@app.cell
def _(mo):
    mo.md(r"""
    ---

    *Next steps: If you have not yet covered 3A --- Attention & Transformers, go there first --- transformers are the foundation of everything in modern NLP. If you want to go cross-modal, see Path B --- Computer Vision for how these same ideas apply to visual understanding, or explore Path C --- Generative Models for the generation side in depth.*
    """)
    return


if __name__ == "__main__":
    app.run()
