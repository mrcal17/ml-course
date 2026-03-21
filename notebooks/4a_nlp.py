import marimo

app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Path A --- Natural Language Processing

    Natural language processing sits at the intersection of linguistics, computer science, and machine learning. It is arguably the subfield that has seen the most dramatic transformation in the deep learning era --- and the one most responsible for the current wave of public interest in AI. This guide maps out the landscape, highlights the concepts that matter most, and points you toward resources and projects that will build real competency.
    """)
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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

    ---

    *Next steps: If you have not yet covered 3A --- Attention & Transformers, go there first --- transformers are the foundation of everything in modern NLP. If you want to go cross-modal, see Path B --- Computer Vision for how these same ideas apply to visual understanding, or explore Path C --- Generative Models for the generation side in depth.*
    """)
    return


if __name__ == "__main__":
    app.run()
