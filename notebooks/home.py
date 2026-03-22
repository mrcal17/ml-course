import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Machine Learning — Complete Course

    A self-paced, interactive course from mathematical foundations to modern deep learning.
    Every module is a reactive notebook — read the lectures, run the code, tweak the parameters, and build real understanding.

    ---

    ## How to Use This Course

    - **Follow the modules in order within each Part** — the dependency graph below shows what feeds into what
    - **Run the code cells** — don't just read, experiment. Change values, break things, rebuild
    - **Interactive widgets** (sliders, dropdowns) let you explore concepts in real-time — this is where the deep learning happens
    - **Animations** visualize key mathematical concepts — watch them, then try to explain what's happening in your own words
    - **Textbook links** open the relevant PDF pages for deeper reading
    - **Practice exercises** are at the end of each module — do them

    To open any module, run: `marimo edit notebooks/<filename>.py`
    Or use: `bash launch.sh 0b` (opens Module 0B)

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 0 — Foundations

    > *Fix your gaps before touching ML. Probability is the weakest link — spend real time on 0D.*

    | # | Module | Topics | Interactive Elements |
    |---|--------|--------|---------------------|
    | 0A | [Python & Data Science Stack](../0a_python/index.html) | NumPy, Pandas, Matplotlib, Jupyter | Executable code throughout |
    | 0B | [Calculus Refresh](../0b_calculus/index.html) | Gradients, chain rule, Jacobians, Hessians | Finite differences explorer, 2 animations |
    | 0C | [Linear Algebra Refresh](../0c_linear_algebra/index.html) | Eigendecomposition, SVD, projections | 2 animations |
    | 0D | [Probability Foundations](../0d_probability/index.html) | Bayes, distributions, expectation, CLT | Distribution explorer, Bayesian updater, 2 animations |
    | 0E | [Statistical Estimation](../0e_estimation/index.html) | MLE, MAP, confidence intervals, bootstrap | Executable derivations |
    | 0F | [Optimization](../0f_optimization/index.html) | Gradient descent, SGD, momentum, Adam | Learning rate explorer, optimizer comparison, 2 animations |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part 1 — Core Machine Learning

    > *Classical ML — the models, the math, and how to evaluate them.*

    | # | Module | Topics | Interactive Elements |
    |---|--------|--------|---------------------|
    | 1A | [The ML Landscape](../1a_ml_landscape/index.html) | Types of learning, bias-variance, curse of dimensionality | Conceptual |
    | 1B | [Linear Regression](../1b_linear_regression/index.html) | OLS, regularization (Ridge/Lasso), probabilistic view | Polynomial degree slider, λ slider, 2 animations |
    | 1C | [Classification](../1c_classification/index.html) | Logistic regression, SVMs, Naive Bayes, metrics | SVM C/kernel explorer, 1 animation |
    | 1D | [Model Selection & Evaluation](../1d_model_selection/index.html) | Cross-validation, bootstrap, hyperparameter tuning | Executable pipelines |
    | 1E | [Trees & Ensembles](../1e_trees_ensembles/index.html) | Decision trees, random forests, gradient boosting | Tree depth slider, ensemble size slider, 1 animation |
    | 1F | [Unsupervised Learning](../1f_unsupervised/index.html) | PCA, K-means, GMMs, t-SNE | K slider, PCA components slider, 1 animation |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part 2 — Deep Learning Foundations

    > *Neural networks, backprop, CNNs, RNNs — the deep learning revolution.*

    | # | Module | Topics | Interactive Elements |
    |---|--------|--------|---------------------|
    | 2A | [Neural Networks & Backpropagation](../2a_neural_networks/index.html) | MLPs, activation functions, backprop, autograd | Hidden neurons slider, 1 animation |
    | 2B | [Optimization for Deep Learning](../2b_dl_optimization/index.html) | Loss landscapes, batch norm, learning rate schedules | Executable code |
    | 2C | [Regularization](../2c_regularization/index.html) | Dropout, early stopping, data augmentation, double descent | Dropout rate slider |
    | 2D | [Convolutional Neural Networks](../2d_cnn/index.html) | Convolution, pooling, ResNet, transfer learning | 1 animation |
    | 2E | [Sequence Models](../2e_sequence_models/index.html) | RNNs, LSTMs, GRUs, seq2seq | 1 animation |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 3 — Modern Deep Learning

    > *Transformers, generative models, and the frontier.*

    | # | Module | Topics | Interactive Elements |
    |---|--------|--------|---------------------|
    | 3A | [Attention & Transformers](../3a_transformers/index.html) | Self-attention, multi-head, positional encoding, scaling laws | 1 animation |
    | 3B | [Generative Models](../3b_generative_models/index.html) | VAEs, GANs, diffusion models, normalizing flows | 1 animation |
    | 3C | [Self-Supervised Learning](../3c_self_supervised/index.html) | BERT, GPT, contrastive learning, CLIP, LoRA | Conceptual |
    | 3D | [Reinforcement Learning](../3d_reinforcement_learning/index.html) | MDPs, Q-learning, policy gradients, PPO, RLHF | 1 animation |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 4 — Specialization

    > *Choose your path.*

    | Path | Module | Focus |
    |------|--------|-------|
    | A | [Natural Language Processing](../4a_nlp/index.html) | Tokenization, LLMs, RAG, evaluation |
    | B | [Computer Vision](../4b_computer_vision/index.html) | Detection, segmentation, ViT, multimodal |
    | C | [Advanced RL](../4c_advanced_rl/index.html) | Model-based RL, SAC, multi-agent, offline RL |
    | D | [Bayesian ML](../4d_bayesian_ml/index.html) | Gaussian processes, BNNs, MCMC, Bayesian optimization |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Dependency Graph

    ```
    0A (Python) → 0B (Calculus) → 0C (Linear Algebra)
                                        │
                    0D (Probability) ←───┘
                         │
                    0E (Estimation)
                         │
                    0F (Optimization)
                         │
                    1A (ML Landscape) → 1B (Regression) → 1C (Classification)
                                                                │
                              1D (Model Selection) ←────────────┘
                                   │
                             1E (Trees/Ensembles)
                                   │
                             1F (Unsupervised)
                                   │
                         2A (Neural Nets) → 2B (DL Optimization) → 2C (Regularization)
                                                                        │
                                              2D (CNNs) ←──────────────┘
                                                   │
                                              2E (Sequences)
                                                   │
                                        3A (Transformers) → 3B (Generative)
                                              │                     │
                                        3D (RL)              3C (Self-Supervised)
                                              │
                                        Part 4 (Specialize)
    ```

    ---

    ## Textbooks

    All textbook PDFs are in the `textbooks/` folder. Click to open:

    | Abbrev. | Title | Authors |
    |---------|-------|---------|
    | MML | Mathematics for Machine Learning | Deisenroth, Faisal, Ong |
    | ISLR | An Introduction to Statistical Learning | James, Witten, Hastie, Tibshirani |
    | ESL | The Elements of Statistical Learning | Hastie, Tibshirani, Friedman |
    | Boyd | Convex Optimization | Boyd, Vandenberghe |
    | DLBook | Deep Learning | Goodfellow, Bengio, Courville |
    | Bishop | Pattern Recognition and Machine Learning | Christopher Bishop |
    | Sutton | Reinforcement Learning | Sutton, Barto |
    | Murphy 1 | Probabilistic ML: An Introduction | Kevin Murphy |
    | Murphy 2 | Probabilistic ML: Advanced Topics | Kevin Murphy |
    | Chan | Introduction to Probability for Data Science | Stanley Chan |
    | Géron | Hands-On ML with Scikit-Learn, Keras & TF | Aurélien Géron |
    | Wasserman | All of Statistics | Larry Wasserman |
    """)
    return


if __name__ == "__main__":
    app.run()
