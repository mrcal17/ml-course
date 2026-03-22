import marimo

app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 0D: Probability Foundations

    > *"Probability theory is nothing but common sense reduced to calculation."*
    > — Pierre-Simon Laplace

    This is the single most important foundations module. If linear algebra gives ML its *structure*, probability gives it its *logic*. Every model you will build, every loss function you will minimize, every prediction you will make — all of it is probability in disguise. I am going to be thorough here because I know this is your weak spot, and I would rather you spend extra time now than struggle with every single concept downstream.

    Let me be direct: you can get through an ML course without deeply understanding eigenvalues. You *cannot* get through it without deeply understanding probability. So we are going to take this slow, build real intuition, and make sure every concept clicks before moving on.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 1. Why Probability Is the Language of ML

    Here is a question: why do ML models output probabilities instead of just answers?

    Because the world is uncertain, and any intelligent system must *represent* that uncertainty rather than pretend it does not exist. When a spam classifier says "this email is 94% likely to be spam," it is doing something fundamentally different from a rule-based system that says "this email IS spam." The probabilistic model knows what it does not know.

    This shows up everywhere in ML:

    - **Classification:** A neural network's softmax output is a probability distribution over classes.
    - **Regression:** Many models predict not just a value but a *distribution* over possible values (a mean and variance).
    - **Generative models:** GPT, diffusion models, VAEs — all are sampling from learned probability distributions.
    - **Training itself:** Maximum likelihood estimation, the most common training paradigm, is literally "find the parameters that make the observed data most probable."
    - **Bayesian inference:** Treating model parameters themselves as random variables with distributions.

    Probability is not a tool that ML *uses*. Probability is what ML *is*. The entire field is an exercise in reasoning under uncertainty using the formal language of probability theory.

    > [MML Ch 6](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) opens with this exact point — read Section 6.1 for a concise statement of why probability is central.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 2. Sample Spaces, Events, and Axioms

    Before we can compute anything, we need to define what we are talking about. Probability theory is built on three foundational objects.

    ### The Setup

    - **Sample space** ($\Omega$): The set of *all possible outcomes* of an experiment. Roll a die? $\Omega = \{1, 2, 3, 4, 5, 6\}$. Measure someone's height? $\Omega = (0, \infty)$.
    - **Event** ($A$): A subset of the sample space. "Rolling an even number" is the event $A = \{2, 4, 6\}$.
    - **Probability function** ($P$): A function that assigns a number to each event, telling you how likely it is.

    ### The Three Axioms (Kolmogorov)

    These are not optional suggestions. They are the *definition* of what it means for something to be a probability:

    1. **Non-negativity:** $P(A) \geq 0$ for any event $A$.
    2. **Normalization:** $P(\Omega) = 1$ — *something* must happen.
    3. **Additivity:** For mutually exclusive events $A$ and $B$ (they cannot both happen): $P(A \cup B) = P(A) + P(B)$.

    That is it. Everything else in probability theory — Bayes' theorem, the central limit theorem, all of it — is a *consequence* of these three axioms. They seem almost trivially obvious, and that is the point. We are building a rigorous framework on top of assumptions that nobody could reasonably dispute.

    > [Chan Ch 1.2–1.3](file:///C:/Users/landa/ml-course/textbooks/Chan-Probability.pdf) gives a careful treatment of axioms with examples.

    ### Why This Matters for ML

    These axioms constrain what a model's outputs can look like. If a neural network outputs "probabilities" that sum to 1.3, those are not probabilities. The softmax function exists precisely to enforce axioms 1 and 2. When you see constraints in ML architectures, they are often just enforcing these axioms.

    ### Discrete vs. Continuous Sample Spaces

    - **Discrete:** Countable outcomes. Coin flips, word tokens, class labels. We assign probabilities to individual outcomes.
    - **Continuous:** Uncountable outcomes. Heights, temperatures, pixel intensities. We *cannot* assign probability to individual points (they all have probability zero). Instead, we assign probability to *intervals*. This is where probability density functions come in — more on that in Section 5.
    """)
    return


@app.cell
def _():
    import numpy as np

    # Sample space: fair die
    omega = np.array([1, 2, 3, 4, 5, 6])
    probs = np.ones(6) / 6  # uniform PMF

    # Axiom 1: all probs >= 0
    assert np.all(probs >= 0), "Non-negativity violated"

    # Axiom 2: total probability = 1
    assert np.isclose(probs.sum(), 1.0), "Normalization violated"

    # Axiom 3: P(even OR odd<=3) for mutually exclusive events {2,4,6} and {1,3}
    A = np.isin(omega, [2, 4, 6])  # even
    B = np.isin(omega, [1, 3])      # odd and <= 3
    P_A = probs[A].sum()             # 3/6
    P_B = probs[B].sum()             # 2/6
    P_union = probs[A | B].sum()     # 5/6 — should equal P_A + P_B
    print(f"P(A)={P_A:.3f}, P(B)={P_B:.3f}, P(A∪B)={P_union:.3f}, P(A)+P(B)={P_A+P_B:.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3. Conditional Probability

    This is where probability starts getting powerful — and where your intuition will start to betray you if you are not careful.

    ### The Definition

    The probability of $A$ given that $B$ has occurred:

    $$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

    Read this as: "To find the probability of $A$ given $B$, take the probability of *both* happening and divide by the probability of $B$." We are *restricting our universe* to only those outcomes where $B$ happened, then asking how often $A$ also happens in that restricted universe.

    **Worked example:** A deck of 52 cards. What is the probability a card is a King, given that it is a face card?

    - $P(\text{King} \cap \text{Face}) = P(\text{King}) = 4/52$ (all Kings are face cards)
    - $P(\text{Face}) = 12/52$
    - $P(\text{King}|\text{Face}) = \frac{4/52}{12/52} = \frac{4}{12} = \frac{1}{3}$

    Makes sense: there are 12 face cards, 4 of which are Kings.

    ### The Multiplication Rule

    Rearranging the definition:

    $$P(A \cap B) = P(A|B) \cdot P(B) = P(B|A) \cdot P(A)$$

    This is how we compute joint probabilities when we know conditionals. It extends to chains:

    $$P(A \cap B \cap C) = P(A) \cdot P(B|A) \cdot P(C|A,B)$$

    This is the **chain rule of probability**, and it is fundamental. Autoregressive language models (GPT, etc.) use exactly this decomposition: $P(\text{sentence}) = P(w_1) \cdot P(w_2|w_1) \cdot P(w_3|w_1,w_2) \cdots$

    ### Independence

    Two events are **independent** if knowing one tells you nothing about the other:

    $$P(A|B) = P(A)$$

    Equivalently: $P(A \cap B) = P(A) \cdot P(B)$.

    Independence is a *strong* assumption. In ML, the Naive Bayes classifier assumes all features are conditionally independent given the class label. This assumption is almost always wrong, but the model often works surprisingly well anyway.

    ### The Counterintuitive Trap

    Here is a classic mistake. Suppose 1% of the population has a disease, and a test is 99% accurate (both sensitivity and specificity). You test positive. What is the probability you actually have the disease?

    Most people say 99%. The real answer is about 50%. We will derive this properly in the next section.

    > [Chan Ch 2.1–2.3](file:///C:/Users/landa/ml-course/textbooks/Chan-Probability.pdf) covers conditional probability with many worked examples.
    """)
    return


@app.cell
def _():
    import numpy as np

    # Conditional probability: P(King | Face card)
    deck = 52
    kings = 4
    face_cards = 12  # J, Q, K of each suit

    # P(King ∩ Face) = P(King), since all kings are face cards
    P_king_and_face = kings / deck
    P_face = face_cards / deck
    P_king_given_face = P_king_and_face / P_face  # Bayes definition

    print(f"P(King | Face) = {P_king_and_face:.4f} / {P_face:.4f} = {P_king_given_face:.4f}")
    print(f"That's 1/{int(1/P_king_given_face)} — exactly 4 kings among 12 face cards")

    # Verify with simulation
    cards = np.arange(deck)
    n_sims = 500_000
    draws = rng.choice(cards, size=n_sims)
    is_face = draws < 12          # first 12 cards are face cards
    is_king = draws < 4           # first 4 are kings
    simulated = is_king[is_face].mean()
    print(f"Simulated P(King | Face) ≈ {simulated:.4f}")
    return


@app.cell
def _():
    import numpy as np

    # Independence test: are two dice rolls independent?
    n = 200_000
    die1 = rng.integers(1, 7, n)
    die2 = rng.integers(1, 7, n)

    # P(die1=6) vs P(die1=6 | die2=6) — should be equal if independent
    P_d1_6 = (die1 == 6).mean()
    P_d1_6_given_d2_6 = (die1[die2 == 6] == 6).mean()
    print(f"P(die1=6) = {P_d1_6:.4f}")
    print(f"P(die1=6 | die2=6) = {P_d1_6_given_d2_6:.4f}")
    print(f"Nearly equal → dice are independent")

    # Also verify: P(A∩B) = P(A)*P(B) for independent events
    P_both_6 = ((die1 == 6) & (die2 == 6)).mean()
    print(f"P(both 6) = {P_both_6:.4f}, P(6)*P(6) = {P_d1_6**2:.4f}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 4. Bayes' Theorem — The Most Important Formula in ML

    I am not being hyperbolic. If you deeply understand Bayes' theorem, you understand what ML is doing at a fundamental level.

    ### Derivation

    Start with the multiplication rule, written two ways:

    $$P(A \cap B) = P(A|B) \cdot P(B) = P(B|A) \cdot P(A)$$

    Set the right-hand sides equal and solve for $P(A|B)$:

    $$\boxed{P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}}$$

    That is Bayes' theorem. Four lines of algebra, and yet this single equation is the engine behind spam filters, medical diagnosis, recommendation systems, and the entire field of Bayesian machine learning.

    ### The Four Pieces

    Each term has a name, and you need to know them cold:

    | Term | Name | Meaning |
    |------|------|---------|
    | $P(A \mid B)$ | **Posterior** | What we want — our updated belief about $A$ after seeing evidence $B$ |
    | $P(B \mid A)$ | **Likelihood** | How probable is the evidence $B$ if $A$ were true? |
    | $P(A)$ | **Prior** | Our belief about $A$ *before* seeing any evidence |
    | $P(B)$ | **Evidence** (or marginal likelihood) | How probable is the evidence overall? Acts as a normalizing constant |

    The conceptual flow is:

    $$\text{Posterior} = \frac{\text{Likelihood} \times \text{Prior}}{\text{Evidence}}$$

    **This is literally what learning is.** You start with a prior belief. You observe data. You update your belief. The posterior becomes your new prior when you see more data. This is how ML models learn from data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Worked Example 1: The Medical Test (False Positive Paradox)

    Let us resolve that counterintuitive trap from Section 3.

    - Disease prevalence: $P(D) = 0.01$ (1% of people have the disease)
    - Test sensitivity: $P(+|D) = 0.99$ (test catches 99% of sick people)
    - Test specificity: $P(-|\neg D) = 0.99$ (test correctly clears 99% of healthy people)

    You test positive. What is $P(D|+)$?

    Apply Bayes:

    $$P(D|+) = \frac{P(+|D) \cdot P(D)}{P(+)}$$

    We need $P(+)$, the total probability of testing positive. Use the law of total probability:

    $$P(+) = P(+|D) \cdot P(D) + P(+|\neg D) \cdot P(\neg D)$$
    $$P(+) = 0.99 \times 0.01 + 0.01 \times 0.99 = 0.0099 + 0.0099 = 0.0198$$

    Therefore:

    $$P(D|+) = \frac{0.99 \times 0.01}{0.0198} = \frac{0.0099}{0.0198} = 0.5$$

    **Only 50%.** Even with a 99% accurate test, a positive result from a rare disease means a coin flip. The prior matters enormously. The rarity of the disease (the prior) overwhelms the test accuracy (the likelihood).

    Think about it with raw numbers. In 10,000 people: 100 are sick, and 99 of them test positive. 9,900 are healthy, and 99 of them *also* test positive (false positives). So of the 198 positive results, only 99 are true positives. That is 50%.
    """)
    return


@app.cell
def _(mo):
    mo.image(src="../animations/rendered/BayesTheoremUpdate.gif")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Worked Example 2: Spam Classification

    This is how the original Naive Bayes spam filter works.

    You receive an email containing the word "lottery." Is it spam?

    - $P(\text{spam}) = 0.3$ — 30% of emails are spam (prior)
    - $P(\text{"lottery"}|\text{spam}) = 0.15$ — 15% of spam emails contain "lottery" (likelihood)
    - $P(\text{"lottery"}|\text{not spam}) = 0.005$ — 0.5% of legitimate emails contain "lottery"

    $$P(\text{spam}|\text{"lottery"}) = \frac{P(\text{"lottery"}|\text{spam}) \cdot P(\text{spam})}{P(\text{"lottery"})}$$

    $$P(\text{"lottery"}) = 0.15 \times 0.3 + 0.005 \times 0.7 = 0.045 + 0.0035 = 0.0485$$

    $$P(\text{spam}|\text{"lottery"}) = \frac{0.15 \times 0.3}{0.0485} = \frac{0.045}{0.0485} \approx 0.928$$

    The word "lottery" bumps our spam probability from 30% (prior) to 93% (posterior). That is Bayesian updating in action.

    ### Why Bayes Is the Foundation of Learning

    Every time a model updates its parameters based on data, it is doing a version of Bayesian reasoning. Maximum likelihood estimation, MAP estimation, full Bayesian inference — these are all points on a spectrum of "how seriously do we take Bayes' theorem?"

    > [Bishop PRML Section 1.2](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) gives an excellent treatment of Bayes in the context of pattern recognition.
    > [Chan Ch 2.4](file:///C:/Users/landa/ml-course/textbooks/Chan-Probability.pdf) works through Bayes' theorem with additional examples.
    > [MML Section 6.3](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) connects Bayes directly to ML.
    """)
    return


@app.cell
def _():
    import numpy as np

    # Verify Bayes' theorem: Medical test example
    P_disease = 0.01
    P_pos_given_disease = 0.99      # sensitivity
    P_pos_given_healthy = 0.01      # 1 - specificity

    # Total probability of positive test (law of total probability)
    P_positive = P_pos_given_disease * P_disease + P_pos_given_healthy * (1 - P_disease)

    # Bayes' theorem
    P_disease_given_pos = (P_pos_given_disease * P_disease) / P_positive

    print("=== Medical Test (Bayes' Theorem) ===")
    print(f"P(+)             = {P_positive:.4f}")
    print(f"P(disease | +)   = {P_disease_given_pos:.4f}  ← only 50%!")

    # Spam example
    P_spam = 0.3
    P_lottery_given_spam = 0.15
    P_lottery_given_ham = 0.005
    P_lottery = P_lottery_given_spam * P_spam + P_lottery_given_ham * (1 - P_spam)
    P_spam_given_lottery = (P_lottery_given_spam * P_spam) / P_lottery

    print(f"\n=== Spam Filter (Bayes' Theorem) ===")
    print(f"P(spam | 'lottery') = {P_spam_given_lottery:.3f}  ← prior 0.30 → posterior 0.93")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ### Interactive: Bayesian Updating — The False Positive Paradox

    Use the slider below to change the disease prevalence (prior probability) and see how the posterior probability $P(\text{disease} \mid \text{positive test})$ changes. The test has sensitivity = 0.99 and specificity = 0.95.
    """)
    return


@app.cell
def _(mo):
    prior_slider = mo.ui.slider(
        start=0.001,
        stop=0.5,
        step=0.001,
        value=0.01,
        label="Prior P(disease)",
    )
    prior_slider
    return (prior_slider,)


@app.cell
def _(mo, prior_slider):
    _sensitivity = 0.99
    _specificity = 0.95
    _prior = prior_slider.value

    _p_pos_given_disease = _sensitivity
    _p_pos_given_no_disease = 1 - _specificity
    _p_positive = _p_pos_given_disease * _prior + _p_pos_given_no_disease * (1 - _prior)
    _posterior = (_p_pos_given_disease * _prior) / _p_positive

    mo.md(rf"""
    **Prior P(disease) = {_prior:.3f}**

    | Quantity | Value |
    |---|---|
    | P(+ \| disease) — sensitivity | {_sensitivity} |
    | P(- \| no disease) — specificity | {_specificity} |
    | P(+) — total positive rate | {_p_positive:.5f} |
    | **P(disease \| +) — posterior** | **{_posterior:.4f} ({_posterior*100:.1f}%)** |

    With a prior of {_prior:.3f} ({_prior*100:.1f}%), a positive test result gives you only a **{_posterior*100:.1f}%** chance of actually having the disease.

    {"**The false positive paradox in action:** Even with a highly accurate test, rare diseases produce mostly false positives!" if _posterior < 0.5 else "With higher prevalence, the posterior becomes more meaningful."}
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 5. Random Variables

    A **random variable** is a function that maps outcomes in the sample space to numbers. That is the formal definition, but practically: a random variable is a quantity whose value is uncertain.

    - $X$ = the number showing on a rolled die (discrete)
    - $Y$ = the height of a randomly selected person (continuous)
    - $Z$ = the number of clicks on an ad (discrete)

    ### Discrete Random Variables: PMF

    A **probability mass function** (PMF) gives the probability of each possible value:

    $$p(x) = P(X = x)$$

    For a fair die: $p(1) = p(2) = \cdots = p(6) = 1/6$.

    Requirements: $p(x) \geq 0$ for all $x$, and $\sum_x p(x) = 1$.

    ### Continuous Random Variables: PDF

    A **probability density function** (PDF) $f(x)$ describes the relative likelihood of values. The probability of $X$ falling in an interval is:

    $$P(a \leq X \leq b) = \int_a^b f(x)\, dx$$

    Requirements: $f(x) \geq 0$ for all $x$, and $\int_{-\infty}^{\infty} f(x)\, dx = 1$.

    **Critical point that confuses people:** A PDF value $f(x)$ is a *density*, not a probability. It *can* be greater than 1. For example, a Uniform(0, 0.5) distribution has $f(x) = 2$ for $x \in [0, 0.5]$. That is perfectly valid because the *area under the curve* still equals 1. Think of it like physical density: a small, heavy object has high density, but its total mass is still finite. Similarly, probability can be "concentrated" into a small region, making the density tall, but the total area (total probability) is always 1.

    ### CDF

    The **cumulative distribution function** works for both discrete and continuous:

    $$F(x) = P(X \leq x)$$

    It goes from 0 to 1, is non-decreasing, and for continuous variables: $f(x) = F'(x)$ — the PDF is the derivative of the CDF.

    > [Chan Ch 3.1–3.3](file:///C:/Users/landa/ml-course/textbooks/Chan-Probability.pdf) covers random variables, PMFs, and PDFs carefully.
    """)
    return


@app.cell
def _():
    import numpy as np

    # Discrete RV: fair die PMF
    outcomes = np.arange(1, 7)
    pmf = np.ones(6) / 6

    # Verify PMF requirements: non-negative, sums to 1
    print("=== Discrete: Fair Die PMF ===")
    for x, p in zip(outcomes, pmf):
        print(f"  P(X={x}) = {p:.4f}")
    print(f"  Sum = {pmf.sum():.1f} ✓")

    # Continuous RV: approximate PDF→probability via sampling
    # P(0.5 < Z < 1.5) for Z ~ N(0,1)
    z_samples = rng.standard_normal(1_000_000)
    P_interval = ((z_samples > 0.5) & (z_samples < 1.5)).mean()
    print(f"\n=== Continuous: Standard Normal ===")
    print(f"  P(0.5 < Z < 1.5) ≈ {P_interval:.4f}")

    # CDF: P(Z <= 1.96) should be ~0.975
    P_cdf = (z_samples <= 1.96).mean()
    print(f"  P(Z ≤ 1.96) ≈ {P_cdf:.4f}  (exact: 0.975)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 6. Common Distributions — Know These Cold

    You will see these distributions over and over again in ML. For each one, I will tell you its formula, what it looks like, what its parameters mean, and where it shows up in ML.

    ### Bernoulli Distribution

    **What it models:** A single yes/no trial. A coin flip. A binary classification.

    $$P(X = x) = p^x (1-p)^{1-x}, \quad x \in \{0, 1\}$$

    - **Parameter:** $p$ = probability of success
    - **Mean:** $E[X] = p$
    - **Variance:** $\text{Var}(X) = p(1-p)$
    - **ML connection:** Binary classification. A logistic regression model outputs $p$, which parameterizes a Bernoulli distribution over the label.

    ### Binomial Distribution

    **What it models:** The number of successes in $n$ independent Bernoulli trials.

    $$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

    - **Parameters:** $n$ = number of trials, $p$ = success probability
    - **Mean:** $E[X] = np$
    - **Variance:** $\text{Var}(X) = np(1-p)$
    - **ML connection:** Batch accuracy. If each prediction is correct with probability $p$, the number correct out of $n$ is Binomial.

    ### Poisson Distribution

    **What it models:** The number of events in a fixed interval when events occur at a constant average rate. Website visits per hour. Typos per page. Particle decay events.

    $$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots$$

    - **Parameter:** $\lambda$ = average rate (both the mean and variance)
    - **Mean:** $E[X] = \lambda$
    - **Variance:** $\text{Var}(X) = \lambda$
    - **ML connection:** Count regression (Poisson regression). Modeling event frequencies.

    ### Uniform Distribution

    **What it models:** Complete ignorance. Every outcome is equally likely.

    Continuous on $[a, b]$:

    $$f(x) = \frac{1}{b - a}, \quad a \leq x \leq b$$

    - **Parameters:** $a, b$ = endpoints
    - **Mean:** $E[X] = \frac{a+b}{2}$
    - **Variance:** $\text{Var}(X) = \frac{(b-a)^2}{12}$
    - **ML connection:** Random initialization of weights. Uninformative priors. Random search over hyperparameters.

    ### Gaussian (Normal) Distribution

    **What it models:** Everything. I am only half joking. The Gaussian is the most important distribution in all of statistics and ML.

    $$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

    - **Parameters:** $\mu$ = mean (center), $\sigma^2$ = variance (spread)
    - **Shape:** The classic bell curve, symmetric around $\mu$
    - **Standard normal:** $\mu = 0, \sigma = 1$, written as $\mathcal{N}(0, 1)$
    - **The 68-95-99.7 rule:** About 68% of values fall within 1$\sigma$ of the mean, 95% within 2$\sigma$, 99.7% within 3$\sigma$

    **Why is the Gaussian everywhere?** The Central Limit Theorem (Section 9). Whenever you add up many independent random effects — measurement errors, genetic influences, market fluctuations — the sum tends toward a Gaussian regardless of the underlying distributions. Nature is full of sums of small effects, so Gaussians are everywhere.

    - **ML connection:** Gaussian noise assumptions in regression. Weight initialization ($\mathcal{N}(0, 0.01)$). Gaussian processes. The entire framework of linear regression assumes Gaussian errors. Batch normalization pushes activations toward Gaussian. Variational autoencoders use Gaussian latent spaces.

    > [MML Section 6.5](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) covers the Gaussian and its properties in depth.
    > [Chan Ch 4.6](file:///C:/Users/landa/ml-course/textbooks/Chan-Probability.pdf) is the dedicated Gaussian section.

    ### Exponential Distribution

    **What it models:** The time between events in a Poisson process. Waiting time until the next bus, the next server request, the next failure.

    $$f(x) = \lambda e^{-\lambda x}, \quad x \geq 0$$

    - **Parameter:** $\lambda$ = rate parameter
    - **Mean:** $E[X] = 1/\lambda$
    - **Variance:** $\text{Var}(X) = 1/\lambda^2$
    - **Key property:** Memoryless. $P(X > s + t \mid X > s) = P(X > t)$. The probability of waiting another $t$ minutes is the same regardless of how long you have already waited.
    - **ML connection:** Survival analysis. Modeling inter-arrival times in streaming data.
    """)
    return


@app.cell
def _():
    import numpy as np
    from scipy import stats

    # Sample from each distribution, compare empirical vs theoretical moments
    n = 100_000
    dists = {
        "Bernoulli(0.3)":    (rng.binomial(1, 0.3, n),    0.3, 0.3*0.7),
        "Binomial(20,0.4)":  (rng.binomial(20, 0.4, n),   8.0, 20*0.4*0.6),
        "Poisson(5)":        (rng.poisson(5, n),           5.0, 5.0),
        "Normal(2,3²)":      (rng.normal(2, 3, n),         2.0, 9.0),
        "Exponential(λ=2)":  (rng.exponential(0.5, n),     0.5, 0.25),
    }
    print(f"{'Distribution':<20} {'E[X] theory':>11} {'E[X] sample':>11} {'Var theory':>11} {'Var sample':>11}")
    print("-" * 68)
    for name, (samples, mu, var) in dists.items():
        print(f"{name:<20} {mu:>11.3f} {samples.mean():>11.3f} {var:>11.3f} {samples.var():>11.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Interactive: Distribution Explorer

    Select a distribution and adjust its parameters to see how the shape changes.
    """)
    return


@app.cell
def _(mo):
    dist_type = mo.ui.dropdown(
        options=["Normal", "Binomial", "Poisson", "Exponential", "Uniform"],
        value="Normal",
        label="Distribution",
    )
    dist_mu = mo.ui.slider(start=-5, stop=5, step=0.1, value=0, label="Mean (mu)")
    dist_sigma = mo.ui.slider(start=0.1, stop=5, step=0.1, value=1, label="Std dev (sigma)")
    dist_n = mo.ui.slider(start=1, stop=50, step=1, value=10, label="n (trials)")
    dist_p = mo.ui.slider(start=0.01, stop=0.99, step=0.01, value=0.5, label="p (probability)")
    dist_lam = mo.ui.slider(start=0.1, stop=20, step=0.1, value=5, label="lambda (rate)")
    dist_a = mo.ui.slider(start=-5, stop=4.9, step=0.1, value=0, label="a (lower)")
    dist_b = mo.ui.slider(start=-4.9, stop=10, step=0.1, value=1, label="b (upper)")
    return (dist_type, dist_mu, dist_sigma, dist_n, dist_p, dist_lam, dist_a, dist_b)


@app.cell
def _(dist_type, dist_mu, dist_sigma, dist_n, dist_p, dist_lam, dist_a, dist_b, mo):
    _d = dist_type.value
    if _d == "Normal":
        _controls = mo.hstack([dist_type, dist_mu, dist_sigma])
    elif _d == "Binomial":
        _controls = mo.hstack([dist_type, dist_n, dist_p])
    elif _d == "Poisson":
        _controls = mo.hstack([dist_type, dist_lam])
    elif _d == "Exponential":
        _controls = mo.hstack([dist_type, dist_lam])
    else:
        _controls = mo.hstack([dist_type, dist_a, dist_b])
    _controls
    return


@app.cell
def _(dist_type, dist_mu, dist_sigma, dist_n, dist_p, dist_lam, dist_a, dist_b):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats

    fig_dist, ax_dist = plt.subplots(figsize=(8, 4))
    _d = dist_type.value

    if _d == "Normal":
        _mu = dist_mu.value
        _sig = dist_sigma.value
        _x = np.linspace(_mu - 4 * _sig, _mu + 4 * _sig, 300)
        _y = stats.norm.pdf(_x, _mu, _sig)
        ax_dist.plot(_x, _y, 'b-', lw=2)
        ax_dist.fill_between(_x, _y, alpha=0.3)
        ax_dist.set_title(f"Normal(mu={_mu:.1f}, sigma={_sig:.1f})")
        ax_dist.set_ylabel("Density")

    elif _d == "Binomial":
        _ni = int(dist_n.value)
        _pi = dist_p.value
        _k = np.arange(0, _ni + 1)
        _pmf = stats.binom.pmf(_k, _ni, _pi)
        ax_dist.bar(_k, _pmf, color='steelblue', alpha=0.7)
        ax_dist.set_title(f"Binomial(n={_ni}, p={_pi:.2f})")
        ax_dist.set_ylabel("P(X = k)")
        ax_dist.set_xlabel("k")

    elif _d == "Poisson":
        _la = dist_lam.value
        _k = np.arange(0, int(_la * 3) + 10)
        _pmf = stats.poisson.pmf(_k, _la)
        ax_dist.bar(_k, _pmf, color='steelblue', alpha=0.7)
        ax_dist.set_title(f"Poisson(lambda={_la:.1f})")
        ax_dist.set_ylabel("P(X = k)")
        ax_dist.set_xlabel("k")

    elif _d == "Exponential":
        _la = dist_lam.value
        _x = np.linspace(0, 5 / _la, 300)
        _y = stats.expon.pdf(_x, scale=1 / _la)
        ax_dist.plot(_x, _y, 'b-', lw=2)
        ax_dist.fill_between(_x, _y, alpha=0.3)
        ax_dist.set_title(f"Exponential(lambda={_la:.1f})")
        ax_dist.set_ylabel("Density")

    else:  # Uniform
        _ai = dist_a.value
        _bi = max(dist_b.value, _ai + 0.1)
        _margin = (_bi - _ai) * 0.3
        _x = np.linspace(_ai - _margin, _bi + _margin, 300)
        _y = stats.uniform.pdf(_x, loc=_ai, scale=_bi - _ai)
        ax_dist.plot(_x, _y, 'b-', lw=2)
        ax_dist.fill_between(_x, _y, alpha=0.3)
        ax_dist.set_title(f"Uniform(a={_ai:.1f}, b={_bi:.1f})")
        ax_dist.set_ylabel("Density")

    ax_dist.set_xlabel("x")
    ax_dist.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_dist
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 7. Expectation, Variance, and Covariance

    You said you know basic expected value and variance, so I will move through the definitions quickly but spend more time on the properties and connections you might have missed.

    ### Expected Value

    The **expected value** (mean) is the probability-weighted average of all possible values.

    Discrete: $\quad E[X] = \sum_x x \cdot p(x)$

    Continuous: $\quad E[X] = \int_{-\infty}^{\infty} x \cdot f(x)\, dx$

    For a function of $X$: $\quad E[g(X)] = \sum_x g(x) \cdot p(x)$ (or the integral version).

    ### Linearity of Expectation — This Is Incredibly Useful

    $$E[aX + bY + c] = aE[X] + bE[Y] + c$$

    This holds **regardless of whether $X$ and $Y$ are independent.** That fact is remarkable and catches people off guard. Even if $X$ and $Y$ are deeply correlated, the expected value of their sum is the sum of their expected values.

    This property makes many calculations tractable. If you need $E[X_1 + X_2 + \cdots + X_n]$, you can just sum $E[X_1] + E[X_2] + \cdots + E[X_n]$. No need to figure out the joint distribution.

    ### Variance — Measuring Spread

    $$\text{Var}(X) = E[(X - \mu)^2]$$

    This is the expected squared deviation from the mean. Let me derive the computational formula:

    $$\text{Var}(X) = E[(X - \mu)^2] = E[X^2 - 2\mu X + \mu^2]$$

    By linearity of expectation:

    $$= E[X^2] - 2\mu E[X] + \mu^2 = E[X^2] - 2\mu^2 + \mu^2$$

    $$\boxed{\text{Var}(X) = E[X^2] - (E[X])^2}$$

    This is the "mean of the square minus the square of the mean." Memorize it — you will use it constantly.

    **Key properties:**
    - $\text{Var}(aX + b) = a^2 \text{Var}(X)$ — shifting does not change spread; scaling scales variance by the square
    - $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y)$ — variance of a sum depends on covariance
    - If $X$ and $Y$ are independent: $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$

    **Standard deviation** $\sigma = \sqrt{\text{Var}(X)}$ is in the same units as $X$, which makes it more interpretable than variance.

    ### Covariance — Do They Move Together?

    $$\text{Cov}(X, Y) = E[(X - \mu_X)(Y - \mu_Y)] = E[XY] - E[X]E[Y]$$

    - $\text{Cov}(X, Y) > 0$: when $X$ is above its mean, $Y$ tends to be above its mean too
    - $\text{Cov}(X, Y) < 0$: they tend to move in opposite directions
    - $\text{Cov}(X, Y) = 0$: no *linear* relationship (but there could still be a nonlinear one!)

    **Correlation** normalizes covariance to $[-1, 1]$:

    $$\rho(X, Y) = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

    ### The Covariance Matrix — Bridging to Multivariate Analysis

    For a random vector $\mathbf{X} = (X_1, X_2, \ldots, X_d)^T$, the **covariance matrix** is:

    $$\boldsymbol{\Sigma} = \text{Cov}(\mathbf{X}) = E[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^T]$$

    This is a $d \times d$ matrix where entry $(i,j)$ is $\text{Cov}(X_i, X_j)$. The diagonal entries are the variances.

    The covariance matrix is symmetric and positive semi-definite. Its eigenvectors point in the directions of maximum variance — this is exactly what PCA (Principal Component Analysis) exploits to do dimensionality reduction. The eigenvalues tell you how much variance each direction captures.

    > [MML Section 6.4](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) covers moments (mean, variance, covariance) with clean notation.
    > [Chan Ch 5.3–5.5](file:///C:/Users/landa/ml-course/textbooks/Chan-Probability.pdf) works through covariance and correlation in detail.
    """)
    return


@app.cell
def _():
    import numpy as np

    # Expected value: E[X] for a loaded die
    outcomes = np.arange(1, 7)
    pmf = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.5])  # loaded toward 6
    E_X = (outcomes * pmf).sum()                        # weighted average
    E_X2 = (outcomes**2 * pmf).sum()

    # Variance two ways
    Var_direct = ((outcomes - E_X)**2 * pmf).sum()      # E[(X-μ)²]
    Var_shortcut = E_X2 - E_X**2                        # E[X²] - (E[X])²
    print(f"E[X]  = {E_X:.3f}")
    print(f"Var(X) via definition = {Var_direct:.3f}")
    print(f"Var(X) via shortcut   = {Var_shortcut:.3f}")

    # Linearity: E[3X+5] = 3*E[X]+5
    print(f"\nE[3X+5] = {3*E_X+5:.3f}")

    # Var(3X+5) = 9*Var(X) — the +5 doesn't affect spread
    print(f"Var(3X+5) = {9*Var_direct:.3f}")
    return


@app.cell
def _():
    import numpy as np

    # Covariance and correlation from data
    rng = np.random.default_rng(0)
    n = 50_000

    # Correlated pair: X ~ N(0,1), Y = 2X + noise
    X = rng.standard_normal(n)
    Y = 2 * X + rng.standard_normal(n) * 0.5

    cov_XY = np.cov(X, Y)[0, 1]           # off-diagonal of covariance matrix
    corr_XY = np.corrcoef(X, Y)[0, 1]     # normalized to [-1, 1]
    print(f"Cov(X,Y)  = {cov_XY:.3f}  (positive — they move together)")
    print(f"Corr(X,Y) = {corr_XY:.3f}  (strong positive linear relationship)")

    # Full covariance matrix for 3 variables
    Z = -X + rng.standard_normal(n) * 0.3
    data = np.stack([X, Y, Z])
    cov_matrix = np.cov(data)
    print(f"\nCovariance matrix (3×3):\n{np.round(cov_matrix, 2)}")
    print("Diagonal = variances, off-diagonal = covariances")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 8. Joint, Marginal, and Conditional Distributions

    This is where two or more random variables interact. If Section 3 was about conditional *events*, this section is about conditional *distributions* — a more general and powerful concept.

    ### Joint Distributions

    A **joint distribution** describes the probability of two (or more) random variables simultaneously.

    Discrete: $\quad p(x, y) = P(X = x, Y = y)$

    Continuous: $\quad f(x, y)$ such that $P(X \in A, Y \in B) = \int_A \int_B f(x, y)\, dy\, dx$

    Think of a joint distribution as a two-dimensional landscape. For discrete variables, it is a table. For continuous variables, it is a surface where height represents density. The total volume under the surface equals 1.

    ### Marginal Distributions — Integrating Out

    The **marginal distribution** of $X$ is obtained by summing (or integrating) over all possible values of $Y$:

    Discrete: $\quad p(x) = \sum_y p(x, y)$

    Continuous: $\quad f_X(x) = \int_{-\infty}^{\infty} f(x, y)\, dy$

    This is called "marginalizing out" $Y$. You are collapsing a 2D distribution down to 1D by asking: "What is the overall distribution of $X$, regardless of $Y$?"

    **ML connection:** This comes up constantly. In a mixture model, you marginalize out the latent cluster assignment. In Bayesian inference, you marginalize out the model parameters to get the predictive distribution. Any time you see an integral over hidden or nuisance variables, that is marginalization.

    ### Conditional Distributions — Slicing the Joint

    The **conditional distribution** of $Y$ given $X = x$:

    $$p(y|x) = \frac{p(x, y)}{p(x)}$$

    Geometrically: take the joint distribution, slice it at a particular value of $X$, and renormalize so it sums/integrates to 1. The slice is the conditional distribution.

    This connects back to Bayes' theorem. Writing Bayes in terms of distributions:

    $$p(\theta | \mathcal{D}) = \frac{p(\mathcal{D} | \theta) \cdot p(\theta)}{p(\mathcal{D})}$$

    where $\theta$ is a model parameter and $\mathcal{D}$ is observed data. The posterior distribution over parameters, given data, is proportional to the likelihood times the prior.

    ### The Chain Rule of Probability (for Distributions)

    For multiple variables:

    $$p(x_1, x_2, \ldots, x_n) = p(x_1) \cdot p(x_2 | x_1) \cdot p(x_3 | x_1, x_2) \cdots p(x_n | x_1, \ldots, x_{n-1})$$

    Any joint distribution can be decomposed into a product of conditionals. Autoregressive models (GPT, WaveNet, PixelCNN) generate data one element at a time using exactly this decomposition.

    ### Independence of Random Variables

    $X$ and $Y$ are independent if and only if:

    $$p(x, y) = p(x) \cdot p(y) \quad \text{for all } x, y$$

    Equivalently: $p(y|x) = p(y)$ — knowing $X$ tells you nothing about $Y$.

    **Conditional independence** is even more important in ML: $X \perp Y \mid Z$ means $X$ and $Y$ are independent *once you know* $Z$. In a Bayesian network, conditional independence structure determines which variables influence which others. This is the backbone of graphical models.

    > [MML Section 6.2](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) covers joint and marginal distributions with matrix notation.
    > [Bishop PRML Section 1.2.1–1.2.4](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) gives an excellent treatment connecting these concepts to ML.
    > [Chan Ch 6](file:///C:/Users/landa/ml-course/textbooks/Chan-Probability.pdf) is dedicated to joint distributions.
    """)
    return


@app.cell
def _():
    import numpy as np

    # Joint distribution table from Exercise P4
    joint = np.array([[0.2, 0.1],   # X=0: Y=0, Y=1
                      [0.3, 0.4]])  # X=1: Y=0, Y=1

    # Marginals: sum over the other variable
    P_X = joint.sum(axis=1)   # sum over Y
    P_Y = joint.sum(axis=0)   # sum over X
    print(f"Marginal P(X): {P_X}  (sums to {P_X.sum():.1f})")
    print(f"Marginal P(Y): {P_Y}  (sums to {P_Y.sum():.1f})")

    # Conditional: P(Y=1 | X=1) = P(X=1,Y=1) / P(X=1)
    P_Y1_given_X1 = joint[1, 1] / P_X[1]
    print(f"\nP(Y=1 | X=1) = {joint[1,1]} / {P_X[1]} = {P_Y1_given_X1:.3f}")

    # Independence check: P(X,Y) == P(X)*P(Y) for all entries?
    independent = np.allclose(joint, np.outer(P_X, P_Y))
    print(f"Independent? {independent}  (product table: {np.round(np.outer(P_X, P_Y), 2)})")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 9. The Law of Large Numbers and the Central Limit Theorem

    These two theorems are why statistics works at all. Without them, we could not learn from data.

    ### The Law of Large Numbers (LLN)

    **Statement (informal):** As you collect more and more data, the sample mean converges to the true population mean.

    If $X_1, X_2, \ldots, X_n$ are independent draws from a distribution with mean $\mu$, then:

    $$\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i \xrightarrow{n \to \infty} \mu$$

    **What this means for ML:** Training on more data gives better estimates. When you compute a loss over a batch of data, that batch average approximates the true expected loss. The bigger the batch, the better the approximation. Stochastic gradient descent works because of the LLN — even small batches give *unbiased* estimates of the true gradient, and over many steps, these estimates average out.

    ### The Central Limit Theorem (CLT)

    **Statement (informal):** The sum (or average) of many independent random variables is approximately Gaussian, *regardless of what the individual variables' distributions look like.*

    More precisely: if $X_1, \ldots, X_n$ are i.i.d. with mean $\mu$ and variance $\sigma^2$, then as $n \to \infty$:

    $$\frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)$$

    **This is why the Gaussian distribution is everywhere.** Any quantity that arises as the sum of many small, independent contributions will be approximately Gaussian. Heights (sum of many genetic and environmental factors). Measurement errors (sum of many small sources of error). Financial returns over long periods. Test scores.

    **What this means for ML:**
    - It justifies the Gaussian noise assumption in linear regression — if errors are the sum of many small, independent effects, the CLT says they will be approximately Gaussian.
    - It explains why batch statistics in batch normalization are approximately Gaussian.
    - It tells you that confidence intervals and hypothesis tests (which assume Gaussianity) work even when the underlying data is not Gaussian, as long as your sample is large enough.

    **Intuitive demonstration:** Roll one die — the distribution is flat (uniform). Roll two dice and average — the distribution is triangular. Roll ten dice and average — it already looks like a bell curve. Roll 100 dice and average — it is essentially Gaussian. The underlying distribution was uniform, but the average became Gaussian. That is the CLT.

    > [Chan Ch 7.2](file:///C:/Users/landa/ml-course/textbooks/Chan-Probability.pdf) covers the LLN and CLT with proofs and visualizations.
    > [MML Section 6.7](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) discusses these limit theorems in the ML context.
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    # Law of Large Numbers: running average converges to true mean
    rng = np.random.default_rng(7)
    true_mean = 3.5  # E[fair die]
    rolls = rng.integers(1, 7, size=5000)
    running_avg = np.cumsum(rolls) / np.arange(1, len(rolls) + 1)

    fig_lln, ax_lln = plt.subplots(figsize=(8, 3))
    ax_lln.plot(running_avg, lw=1)
    ax_lln.axhline(true_mean, color='r', linestyle='--', label=f'True mean = {true_mean}')
    ax_lln.set_xlabel('Number of rolls')
    ax_lln.set_ylabel('Running average')
    ax_lln.set_title('Law of Large Numbers: sample mean → population mean')
    ax_lln.legend()
    ax_lln.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_lln
    return


@app.cell
def _(mo):
    mo.image(src="../animations/rendered/CentralLimitTheorem.gif")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 10. Information Theory Basics

    Information theory started as a theory of communication (Shannon, 1948), but it turns out to be deeply connected to probability and ML. I will keep this section brief but cover the concepts you will encounter constantly.

    ### Entropy — Measuring Uncertainty

    The **entropy** of a discrete random variable $X$ with PMF $p(x)$:

    $$H(X) = -\sum_x p(x) \log p(x)$$

    (We typically use $\log_2$ for bits or $\ln$ for nats.)

    Entropy measures *how surprised you expect to be* when you observe $X$. High entropy means high uncertainty (outcomes are spread out). Low entropy means low uncertainty (one outcome dominates).

    - A fair coin: $H = -0.5 \log 0.5 - 0.5 \log 0.5 = 1$ bit. Maximum uncertainty.
    - A biased coin ($p = 0.99$): $H \approx 0.08$ bits. Almost no uncertainty.
    - A sure thing ($p = 1$): $H = 0$. No surprise at all.

    ### KL Divergence — Distance Between Distributions

    The **Kullback-Leibler divergence** from distribution $q$ to distribution $p$:

    $$D_{\text{KL}}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)}$$

    KL divergence measures how much *extra* surprise you experience if you think the distribution is $q$ but it is actually $p$. Key properties:

    - $D_{\text{KL}}(p \| q) \geq 0$ (always non-negative)
    - $D_{\text{KL}}(p \| q) = 0$ if and only if $p = q$
    - It is **not symmetric**: $D_{\text{KL}}(p \| q) \neq D_{\text{KL}}(q \| p)$ in general (so it is not a true "distance")

    **ML connection:** Variational inference minimizes $D_{\text{KL}}(q \| p)$ where $q$ is an approximate posterior and $p$ is the true posterior. The choice of KL direction matters — forward vs. reverse KL produces different approximation behaviors.

    ### Cross-Entropy — The Loss Function You Will Use Everywhere

    The **cross-entropy** between true distribution $p$ and predicted distribution $q$:

    $$H(p, q) = -\sum_x p(x) \log q(x)$$

    Notice the relationship:

    $$H(p, q) = H(p) + D_{\text{KL}}(p \| q)$$

    Since $H(p)$ is fixed (it is a property of the true distribution), **minimizing cross-entropy is equivalent to minimizing KL divergence.** This is why cross-entropy loss works: by minimizing cross-entropy between the true labels and your model's predictions, you are making your model's distribution as close to the true distribution as possible.

    When you train a classifier with cross-entropy loss, you are doing maximum likelihood estimation. When you train a binary classifier with binary cross-entropy, the loss for a single example with true label $y \in \{0, 1\}$ and predicted probability $\hat{p}$ is:

    $$\mathcal{L} = -[y \log \hat{p} + (1-y) \log(1-\hat{p})]$$

    This is cross-entropy between a Bernoulli($y$) and a Bernoulli($\hat{p}$). Every time you call `nn.CrossEntropyLoss()` in PyTorch, this is what is happening under the hood.

    > [MML Section 6.6](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) introduces entropy and KL divergence.
    > [Bishop PRML Section 1.6](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) covers information theory and its connection to ML beautifully.
    > [Murphy PML1 Section 2.8](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf) provides a modern treatment with ML applications.
    """)
    return


@app.cell
def _():
    import numpy as np

    def entropy(p):
        """H(p) in nats (using ln)."""
        p = np.asarray(p, dtype=float)
        p = p[p > 0]
        return -np.sum(p * np.log(p))

    def kl_divergence(p, q):
        """D_KL(p || q) in nats."""
        p, q = np.asarray(p, float), np.asarray(q, float)
        mask = p > 0
        return np.sum(p[mask] * np.log(p[mask] / q[mask]))

    def cross_entropy(p, q):
        """H(p, q) in nats."""
        p, q = np.asarray(p, float), np.asarray(q, float)
        mask = p > 0
        return -np.sum(p[mask] * np.log(q[mask]))

    # Compare: fair die vs loaded die
    p_fair = np.ones(6) / 6
    p_loaded = np.array([0.4, 0.2, 0.15, 0.1, 0.1, 0.05])

    print(f"Entropy(fair)   = {entropy(p_fair):.4f} nats  (maximum uncertainty)")
    print(f"Entropy(loaded) = {entropy(p_loaded):.4f} nats  (less uncertain)")
    print(f"KL(fair || loaded) = {kl_divergence(p_fair, p_loaded):.4f}  (asymmetric!)")
    print(f"KL(loaded || fair) = {kl_divergence(p_loaded, p_fair):.4f}")
    print(f"Cross-entropy H(fair, loaded) = {cross_entropy(p_fair, p_loaded):.4f}")
    print(f"Verify: H(p) + KL(p||q) = {entropy(p_fair) + kl_divergence(p_fair, p_loaded):.4f} = H(p,q)")
    return


@app.cell
def _():
    import numpy as np

    # Binary cross-entropy loss — the classification workhorse
    # True label y=1, model predicts p_hat
    y_true = 1
    p_hats = [0.01, 0.1, 0.5, 0.9, 0.99]

    print("Binary cross-entropy loss for y=1:")
    for p_hat in p_hats:
        bce = -(y_true * np.log(p_hat) + (1 - y_true) * np.log(1 - p_hat))
        print(f"  p_hat={p_hat:.2f} → loss={bce:.4f}")
    print("Lower p_hat → higher loss (model is wrong and gets punished)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Summary: The Key Ideas

    | Concept | One-line summary | Where it shows up in ML |
    |---|---|---|
    | Bayes' theorem | Update beliefs with evidence | All Bayesian methods, posterior inference |
    | Conditional probability | Restrict to relevant outcomes | Every conditional model, chain rule decompositions |
    | Gaussian distribution | Bell curve, sum of small effects | Regression noise, weight init, latent spaces |
    | Covariance matrix | Multivariate spread and correlation | PCA, Gaussian processes, feature analysis |
    | Marginalization | Integrate out nuisance variables | Mixture models, Bayesian prediction |
    | CLT | Sums become Gaussian | Justifies Gaussian assumptions |
    | Cross-entropy | Measures prediction quality | The default classification loss function |
    | KL divergence | Distribution mismatch | VAEs, variational inference, regularization |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Practice Exercises

    ### Pen-and-Paper Problems

    **P1.** You have two coins. Coin A is fair ($p = 0.5$). Coin B is biased ($p = 0.8$). You pick a coin uniformly at random and flip it. It comes up heads. What is the probability you picked Coin A? (Use Bayes' theorem.)

    **P2.** Derive $\text{Var}(aX + b) = a^2 \text{Var}(X)$ from the definition $\text{Var}(X) = E[(X - \mu)^2]$.

    **P3.** Suppose $X \sim \text{Poisson}(\lambda = 3)$. Compute $P(X = 0)$, $P(X = 1)$, and $P(X \leq 2)$.

    **P4.** Two discrete random variables have the following joint distribution:

    |  | $Y=0$ | $Y=1$ |
    |---|---|---|
    | $X=0$ | 0.2 | 0.1 |
    | $X=1$ | 0.3 | 0.4 |

    Compute: (a) the marginal distributions of $X$ and $Y$, (b) $P(Y=1 \mid X=1)$, (c) are $X$ and $Y$ independent?

    **P5.** Prove that if $X$ and $Y$ are independent, then $\text{Cov}(X, Y) = 0$. (Hint: use $E[XY] = E[X]E[Y]$ for independent variables.)

    **P6.** Compute the entropy of a 4-sided fair die and a loaded die with probabilities $(0.7, 0.1, 0.1, 0.1)$. Which has higher entropy and why?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Python Simulation Exercises

    **S1. Verify the False Positive Paradox (Bayes' theorem)**
    """)
    return


@app.cell
def _():
    import numpy as np

    # Simulate 1,000,000 people
    n_people = 1_000_000
    prevalence = 0.01
    sensitivity = 0.99
    specificity = 0.99

    # Generate disease status
    has_disease = rng.random(n_people) < prevalence

    # Generate test results
    test_positive = np.where(
        has_disease,
        rng.random(n_people) < sensitivity,   # true positive
        rng.random(n_people) < (1 - specificity)  # false positive
    )

    # Among those who tested positive, what fraction actually has the disease?
    positive_mask = test_positive
    p_disease_given_positive = has_disease[positive_mask].mean()
    print(f"P(disease | positive test) = {p_disease_given_positive:.3f}")
    # Should be approximately 0.50
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **S2. Verify the Central Limit Theorem**
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    # Take means of increasing numbers of Uniform(0,1) samples
    sample_sizes = [1, 2, 5, 30]
    fig_clt, axes_clt = plt.subplots(1, 4, figsize=(16, 3))

    for ax, n in zip(axes_clt, sample_sizes):
        means = [rng.uniform(0, 1, n).mean() for _ in range(10000)]
        ax.hist(means, bins=50, density=True, alpha=0.7)
        ax.set_title(f'Mean of {n} uniform samples')
        ax.set_xlim(0, 1)

    plt.tight_layout()
    fig_clt
    # Watch the distribution go from flat (n=1) to Gaussian (n=30)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **S3. Explore Entropy**
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    def entropy(probs):
        """Compute entropy in bits."""
        probs = np.array(probs)
        probs = probs[probs > 0]  # avoid log(0)
        return -np.sum(probs * np.log2(probs))

    # Vary bias of a coin from 0 to 1
    ps = np.linspace(0.001, 0.999, 100)
    entropies = [entropy([p, 1 - p]) for p in ps]

    fig_entropy, ax_entropy = plt.subplots(figsize=(6, 4))
    ax_entropy.plot(ps, entropies)
    ax_entropy.set_xlabel('P(heads)')
    ax_entropy.set_ylabel('Entropy (bits)')
    ax_entropy.set_title('Entropy of a Bernoulli distribution')
    ax_entropy.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_entropy
    # Maximum entropy at p=0.5 (maximum uncertainty)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **S4. Bayesian Updating — Watch the Posterior Evolve**
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import beta

    # You have a coin with unknown bias. You flip it and observe outcomes.
    # Prior: Beta(1,1) = Uniform (we know nothing)
    # After each flip, posterior updates.

    rng = np.random.default_rng(42)
    true_p = 0.7
    flips = rng.random(100) < true_p  # 100 flips of a biased coin

    x_beta = np.linspace(0, 1, 200)
    a_prior, b_prior = 1, 1  # Beta prior parameters

    fig_bayes, axes_bayes = plt.subplots(2, 3, figsize=(14, 8))
    checkpoints = [0, 1, 3, 10, 30, 100]

    for ax, n_obs in zip(axes_bayes.flatten(), checkpoints):
        a_post = a_prior + flips[:n_obs].sum()
        b_post = b_prior + n_obs - flips[:n_obs].sum()
        ax.plot(x_beta, beta.pdf(x_beta, a_post, b_post), 'b-', lw=2)
        ax.axvline(true_p, color='r', linestyle='--', label=f'True p={true_p}')
        ax.set_title(f'After {n_obs} flips (a={a_post:.0f}, b={b_post:.0f})')
        ax.set_xlim(0, 1)
        ax.legend()

    plt.tight_layout()
    fig_bayes
    # Watch the posterior concentrate around the true value as data accumulates
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## What Comes Next

    With probability foundations in place, you have the language needed for the rest of this course. In **Part 1**, we will start building actual ML models, and you will see every concept from this module in action:

    - Bayes' theorem drives Naive Bayes classification and all Bayesian methods
    - The Gaussian distribution underlies linear regression
    - Cross-entropy is the loss function for logistic regression and neural networks
    - Joint and conditional distributions are the foundation of graphical models
    - The covariance matrix drives PCA and feature analysis

    You do not need to have memorized every formula. But you do need the *intuition*: what conditional probability means, why Bayes' theorem flips your priors, how distributions describe uncertainty, and why cross-entropy measures prediction quality. If you have that intuition, the formulas will follow.

    > **Primary references for this module:**
    > - [Chan — Introduction to Probability for Data Science](file:///C:/Users/landa/ml-course/textbooks/Chan-Probability.pdf) — Chapters 1–7 cover everything in this lecture with excellent examples and visualizations
    > - [MML Chapter 6](file:///C:/Users/landa/ml-course/textbooks/MML.pdf) — Probability and Distributions, compact and ML-focused
    > - [Bishop PRML Chapter 1](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) — Introduction, particularly Sections 1.2 (Probability Theory) and 1.6 (Information Theory)
    > - [Murphy PML1 Chapter 2](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML1.pdf) — Probability: Univariate Models, a modern and thorough treatment
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Code It: Implementation Exercises

    Now it is your turn. Each exercise below gives you a problem and a skeleton — fill in the missing code. These reinforce the math-to-code translation for every major concept in this module.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 1: Bayes' Theorem from Scratch

    A factory has two machines. Machine A produces 60% of items, Machine B produces 40%. Machine A has a 2% defect rate, Machine B has a 5% defect rate. An item is found defective. What is the probability it came from Machine A?

    Compute `P_A_given_defective` using Bayes' theorem.
    """)
    return


@app.cell
def _():
    # Exercise 1: Bayes' theorem — factory defect problem
    P_A = 0.6            # prior: item from Machine A
    P_B = 0.4            # prior: item from Machine B
    P_def_A = 0.02       # P(defective | Machine A)
    P_def_B = 0.05       # P(defective | Machine B)

    # TODO: compute total probability of defect (law of total probability)
    P_defective = ...

    # TODO: apply Bayes' theorem
    P_A_given_defective = ...

    # print(f"P(Machine A | defective) = {P_A_given_defective:.4f}")
    # Expected answer: ~0.375
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 2: Simulate the Law of Large Numbers

    Roll a biased die (P(6) = 0.5, others share the remaining 0.5 equally) 10,000 times. Plot the running mean and show it converges to the true expected value.

    Hint: compute the true E[X] from the PMF first.
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    # Exercise 2: LLN with a biased die
    pmf = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.5])  # P(1)..P(6)
    outcomes = np.arange(1, 7)

    # TODO: compute the true expected value from the PMF
    true_mean = ...

    # TODO: sample 10,000 rolls using np.random.choice with the pmf as weights
    rolls = ...

    # TODO: compute running average (cumulative sum / index)
    running_avg = ...

    # Uncomment to plot:
    # fig, ax = plt.subplots(figsize=(8, 3))
    # ax.plot(running_avg, lw=1)
    # ax.axhline(true_mean, color='r', linestyle='--', label=f'E[X] = {true_mean:.2f}')
    # ax.set_xlabel('Number of rolls')
    # ax.set_ylabel('Running average')
    # ax.set_title('LLN: biased die')
    # ax.legend()
    # ax.grid(True, alpha=0.3)
    # plt.tight_layout()
    # fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 3: Variance — Two Ways

    Given a discrete distribution with outcomes [1, 2, 3, 4, 5] and PMF [0.1, 0.2, 0.4, 0.2, 0.1], compute the variance using:
    1. The definition: $\text{Var}(X) = E[(X - \mu)^2]$
    2. The shortcut: $\text{Var}(X) = E[X^2] - (E[X])^2$

    Verify they match.
    """)
    return


@app.cell
def _():
    import numpy as np

    # Exercise 3: Compute variance two ways
    outcomes = np.array([1, 2, 3, 4, 5])
    pmf = np.array([0.1, 0.2, 0.4, 0.2, 0.1])

    # TODO: compute E[X]
    E_X = ...

    # TODO: compute Var(X) via definition E[(X - mu)^2]
    var_definition = ...

    # TODO: compute Var(X) via shortcut E[X^2] - (E[X])^2
    E_X2 = ...
    var_shortcut = ...

    # print(f"E[X] = {E_X:.3f}")
    # print(f"Var (definition) = {var_definition:.3f}")
    # print(f"Var (shortcut)   = {var_shortcut:.3f}")
    # print(f"Match: {np.isclose(var_definition, var_shortcut)}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Exercise 4: Cross-Entropy Loss

    A 3-class classifier outputs predicted probabilities `q = [0.7, 0.2, 0.1]` for a sample whose true class is 0 (one-hot: `p = [1, 0, 0]`).

    1. Compute the cross-entropy loss $H(p, q) = -\sum p_i \log q_i$
    2. Now suppose the model predicts `q = [0.3, 0.4, 0.3]` instead. Compute the loss again.
    3. Which prediction has lower loss? Why?
    """)
    return


@app.cell
def _():
    import numpy as np

    # Exercise 4: Cross-entropy loss for classification
    p_true = np.array([1, 0, 0])  # one-hot label: class 0

    q_good = np.array([0.7, 0.2, 0.1])  # confident and correct
    q_bad = np.array([0.3, 0.4, 0.3])   # uncertain and wrong

    # TODO: compute cross-entropy H(p, q) = -sum(p * log(q))
    loss_good = ...
    loss_bad = ...

    # print(f"Loss (good prediction): {loss_good:.4f}")
    # print(f"Loss (bad prediction):  {loss_bad:.4f}")
    # print(f"Better model has lower loss: {'good' if loss_good < loss_bad else 'bad'}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Exercise 5: Simulate the CLT with an Exponential Distribution

    The exponential distribution is heavily right-skewed — nothing like a Gaussian. Yet the CLT says that the mean of many exponential samples will be approximately Gaussian.

    Draw `n_samples=10000` means, each computed from `k` exponential($\lambda=2$) draws, for `k` in [1, 5, 30, 100]. Plot histograms and watch the shape become Gaussian.
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    # Exercise 5: CLT with exponential distribution
    lam = 2.0
    n_samples = 10_000
    k_values = [1, 5, 30, 100]

    fig, axes = plt.subplots(1, 4, figsize=(16, 3))

    for ax, k in zip(axes, k_values):
        # TODO: generate (n_samples, k) exponential draws and take row means
        # Hint: rng.exponential(scale=1/lam, size=(n_samples, k)).mean(axis=1)
        means = ...

        # Uncomment to plot:
        # ax.hist(means, bins=50, density=True, alpha=0.7)
        # ax.set_title(f'Mean of {k} Exp samples')
        pass

    # plt.tight_layout()
    # fig
    return


if __name__ == "__main__":
    app.run()
