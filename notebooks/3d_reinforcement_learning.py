import marimo

app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Module 3D: Reinforcement Learning

    Everything you have learned so far --- regression, classification, generative models, transformers --- falls into two paradigms: supervised learning (given inputs and correct outputs, learn the mapping) and unsupervised learning (given inputs only, find structure). Reinforcement learning is neither. It is a fundamentally different way of thinking about learning, and it is the framework behind game-playing agents, robotic control, and the fine-tuning process that makes modern language models useful.

    This module is your introduction. We will build the theory from the ground up, starting from the mathematical framework, working through the classical algorithms, and arriving at the deep RL methods and RLHF pipelines that dominate current practice. The primary reference is [Sutton & Barto, Chapter 1](file:///C:/Users/landa/ml-course/textbooks/Sutton-RL.pdf), which gives an excellent high-level motivation before diving into formalism.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 1. A Different Paradigm

    In supervised learning, someone hands you a dataset of (input, correct answer) pairs. You optimize a loss function. The data is fixed. In reinforcement learning, there is no dataset at all --- at least not one that exists before the agent starts acting.

    The setup is this: an **agent** exists inside an **environment**. At each time step, the agent observes a **state**, chooses an **action**, and receives a **reward**. The action changes the environment, producing a new state. The agent's goal is to learn a **policy** --- a mapping from states to actions --- that maximizes the total reward accumulated over time.

    This creates a feedback loop that does not exist in supervised learning. The agent's actions affect what data it sees next. If it always exploits the best-known action, it may miss better options it has never tried. If it always explores random actions, it wastes time on things it already knows are bad. This is the **exploration-exploitation tradeoff**, and it is unique to RL. See [Sutton & Barto, Section 2.1](file:///C:/Users/landa/ml-course/textbooks/Sutton-RL.pdf) for a thorough treatment via the multi-armed bandit problem.

    There is another subtlety: **delayed reward**. In chess, you may not know whether a move was good until fifty moves later when you win or lose. The agent must solve a **credit assignment problem** --- figuring out which past actions contributed to current rewards. This is fundamentally harder than the gradient-based credit assignment you are used to in backpropagation.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 2. Markov Decision Processes (MDPs)

    The mathematical framework for RL is the **Markov Decision Process**. An MDP is defined by a tuple $(S, A, P, R, \gamma)$:

    - **$S$**: the set of states the environment can be in
    - **$A$**: the set of actions available to the agent (possibly state-dependent, $A(s)$)
    - **$P(s' | s, a)$**: the **transition probability** --- the probability of moving to state $s'$ given current state $s$ and action $a$
    - **$R(s, a, s')$**: the **reward** received for the transition $s \xrightarrow{a} s'$
    - **$\gamma \in [0, 1)$**: the **discount factor**, which controls how much the agent values future rewards relative to immediate ones

    The defining property is the **Markov property**: the probability of the next state depends only on the current state and action, not on the history of how the agent got there. Formally:

    $$P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, \ldots) = P(s_{t+1} | s_t, a_t)$$

    This is a strong assumption, but it is also what makes the problem tractable. If the current state encodes everything relevant about the past, then the agent does not need memory --- it only needs to react to the present. In practice, we design states to satisfy this property (or approximately satisfy it). See [Sutton & Barto, Section 3.1](file:///C:/Users/landa/ml-course/textbooks/Sutton-RL.pdf) for the formal MDP definition.

    **Episodic vs. continuing tasks.** Some tasks have a natural endpoint --- a game ends, a robot reaches its target, an episode terminates. These are **episodic tasks**. Others run indefinitely --- a thermostat controlling room temperature, a trading agent in a market. These are **continuing tasks**. The discount factor $\gamma$ is especially important for continuing tasks, because without it the total reward could be infinite.

    **The agent's goal** is to find a policy $\pi$ that maximizes the **expected cumulative discounted return**:

    $$\max_\pi \; \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t R_t \right]$$

    When $\gamma = 0$, the agent is myopic --- it only cares about the immediate reward. As $\gamma \to 1$, the agent is far-sighted, valuing future rewards almost as much as present ones. Choosing $\gamma$ is a modeling decision that shapes the agent's behavior. See [Sutton & Barto, Section 3.3](file:///C:/Users/landa/ml-course/textbooks/Sutton-RL.pdf) for the discussion of returns and episodes.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 3. Value Functions and the Bellman Equations

    A **policy** $\pi(a|s)$ gives the probability of taking action $a$ in state $s$. Given a policy, we want to evaluate how good it is. We do this with value functions.

    **State-value function.** The value of a state $s$ under policy $\pi$ is the expected return starting from $s$ and following $\pi$ thereafter:

    $$V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t R_t \;\middle|\; S_0 = s \right]$$

    **Action-value function.** The value of taking action $a$ in state $s$ and then following $\pi$:

    $$Q^\pi(s, a) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t R_t \;\middle|\; S_0 = s, A_0 = a \right]$$

    The relationship between them is straightforward: $V^\pi(s) = \sum_a \pi(a|s) \, Q^\pi(s, a)$. You average the action-values over the policy.

    **The Bellman equation.** Here is where the recursion comes in. Let me derive it step by step.

    Start with the definition of $V^\pi(s)$:

    $$V^\pi(s) = \mathbb{E}_\pi[R_0 + \gamma R_1 + \gamma^2 R_2 + \cdots \mid S_0 = s]$$

    Factor out the first reward:

    $$= \mathbb{E}_\pi[R_0 + \gamma(R_1 + \gamma R_2 + \cdots) \mid S_0 = s]$$

    The term in parentheses is the return starting from $S_1$, which is $V^\pi(S_1)$ by definition. So:

    $$= \mathbb{E}_\pi[R_0 + \gamma V^\pi(S_1) \mid S_0 = s]$$

    Now expand this expectation over actions and next states. The agent chooses action $a$ with probability $\pi(a|s)$, then the environment transitions to $s'$ with probability $P(s'|s,a)$, yielding reward $R(s,a,s')$:

    $$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \Big[ R(s,a,s') + \gamma V^\pi(s') \Big]$$

    This is the **Bellman equation for $V^\pi$**. It says: the value of a state equals the expected immediate reward plus the discounted value of the next state, averaged over all possible actions and transitions. It is a system of $|S|$ equations in $|S|$ unknowns. See [Sutton & Barto, Section 3.5](file:///C:/Users/landa/ml-course/textbooks/Sutton-RL.pdf) for this derivation.

    The **optimal value functions** $V^*(s)$ and $Q^*(s,a)$ correspond to the best possible policy. The Bellman optimality equation replaces the average over actions with a maximum:

    $$V^*(s) = \max_a \sum_{s'} P(s'|s,a) \Big[ R(s,a,s') + \gamma V^*(s') \Big]$$

    If you have $Q^*$, the optimal policy is trivial: just pick $\arg\max_a Q^*(s,a)$. This is why much of RL focuses on learning $Q^*$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 4. Dynamic Programming (When You Know the Model)

    If you know the transition probabilities $P(s'|s,a)$ and the rewards $R(s,a,s')$ --- that is, you have a complete model of the environment --- you can solve the MDP exactly using **dynamic programming**.

    **Policy evaluation.** Given a fixed policy $\pi$, compute $V^\pi$ by iterating the Bellman equation as an update rule. Initialize $V(s)$ arbitrarily, then sweep through all states, repeatedly applying:

    $$V(s) \leftarrow \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)\Big[R(s,a,s') + \gamma V(s')\Big]$$

    This converges to $V^\pi$. See [Sutton & Barto, Section 4.1](file:///C:/Users/landa/ml-course/textbooks/Sutton-RL.pdf).

    **Policy improvement.** Once you have $V^\pi$, you can improve $\pi$ by acting greedily:

    $$\pi'(s) = \arg\max_a \sum_{s'} P(s'|s,a)\Big[R(s,a,s') + \gamma V^\pi(s')\Big]$$

    The **policy improvement theorem** guarantees that $\pi'$ is at least as good as $\pi$, and strictly better unless $\pi$ was already optimal.

    **Policy iteration** alternates between evaluation and improvement: evaluate $\pi$ to get $V^\pi$, improve to get $\pi'$, evaluate $\pi'$ to get $V^{\pi'}$, and so on. This converges to the optimal policy in a finite number of iterations. See [Sutton & Barto, Section 4.3](file:///C:/Users/landa/ml-course/textbooks/Sutton-RL.pdf).

    **Value iteration** collapses evaluation and improvement into a single update by applying the Bellman optimality equation directly:

    $$V(s) \leftarrow \max_a \sum_{s'} P(s'|s,a)\Big[R(s,a,s') + \gamma V(s')\Big]$$

    This converges to $V^*$, from which you extract the optimal policy.

    **The catch:** DP requires a complete model. In nearly all interesting problems --- robotics, games with complex physics, real-world decision-making --- you do not have $P(s'|s,a)$. This motivates model-free methods.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 5. Monte Carlo Methods (Model-Free, Episodic)

    Monte Carlo methods learn from experience --- actual episodes of interaction with the environment. No model required. The idea is simple: to estimate $V(s)$, just run many episodes, and average the returns observed after visiting $s$.

    **First-visit MC** averages the return following the *first* time $s$ is visited in each episode. **Every-visit MC** averages the return following *every* time $s$ is visited. Both converge to $V^\pi(s)$ as the number of episodes grows. See [Sutton & Barto, Section 5.1](file:///C:/Users/landa/ml-course/textbooks/Sutton-RL.pdf).

    For **control** (finding the optimal policy), we estimate $Q(s,a)$ instead of $V(s)$, because with $Q$-values you can improve the policy without needing a model --- just pick $\arg\max_a Q(s,a)$.

    There is a problem: if the policy is deterministic, many state-action pairs may never be visited. Two solutions:

    1. **Exploring starts**: every state-action pair has a nonzero probability of being the starting point of an episode. This is often impractical.
    2. **$\varepsilon$-greedy policies**: with probability $1 - \varepsilon$ take the greedy action, with probability $\varepsilon$ take a random action. This ensures all state-action pairs are visited eventually.

    MC methods have a clear advantage: they do not bootstrap (use their own estimates to update). This means they are unbiased --- they converge to the true value. But they have high variance, because a single return depends on many random transitions, and they can only learn after complete episodes.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 6. Temporal Difference Learning --- The Key Innovation

    Temporal difference (TD) learning combines the best of DP and MC. Like MC, it learns from experience without a model. Like DP, it bootstraps --- it updates estimates based on other estimates, without waiting for the episode to end. TD is arguably the single most important idea in RL. See [Sutton & Barto, Section 6.1](file:///C:/Users/landa/ml-course/textbooks/Sutton-RL.pdf).

    **TD(0) update.** After observing a transition $s \xrightarrow{a} r, s'$, update:

    $$V(s) \leftarrow V(s) + \alpha \Big[ R + \gamma V(s') - V(s) \Big]$$

    The term $\delta = R + \gamma V(s') - V(s)$ is the **TD error**. It measures the difference between the current estimate $V(s)$ and the improved estimate $R + \gamma V(s')$. The learning rate $\alpha$ controls how far you step toward the new estimate.

    Compare this to the MC update, which would be $V(s) \leftarrow V(s) + \alpha[G - V(s)]$, where $G$ is the actual return at the end of the episode. TD replaces the actual return $G$ with the bootstrapped estimate $R + \gamma V(s')$. This means TD can update after every single step, not just at the end of an episode.

    **Bias-variance tradeoff.** TD introduces bias because $V(s')$ is itself an estimate, not the true value. MC is unbiased because $G$ is a sample of the true return. But MC has high variance because $G$ depends on the entire trajectory --- every random action and transition until the episode ends. TD's estimates depend on fewer random variables (just the immediate reward and next state), so they have lower variance. In practice, TD's lower variance usually wins, especially in continuing tasks where MC is not even applicable. See [Sutton & Barto, Section 6.2](file:///C:/Users/landa/ml-course/textbooks/Sutton-RL.pdf) for the comparison.
    """)
    return


@app.cell
def _(mo):
    mo.image(src="../animations/rendered/TDLearningUpdate.gif")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### TD Control: SARSA and Q-Learning

    To find optimal policies, we need TD methods for action-values.

    **SARSA** (State, Action, Reward, State, Action) is **on-policy** TD control. At each step, the agent is in state $s$, takes action $a$ (from its current policy), observes reward $R$ and next state $s'$, then picks $a'$ from the current policy, and updates:

    $$Q(s,a) \leftarrow Q(s,a) + \alpha \Big[ R + \gamma Q(s',a') - Q(s,a) \Big]$$

    Because SARSA uses the action $a'$ actually chosen by the current policy, it learns the value of the policy it is following (including its exploration). See [Sutton & Barto, Section 6.4](file:///C:/Users/landa/ml-course/textbooks/Sutton-RL.pdf).

    **Q-learning** is **off-policy** TD control --- and one of the most important algorithms in all of RL. The update is:

    $$Q(s,a) \leftarrow Q(s,a) + \alpha \Big[ R + \gamma \max_{a'} Q(s',a') - Q(s,a) \Big]$$

    The critical difference is the $\max$. Instead of using the action $a'$ that the agent actually takes, Q-learning uses the best possible action. This means Q-learning learns $Q^*$ --- the optimal action-value function --- regardless of what policy the agent follows during training (the "behavior policy"). The behavior policy just needs to keep exploring. This separation of the learned policy from the behavior policy is what "off-policy" means. See [Sutton & Barto, Section 6.5](file:///C:/Users/landa/ml-course/textbooks/Sutton-RL.pdf).

    Q-learning was a landmark result. With tabular representations (a table entry for every state-action pair), it provably converges to $Q^*$ under reasonable conditions. But what happens when the state space is enormous --- say, raw pixels from a video game? You cannot maintain a table with billions of entries. This is where deep learning enters the picture.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 7. Deep RL --- When Tables Are Not Enough

    ### Deep Q-Networks (DQN)

    The idea is natural: replace the Q-table with a neural network $Q(s, a; \theta)$ that takes a state as input and outputs Q-values for all actions. Train the network by minimizing the TD error:

    $$L(\theta) = \mathbb{E}\Big[\big(R + \gamma \max_{a'} Q(s', a'; \theta^{-}) - Q(s, a; \theta)\big)^2\Big]$$

    But naively applying this does not work. Neural network function approximation with RL is notoriously unstable. DeepMind's 2013/2015 DQN paper introduced two crucial tricks:

    **Experience replay.** Instead of training on transitions in order, store transitions $(s, a, r, s')$ in a replay buffer and sample random mini-batches. This breaks the temporal correlations between consecutive transitions (which violate the i.i.d. assumption that SGD relies on) and makes much more efficient use of data by reusing transitions multiple times.

    **Target network.** Use a separate "target" network with parameters $\theta^{-}$ to compute the target $R + \gamma \max_{a'} Q(s', a'; \theta^{-})$. The target network parameters are copied from the main network only every $C$ steps (or updated via a slow exponential moving average). This prevents the target from changing with every gradient step, which would create a moving-target problem that destabilizes training.

    With these two innovations, DQN learned to play Atari 2600 games from raw pixels at superhuman levels --- using the same architecture and hyperparameters across dozens of different games. This was a watershed moment for RL. See [Sutton & Barto, Section 16.5](file:///C:/Users/landa/ml-course/textbooks/Sutton-RL.pdf) and [Murphy PML2, Section 35.2](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML2.pdf) for discussion.

    **Variants.** Two quick improvements worth knowing:

    - **Double DQN** addresses the overestimation bias in Q-learning. Standard Q-learning uses $\max_{a'} Q(s', a')$ as the target, which systematically overestimates values because it uses the same network to both select and evaluate the best action. Double DQN decouples these: use the online network to select $a^* = \arg\max_{a'} Q(s', a'; \theta)$, but evaluate it with the target network: $Q(s', a^*; \theta^{-})$.
    - **Dueling DQN** separates the network into two streams: one estimates $V(s)$, the other estimates the advantage $A(s,a) = Q(s,a) - V(s)$. They combine at the end: $Q(s,a) = V(s) + A(s,a)$. This is beneficial because many states have similar values regardless of action --- the network can learn the state value separately from the relative advantage of each action.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 8. Policy Gradient Methods

    DQN and its variants learn a value function and derive the policy from it (take the action with the highest Q-value). **Policy gradient methods** take a completely different approach: directly parameterize the policy $\pi_\theta(a|s)$ and optimize it by gradient ascent on expected return.

    Why would you want this? Several reasons:
    - For continuous action spaces (robotic control), you cannot do $\arg\max_a Q(s,a)$ --- there are infinitely many actions. A policy network can directly output continuous actions.
    - Policy gradient methods can learn stochastic policies, which are sometimes optimal (e.g., in partially observable or adversarial settings).
    - Value-based methods can be unstable with function approximation. Policy gradient methods have nicer convergence properties.

    ### REINFORCE

    The simplest policy gradient algorithm is **REINFORCE**. The key insight is the **policy gradient theorem**. Let $J(\theta) = \mathbb{E}_{\pi_\theta}[\sum_t \gamma^t R_t]$ be the expected return. Then:

    $$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t\right]$$

    where $G_t = \sum_{k=t}^{T} \gamma^{k-t} R_k$ is the return from time $t$ onward. See [Sutton & Barto, Section 13.2](file:///C:/Users/landa/ml-course/textbooks/Sutton-RL.pdf) for the full derivation.

    **The log-derivative trick.** Let me derive why $\nabla_\theta \log \pi_\theta$ appears. We want $\nabla_\theta \mathbb{E}_{\pi_\theta}[G]$, which involves differentiating through the probability distribution over trajectories. For a trajectory $\tau = (s_0, a_0, r_0, s_1, a_1, \ldots)$:

    $$\nabla_\theta \mathbb{E}[G] = \nabla_\theta \sum_\tau P(\tau; \theta) G(\tau)$$

    $$= \sum_\tau \nabla_\theta P(\tau; \theta) \cdot G(\tau)$$

    Now apply the identity $\nabla_\theta P(\tau; \theta) = P(\tau; \theta) \cdot \nabla_\theta \log P(\tau; \theta)$. (This follows from $\nabla \log f = \nabla f / f$.)

    $$= \sum_\tau P(\tau; \theta) \cdot \nabla_\theta \log P(\tau; \theta) \cdot G(\tau)$$

    $$= \mathbb{E}_{\pi_\theta}\Big[\nabla_\theta \log P(\tau; \theta) \cdot G(\tau)\Big]$$

    The trajectory probability factors as $P(\tau; \theta) = p(s_0) \prod_t P(s_{t+1}|s_t,a_t) \cdot \pi_\theta(a_t|s_t)$. Taking the log, only the $\pi_\theta$ terms depend on $\theta$. The transition dynamics $P(s_{t+1}|s_t,a_t)$ vanish from the gradient. This is remarkable --- you can compute the policy gradient without knowing the environment dynamics.

    **The variance problem.** REINFORCE is unbiased but has extremely high variance. The return $G_t$ is noisy --- it is the sum of many stochastic rewards. Multiplying this noisy signal by $\nabla \log \pi_\theta$ produces gradients that fluctuate wildly.

    **Baselines.** The standard fix is to subtract a **baseline** $b(s_t)$ from the return: replace $G_t$ with $G_t - b(s_t)$. A common choice is $b(s_t) = V(s_t)$, the estimated value of the state. This does not introduce bias (because the baseline does not depend on the action), but dramatically reduces variance. See [Sutton & Barto, Section 13.4](file:///C:/Users/landa/ml-course/textbooks/Sutton-RL.pdf).

    ### Actor-Critic

    The natural extension is to learn both a policy (the **actor**) and a value function (the **critic**). The critic estimates $V(s)$, and the actor updates using the **advantage function**:

    $$A(s_t, a_t) = Q(s_t, a_t) - V(s_t) \approx R_t + \gamma V(s_{t+1}) - V(s_t)$$

    Notice: the advantage is just the TD error $\delta_t$. If $A > 0$, the action was better than average for that state --- increase its probability. If $A < 0$, it was worse --- decrease it. This is the core actor-critic update. See [Sutton & Barto, Section 13.5](file:///C:/Users/landa/ml-course/textbooks/Sutton-RL.pdf).

    ### PPO (Proximal Policy Optimization)

    PPO, introduced by OpenAI in 2017, is the workhorse of modern policy gradient methods. The problem it solves: standard policy gradient updates can be too large, causing the policy to change drastically and collapse performance. TRPO (Trust Region Policy Optimization) addressed this with constrained optimization, but was complex to implement.

    PPO simplifies things with a **clipped surrogate objective**. Let $r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\theta_{\text{old}}}(a_t|s_t)$ be the probability ratio. PPO optimizes:

    $$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\big(r_t(\theta) A_t, \;\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\big)\Big]$$

    where $\epsilon \approx 0.2$. The clipping prevents the ratio from moving too far from 1, ensuring the new policy does not deviate too much from the old one. When the advantage is positive and the ratio exceeds $1 + \epsilon$, the clipped term caps the objective --- preventing excessively large updates in the direction of this action. When the advantage is negative and the ratio falls below $1 - \epsilon$, the same effect applies. This gives stable, monotonic improvement in practice. PPO is used in robotics, game-playing, and as the RL optimizer in RLHF.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 9. RLHF --- Reinforcement Learning from Human Feedback

    This is where RL meets large language models, and it is arguably the reason you are hearing about RL now. The problem: a pretrained language model generates fluent text, but it may be unhelpful, dishonest, or unsafe. Supervised fine-tuning on curated demonstrations helps, but it is hard to specify what "good" output looks like for every possible input. RLHF offers an alternative.

    ### The Three-Step Pipeline

    **Step 1: Supervised Fine-Tuning (SFT).** Start with a pretrained LLM. Fine-tune it on a dataset of (prompt, high-quality response) pairs. This gives you a model that can follow instructions, but it is not yet optimized for human preferences.

    **Step 2: Reward Model Training.** Collect human comparisons: given a prompt, show two model responses to a human annotator and ask which is better. Train a reward model $r_\phi(x, y)$ that takes a prompt $x$ and response $y$ and outputs a scalar score. The reward model is trained to assign higher scores to preferred responses using a Bradley-Terry preference model:

    $$P(\text{response } y_1 \succ y_2) = \sigma\big(r_\phi(x, y_1) - r_\phi(x, y_2)\big)$$

    where $\sigma$ is the sigmoid function.

    **Step 3: RL Optimization.** Treat the SFT model as the initial policy $\pi_\theta$. Treat the reward model as the environment's reward signal. Optimize the policy using PPO to maximize:

    $$J(\theta) = \mathbb{E}_{x \sim \mathcal{D},\; y \sim \pi_\theta(\cdot|x)}\big[r_\phi(x, y)\big] - \beta \, D_{\text{KL}}\big[\pi_\theta \| \pi_{\text{ref}}\big]$$

    The KL divergence penalty keeps the policy from drifting too far from the reference policy (the SFT model), preventing reward hacking --- the model finding degenerate outputs that score highly with the reward model but are actually nonsensical.

    This pipeline was used to create InstructGPT and is the standard approach for aligning language models. It connects everything in this module: the LLM is the policy, the reward model is the environment, and PPO is the optimizer. See [Murphy PML2, Section 35.4](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML2.pdf) for extended discussion.

    ### DPO: Direct Preference Optimization

    RLHF is effective but complex --- you need to train a separate reward model, run PPO (which is finicky to tune), and manage the KL penalty. **DPO** (Rafailov et al., 2023) sidesteps the reward model entirely.

    The key insight is that the optimal policy under the RLHF objective has a closed-form relationship with the reward function. You can rearrange the math to express the reward in terms of the policy and reference policy, then substitute this into the preference likelihood. The result is a loss function that directly optimizes the policy on preference data:

    $$L_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

    where $y_w$ is the preferred (winning) response and $y_l$ is the dispreferred (losing) response. This is a standard supervised loss --- no RL loop, no reward model, no PPO. It is simpler and, in many settings, performs comparably to RLHF. DPO has become widely adopted as a lightweight alternative to full RLHF.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 10. Open Challenges

    RL has produced remarkable results, but significant challenges remain.

    **Sample efficiency.** RL agents need enormous amounts of experience. DQN required roughly 200 million frames (about 38 days of real-time gameplay) to learn a single Atari game. Real-world applications --- robotics, healthcare, education --- cannot afford millions of trial-and-error interactions. Model-based RL, where the agent learns a model of the environment and plans inside it, is the leading approach to improving sample efficiency, but learning accurate world models is itself extremely difficult.

    **Sim-to-real transfer.** Training in simulation is cheap and safe, but policies trained in simulation often fail in the real world due to differences between the simulator and reality (the "reality gap"). Domain randomization (varying simulation parameters during training) helps, but bridging this gap remains an active research area.

    **Multi-agent RL.** Most RL theory assumes a single agent. When multiple agents interact --- competing, cooperating, or both --- the environment becomes non-stationary from each agent's perspective (the other agents are learning and changing their behavior). Game-theoretic approaches like self-play have produced breakthroughs (AlphaGo, OpenAI Five), but general multi-agent RL is far from solved.

    **Safety and reward hacking.** Agents optimize the reward signal they are given, not necessarily what you intended. A cleaning robot rewarded for "not seeing dirt" might learn to close its eyes. An RLHF-trained model might learn to produce outputs that fool the reward model rather than genuinely satisfy human preferences. Reward specification --- ensuring the reward signal captures what you actually want --- is an unsolved and possibly fundamental problem. This is sometimes called the **alignment problem** and is one of the central concerns in AI safety research.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Summary

    | Method | Model needed? | Bootstraps? | Complete episodes? | On/Off-policy |
    |---|---|---|---|---|
    | Dynamic Programming | Yes | Yes | N/A | N/A |
    | Monte Carlo | No | No | Yes | Both |
    | TD / SARSA | No | Yes | No | On-policy |
    | Q-learning | No | Yes | No | Off-policy |
    | DQN | No | Yes | No | Off-policy |
    | Policy Gradient / PPO | No | No* | No | On-policy |

    *Policy gradient methods use value function baselines that bootstrap, but the gradient estimate itself does not rely on bootstrapping.

    The conceptual arc of this module: MDPs formalize the problem -> Bellman equations reveal recursive structure -> DP solves it with a model -> MC and TD solve it without a model -> deep networks extend it to massive state spaces -> policy gradients handle continuous actions and direct optimization -> RLHF brings it all to language models.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Key References

    - [Sutton & Barto, Ch. 1: Introduction](file:///C:/Users/landa/ml-course/textbooks/Sutton-RL.pdf) --- motivation, agent-environment interface
    - [Sutton & Barto, Ch. 3: MDPs](file:///C:/Users/landa/ml-course/textbooks/Sutton-RL.pdf) --- formal framework, Bellman equations
    - [Sutton & Barto, Ch. 4: Dynamic Programming](file:///C:/Users/landa/ml-course/textbooks/Sutton-RL.pdf) --- policy iteration, value iteration
    - [Sutton & Barto, Ch. 5: Monte Carlo Methods](file:///C:/Users/landa/ml-course/textbooks/Sutton-RL.pdf) --- model-free estimation
    - [Sutton & Barto, Ch. 6: Temporal-Difference Learning](file:///C:/Users/landa/ml-course/textbooks/Sutton-RL.pdf) --- TD(0), SARSA, Q-learning
    - [Sutton & Barto, Ch. 13: Policy Gradient Methods](file:///C:/Users/landa/ml-course/textbooks/Sutton-RL.pdf) --- REINFORCE, actor-critic
    - [Murphy PML2, Ch. 35: Reinforcement Learning](file:///C:/Users/landa/ml-course/textbooks/Murphy-PML2.pdf) --- deep RL, DQN, policy gradients
    - [Goodfellow et al., Part III overview](file:///C:/Users/landa/ml-course/textbooks/DLBook.pdf) --- deep learning context for RL
    - [Bishop PRML, Section 1.2.6: Bayesian Decision Theory](file:///C:/Users/landa/ml-course/textbooks/Bishop-PRML.pdf) --- connections to decision-making under uncertainty
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Practice

    1. **Bellman by hand.** Consider a 3-state MDP with states {A, B, C}, two actions {left, right}, deterministic transitions, and known rewards. Write the Bellman equation for each state under a given policy, then solve the linear system to find $V^\pi$. Verify by running policy evaluation iteratively.

    2. **Implement tabular Q-learning** on a simple gridworld (e.g., 5x5 grid, goal in corner, -1 reward per step, 0 at goal). Vary $\varepsilon$ in $\varepsilon$-greedy exploration and $\alpha$. Plot the number of episodes needed to converge to the optimal policy. What happens if $\alpha$ is too large? Too small?

    3. **MC vs TD comparison.** On the same gridworld, implement first-visit MC and TD(0) for policy evaluation. Run both with the same random policy for 1000 episodes. Plot $V(s)$ estimates over time for a few states. Which method converges faster? Which has more variance between runs?

    4. **DQN on CartPole.** Using PyTorch, implement DQN with experience replay and a target network for the CartPole-v1 environment (from Gymnasium). Train until the agent consistently balances the pole for 500 steps. Then remove experience replay and retrain --- what happens to stability?

    5. **REINFORCE on CartPole.** Implement the REINFORCE algorithm (without baseline) for the same CartPole environment. Observe the high variance in training curves. Then add a state-value baseline and retrain. Compare the learning curves with and without the baseline.

    6. **Conceptual: RLHF pipeline.** Suppose you have a language model that generates product descriptions. Design an RLHF pipeline: What would the reward model be trained on? What preference data would you collect? How would you detect reward hacking? Write a 1-page design document covering each step.

    7. **Exploration-exploitation.** Implement the $\varepsilon$-greedy, UCB (Upper Confidence Bound), and Thompson Sampling strategies for a 10-armed bandit problem. Run 10,000 steps and plot average reward over time for each strategy. Which performs best? Refer to [Sutton & Barto, Chapter 2](file:///C:/Users/landa/ml-course/textbooks/Sutton-RL.pdf).
    """)
    return


if __name__ == "__main__":
    app.run()
