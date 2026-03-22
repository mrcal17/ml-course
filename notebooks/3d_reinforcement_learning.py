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


@app.cell(hide_code=True)
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
def _(mo, np):
    # Discounted return: G = sum_{t=0}^{T} gamma^t * R_t
    rewards = np.array([1.0, 0.0, 0.0, 0.0, 10.0])  # reward at each step
    gamma = 0.9

    # Compute discounted return from each timestep
    T = len(rewards)
    G = np.zeros(T)
    G[-1] = rewards[-1]
    for t in range(T - 2, -1, -1):
        G[t] = rewards[t] + gamma * G[t + 1]  # G_t = R_t + gamma * G_{t+1}

    mo.md(f"""
    **Code: Discounted Returns**

    Rewards: `{rewards.tolist()}`, gamma = `{gamma}`

    Discounted return from each timestep: `{np.round(G, 3).tolist()}`

    Notice how the big reward at t=4 propagates backward, discounted by gamma each step.
    """)
    return (G,)


@app.cell(hide_code=True)
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
def _(mo, np):
    # Bellman equation: V(s) = sum_a pi(a|s) * sum_s' P(s'|s,a) * [R(s,a,s') + gamma * V(s')]
    # 3-state MDP: states 0, 1, 2; two actions: 0 (left), 1 (right)

    n_states, n_actions = 3, 2
    gamma_bellman = 0.9

    # Transition probabilities P[s, a, s'] — deterministic for clarity
    P = np.zeros((n_states, n_actions, n_states))
    P[0, 0, 0] = 1.0;  P[0, 1, 1] = 1.0  # state 0: left->0, right->1
    P[1, 0, 0] = 1.0;  P[1, 1, 2] = 1.0  # state 1: left->0, right->2
    P[2, 0, 1] = 1.0;  P[2, 1, 2] = 1.0  # state 2: left->1, right->2

    # Rewards R[s, a, s']
    R = np.zeros((n_states, n_actions, n_states))
    R[1, 1, 2] = 1.0   # reward for reaching state 2 from state 1
    R[0, 1, 1] = 0.5   # small reward for moving right from state 0

    # Uniform random policy: pi(a|s) = 0.5 for all s, a
    pi = np.ones((n_states, n_actions)) / n_actions

    # Solve Bellman equation iteratively (policy evaluation)
    V = np.zeros(n_states)
    for _ in range(100):
        V_new = np.zeros(n_states)
        for s in range(n_states):
            for a in range(n_actions):
                for s_next in range(n_states):
                    # Bellman equation: weighted sum over actions and transitions
                    V_new[s] += pi[s, a] * P[s, a, s_next] * (R[s, a, s_next] + gamma_bellman * V[s_next])
        V = V_new

    mo.md(f"""
    **Code: Bellman Equation — Policy Evaluation**

    3-state MDP with uniform random policy (pi = 0.5 for each action).

    Converged state values: V = `{np.round(V, 3).tolist()}`

    State 2 has the highest value because the reward is for reaching it.
    """)
    return (P, R, V, gamma_bellman, n_actions, n_states, pi)


@app.cell(hide_code=True)
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
def _(P, R, gamma_bellman, mo, n_actions, n_states, np):
    # Value iteration: V(s) <- max_a sum_s' P(s'|s,a) * [R(s,a,s') + gamma * V(s')]
    V_star = np.zeros(n_states)
    for _vi in range(100):
        V_new_vi = np.zeros(n_states)
        for s in range(n_states):
            action_values = np.zeros(n_actions)
            for a in range(n_actions):
                for s_next in range(n_states):
                    action_values[a] += P[s, a, s_next] * (R[s, a, s_next] + gamma_bellman * V_star[s_next])
            V_new_vi[s] = np.max(action_values)  # Bellman optimality: max over actions
        V_star = V_new_vi

    # Extract optimal policy: greedy w.r.t. V*
    optimal_policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        action_values = np.zeros(n_actions)
        for a in range(n_actions):
            for s_next in range(n_states):
                action_values[a] += P[s, a, s_next] * (R[s, a, s_next] + gamma_bellman * V_star[s_next])
        optimal_policy[s] = np.argmax(action_values)

    action_names = ["left", "right"]
    mo.md(f"""
    **Code: Value Iteration**

    Optimal values: V* = `{np.round(V_star, 3).tolist()}`

    Optimal policy: `{[action_names[a] for a in optimal_policy]}`

    The agent learns to go right from every state to reach the reward.
    """)
    return (V_star,)


@app.cell(hide_code=True)
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
def _(mo, np):
    # First-visit Monte Carlo policy evaluation on a simple chain MDP
    # States: 0 -> 1 -> 2 -> 3 (terminal), reward +1 at terminal
    rng_mc = np.random.default_rng(42)
    n_states_mc = 4  # state 3 is terminal
    gamma_mc = 0.9
    n_episodes_mc = 500

    returns_per_state = {s: [] for s in range(n_states_mc - 1)}

    for _ in range(n_episodes_mc):
        # Generate episode: random walk right with p=0.7, left with p=0.3
        s = 0
        episode = []
        while s < n_states_mc - 1:
            a = 1 if rng_mc.random() < 0.7 else 0  # mostly go right
            s_next = min(s + 1, n_states_mc - 1) if a == 1 else max(s - 1, 0)
            r = 1.0 if s_next == n_states_mc - 1 else 0.0
            episode.append((s, r))
            s = s_next

        # Compute returns backward and record first visits
        G_mc = 0.0
        visited = set()
        for s_ep, r_ep in reversed(episode):
            G_mc = r_ep + gamma_mc * G_mc
            if s_ep not in visited:  # first-visit MC
                visited.add(s_ep)
                returns_per_state[s_ep].append(G_mc)

    V_mc = {s: np.mean(rets) if rets else 0 for s, rets in returns_per_state.items()}

    mo.md(f"""
    **Code: First-Visit Monte Carlo**

    Chain MDP: 0 -> 1 -> 2 -> 3(terminal, reward=1). Policy: go right 70%, left 30%.

    MC value estimates after {n_episodes_mc} episodes:
    V(0) = `{V_mc[0]:.3f}`, V(1) = `{V_mc[1]:.3f}`, V(2) = `{V_mc[2]:.3f}`

    Values increase closer to the terminal reward, as expected.
    """)
    return


@app.cell(hide_code=True)
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
def _(mo, np):
    # TD(0) policy evaluation on the same chain MDP
    rng_td = np.random.default_rng(42)
    n_states_td = 4  # state 3 is terminal
    gamma_td = 0.9
    alpha_td = 0.1

    V_td = np.zeros(n_states_td)
    td_errors = []

    for _ in range(500):
        s = 0
        while s < n_states_td - 1:
            a = 1 if rng_td.random() < 0.7 else 0
            s_next = min(s + 1, n_states_td - 1) if a == 1 else max(s - 1, 0)
            r = 1.0 if s_next == n_states_td - 1 else 0.0

            # TD(0) update: V(s) += alpha * [R + gamma * V(s') - V(s)]
            td_error = r + gamma_td * V_td[s_next] - V_td[s]
            V_td[s] += alpha_td * td_error
            td_errors.append(td_error)
            s = s_next

    mo.md(f"""
    **Code: TD(0) Learning**

    Same chain MDP, same policy. TD updates *every step* (no waiting for episode end).

    TD value estimates: V(0) = `{V_td[0]:.3f}`, V(1) = `{V_td[1]:.3f}`, V(2) = `{V_td[2]:.3f}`

    Mean |TD error| over last 100 steps: `{np.mean(np.abs(td_errors[-100:])):.4f}` (should be small when converged)
    """)
    return


@app.cell
def _(mo):
    mo.image(src="../animations/rendered/TDLearningUpdate.gif")
    return


@app.cell(hide_code=True)
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
def _(mo, np):
    # Tabular Q-learning on a 5-state chain: 0-1-2-3-4, goal=4
    rng_q = np.random.default_rng(0)
    n_s, n_a = 5, 2  # actions: 0=left, 1=right
    Q_table = np.zeros((n_s, n_a))
    alpha_q, gamma_q, epsilon_q = 0.1, 0.9, 0.1

    for _ in range(1000):
        s = 0
        while s != n_s - 1:
            # Epsilon-greedy action selection
            if rng_q.random() < epsilon_q:
                a = rng_q.integers(n_a)
            else:
                a = np.argmax(Q_table[s])

            s_next = max(0, s - 1) if a == 0 else min(n_s - 1, s + 1)
            r = 1.0 if s_next == n_s - 1 else -0.01

            # Q-learning update: use max over next actions (off-policy)
            Q_table[s, a] += alpha_q * (r + gamma_q * np.max(Q_table[s_next]) - Q_table[s, a])
            s = s_next

    learned_policy = ["left" if np.argmax(Q_table[s]) == 0 else "right" for s in range(n_s - 1)]
    mo.md(f"""
    **Code: Tabular Q-Learning**

    5-state chain, goal at state 4. Epsilon-greedy exploration (eps={epsilon_q}).

    Learned Q-table (rows=states, cols=[left, right]):
    ```
    {np.array2string(np.round(Q_table, 3), separator=', ')}
    ```

    Learned policy: `{learned_policy}` — agent goes right from every state.
    """)
    return


@app.cell(hide_code=True)
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
def _(mo, np):
    # Experience replay buffer — the core DQN data structure
    class ReplayBuffer:
        def __init__(self, capacity):
            self.capacity = capacity
            self.buffer = []
            self.pos = 0

        def push(self, transition):
            """Store (s, a, r, s_next, done) tuple"""
            if len(self.buffer) < self.capacity:
                self.buffer.append(transition)
            else:
                self.buffer[self.pos] = transition  # overwrite oldest
            self.pos = (self.pos + 1) % self.capacity

        def sample(self, batch_size, rng):
            """Sample random mini-batch — breaks temporal correlations"""
            idxs = rng.integers(0, len(self.buffer), size=batch_size)
            return [self.buffer[i] for i in idxs]

    # Demo: fill buffer and sample a batch
    rng_buf = np.random.default_rng(42)
    buf = ReplayBuffer(capacity=1000)
    for i in range(50):
        buf.push((i % 5, i % 2, np.random.randn(), (i + 1) % 5, i == 49))

    batch = buf.sample(4, rng_buf)
    mo.md(f"""
    **Code: Experience Replay Buffer**

    Buffer stores transitions `(s, a, r, s', done)` and samples random mini-batches.

    Buffer size: {len(buf.buffer)} transitions. Sample batch of 4:
    ```
    {chr(10).join(str(t) for t in batch)}
    ```

    Random sampling breaks correlations between consecutive transitions.
    """)
    return


@app.cell(hide_code=True)
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
def _(mo, np):
    # REINFORCE gradient estimate (softmax policy, tabular)
    # Policy: pi(a|s) = softmax(theta[s, :])
    rng_pg = np.random.default_rng(42)
    n_s_pg, n_a_pg = 3, 2
    theta_pg = np.zeros((n_s_pg, n_a_pg))

    def softmax(x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    # Simulate one episode: states 0->1->2(terminal), reward at end
    episode_pg = [(0, 1, 0.0), (1, 1, 1.0)]  # (state, action, reward)
    gamma_pg = 0.9

    # Compute returns G_t backward
    returns_pg = np.zeros(len(episode_pg))
    returns_pg[-1] = episode_pg[-1][2]
    for t in range(len(episode_pg) - 2, -1, -1):
        returns_pg[t] = episode_pg[t][2] + gamma_pg * returns_pg[t + 1]

    # Compute REINFORCE gradient: sum_t grad_log_pi(a_t|s_t) * G_t
    grad_theta = np.zeros_like(theta_pg)
    for t, (s, a, r) in enumerate(episode_pg):
        probs = softmax(theta_pg[s])
        # grad log pi(a|s) = e_a - pi  (for softmax parameterization)
        grad_log_pi = -probs.copy()
        grad_log_pi[a] += 1.0
        grad_theta[s] += grad_log_pi * returns_pg[t]  # weight by return

    mo.md(f"""
    **Code: REINFORCE Gradient**

    Softmax policy over 2 actions, 3 states. One episode: s0->s1->terminal (reward=1).

    Returns: `{np.round(returns_pg, 3).tolist()}`

    Policy gradient (per state-action):
    ```
    {np.array2string(np.round(grad_theta, 3), separator=', ')}
    ```

    Positive gradient for action 1 (right) = gradient ascent will increase its probability.
    """)
    return


@app.cell
def _(mo, np):
    # PPO clipped objective — the key computation
    def ppo_clipped_objective(ratio, advantage, epsilon=0.2):
        """
        ratio = pi_new(a|s) / pi_old(a|s)
        advantage = A(s, a)
        Returns clipped surrogate objective (per sample)
        """
        clipped_ratio = np.clip(ratio, 1 - epsilon, 1 + epsilon)
        return np.minimum(ratio * advantage, clipped_ratio * advantage)

    # Demo: show how clipping prevents large updates
    ratios = np.linspace(0.5, 2.0, 100)
    adv_pos = ppo_clipped_objective(ratios, advantage=1.0)
    adv_neg = ppo_clipped_objective(ratios, advantage=-1.0)

    mo.md(f"""
    **Code: PPO Clipped Objective**

    When advantage > 0 (good action): objective is capped at ratio = 1.2
    - At ratio=1.5: unclipped = {1.5 * 1.0:.1f}, clipped = {ppo_clipped_objective(np.array([1.5]), 1.0)[0]:.1f}

    When advantage < 0 (bad action): objective is capped at ratio = 0.8
    - At ratio=0.5: unclipped = {0.5 * -1.0:.1f}, clipped = {ppo_clipped_objective(np.array([0.5]), -1.0)[0]:.1f}

    The clip prevents the policy from changing too much in one update.
    """)
    return


@app.cell(hide_code=True)
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
def _(mo, np):
    # Bradley-Terry preference model & DPO loss
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    # Reward model scores for two responses
    r_win, r_lose = 2.5, 1.0
    # P(y_win preferred) = sigma(r_win - r_lose)
    p_prefer = sigmoid(r_win - r_lose)

    # DPO loss: L = -log(sigma(beta * (log_ratio_win - log_ratio_lose)))
    beta_dpo = 0.1
    # Simulated log-probability ratios: log(pi_theta(y|x) / pi_ref(y|x))
    log_ratio_win = 0.5   # policy slightly prefers winner vs reference
    log_ratio_lose = -0.3  # policy slightly dislikes loser vs reference

    dpo_loss = -np.log(sigmoid(beta_dpo * (log_ratio_win - log_ratio_lose)))

    mo.md(f"""
    **Code: Bradley-Terry Model & DPO Loss**

    Reward model scores: r(win)={r_win}, r(lose)={r_lose}
    P(win preferred) = sigma({r_win}-{r_lose}) = **{p_prefer:.3f}**

    DPO loss (no reward model needed):
    - log(pi/pi_ref) for winner: {log_ratio_win}, for loser: {log_ratio_lose}
    - beta = {beta_dpo}
    - DPO loss = **{dpo_loss:.4f}**

    Lower loss = policy better separates preferred from dispreferred responses.
    """)
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Code It: Implementation Exercises

    The exercises below give you skeleton code with `TODO` placeholders. Fill in the missing pieces to implement core RL algorithms from scratch in numpy.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 1: Policy Evaluation via the Bellman Equation

    Given a 4-state MDP (states 0-3, two actions, known transitions and rewards), implement iterative policy evaluation. Sweep through all states and apply the Bellman update until values converge within a tolerance.
    """)
    return


@app.cell
def _(np):
    def policy_evaluation_exercise(P_ex, R_ex, pi_ex, gamma_ex, tol=1e-6):
        """
        Iterative policy evaluation.

        Args:
            P_ex: transition probs, shape (n_states, n_actions, n_states)
            R_ex: rewards, shape (n_states, n_actions, n_states)
            pi_ex: policy, shape (n_states, n_actions) — pi[s,a] = P(a|s)
            gamma_ex: discount factor
            tol: convergence tolerance

        Returns:
            V: value function, shape (n_states,)
        """
        n_states_ex = P_ex.shape[0]
        V_ex = np.zeros(n_states_ex)

        for _iteration in range(10000):
            V_old = V_ex.copy()
            for s in range(n_states_ex):
                # TODO: Compute V_ex[s] using the Bellman equation
                # V(s) = sum_a pi(a|s) * sum_s' P(s'|s,a) * [R(s,a,s') + gamma * V(s')]
                V_ex[s] = 0.0  # Replace this line

            # Check convergence
            if np.max(np.abs(V_ex - V_old)) < tol:
                break

        return V_ex

    # Test MDP
    _P_test = np.zeros((4, 2, 4))
    _P_test[0, 0, 0] = 1; _P_test[0, 1, 1] = 1
    _P_test[1, 0, 0] = 1; _P_test[1, 1, 2] = 1
    _P_test[2, 0, 1] = 1; _P_test[2, 1, 3] = 1
    _P_test[3, 0, 3] = 1; _P_test[3, 1, 3] = 1  # terminal
    _R_test = np.zeros((4, 2, 4))
    _R_test[2, 1, 3] = 10.0  # big reward for reaching state 3
    _pi_test = np.ones((4, 2)) / 2  # uniform random

    _V_result = policy_evaluation_exercise(_P_test, _R_test, _pi_test, 0.9)
    print(f"Your V: {np.round(_V_result, 3)}")
    # Expected: state 3 should have 0 (terminal), states closer should have positive values
    return (policy_evaluation_exercise,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 2: Tabular Q-Learning on a Gridworld

    Implement Q-learning with epsilon-greedy exploration on a 4x4 gridworld. The agent starts at (0,0), goal is at (3,3) with reward +1, every step costs -0.01. Actions: up/down/left/right.
    """)
    return


@app.cell
def _(np):
    def q_learning_gridworld(n_episodes_ex=2000, alpha_ex=0.1, gamma_ex=0.95,
                             epsilon_ex=0.1, grid_size=4):
        """
        Tabular Q-learning on a gridworld.

        Returns:
            Q: learned Q-table, shape (grid_size*grid_size, 4)
            episode_rewards: total reward per episode
        """
        rng_ex = np.random.default_rng(42)
        n_states_ex = grid_size * grid_size
        n_actions_ex = 4  # 0=up, 1=down, 2=left, 3=right
        Q_ex = np.zeros((n_states_ex, n_actions_ex))
        goal = n_states_ex - 1
        episode_rewards_ex = []

        # Movement deltas: up, down, left, right
        dx = [-1, 1, 0, 0]
        dy = [0, 0, -1, 1]

        def step(state, action):
            row, col = state // grid_size, state % grid_size
            nr, nc = row + dx[action], col + dy[action]
            nr, nc = max(0, min(grid_size-1, nr)), max(0, min(grid_size-1, nc))
            next_state = nr * grid_size + nc
            reward = 1.0 if next_state == goal else -0.01
            done = next_state == goal
            return next_state, reward, done

        for _ep in range(n_episodes_ex):
            s = 0
            total_reward = 0
            for _step_count in range(200):
                # TODO: Implement epsilon-greedy action selection
                # With probability epsilon, pick random action; else pick argmax Q[s]
                a = 0  # Replace this line

                s_next, r, done = step(s, a)
                total_reward += r

                # TODO: Q-learning update
                # Q[s,a] += alpha * (r + gamma * max_a' Q[s',a'] - Q[s,a])
                pass  # Replace this line

                s = s_next
                if done:
                    break
            episode_rewards_ex.append(total_reward)

        return Q_ex, episode_rewards_ex

    _Q_grid, _rewards_grid = q_learning_gridworld()
    print(f"Mean reward (last 100 eps): {np.mean(_rewards_grid[-100:]):.3f}")
    # Should be close to 0.94 (reward 1 minus ~6 steps * 0.01) when converged
    return (q_learning_gridworld,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 3: REINFORCE with Baseline

    Implement the REINFORCE policy gradient algorithm with a value-function baseline on a simple bandit-like problem. The environment has 1 state and 3 actions with different expected rewards.
    """)
    return


@app.cell
def _(np):
    def reinforce_with_baseline(n_iters=1000, alpha_policy=0.1, alpha_value=0.01,
                                gamma_rf=0.99):
        """
        REINFORCE with a learned value-function baseline.
        Environment: 1 state, 3 actions with rewards N(mu_a, 1).
        True means: [0.2, 0.8, 0.5]

        Returns:
            theta: policy parameters, shape (3,)
            policy_history: action probabilities over time
        """
        rng_rf = np.random.default_rng(42)
        true_means = np.array([0.2, 0.8, 0.5])
        n_actions_rf = len(true_means)

        theta_rf = np.zeros(n_actions_rf)  # softmax policy params
        V_baseline = 0.0  # scalar baseline (one state)
        policy_history_rf = []

        def softmax_rf(x):
            e = np.exp(x - np.max(x))
            return e / e.sum()

        for _it in range(n_iters):
            probs = softmax_rf(theta_rf)
            policy_history_rf.append(probs.copy())

            # TODO: Sample action from policy (use rng_rf.choice with p=probs)
            a = 0  # Replace this line

            # TODO: Get reward (sample from N(true_means[a], 1))
            r = 0.0  # Replace this line

            # TODO: Compute advantage = reward - baseline
            advantage = 0.0  # Replace this line

            # TODO: Update policy params: theta[a] += alpha * advantage * (1 - probs[a])
            #        and theta[other] -= alpha * advantage * probs[other]
            # (This is grad log softmax(a) * advantage)
            pass  # Replace this line

            # TODO: Update baseline: V_baseline += alpha_value * (r - V_baseline)
            pass  # Replace this line

        return theta_rf, np.array(policy_history_rf)

    _theta_rf, _hist_rf = reinforce_with_baseline()
    _final_probs = np.exp(_theta_rf - np.max(_theta_rf))
    _final_probs = _final_probs / _final_probs.sum()
    print(f"Final policy: {np.round(_final_probs, 3)}")
    # Should converge to mostly selecting action 1 (highest mean reward = 0.8)
    return (reinforce_with_baseline,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 4: Multi-Armed Bandit — Exploration Strategies

    Implement and compare three exploration strategies: epsilon-greedy, UCB (Upper Confidence Bound), and a simple softmax (Boltzmann) exploration on a 5-armed bandit.
    """)
    return


@app.cell
def _(np):
    def bandit_comparison(n_steps=5000, n_arms=5):
        """
        Compare exploration strategies on a k-armed bandit.
        True arm means drawn from N(0, 1).

        Returns:
            rewards_dict: dict mapping strategy name -> array of rewards per step
        """
        rng_bandit = np.random.default_rng(123)
        true_values = rng_bandit.standard_normal(n_arms)
        print(f"True arm values: {np.round(true_values, 2)}")

        results = {}

        # --- Epsilon-greedy ---
        Q_eg = np.zeros(n_arms)
        N_eg = np.zeros(n_arms)
        rewards_eg = []
        epsilon_bandit = 0.1
        for t in range(n_steps):
            # TODO: epsilon-greedy action selection
            # With prob epsilon pick random arm, else pick argmax Q_eg
            a = 0  # Replace

            r = true_values[a] + rng_bandit.standard_normal()
            N_eg[a] += 1
            # TODO: Incremental mean update: Q[a] += (1/N[a]) * (r - Q[a])
            pass  # Replace

            rewards_eg.append(r)
        results["epsilon-greedy"] = np.array(rewards_eg)

        # --- UCB ---
        Q_ucb = np.zeros(n_arms)
        N_ucb = np.zeros(n_arms)
        rewards_ucb = []
        c_ucb = 2.0
        for t in range(n_steps):
            # TODO: UCB action selection
            # If any arm unvisited, pick it. Otherwise:
            # a = argmax_a [ Q[a] + c * sqrt(ln(t+1) / N[a]) ]
            a = 0  # Replace

            r = true_values[a] + rng_bandit.standard_normal()
            N_ucb[a] += 1
            Q_ucb[a] += (1 / N_ucb[a]) * (r - Q_ucb[a])
            rewards_ucb.append(r)
        results["UCB"] = np.array(rewards_ucb)

        # Print summary
        for name, rews in results.items():
            print(f"{name}: mean reward (last 500) = {np.mean(rews[-500:]):.3f}")

        return results

    _bandit_results = bandit_comparison()
    return (bandit_comparison,)


if __name__ == "__main__":
    app.run()
