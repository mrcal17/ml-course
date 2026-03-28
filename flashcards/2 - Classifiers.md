#flashcards/part1

## Discriminative vs. Generative Classifiers

What is the core difference between discriminative and generative classifiers?
?
Discriminative models learn P(y|x) directly; they focus on the decision boundary. Generative models learn P(x|y) and P(y), then use Bayes' rule to get P(y|x). Generative models learn the full data distribution per class.

When do generative classifiers outperform discriminative ones?
?
With small datasets. Their distributional assumptions act as a strong inductive bias that regularizes learning. Ng & Jordan (2002) showed generative models reach lower error faster with limited data. Discriminative models win asymptotically with enough data.

What assumptions does LDA make, and what is its decision boundary?
?
LDA assumes P(x|y=k) ~ N(mu_k, Sigma) with a shared covariance matrix across classes but different means. The boundary is linear: delta_k(x) = x^T Sigma^(-1) mu_k - (1/2) mu_k^T Sigma^(-1) mu_k + log pi_k.

What is Fisher's geometric interpretation of LDA?
?
LDA finds the projection direction that maximizes the ratio of between-class variance to within-class variance. It separates the class means as much as possible while keeping each class tight.

What assumption makes Naive Bayes "naive," and why does it still work?
?
It assumes features are conditionally independent given the class: P(x|y=k) = product_j P(x_j|y=k). This is almost always wrong, but it works because classification only needs to rank classes correctly, not estimate probabilities accurately. The decision boundary can still be good even if the probabilities are miscalibrated.

Write the SVM objective and explain each term.
?
min (1/2)||w||^2 + C * sum(xi_i), subject to y_i(w^T x_i + b) >= 1 - xi_i. The first term maximizes the margin (smaller ||w|| = wider margin). C controls the penalty for misclassifications. xi_i are slack variables allowing soft margin violations.

What is the SVM hinge loss, and how does it differ from logistic loss?
?
Hinge loss: max(0, 1 - y * f(x)). It is exactly zero when a point is on the correct side of the margin (y*f(x) >= 1), creating sparse solutions (only support vectors matter). Logistic loss: log(1 + e^(-y*f(x))) is always positive; it never completely ignores correctly classified points.

How does the kernel trick enable nonlinear SVMs without computing high-dimensional features?
?
The SVM dual formulation only needs inner products x_i^T x_j. A kernel K(x_i, x_j) = phi(x_i)^T phi(x_j) computes the inner product in a high-dimensional (or infinite-dimensional) feature space without ever constructing phi explicitly. RBF kernel corresponds to infinite-dimensional features.

Name the three common SVM kernels and when to use each.
?
Linear: K = x^T z. Use when data is roughly linearly separable or p >> n. Polynomial: K = (x^T z + c)^d. Use for feature interactions. RBF: K = exp(-gamma||x-z||^2). Default choice; captures complex boundaries. Corresponds to infinite-dimensional features.

Why don't SVMs output probabilities natively?
?
SVMs optimize the margin, not a probabilistic objective. The decision function f(x) = w^T x + b gives a signed distance from the boundary, not a probability. Platt scaling fits a sigmoid to the SVM outputs on a validation set to convert distances to probabilities.
