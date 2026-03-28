#flashcards/part1

## Key Foundations (Section 0 concepts critical to Part 1)

What are the three components of the bias-variance decomposition?
?
MSE = Bias^2 + Variance + sigma^2. Bias: systematic error from wrong assumptions (underfitting). Variance: sensitivity to training data (overfitting). sigma^2: irreducible noise from the data itself. You cannot compute this decomposition in practice because it requires knowing the true function.

How does MLE relate to OLS, logistic regression, and cross-entropy?
?
OLS is MLE under Gaussian noise. Logistic regression's cross-entropy loss is the negative log-likelihood of a Bernoulli model. Minimizing cross-entropy = maximizing likelihood. This connection means all these models share the same theoretical foundation: find parameters that make the observed data most probable.

How does MAP estimation connect to regularization?
?
MAP = MLE + prior: theta-hat = argmax[log p(D|theta) + log p(theta)]. A Gaussian prior on w gives log p(w) proportional to -||w||^2, which adds an L2 penalty (Ridge). A Laplace prior gives log p(w) proportional to -||w||_1, which adds an L1 penalty (Lasso). Regularization strength lambda corresponds to the prior's precision.

Why does the covariance matrix appear everywhere in Part 1?
?
PCA eigendecomposes it to find principal components. LDA uses within-class and between-class covariance. GMMs give each component its own covariance matrix. OLS uses (X^T X)^(-1), which is proportional to the inverse covariance. Ridge adds lambda*I to fix ill-conditioned covariance. It is the central object connecting linear algebra to statistical modeling.
