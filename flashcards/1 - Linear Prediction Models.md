#flashcards/part1

## Linear Prediction Models

What is the closed-form solution for Ordinary Least Squares?
?
w* = (X^T X)^(-1) X^T y. It minimizes ||y - Xw||^2, the sum of squared residuals. Geometrically, it projects y onto the column space of X.

When does OLS fail, and what should you use instead?
?
OLS fails when p is large relative to n, or when features are correlated (multicollinearity). X^T X becomes ill-conditioned or singular. Use Ridge (L2) to fix conditioning, or Lasso (L1) if you also want feature selection.

How does Ridge Regression modify the OLS solution, and why does it work?
?
w* = (X^T X + lambda*I)^(-1) X^T y. Adding lambda*I to X^T X guarantees invertibility and shrinks all coefficients toward zero. Bayesian interpretation: it is equivalent to MAP estimation with a Gaussian prior on w.

What is the key geometric difference between L1 and L2 regularization that explains sparsity?
?
L2's constraint region is a sphere; the OLS solution touches it at an arbitrary point, shrinking all weights. L1's constraint region is a diamond with corners on the axes; the OLS solution is more likely to touch a corner, which sets some weights exactly to zero.

How does Lasso differ from Ridge in handling correlated features?
?
Lasso arbitrarily picks one feature from a correlated group and zeroes out the rest. Ridge keeps all correlated features but shrinks their coefficients proportionally. Elastic Net combines both to get sparsity while handling correlation.

What two hyperparameters does Elastic Net introduce, and when is it preferred?
?
lambda_1 (L1 strength) and lambda_2 (L2 strength). Preferred when you want sparsity (from L1) but also have correlated features that L2 handles better than pure Lasso.

Why is polynomial regression still "linear" in the ML sense?
?
It is linear in the parameters w, even though it is nonlinear in the input x. The model is y = w^T phi(x) where phi(x) = [1, x, x^2, ..., x^d]. You apply the same OLS math to the expanded feature matrix.

What are the two main failure modes of polynomial regression?
?
1. Overfitting for high degree d, especially with limited data. 2. Runge phenomenon: oscillation at the edges of the data range. Both worsen with the curse of dimensionality when applied to multiple input features.

Write the logistic regression model and its loss function.
?
Model: P(y=1|x) = sigma(w^T x) = 1 / (1 + e^(-w^T x)). Loss: binary cross-entropy, -sum[y log(y-hat) + (1-y) log(1-y-hat)]. The gradient has the same form as OLS: X^T(sigma(Xw) - y).

Why does logistic regression have no closed-form solution, unlike OLS?
?
The cross-entropy loss, while convex, is nonlinear in w due to the sigmoid function. You need iterative methods like gradient descent or Newton's method. OLS has a closed form because its loss is quadratic in w.

How does Softmax generalize logistic regression to K classes?
?
P(y=k|x) = e^(w_k^T x) / sum_j e^(w_j^T x). Each class gets its own weight vector w_k, and the denominator normalizes to a valid probability distribution. Loss is categorical cross-entropy. For K=2, it reduces to standard logistic regression.

What connects OLS, Ridge, and MLE/MAP?
?
OLS is MLE under Gaussian noise assumptions. Ridge is MAP with a Gaussian prior on w. Lasso is MAP with a Laplace prior on w. The regularization parameter lambda corresponds to the ratio of noise variance to prior variance.
