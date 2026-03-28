#flashcards/part1

## Evaluation Metrics and Model Selection

When is accuracy a misleading metric?
?
When classes are imbalanced. A classifier that predicts the majority class every time achieves high accuracy but is useless. Use precision, recall, F1, or AUC instead.

What is the difference between precision and recall, and when does each matter?
?
Precision = TP / (TP + FP): of everything you predicted positive, how many were correct. Recall = TP / (TP + FN): of all actual positives, how many did you catch. Precision matters when false positives are costly (spam filter). Recall matters when false negatives are costly (cancer screening).

Why is AUC-ROC preferred over accuracy for comparing classifiers?
?
AUC-ROC evaluates ranking quality across all classification thresholds, not just one. It measures the probability that the model ranks a random positive higher than a random negative. It is threshold-independent and works on imbalanced data.

When should you use PR-AUC instead of ROC-AUC?
?
When data is highly imbalanced. ROC-AUC can look good because TNs dominate the FPR denominator, masking poor performance on the minority class. PR-AUC focuses only on the positive class and is more sensitive to how well you detect rare events.

What is the purpose of nested cross-validation?
?
To get an unbiased estimate of the full model selection pipeline. The outer loop evaluates generalization; the inner loop tunes hyperparameters. Without nesting, the CV score is optimistically biased because the same data informed both tuning and evaluation.

How does stratified k-fold differ from regular k-fold?
?
Stratified k-fold preserves the class proportions in each fold. Regular k-fold splits randomly, which can produce folds with skewed class distributions, especially with imbalanced data or small datasets.

What is the difference between AIC and BIC?
?
Both penalize model complexity to prevent overfitting. AIC = 2k - 2ln(L-hat); lighter penalty, tends to select more complex models. BIC = k*ln(n) - 2ln(L-hat); heavier penalty that grows with sample size, favoring simpler models. BIC is consistent (selects the true model as n grows).

How does the bootstrap estimate standard errors?
?
Sample n points with replacement from the data. Each bootstrap sample contains ~63.2% unique points. Repeat B times, compute the statistic on each sample, then use the standard deviation of those B statistics as the standard error estimate.

What is the OOB (out-of-bag) error and why is it useful?
?
Each bootstrap sample leaves out ~36.8% of the data. These out-of-bag points act as a free validation set. For Random Forests, OOB error is computed by predicting each point using only the trees that did not include it in training. No separate validation split needed.

What is the .632 bootstrap estimator?
?
Err = 0.368 * Err_train + 0.632 * Err_oob. It corrects the bias of pure OOB estimates by blending training error and OOB error. The weights come from the expected fraction of unique points in a bootstrap sample (63.2%).

How does Random Search compare to Grid Search for hyperparameter tuning?
?
Random Search samples hyperparameter combinations randomly rather than exhaustively. It explores more unique values per dimension, which matters because some hyperparameters matter more than others. Bergstra & Bengio (2012) showed Random Search finds good configurations faster because it does not waste evaluations on unimportant dimensions.

What does R^2 measure, and can it be negative?
?
R^2 = 1 - SS_res / SS_tot. It is the fraction of variance explained by the model. R^2 = 1 means perfect prediction; R^2 = 0 means the model is no better than predicting the mean. It can be negative on a test set if the model is worse than predicting the mean.
