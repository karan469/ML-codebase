# Evaluation and loss metrics
*https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc*

## 1. Accuracy

- Accuracy is useful when dataset is balanced and both classes (in case of binary classification) are equally important with respect to the buisness model.
- Keep in mind, the threshold used in order to report accuracy. By default it could be 0.5 but that might gives us a false sense of achievement if the predictions are, say, slightly higher and lower than 0.5 (where it is evident that the model didn't really learnt much)

## 2. F1 Score

Precision: How many selected items are relevant. TP/(TP+TN)

Recall: How many relevant items are selected. TP/(TP+FP)

F1: Harmonic mean of Precision and Recall (with threshold regulated by beta)

When to use: When you care more about the positive class than negative class.

## 3. ROC AUC

## 4. PR AUC | Average Precision

## Evaluation metrics evaulation
