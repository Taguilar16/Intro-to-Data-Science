The goal of this assignment was to identify the best machine learning model to predict customer churn for a telecom company. 

Using PyCaret’s ClassificationExperiment, I trained multiple classification models and specified “Churn” as the target variable. I selected accuracy as the primary metric for model comparison. While recall or F1 score are often better metrics for churn prediction (since missing churners can be costly), Logistic Regression performed strongly across nearly all metrics, including recall and F1, and was therefore selected as the best-performing model.

The final selected model was Logistic Regression.

he ROC curve showed an AUC of approximately 0.84. This means: There is an 84% probability that the model will rank a randomly chosen churner higher than a randomly chosen non-churner. In other words, the model has good overall discriminative ability between churners and non-churners.

The Confusion Matrix revealed that the model performs very well at predicting non-churners, but struggles more with identifying churners. Nearly 50% of actual churners were misclassified as non-churners. This suggests that while the model has reasonable overall performance, it may require threshold tuning or a different metric focus (such as recall) if the business priority is reducing churn risk.

The trained model was saved to disk and loaded into a Python script that accepts a pandas DataFrame as input and returns:

prediction_label (Yes/No churn prediction)
prediction_score (probability associated with the predicted class)

Using new customer data, the model outputs churn predictions along with probability scores. These probability scores are especially useful for identifying high-risk customers so that preventative retention strategies can be implemented.