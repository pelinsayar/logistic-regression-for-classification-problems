# logistic-regression-in-classification-problems
This study develops a basic classification model to predict diabetes using features such as age, pregnancies, glucose, blood pressure, insulin, BMI, and outcome. Data preprocessing included outlier analysis and standardization. Logistic regression evaluated performance, focusing on metrics and overfitting.
Logistic Regression in Classification Problems

# Logistic Regression in Classification Problems

Example: Diabetes Prediction

Diabetes is a major global health issue, and with early diagnosis and proper treatment, it can be managed effectively. In this study, a simple classification model without feature engineering has been developed to predict whether an individual has diabetes. The dataset includes features such as age, number of pregnancies, glucose level, blood pressure, skin thickness, insulin level, body mass index (BMI), and diabetes pedigree function, along with the outcome (Outcome) indicating whether the individual has diabetes.

In the study, data preprocessing steps such as outlier analysis, normalization methods, and standardization were applied before modeling. Then, logistic regression classification model were used to evaluate performance, success was analyzed through metrics, and overfitting/underfitting was checked. This process plays a critical role in understanding the basic dynamics of classification problems and developing a predictive system that can be used in real-life scenarios.


Outlier Detection and Handling

This step identifies and handles outliers in the dataset. The outlier_thresholds function calculates upper and lower bounds for a given column using interquartile range (IQR). The check_outlier function checks for outliers in each column. If outliers exist, the replace_with_thresholds function replaces them with the calculated thresholds. For this dataset, outliers were detected and handled only in the "Insulin" variable.


Standardization

RobustScaler is a scaling method that is particularly useful for datasets with outliers, as it is less sensitive to extreme values. Therefore, it is preferred in datasets like the diabetes dataset, where some variables contain outliers.

In the diabetes dataset, some variables (e.g., Glucose, Insulin, BMI) have outliers. These outliers can distort the results of standard scaling methods (e.g., StandardScaler or MinMaxScaler). For example, Age typically ranges between 20 and 80, while variables like Glucose and Insulin may have wider and more varied distributions. Therefore, a scaling method is needed to eliminate scale differences.

For instance, StandardScaler uses the mean and standard deviation to normalize the data distribution. However, outliers significantly affect these metrics, which can lead to incorrect scaling results.


RobustScaler scales the data using the median and interquartile range (IQR). This method minimizes the impact of outliers. The process works as follows:

Each value is centered by subtracting the median. The data is then scaled by the interquartile range (IQR, which is Q3 — Q1, or the 75th and 25th percentiles).

Evaluation of Model Performance Metrics


Accuracy: 0.772 Precision: 0.719 Recall: 0.574 F1-Score: 0.637 AUC (Area Under Curve): 0.832

Accuracy:

The model’s accuracy is 77.2%. However, accuracy can be misleading, especially in imbalanced datasets.

If the proportion of diabetic patients in the dataset is low, a model that classifies all instances as “Not Diabetic” could still achieve high accuracy. Therefore, accuracy alone is not a sufficient metric of success, especially in critical fields like healthcare, where more critical metrics should be prioritized.

Recall:

Our recall value is 0.5747, meaning we correctly identified 57% of individuals who actually have diabetes.

However, this also means that we missed 43% of diabetic patients (False Negatives). For instance, telling a diabetic patient “You are not diabetic” can lead to serious health issues. In such cases, it is crucial to focus on improving the recall value.

Precision:

Our precision value is 0.7192, meaning that 72% of those classified as diabetic by the model are indeed diabetic.

However, this also implies that 28% of the predictions were false positives (non-diabetic individuals misclassified as diabetic). This could lead to unnecessary diagnostic procedures and treatments. Improving precision can help avoid such misdiagnoses.

F1-Score:

The F1-Score, the harmonic mean of precision and recall, is 0.6371.

The F1-Score is important for evaluating overall performance in imbalanced datasets. A higher F1-Score indicates better success in recognizing the positive class.

AUC (Area Under Curve):

Our AUC value is 0.8327, indicating the model is generally successful.

The AUC metric evaluates the model’s ability to distinguish the positive class across all possible thresholds. An AUC of 0.83 shows that the model has a high ability to correctly differentiate between classes.

Why 10-Fold Cross-Validation Was Used?

Cross-validation is a technique used to evaluate a model’s performance in a more reliable and generalizable way.

5-Fold Cross-Validation: The dataset is split into 5 parts. In each iteration, one part is used as the test set while the remaining 4 parts serve as the training set. This ensures that every data point is tested at least once. This method is especially useful when the dataset is small, as it helps us better understand the model’s generalizable performance. Cross-validation allows us to detect issues like overfitting or underfitting. It provides insight into whether the model is consistently working across the entire dataset.

Is There Overfitting or Underfitting in the Model?

Overfitting:

If the model showed high performance only on the training set but failed on the test set, we would suspect overfitting. However, since the cross-validation results are consistent, overfitting is not observed in this case.

Underfitting:

The relatively low values of metrics like Recall and Precision (especially Recall) indicate that the model is struggling to identify the positive class effectively. This suggests that the model may need a more complex structure (e.g., hyperparameter optimization or trying different algorithms).
