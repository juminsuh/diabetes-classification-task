# diabetes-classification-task

## Dataset from Kaggle
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/data

## ✅ **Visualization**
I aimed to analyze how each feature influences the outcome by not simply visualizing the data distribution for yes/no classes but by setting appropriate cutoffs for each feature based on medical knowledge. Then, I visualized the proportions of yes/no classes below and above the cutoff for each feature.

![image](https://github.com/user-attachments/assets/85408b2f-6fca-4de8-883f-a5793ba4a3f2)

## ✅ **Preprocessing**

- There were no NaN values. However, the value 0 in [Glucose, BloodPressure, SkinThickness, Insulin, BMI] is considered meaningless, so I replaced 0 values with np.nan. These missing values are then appropriately filled using either the mean or median based on the data distribution.

- Outliers are replaced with nan values and subsequently filled using a KNN imputer.

## ✅ **Model**

I tried oversampling method called SMOTE, but the performance was not that good. So I just optimized the model using lightgbm with bayesian optimization. Below code is the optimal hyperparameters which yielded the highest f1 score, which was **0.7414** for training set. 

```
model = LGBMClassifier(
    learning_rate=0.02885,
      max_depth=int(34.71),
      n_estimators=int(15.62),
      num_leaves=int(71.1),
      reg_alpha= 0.02116,
      reg_lambda=0.02655)
```

## ✅ **Result**

### Confusion Matrix 

||predicted 0|predicted 1|
|------|---|---|
|actual 0|81|24|
|actual 1|6|43|

### Evaluation 

|f1|0.7413793103448276|
|------|---|
|recall|0.8775510204081632|
|precision|0.6417910447761194|
|accuracy|0.8051948051948052|

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.93      | 0.77   | 0.84     | 105     |
| 1     | 0.64      | 0.88   | 0.74     | 49      |

| Metric       | Value |
|--------------|-------|
| **Accuracy** | 0.81  |
| **Macro Avg** | Precision: 0.79, Recall: 0.82, F1-Score: 0.79 |
| **Weighted Avg** | Precision: 0.84, Recall: 0.81, F1-Score: 0.81 |
