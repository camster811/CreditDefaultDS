# Figures to Include in the Credit Default Report

The following figures should be extracted from the notebook and included in the report at the specified locations:

## Section 2.2: Exploratory Data Analysis (EDA)

1. **Class Distribution - No resampling**
   - Bar chart showing the proportion of defaulters vs non-defaulters
   - Located in the "Key Statistics and Data Presentation" section of the notebook
   - Shows 22.1% defaulters vs 77.9% non-defaulters

2. **Credit Limit by Default**
   - Density plot showing the distribution of credit limits for defaulters and non-defaulters
   - Located in the "Data visualizations" section of the notebook
   - Shows defaulters tend to have lower credit limits

3. **Default Proportions by Repayment Status**
   - Stacked bar chart showing default proportions by repayment status (PAY_1)
   - Located in the "Data visualizations" section of the notebook
   - Shows strong relationship between repayment status and default probability

4. **Correlation Matrix**
   - Heatmap showing correlations between variables
   - Located in the "Data visualizations" section of the notebook
   - Shows moderate positive correlation between payment history variables and default

5. **Age Distribution by Default**
   - Density plot showing age distribution for defaulters and non-defaulters
   - Located in the "Data visualizations" section of the notebook
   - Shows younger clients are slightly more likely to default

6. **Bill Amounts Across Months by Default**
   - Line plot showing bill amounts across months for defaulters and non-defaulters
   - Located in the "Data visualizations" section of the notebook
   - Shows variation in bill amounts across months

## Section 3: Results

7. **Logistic Regression Evaluation**
   - Confusion Matrix and Precision-Recall Curve
   - Located in the "Model Evaluation" section of the notebook
   - Shows model performance metrics visually

8. **SVM Evaluation**
   - Confusion Matrix and Precision-Recall Curve
   - Located in the "SVM Evaluation" section of the notebook
   - Shows model performance metrics visually

9. **Decision Tree Evaluation**
   - Confusion Matrix and Precision-Recall Curve
   - Located in the "Tree Methods" section of the notebook
   - Shows model performance metrics visually

10. **Random Forest Evaluation**
    - Confusion Matrix and Precision-Recall Curve
    - Located in the "Random Forest" section of the notebook
    - Shows model performance metrics visually

11. **Model Performance Comparison**
    - Bar charts comparing all models across different metrics
    - Located in the "Conclusion" section of the notebook
    - Shows comparative performance of all models and methods

12. **F1-Score Comparison**
    - Heatmap comparing F1-scores across models and methods
    - Located in the "Conclusion" section of the notebook
    - Shows which model-method combination performed best

## Appendix A: Detailed Exploratory Data Analysis

13. **Box plot of numeric normalized features**
    - Box plot showing the distribution of normalized numeric features
    - Located in the "Normalize data" section of the notebook
    - Shows the distribution of features after normalization

14. **Explained Variance by Principal Components**
    - Bar chart showing explained variance by principal components
    - Located in the "PCA" section of the notebook
    - Shows how much variance each principal component explains

15. **Cumulative Explained Variance by Principal Components**
    - Line plot showing cumulative explained variance
    - Located in the "PCA" section of the notebook
    - Shows how many components are needed to explain a certain percentage of variance

16. **Class Distribution after K-means sampling**
    - Bar chart showing class distribution after K-means sampling
    - Located in the "K-Means sampling" section of the notebook
    - Shows balanced class distribution after sampling

17. **Class Distribution after KMeansSMOTE**
    - Bar chart showing class distribution after KMeansSMOTE
    - Located in the "K Means SMOTE" section of the notebook
    - Shows balanced class distribution after oversampling

## Appendix B: Additional Results

18. **Decision Tree visualization**
    - Tree diagram showing the structure of the decision tree model
    - Located in the "Tree Methods" section of the notebook
    - Shows the decision rules learned by the model
