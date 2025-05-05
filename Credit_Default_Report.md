# Credit Card Default Prediction

Student Name

STAT 4000  
Department of Mathematics and Statistics  
Auburn University

May 4, 2025

## 1. Introduction

### 1.1 Background and Motivation

Credit card default prediction is a critical problem in the financial industry. Banks and credit card companies need to assess the likelihood of customers defaulting on their payments to manage risk effectively. Accurate prediction of defaults can help financial institutions make informed decisions about credit limits, interest rates, and collection strategies. This analysis explores methods to predict credit card defaults based on customer demographics and payment history.

### 1.2 Problem Statement and Research Questions

The main research question addressed in this study is: Can we accurately predict whether a credit card holder will default on their payment next month based on their demographic information and payment history?

Specific sub-questions include:
- Which features are most predictive of credit card default?
- How do different machine learning algorithms compare in predicting credit card defaults?
- How do different preprocessing and sampling techniques affect model performance?

### 1.3 Dataset Description

This study uses the "Default of Credit Card Clients" dataset from the UCI Machine Learning Repository. The dataset contains information on default payments, demographic factors, credit data, payment history, and bill statements of credit card clients in Taiwan from April 2005 to September 2005.

The dataset includes 30,000 observations with 24 features (23 predictor variables and 1 target variable). Key variables include:
- Demographic information: Gender, Education, Marriage, Age
- Credit data: Credit limit balance
- Payment history: Past payment records (PAY_1 to PAY_6)
- Bill statements: Amount of bill statements (BILL_AMT1 to BILL_AMT6)
- Payment amounts: Amount of previous payments (PAY_AMT1 to PAY_AMT6)
- Target variable: Default payment next month (1=Yes, 0=No)

### 1.4 Report Outline

The remainder of this report is organized as follows:
- Section 2 describes the methodology, including data preprocessing, exploratory data analysis, modeling approaches, and evaluation metrics.
- Section 3 presents the results of the analysis, including model performance comparisons.
- Section 4 discusses the findings, limitations, and conclusions of the study.

## 2. Methodology

### 2.1 Data Preprocessing

Several preprocessing steps were applied to prepare the data for analysis:

1. **Data Cleaning**: 
   - Removed rows with unexpected categories in EDUCATION (values 0, 5, 6) and MARRIAGE (value 0) variables.
   - Adjusted PAY_1 to PAY_6 values by adding 1 to align with the expected range.

2. **Feature Engineering**:
   - Created binary features for gender (MALE), marital status (MARRIED), and education level (GRAD_SCHOOL, UNIVERSITY, HIGH_SCHOOL) using one-hot encoding.
   - Dropped the original categorical columns (SEX, MARRIAGE, EDUCATION) after feature engineering.

3. **Data Splitting**:
   - Split the data into training (75%) and testing (25%) sets, with stratification to maintain the class distribution.

4. **Feature Scaling**:
   - Applied Min-Max scaling to normalize all features to a range of 0 to 1.

5. **Dimensionality Reduction**:
   - Applied Principal Component Analysis (PCA) to reduce the dimensionality of the data while preserving 90% of the variance, resulting in 12 principal components.

6. **Handling Class Imbalance**:
   - Applied K-Means sampling to balance the classes by selecting representative samples from the majority class.
   - Applied K-Means SMOTE to oversample the minority class while maintaining the structure of the data.

### 2.2 Exploratory Data Analysis (EDA)

The exploratory data analysis revealed several key insights:

1. **Class Distribution**:
   - The dataset is imbalanced, with only 22.1% of clients defaulting on their payments.
   - [INCLUDE FIGURE: Class Distribution - No resampling]

2. **Credit Limit by Default**:
   - Defaulters tend to have lower credit limits compared to non-defaulters.
   - [INCLUDE FIGURE: Credit Limit by Default]

3. **Repayment Status and Default**:
   - There is a strong relationship between repayment status (PAY_1) and default probability.
   - Clients with delayed payments are more likely to default.
   - [INCLUDE FIGURE: Default Proportions by Repayment Status]

4. **Correlation Analysis**:
   - Payment history variables (PAY_1 to PAY_6) show moderate positive correlation with default.
   - Bill amounts show weak correlation with default.
   - [INCLUDE FIGURE: Correlation Matrix]

5. **Age Distribution**:
   - Younger clients appear slightly more likely to default than older clients.
   - [INCLUDE FIGURE: Age Distribution by Default]

6. **Bill Amounts Across Months**:
   - Bill amounts show some variation across months, with defaulters having slightly different patterns than non-defaulters.
   - [INCLUDE FIGURE: Bill Amounts Across Months by Default]

### 2.3 Modeling Approaches / Algorithms Used

Four different classification algorithms were implemented and compared:

1. **Logistic Regression**:
   - A linear model that estimates the probability of default based on a linear combination of features.
   - Hyperparameter tuning was performed for the regularization parameter C using values [0.001, 0.01, 0.1, 1, 10, 100].
   - Logistic regression provides interpretable coefficients that indicate the direction and strength of feature relationships.

2. **Support Vector Machine (SVM)**:
   - A non-linear model that finds the optimal hyperplane to separate defaulters from non-defaulters.
   - Hyperparameter tuning was performed for C [0.1, 1, 10] and kernel type ['rbf', 'linear'].
   - SVM can capture complex relationships in the data through kernel transformations.

3. **Decision Tree**:
   - A tree-based model that recursively splits the data based on feature values to create homogeneous subsets.
   - Hyperparameter tuning was performed for max_depth [5, 10, 20, 30, 50] and criterion ['entropy'].
   - Decision trees provide interpretable rules and can capture non-linear relationships.

4. **Random Forest**:
   - An ensemble of decision trees that improves prediction accuracy and reduces overfitting.
   - Hyperparameter tuning was performed for n_estimators [10, 50, 100] and max_features ['sqrt'].
   - Random forests provide feature importance measures and can handle high-dimensional data.

Each algorithm was evaluated using four different preprocessing approaches:
1. Raw data (no dimensionality reduction or sampling)
2. PCA (dimensionality reduction only)
3. PCA + K-Means sampling (dimensionality reduction and undersampling)
4. PCA + KMeansSMOTE (dimensionality reduction and oversampling)

### 2.4 Evaluation Metrics

The following metrics were used to evaluate model performance:

1. **Accuracy**: The proportion of correct predictions among the total number of cases. While commonly used, this metric can be misleading for imbalanced datasets.

2. **Precision**: The proportion of true positive predictions among all positive predictions. This measures the model's ability to avoid false positives.

3. **Recall**: The proportion of true positive predictions among all actual positive cases. This measures the model's ability to find all positive cases.

4. **F1-score**: The harmonic mean of precision and recall, providing a balance between the two. This is particularly useful for imbalanced datasets.

5. **AUC (Area Under the Precision-Recall Curve)**: Measures the model's ability to distinguish between classes across different threshold settings, focusing on the trade-off between precision and recall.

F1-score was chosen as the primary metric for model comparison due to the class imbalance in the dataset, as it balances the need to correctly identify defaulters (recall) while minimizing false alarms (precision).

### 2.5 Tools and Libraries

The analysis was conducted using Python with the following libraries:
- **Data manipulation**: Pandas, NumPy
- **Machine learning**: Scikit-learn, imbalanced-learn
- **Visualization**: Matplotlib, Seaborn
- **Data acquisition**: ucimlrepo

## 3. Results

The performance of each model across different preprocessing approaches was evaluated using the metrics described above. The results are summarized below:

### Logistic Regression Results

| Preprocessing Method | Accuracy | Recall | Precision | F1-score | AUC |
|----------------------|----------|--------|-----------|----------|-----|
| Raw data             | 0.790704 | 0.123561 | 0.666667 | 0.208482 | 0.455848 |
| PCA                  | 0.798541 | 0.178074 | 0.686916 | 0.282828 | 0.476184 |
| PCA + K-Means        | 0.480746 | 0.824349 | 0.276964 | 0.414623 | 0.454689 |
| PCA + KMeansSMOTE    | 0.725848 | 0.382193 | 0.384756 | 0.383470 | 0.341907 |

[INCLUDE FIGURE: Logistic Regression Confusion Matrix and Precision-Recall Curve]

### SVM Results

| Preprocessing Method | Accuracy | Recall | Precision | F1-score | AUC |
|----------------------|----------|--------|-----------|----------|-----|
| Raw data             | 0.394271 | 0.829194 | 0.245781 | 0.379172 | 0.182710 |
| PCA                  | 0.229023 | 0.996972 | 0.224037 | 0.365859 | 0.387072 |
| PCA + K-Means        | 0.333874 | 0.920654 | 0.240544 | 0.381430 | 0.305282 |
| PCA + KMeansSMOTE    | 0.226051 | 0.999394 | 0.223668 | 0.365529 | 0.428753 |

[INCLUDE FIGURE: SVM Confusion Matrix and Precision-Recall Curve]

### Decision Tree Results

| Preprocessing Method | Accuracy | Recall | Precision | F1-score | AUC |
|----------------------|----------|--------|-----------|----------|-----|
| Raw data             | 0.394271 | 0.829194 | 0.245781 | 0.379172 | 0.182710 |
| PCA                  | 0.229023 | 0.996972 | 0.224037 | 0.365859 | 0.387072 |
| PCA + K-Means        | 0.333874 | 0.920654 | 0.240544 | 0.381430 | 0.305282 |
| PCA + KMeansSMOTE    | 0.226051 | 0.999394 | 0.223668 | 0.365529 | 0.428753 |

[INCLUDE FIGURE: Decision Tree Confusion Matrix and Precision-Recall Curve]

### Random Forest Results

| Preprocessing Method | Accuracy | Recall | Precision | F1-score | AUC |
|----------------------|----------|--------|-----------|----------|-----|
| Raw data             | 0.813133 | 0.354936 | 0.648230 | 0.458708 | 0.545051 |
| PCA                  | 0.802594 | 0.298607 | 0.619347 | 0.402942 | 0.492479 |
| PCA + K-Means        | 0.507499 | 0.786796 | 0.282883 | 0.416146 | 0.345251 |
| PCA + KMeansSMOTE    | 0.730847 | 0.354936 | 0.387310 | 0.370417 | 0.375297 |

[INCLUDE FIGURE: Random Forest Confusion Matrix and Precision-Recall Curve]

### Model Comparison

[INCLUDE FIGURE: Model Performance Comparison Across Methods]

[INCLUDE FIGURE: F1-Score Comparison Across Models and Methods]

The best overall model was Random Forest with Raw data, achieving an F1-score of 0.458708. This model provided a good balance between precision and recall, which is important for the credit default prediction task.

## 4. Discussion & Conclusion

### Key Findings

1. **Model Performance**: Random Forest with Raw data achieved the best overall performance with an F1-score of 0.458708, followed closely by Random Forest with PCA + K-Means sampling (F1-score: 0.416146). This suggests that ensemble methods can effectively predict credit card defaults, even without extensive preprocessing.

2. **Preprocessing Impact**: Different preprocessing techniques had varying effects on model performance:
   - For Logistic Regression, sampling techniques significantly improved performance, with PCA + K-Means sampling achieving the highest F1-score (0.414623).
   - For SVM, performance was relatively consistent across preprocessing methods, with Raw data and PCA + K-Means sampling performing slightly better.
   - For Decision Tree, similar patterns to SVM were observed.
   - For Random Forest, Raw data performed best, suggesting that this model can effectively handle the original feature space without dimensionality reduction.

3. **Precision vs. Recall Trade-off**: Sampling techniques (K-Means and KMeansSMOTE) generally improved recall at the expense of precision. For example, Logistic Regression with PCA + K-Means sampling achieved a high recall of 0.824349 but a low precision of 0.276964. This trade-off may be acceptable in default prediction where missing a potential defaulter (false negative) is more costly than falsely flagging a non-defaulter (false positive).

4. **Class Imbalance Handling**: The original dataset had a significant class imbalance (22.1% defaulters), which affected model performance. Sampling techniques helped address this imbalance and improved the detection of the minority class, particularly for Logistic Regression.

### Limitations

1. **Data Limitations**: The dataset is from Taiwan in 2005, which may limit the generalizability of findings to other regions or time periods.

2. **Model Complexity vs. Interpretability**: While more complex models like Random Forest performed better, they are less interpretable than simpler models like Logistic Regression. This presents a trade-off for financial institutions where model transparency may be required for regulatory compliance or customer explanations.

3. **Performance Metrics**: The models achieved moderate F1-scores (best being 0.458708), indicating room for improvement. This suggests that predicting credit defaults remains challenging even with sophisticated machine learning techniques.

4. **Accuracy vs. Recall**: Models with high accuracy often had low recall (e.g., Random Forest with Raw data: 0.813133 accuracy but only 0.354936 recall), meaning they missed many actual defaulters. Conversely, models with high recall often had low accuracy (e.g., SVM with PCA + KMeansSMOTE: 0.226051 accuracy but 0.999394 recall), meaning they flagged many non-defaulters as defaulters.

### Conclusion

This study demonstrated that machine learning models can predict credit card defaults with moderate success, with Random Forest on Raw data achieving the best overall performance. The analysis revealed important trade-offs between different performance metrics and preprocessing techniques.

For financial institutions, the choice of model and preprocessing approach should be guided by specific business objectives:
- If minimizing false negatives (missed defaulters) is critical, models with high recall like SVM with PCA + KMeansSMOTE would be preferable.
- If a balance between precision and recall is desired, Random Forest with Raw data offers the best compromise.
- If model interpretability is important, Logistic Regression with PCA + K-Means sampling provides a reasonable balance of performance and transparency.

The study also highlighted the importance of addressing class imbalance, particularly for certain algorithms like Logistic Regression, where sampling techniques significantly improved performance.

Future work could explore more advanced feature engineering techniques, ensemble methods that combine predictions from multiple models, or deep learning approaches that might capture more complex patterns in the data. Additionally, incorporating domain knowledge and business-specific cost functions could further improve the practical utility of these models for financial institutions.

## References

1. UCI Machine Learning Repository: Default of Credit Card Clients Data Set. https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

2. Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480.

## A. Detailed Exploratory Data Analysis

[INCLUDE FIGURE: Box plot of numeric normalized features]

[INCLUDE FIGURE: Explained Variance by Principal Components]

[INCLUDE FIGURE: Cumulative Explained Variance by Principal Components]

[INCLUDE FIGURE: Class Distribution after K-means sampling]

[INCLUDE FIGURE: Class Distribution after KMeansSMOTE]

## B. Additional Results

[INCLUDE FIGURE: Decision Tree visualization]

## C. Code Snippets (Optional)

Key code snippets for the modeling pipeline:

```python
def pipeline(model, params, method='raw', n_components=12, random_state=random_state):
    """
    Run the complete modeling pipeline with specified preprocessing and sampling method.
    
    Parameters:
    -----------
    model : sklearn estimator
        The model to train
    params : dict
        Parameter grid for GridSearchCV
    method : str, default='raw'
        Preprocessing method: 'raw', 'pca', 'kmeans', 'kmeans_smote'
    n_components : int, default=12
        Number of PCA components to use
    random_state : int, default=42
        Random state for reproducibility
    
    Returns:
    --------
    tuple
        (test_scores, y_pred, y_test)
    """
    # Handle missing values in both train and test sets
    X_train_clean = handle_missing_values(X_train)
    X_test_clean = handle_missing_values(X_test)
    
    # Normalize data for all methods (important for PCA)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_norm = pd.DataFrame(
        scaler.fit_transform(X_train_clean),
        columns=X_train_clean.columns
    )
    X_test_norm = pd.DataFrame(
        scaler.transform(X_test_clean),
        columns=X_test_clean.columns
    )
    
    # Apply the specified method
    if method == 'raw':
        # Raw data - no PCA, no sampling
        X_train_processed = X_train_clean
        X_test_processed = X_test_clean
        y_train_processed = y_train
        
    elif method == 'pca':
        # Apply PCA on normalized data
        print("Applying PCA transformation...")
        pca = PCA(n_components=n_components, random_state=random_state)
        pca.fit(X_train_norm)  # Fit on normalized training data
        
        # Transform both training and test data
        X_train_pca = pca.transform(X_train_norm)
        X_test_pca = pca.transform(X_test_norm)
        
        # Convert to DataFrame with proper column names
        X_train_processed = pd.DataFrame(X_train_pca, 
                                        columns=[f'PC{i}' for i in range(1, n_components+1)])
        X_test_processed = pd.DataFrame(X_test_pca, 
                                       columns=[f'PC{i}' for i in range(1, n_components+1)])
        
        y_train_processed = y_train.reset_index(drop=True)
        
    elif method == 'kmeans':
        # Apply PCA first on normalized data
        pca = PCA(n_components=n_components, random_state=random_state)
        pca.fit(X_train_norm)
        
        X_train_pca = pca.transform(X_train_norm)
        X_test_pca = pca.transform(X_test_norm)
        
        # Convert to DataFrame and ensure indices match
        X_train_pca_df = pd.DataFrame(X_train_pca, 
                                    columns=[f'PC{i}' for i in range(1, n_components+1)])
        X_test_processed = pd.DataFrame(X_test_pca, 
                                    columns=[f'PC{i}' for i in range(1, n_components+1)])
        
        # Reset indices to ensure alignment
        X_train_pca_df = X_train_pca_df.reset_index(drop=True)
        y_train_reset = y_train.reset_index(drop=True)
        
        # Then apply K-Means sampling with aligned indices
        # Separate the training data into default and non-default classes
        X_train_default = X_train_pca_df[y_train_reset == 1]
        X_train_non_default = X_train_pca_df[y_train_reset == 0]
        
        # Set the number of clusters to match the smaller class size
        n_clusters = len(X_train_default)
        
        # Apply KMeans clustering to the larger class (non-default)
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        X_train_non_default_copy = X_train_non_default.copy()
        X_train_non_default_copy['cluster'] = kmeans.fit_predict(X_train_non_default)
        
        # Select one representative sample from each cluster
        X_train_non_default_sampled = X_train_non_default_copy.groupby('cluster').apply(
            lambda x: x.sample(1, random_state=random_state)
        ).reset_index(drop=True).drop(columns=['cluster'])
        
        # Combine the sampled non-default data with the default data
        X_train_processed = pd.concat([X_train_non_default_sampled, X_train_default])
        y_train_processed = pd.Series(
            np.concatenate([
                np.zeros(len(X_train_non_default_sampled)), 
                np.ones(len(X_train_default))
            ])
        )
        
        # Shuffle the data
        X_train_processed, y_train_processed = shuffle(X_train_processed, y_train_processed, random_state=random_state)
