# Binary Classification Project

This notebook demonstrates a binary classification workflow using a Decision Tree Classifier. The process involves data loading, preprocessing, model training, evaluation, and visualization of the decision boundary.

## 1. Data Loading and Initial Exploration

The dataset `Lab_Exam_binary_classification_dataset.csv` is loaded into a Pandas DataFrame. Initial steps include:
- Displaying the first few rows (`df.head()`).
- Checking data types and non-null counts (`df.info()`).
- Viewing descriptive statistics (`df.describe()`).

## 2. Data Preprocessing

- **Missing Values**: Missing values in the 'Target' column are identified and handled by dropping the corresponding rows (`df.dropna()`).
- **Target Encoding**: The 'Target' column, which contains 'Yes' and 'No' values, is converted to numerical representation (1 for 'Yes', 0 for 'No').

## 3. Exploratory Data Analysis (EDA)

Visualizations are used to understand the data characteristics:
- **Target Variable Distribution**: A count plot shows the distribution of the 'Target' variable.
- **Correlation Matrix**: A heatmap displays the correlation between features.
- **Scatter Plot of Features**: A scatter plot visualizes the relationship between 'Feature1' and 'Feature2', colored by the 'Target' variable.
- **Pair Plot**: A pair plot provides pairwise relationships between features and the target.
- **Box Plot**: Box plots are used to identify potential outliers and distribution of 'Feature1' and 'Feature2'.

## 4. Model Training

- **Data Splitting**: The dataset is split into training and testing sets (`X_train`, `X_test`, `y_train`, `y_test`) using a 80/20 ratio.
- **Feature Scaling**: `StandardScaler` is applied to scale the features (`Feature1`, `Feature2`) to ensure that features with larger values do not disproportionately influence the model.
- **Model Instantiation and Training**: A `DecisionTreeClassifier` is initialized with `random_state=42` and trained on the scaled training data.

## 5. Model Evaluation

The trained model's performance is evaluated using standard classification metrics on the test set:
- **Predictions**: Predictions are made on the scaled test set (`y_pred`).
- **Accuracy Score**: Overall accuracy of the model.
- **Confusion Matrix**: A matrix showing true positive, true negative, false positive, and false negative counts.
- **Classification Report**: Provides precision, recall, and f1-score for each class.

## 6. Visualization of Decision Boundary

A plot is generated to visualize the decision boundary of the trained Decision Tree Classifier. This helps in understanding how the model separates the two classes based on 'Feature1' and 'Feature2' in the scaled feature space.
