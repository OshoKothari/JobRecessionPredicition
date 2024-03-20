
## Job Recession Prediciton

This repository contains Python code for conducting a machine learning analysis related to predicting job recessions in India. The project is organized into several sections:

1. **Data Collection:** The dataset (`new_data.csv`) is loaded using Pandas to gather information about economic indicators such as GDP growth, inflation, industrial production, unemployment rate, and the recession indicator.

2. **Data Preprocessing:** Missing values in the dataset are handled by dropping rows with missing values. Descriptive statistics are then computed for numerical columns to understand the data better.

3. **Data Exploration and Visualization:** Data exploration includes creating pair plots and correlation heatmaps to visualize relationships between numerical features. Box plots are used to identify outliers in the data.

4. **Feature Engineering:** Categorical variables are encoded using one-hot encoding, and numerical and categorical transformers are defined using Scikit-Learn's preprocessing tools.

5. **Model Building:** Several machine learning models are trained using Scikit-Learn's Pipeline functionality, including Random Forest, Support Vector Machine (SVM), Logistic Regression, Gradient Boosting, and K-Nearest Neighbors (KNN). Class weights are applied to handle imbalanced data.

6. **Model Evaluation:** Classification reports, confusion matrices, and ROC curves are generated to evaluate the performance of each model on a test dataset.

7. **User Input and Prediction:** A user input loop allows users to input economic indicators for a specific year and quarter, choose a model for prediction, and compare the predictions of different models.

8. **Algorithm Comparison:** The performance metrics (accuracy, precision, recall, F1-score) of different models are compared using a bar plot visualization.

9. **Particle Swarm Optimization (PSO):** Performance metrics of models optimized using PSO are also displayed in a separate bar plot.

The code is well-documented with comments to explain each step and facilitate understanding and further development of the machine learning analysis. It provides a comprehensive framework for analyzing economic data and predicting job recessions using machine learning algorithms.
