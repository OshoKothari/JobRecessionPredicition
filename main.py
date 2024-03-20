import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load your dataset
df = pd.read_csv('new_data.csv')

# Data Collection: Display basic information about the dataset
print("Head of the dataset:")
print(df.head())

print("\nTail of the dataset:")
print(df.tail())

print("\nSize of the dataset:")
print(df.size)

print("\nShape of the dataset:")
print(df.shape)

print("\nInfo about the dataset:")
print(df.info())

# Data Preprocessing: Handling missing values
print("\nHandling missing values:")
print("Number of missing values per column:")
print(df.isnull().sum())

# Drop rows with missing values
df.dropna(inplace=True)

# Data Exploration: Descriptive statistics
print("\nDescriptive statistics of numerical columns:")
print(df.describe())

# Data Visualization: Pair plot for numerical features
numerical_features = ['Year', 'GDP_Growth', 'Inflation', 'Industrial_Production', 'Unemployment_Rate', 'Recession_Indicator']
pair_plot_data = df[numerical_features]
sns.pairplot(pair_plot_data, hue='Recession_Indicator', diag_kind='kde')
plt.suptitle('Pair Plot of Numerical Features')
plt.show()

# Data Visualization: Correlation heatmap for numerical features
numeric_data = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()

# Outliers Identification: Box plots for numerical features
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features[:-1], 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x='Recession_Indicator', y=feature, data=df)
    plt.title(f'Box plot of {feature}')
plt.tight_layout()
plt.show()

# Encode categorical variables using one-hot encoding
X = df.drop('Recession_Indicator', axis=1)
y = df['Recession_Indicator']

# Define numerical and categorical features
numerical_features = ['Year', 'GDP_Growth', 'Inflation', 'Industrial_Production', 'Unemployment_Rate']
categorical_features = ['Quarter']

# Create transformers for numerical and categorical features
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder())
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Append classifier to preprocessing pipeline with class weights
rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(class_weight='balanced'))
])
rf_model.fit(X_train, y_train)

# Train Support Vector Machine model with class weights
svm_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(class_weight='balanced', probability=True))
])
svm_model.fit(X_train, y_train)

# Train Logistic Regression model with class weights
lr_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42))
])
lr_model.fit(X_train, y_train)

# Train Gradient Boosting model with class weights
gb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier())
])
gb_model.fit(X_train, y_train)

# Train K-Nearest Neighbors model
knn_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier())
])
knn_model.fit(X_train, y_train)

# Display classification reports and confusion matrices
def display_classification_reports_and_confusion_matrices(model, X_test, y_test, title):
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, zero_division=1)
    confusion_mat = confusion_matrix(y_test, predictions)
    
    print(f"Classification Report - {title}:")
    print(report)
    
    plt.figure(figsize=(4, 3))
    sns.heatmap(confusion_mat, annot=True, cmap='Blues', fmt='g', xticklabels=['Not Recession', 'Recession'], yticklabels=['Not Recession', 'Recession'])
    plt.title(f'Confusion Matrix - {title}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Display classification reports and confusion matrices for each model
for model, title in zip([rf_model, svm_model, lr_model, gb_model, knn_model],
                        ['Random Forest', 'Support Vector Machine', 'Logistic Regression', 'Gradient Boosting', 'K-Nearest Neighbors']):
    display_classification_reports_and_confusion_matrices(model, X_test, y_test, title)

# ROC Curve visualization in a single figure
plt.figure(figsize=(10, 8))
for model, title in zip([rf_model, svm_model, lr_model, gb_model, knn_model],
                        ['Random Forest', 'Support Vector Machine', 'Logistic Regression', 'Gradient Boosting', 'K-Nearest Neighbors']):
    probabilities = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    auc = roc_auc_score(y_test, probabilities)
    plt.plot(fpr, tpr, label=f'{title} (AUC = {auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='lower right')
plt.show()

# User input loop
stop_condition = False

while not stop_condition:
    user_year_input = input("Enter the year for prediction (or type 'stop' to end): ")

    if user_year_input.lower() == 'stop':
        stop_condition = True
        break

    # Convert user_year_input to integer
    try:
        user_year = int(user_year_input)
    except ValueError:
        print("Invalid year input. Please enter a valid year or type 'stop' to end.")
        continue

    user_quarter = input("Enter the quarter for prediction (Q1, Q2, Q3, Q4): ").upper()

    # Check for valid quarter input
    valid_quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    if user_quarter not in valid_quarters:
        print("Invalid quarter input. Please enter Q1, Q2, Q3, or Q4.")
        continue

    # Create user input DataFrame
    user_input = pd.DataFrame({
        'Year': [user_year],
        'Quarter': [user_quarter],
        'GDP_Growth': float(input("Enter GDP Growth (%): ")),
        'Inflation': float(input("Enter Inflation (%): ")),
        'Industrial_Production': float(input("Enter Industrial Production (%): ")),
        'Unemployment_Rate': float(input("Enter Unemployment Rate (%): ")),
    })

    # Model selection for prediction
    selected_model = input("Choose a model for prediction (RF, SVM, LR, GB, KNN): ").upper()
    selected_model_dict = {'RF': rf_model, 'SVM': svm_model, 'LR': lr_model, 'GB': gb_model, 'KNN': knn_model}

    if selected_model in selected_model_dict:
        model = selected_model_dict[selected_model]

        # Make predictions for user input
        user_prediction = model.predict_proba(user_input)[0, 1] * 100

        # Make binary predictions based on the threshold
        threshold = 50
        recession_prediction = 1 if user_prediction >= threshold else 0

        # Display user predictions and confidence levels in percentage format
        print(f"\nPrediction for {user_quarter} {user_year} using {selected_model} model:")
        print(f"{selected_model} Prediction: {user_prediction:.2f}% (Recession: {recession_prediction})")

        # Compare selected model with other models using the same input
        print("\nComparison with other models:")
        for other_model, other_title in zip([rf_model, svm_model, lr_model, gb_model, knn_model],
                                           ['Random Forest', 'Support Vector Machine', 'Logistic Regression', 'Gradient Boosting', 'K-Nearest Neighbors']):
            if other_model != model:
                other_prediction = other_model.predict_proba(user_input)[0, 1] * 100
                other_recession_prediction = 1 if other_prediction >= threshold else 0
                print(f"{other_title} Prediction: {other_prediction:.2f}% (Recession: {other_recession_prediction})")

        # Visualize the comparison of selected model with other models
        plt.figure(figsize=(8, 5))
        model_probabilities = [model.predict_proba(user_input)[0, 1] * 100 for model in [rf_model, svm_model, lr_model, gb_model, knn_model]]
        model_titles = ['Random Forest', 'Support Vector Machine', 'Logistic Regression', 'Gradient Boosting', 'K-Nearest Neighbors']
        sns.barplot(x=model_probabilities, y=model_titles, color='skyblue', label=selected_model)
        plt.title(f'Comparison of {selected_model} with Other Models')
        plt.xlabel('Probability (%)')
        plt.ylabel('Models')
        plt.xlim([0, 100])
        plt.legend()
        plt.show()

    else:
        print("Invalid model selection. Please choose from RF, SVM, LR, GB, or KNN.")

# Comparison of algorithms with provided values
new_data = {
    'Model': ['Random Forest', 'Support Vector Machine', 'Logistic Regression', 'Gradient Boosting', 'K-Nearest Neighbors'],
    'Accuracy': [100, 95, 90, 100, 96],
    'Precision': [100, 82, 67, 100, 90],
    'Recall': [100, 100, 94, 100, 90],
    'F1-Score': [100, 90, 78, 100, 90]
}

df_comparison = pd.DataFrame(new_data)

colors = ['#2C3E50', '#E74C3C', '#F39C12', '#16A085']

df_comparison_melted = pd.melt(df_comparison, id_vars=['Model'], var_name='Metric', value_name='Value')

# Plot using Seaborn with adjusted bar width
plt.figure(figsize=(16, 6))

# Move grid lines to the background
sns.set(style="whitegrid")
ax = sns.barplot(x='Model', y='Value', hue='Metric', data=df_comparison_melted, palette=colors, saturation=0.8,
                 dodge=True, ci=None, zorder=40, capsize=0.05)

# Adjust bar width to create more gap between bars
for container in ax.containers:
    container.width = 0.6

plt.xlabel('Classifiers', fontsize=16)
plt.ylabel('Percentage', fontsize=16)
plt.title('Performance Metrics of Models with Particle Swarm Optimization', fontsize=14)

plt.yticks(fontsize=12)
plt.ylim(0, 100)
plt.yticks(range(0, 101, 10))

plt.xticks(fontsize=14)

# Move legend outside the plot and to the upper right
plt.legend(title='Metric', loc='upper right', bbox_to_anchor=(1.25, 1), fontsize=12)

# Remove top and right spines
sns.despine()

plt.tight_layout()

plt.show()
