# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor  # For feature importance
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv('winequality-red.csv', delimiter=',', encoding='utf-8')

# Display dataset info
print("Dataset Information:")
print(data.info())
print("\nFirst 5 Rows of the Dataset:")
print(data.head())

# Check for missing values
print("\nMissing Values in Each Column:")
print(data.isnull().sum())

# Exclude the 'Id' column before calculating correlations
correlation_matrix = data.drop(['Id'], axis=1).corr()

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()

# Feature scaling (important for clustering)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.drop(['Id'], axis=1))  # Exclude 'Id' column for scaling

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['quality_label'] = kmeans.fit_predict(scaled_data)

# Assign synthetic quality scores and labels to clusters
cluster_scores = {0: 50, 1: 75, 2: 100}  # Example: Map clusters to scores
cluster_labels = {0: 'low', 1: 'medium', 2: 'high'}  # Map clusters to quality labels
data['quality_score'] = data['quality_label'].map(cluster_scores)
data['quality_label_name'] = data['quality_label'].map(cluster_labels)

# Visualize the clusters in 2D using PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Add the cluster labels (quality labels) to the PCA dataframe for visualization
pca_df['quality_label_name'] = data['quality_label_name']

# Scatter plot of clusters in 2D space using PCA
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='quality_label_name', palette='Set1', data=pca_df, s=100, alpha=0.6)
plt.title('Wine Clusters with Quality')
plt.show()

# Distribution of clusters (simulating quality detection)
sns.countplot(x='quality_label_name', data=data, palette='viridis')
plt.title('Wine Quality Clusters Distribution')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.show()

# Evaluate the clustering by checking the number of wines in each cluster
print("\nCluster Sizes:")
print(data['quality_label_name'].value_counts())

# Train a regression model to predict quality score
X = data.drop(['quality_label', 'quality_score', 'Id', 'quality_label_name'], axis=1)  # Features
y = data['quality_score']  # Target quality score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict scores
y_pred = regressor.predict(X_test)

# Evaluate regression model
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R-squared (R²):", r2_score(y_test, y_pred))

# Rank wines by predicted quality score
data['predicted_quality_score'] = regressor.predict(X)
ranked_data = data.sort_values(by='predicted_quality_score', ascending=False)

print("\nTop 10 Wines by Predicted Quality Score:")
print(ranked_data[['Id', 'predicted_quality_score']].head(10))

# Visualize the ranked wines
plt.figure(figsize=(10, 6))
sns.barplot(x='Id', y='predicted_quality_score', data=ranked_data.head(10), palette='viridis')
plt.title('Top 10 Wines Ranked by Quality Score')
plt.xlabel('Wine ID')
plt.ylabel('Predicted Quality Score')
plt.xticks(rotation=45)
plt.show()

# Train a Random Forest Regressor to determine feature importance
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Get feature importance
feature_importance = rf_regressor.feature_importances_

# Create a DataFrame for feature importance
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
})

# Sort features by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Display top 5 features influencing the quality prediction
print("\nTop Features Influencing Wine Quality:")
print(importance_df.head())

# Visualize feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importance in Wine Quality Prediction')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Train a Logistic Regression model to predict cluster probabilities
classifier = LogisticRegression(max_iter=1000, random_state=42)
classifier.fit(X_train, data.loc[X_train.index, 'quality_label'])  # Train with original cluster labels

# Predict probabilities for the entire dataset
probabilities = classifier.predict_proba(X)

# Calculate batch probability distribution
batch_probabilities = np.mean(probabilities, axis=0)  # Average probabilities across all samples
batch_probability_distribution = {
    cluster_labels[cls]: batch_probabilities[idx] for idx, cls in enumerate(classifier.classes_)
}

print("\nProbability Distribution for Entire Batch:")
for quality, probability in batch_probability_distribution.items():
    print(f"{quality.capitalize()} Quality: {probability:.2%}")

# Visualize batch probability distribution
plt.figure(figsize=(8, 6))
sns.barplot(x=list(batch_probability_distribution.keys()), y=list(batch_probability_distribution.values()), palette='viridis')
plt.title('Probability Distribution for Wine Quality Categories')
plt.xlabel('Quality Category')
plt.ylabel('Probability')
plt.ylim(0, 1)
plt.show()


# Calculate batch probability distribution and convert to percentage
batch_probabilities = np.mean(probabilities, axis=0)  # Average probabilities across all samples
batch_probability_distribution = {
    cluster_labels[cls]: batch_probabilities[idx] * 100  # Multiply by 100 to convert to percentage
    for idx, cls in enumerate(classifier.classes_)
}

print("\nProbability Distribution for Entire Batch (in percentage):")
for quality, probability in batch_probability_distribution.items():
    print(f"{quality.capitalize()} Quality: {probability:.2f}%")  # Print as percentage

# Visualize batch probability distribution in percentage
plt.figure(figsize=(8, 6))
sns.barplot(x=list(batch_probability_distribution.keys()), 
            y=list(batch_probability_distribution.values()), 
            palette='viridis')
plt.title('Probability Distribution for Wine Quality Categories (in Percentage)')
plt.xlabel('Quality Category')
plt.ylabel('Probability (%)')
plt.ylim(0, 100)  # Set y-axis to range from 0 to 100
plt.show()
