# File upload from Google Colab
from google.colab import files
uploaded = files.upload()

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Enable inline plotting
%matplotlib inline

# Step 1: Read CSV file
df = pd.read_csv('data.csv.xls')  # Reading the dataset (replace 'data.csv.xls' with your file name if needed)
df.head()  # Display the first few rows of the dataset
df.info()  # Show information about the dataset (e.g., column names, data types, null values)

# Step 2: Convert integer columns to object type
int_columns = df.select_dtypes(include=['int64']).columns  # Selecting integer columns
df[int_columns] = df[int_columns].astype('object')  # Converting integer columns to object type
df.dtypes  # Checking data types after conversion

# Step 3: Basic Data Exploration
df.describe()  # Summary statistics of numerical features
df.isnull().sum()  # Checking missing values
df.duplicated().sum()  # Checking for duplicate rows
df.dropna(inplace=True)  # Removing rows with missing values
df.drop_duplicates(inplace=True)  # Removing duplicate rows

# Step 4: Correlation Analysis
numerical_df = df.select_dtypes(include=[np.number])  # Selecting numerical columns
correlation_matrix = numerical_df.corr()  # Computing correlation matrix

# Plotting the Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()

# Step 5: Visualization of Numerical Features
df.hist(bins=20, figsize=(15, 10), color='blue')  # Histogram of numerical features
plt.suptitle('Histograms of Numerical Features')
plt.show()

# Boxplot for numerical features
plt.figure(figsize=(10, 6))
sns.boxplot(data=df.select_dtypes(include=[np.number]))
plt.title('Boxplot for Numerical Features')
plt.show()

# Countplot for categorical features
for column in df.select_dtypes(include=['object']):
    plt.figure(figsize=(10, 5))
    sns.countplot(x=column, data=df, palette='viridis')
    plt.title(f'Countplot of {column}')
    plt.xticks(rotation=45)
    plt.show()

# Step 6: Bivariate Analysis
if len(df.select_dtypes(include=[np.number]).columns) >= 2:
    num_cols = df.select_dtypes(include=[np.number]).columns[:2]  # Selecting first two numerical columns
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=num_cols[0], y=num_cols[1], data=df, hue=df.select_dtypes(include=['object']).columns[0])
    plt.title(f'Scatterplot: {num_cols[0]} vs {num_cols[1]}')
    plt.show()

# Barplot of categorical vs numerical feature
plt.figure(figsize=(10, 6))
sns.barplot(x=df.select_dtypes(include=['object']).columns[0], 
            y=df.select_dtypes(include=[np.number]).columns[0], 
            data=df)
plt.title('Barplot of Categorical vs Numerical Feature')
plt.xticks(rotation=45)
plt.show()

# Pairplot for numerical features
sns.pairplot(df.select_dtypes(include=[np.number]))
plt.suptitle('Pairplot of Numerical Features', y=1.02)
plt.show()

# Boxplot to visualize distribution of a numerical variable across categories
for col in df.select_dtypes(include=['object']):
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=col, y=df.select_dtypes(include=[np.number]).columns[0], data=df, palette='Set3')
    plt.title(f'Boxplot of {col} vs Numerical Feature')
    plt.xticks(rotation=45)
    plt.show()

# Saving visualizations
plt.savefig('visualization_output.png')

# Step 7: Export Cleaned Dataset
df.to_csv('cleaned_dataset.csv', index=False)  # Saving the cleaned dataset
