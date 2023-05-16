import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# read data set
data = pd.read_csv(
    "/Users/avivrab/Documents/CS-workshop/CS-Brain-Research-Workshop/mental_strategies_data.csv"
)
df = pd.DataFrame(data)


print("before", df.shape)
# Using Pearson Correlation
# plt.figure(figsize=(20, 20))
# cor = df.corr()
# sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
# plt.show()

# # removing features with low variance
# sel = VarianceThreshold(threshold=(0.8 * (1 - 0.8)))
# sel.fit_transform(df)
# # selecting univariate features
# X, y = df(return_X_y=True)
# print(X.shape)
# X_new = SelectKBest(f_classif, k=2).fit_transform(X, y)
# print(X_new.shape)


# X = data.drop("target_column", axis=1)  # Features (independent variables)
# y = data["target_column"]  # Target variable (dependent variable)

###----------------- remove low variance columns -----------------###
variances = df.var()
variance_threshold = 0.025
low_variance_columns = variances[variances < variance_threshold].index
df_filtered = df.drop(low_variance_columns, axis="columns")
df_filtered.to_csv(
    "/Users/avivrab/Documents/CS-workshop/CS-Brain-Research-Workshop/modified_file.csv",
    index=False,
)
###---------------------------------------------------------------###
print("after op 1 :", df_filtered.shape)
###--------------- remove highly correlated columns ---------------###

# Calculate correlation matrix
correlation_matrix = df_filtered.corr()
# Define correlation threshold
correlation_threshold = 0.75

# Find highly correlated features
highly_correlated = set()  # Set to store correlated features

# Iterate over correlation matrix
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
            colname = correlation_matrix.columns[i]
            highly_correlated.add(colname)

# Remove highly correlated features from the dataset
df_filtered = df_filtered.drop(highly_correlated, axis="columns")
# Save filtered dataset to CSV
df_filtered.to_csv(
    "/Users/avivrab/Documents/CS-workshop/CS-Brain-Research-Workshop/filtered_dataset.csv",
    index=False,
)

###---------------------------------------------------------------###
print("after op 2: ", df_filtered.shape)

# TODO: make features: uklidean distance, cosine
