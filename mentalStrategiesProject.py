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

###----01------------------PREPROCESSING DATA---------------------###
###----------------- remove low variance columns -----------------###
variances = df.loc[:, ~df.columns.isin(["sub_num", "session_num"])].var()
variance_threshold = 0.025
low_variance_columns = variances[variances < variance_threshold].index
df_filtered = df.drop(low_variance_columns, axis="columns")
df_filtered.to_csv(
    "/Users/avivrab/Documents/CS-workshop/CS-Brain-Research-Workshop/modified_file.csv",
    index=False,
)
###--------------- remove highly correlated columns ---------------###

# Calculate correlation matrix
correlation_matrix = df_filtered.loc[
    :, ~df_filtered.columns.isin(["sub_num", "session_num"])
].corr()
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
correlation_matrix = correlation_matrix.drop(highly_correlated, axis="columns")
# Save filtered dataset to CSV
df_filtered.to_csv(
    "/Users/avivrab/Documents/CS-workshop/CS-Brain-Research-Workshop/filtered_dataset.csv",
    index=False,
)

###------------------FEATURE EXTRACTION FUNCS---------------------###


def softCos(a, b, matrix):
    # nominator calculation
    sum_nominator = 0
    for i in range(len(a)):
        for j in range(len(b)):
            sum_nominator += (matrix[i, j]) * (a[i]) * (b[j])

    # denominator nrom values calculation
    sum_norm_a = 0
    for i in range(len(a)):
        for j in range(len(a)):
            sum_norm_a += matrix[i, j] * (a[i]) * (a[j])
    norm_a = np.sqrt(sum_norm_a)

    sum_norm_b = 0
    for i in range(len(b)):
        for j in range(len(b)):
            sum_norm_b += matrix[i, j] * (b[i]) * (b[j])
    norm_b = np.sqrt(sum_norm_b)

    # calculate softcosine between a and b
    sftcos = sum_nominator / (norm_a * norm_b)

    return sftcos


def cosineWithinSubject(table, subject_features):
    rows_arr = []
    for row in range(len(table) - 1):
        columns_arr = []
        for column in range(row + 1, len(table)):
            columns_arr.append(
                softCos(session[row], session[column], correlation_matrix)
            )
        rows_arr.append(columns_arr)
    subject_features.append(np.array(rows_arr).mean())


def meanVectorForEachSession(session):
    return np.array(session.mean(axis=1))


###----02---PROCESSING WITHIN SUBJECT: FEATURE EXTRACTION---------###

per_subject = df_filtered.groupby(["sub_num"])
subjects = []
for subject_num, subject in enumerate(per_subject):
    subject_features = []
    all_sessions = subject.groupby(["session_num"])
    strategies_cosines = []
    mean_sessions = []
    # cosine for session (5 runs)
    for session_num, session in enumerate(all_sessions):
        cosineWithinSubject(session, subject_features)  #### 6 features
        mean_sessions.append(
            meanVectorForEachSession(session)
        )  # mean_sessions is of 6th dim for each run
    cosineWithinSubject(mean_sessions, subject_features)
    subjects.append(subject_features)
###---------------------------------------------------------------###
