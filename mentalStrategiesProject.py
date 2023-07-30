import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import sklearn as sklearn
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn import cluster
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import math

# read data set
data = pd.read_csv(
    "/Users/avivrab/Documents/CS-workshop/CS-Brain-Research-Workshop/mental_strategies_data.csv"
)
df = pd.DataFrame(data)

###----01------------------PREPROCESSING DATA---------------------###
df = df.fillna(0)

###----------------- extract success rates -----------------------###
success_cols_names = ["Qhat"]
df_success =  df.loc[:, df.columns.isin(success_cols_names)]
df_filtered = df.drop(success_cols_names, axis="columns")

###----------------- remove low variance columns -----------------###
variances = df.loc[:, ~df.columns.isin(["sub_num", "session_num"])].var()
variance_threshold = 0.025
low_variance_columns = variances[variances < variance_threshold].index
df_filtered = df_filtered.drop(low_variance_columns, axis="columns")

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
correlation_matrix = correlation_matrix.drop(highly_correlated, axis="columns").drop(
    highly_correlated, axis="rows"
)
# Save filtered dataset to CSV
df_filtered.to_csv(
    "/Users/avivrab/Documents/CS-workshop/CS-Brain-Research-Workshop/filtered_dataset.csv",
    index=False,
)
###------------------FEATURE EXTRACTION FUNCS---------------------###

def softCos(table, a, b, matrix):
    # nominator calculation
    sum_nominator = 0
    for i in range(len(a)):
        for j in range(len(b)):
            sum_nominator += matrix.iloc[i, j] * (a.iloc[i]) * (b.iloc[j])

    # denominator nrom values calculation
    sum_norm_a = 0
    for i in range(len(a)):
        for j in range(len(a)):
            sum_norm_a += matrix.iloc[i, j] * (a.iloc[i]) * (a.iloc[j])
    norm_a = np.sqrt(sum_norm_a)

    sum_norm_b = 0
    for i in range(len(b)):
        for j in range(len(b)):
            sum_norm_b += matrix.iloc[i, j] * (b.iloc[i]) * (b.iloc[j])
    norm_b = np.sqrt(sum_norm_b)

    # calculate softcosine between a and b
    sftcos = sum_nominator / (norm_a * norm_b)

    return sftcos


def cosineWithinSubject(table, subject_features, classify_sessions):
    if classify_sessions == "within":
        soft_cosine_arr = []
    for run_a in range(len(table) - 1):
        for run_b in range(run_a + 1, len(table)):
            if classify_sessions == "within":
                soft_cosine_arr.append(
                    softCos(table,table.iloc[run_a], table.iloc[run_b], correlation_matrix)
                )
            else:  # classify == "between"
                subject_features.append(
                    softCos(table,table.iloc[run_a], table.iloc[run_b], correlation_matrix)
                )
    if classify_sessions == "within":
        subject_features.append(np.array(soft_cosine_arr).mean())


def meanVectorForEachSession(session):
    return np.array(session).mean(axis=0)


###----02---PROCESSING WITHIN SUBJECT: FEATURE EXTRACTION---------###
data_preprocessed = 1
if not data_preprocessed:
    per_subject = df_filtered.groupby(["sub_num"])
    success_per_subject = df_success.groupby(["sub_num"])
    subjects = []
    last_session_success_rates = []
    last_minus_first_session_success_rates = []
    for subject_tuple in per_subject:
        subject_trial = subject_tuple[1]
        subject_num = subject_tuple[0][0]
        subject_trial_success = success_per_subject.get_group(subject_num)
        subject_features = []
        subject_sessions = subject_trial.groupby(["session_num"])
        subject_sessions_success = subject_trial_success.groupby(["session_num"])

        #subject has less than 5 sessions
        if len(subject_sessions) < 5:
            continue
        
        session_nums = subject_sessions_success.groups.keys()
        first_session_mean = subject_sessions_success.get_group(min(session_nums))["Qhat"].mean()
        last_session_mean = subject_sessions_success.get_group(max(session_nums))["Qhat"].mean()

        last_session_success_rates.append(last_session_mean)
        last_minus_first_session_success_rates.append(last_session_mean - first_session_mean)

        strategies_cosines = []
        mean_sessions = []
        # cosine for session (5 runs)
        for session_tuple in subject_sessions:
            session = session_tuple[1].drop(["session_num", "sub_num"], axis="columns")
            session_num = session_tuple[0][0]
            # session has less than 3 runs -> don't perform cosine within session
            if len(session) >= 3:
                # print(f"subject {subject_num} has {len(session)} runs in session {session_num}")
                cosineWithinSubject(
                    session, subject_features, classify_sessions="within"
                )  #### 6 features
            else:
                subject_features.append(0)
            mean_sessions.append(
                meanVectorForEachSession(session)
            )  # mean_sessions is of 6th dim for each run
        cosineWithinSubject(
            pd.DataFrame(mean_sessions), subject_features, classify_sessions = "between"
        )
        subjects.append(subject_features)
        pd.DataFrame(subjects).to_csv("/Users/avivrab/Documents/CS-workshop/CS-Brain-Research-Workshop/subjects_features.csv",
        index=False,)
        pd.DataFrame(last_session_success_rates).to_csv("/Users/avivrab/Documents/CS-workshop/CS-Brain-Research-Workshop/last_session_success_rates.csv",
        index=False,)
        pd.DataFrame(last_minus_first_session_success_rates).to_csv("/Users/avivrab/Documents/CS-workshop/CS-Brain-Research-Workshop/last_minus_first_session_success_rates.csv",
        index=False,)
        data_preprocessed = 1
else:
    subjects = pd.DataFrame(pd.read_csv("/Users/avivrab/Documents/CS-workshop/CS-Brain-Research-Workshop/subjects_features.csv"))
    last_session_success_rates = pd.DataFrame(pd.read_csv("/Users/avivrab/Documents/CS-workshop/CS-Brain-Research-Workshop/last_session_success_rates.csv"))
    last_minus_first_session_success_rates = pd.DataFrame(pd.read_csv("/Users/avivrab/Documents/CS-workshop/CS-Brain-Research-Workshop/last_minus_first_session_success_rates.csv"))

###---------------------------------------------------------------###

###-------03---PROCESSING BETWEEN SUBJECT: PREDICTION MODELS---------###

# Data preprocessing to handle inhomogeneous shape
# We'll pad the sequences to a fixed length using zeros
for subject_index in range(len(subjects)):
    subject = subjects.iloc[subject_index]
    subjects.iloc[subject_index] = subject.fillna(0)
pd.DataFrame(subjects).to_csv("/Users/avivrab/Documents/CS-workshop/CS-Brain-Research-Workshop/subjects_padded.csv",
        index=False,)
# Split the data into training and test sets (50% for training, 50% for testing)
processed_X_train, processed_X_test, processed_y_train, processed_y_test = train_test_split(subjects, last_session_success_rates, test_size=0.5, random_state=42)
 
###--------------------COMPARE MODELS-----------------------------###

for subject_index in range(len(df_filtered)):
    subject = df_filtered.iloc[subject_index]
    df_filtered.iloc[subject_index] = subject.fillna(0)

# Split the data into training and test sets (50% for training, 50% for testing)
unprocessed_X_train, unprocessed_X_test, unprocessed_y_train, unprocessed_y_test = train_test_split(df_filtered, df_success, test_size=0.5, random_state=42)
 

###----------------------REGRESSION MODELS------------------------###

def regression(model, x_train, y_train, x_test):
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return predictions


###---------------------TEST-----------------------------###

#linear models
linear_regression_model = sklearn.linear_model.LinearRegression()
random_forest_model = sklearn.ensemble.RandomForestRegressor()
ridge_model = sklearn.linear_model.Ridge()
lasso_model = sklearn.linear_model.Lasso()

#clustering models
knn_model = sklearn.neighbors.KNeighborsRegressor()
kmeans = sklearn.cluster.KMeans(n_clusters=2, random_state=0)

model = kmeans
processed = 0

if processed:
    predictions = regression(model, processed_X_train, processed_y_train, processed_X_test)
    y_test = processed_y_test
else:
    predictions = regression(model, unprocessed_X_train, unprocessed_y_train, unprocessed_X_test)
    y_test = unprocessed_y_test


plt.scatter(y_test, predictions)
plt.xlabel("Actual Success Rates")
plt.ylabel("Predicted Success Rates")
plt.title(f"{model}: Actual vs. Predicted (Test Data)")
plt.show()