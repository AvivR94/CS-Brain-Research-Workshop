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
import json
import os

def runApp():
    def load_variables(filename):
        script_dir = os.path.dirname(__file__)  # Directory of the script
        abs_file_path = os.path.join(script_dir, filename)
        with open(abs_file_path, 'r') as file:
            variables_dict = json.load(file)
            return variables_dict

    # loaded_variables = load_variables("user_variables.json")
    # print(loaded_variables)
    # how it should be:
    # variable1 = loaded_variables["variable1"]
    # variable2 = loaded_variables["variable2"]
    # option1 = loaded_variables["option1"]
    # option2 = loaded_variables["option2"]
    ###-----------------SET VARIABLES-----------------------------------###             
    data_file_path = "/Users/avivrab/Documents/CS-workshop/CS-Brain-Research-Workshop/mental_strategies_data.csv"
    success_cols_names = ["Qhat"]
    subject_num_col_name = "sub_num"
    session_num_col_name = "session_num"
    correlation_threshold = 0.75
    filtered_data_file_path = "/Users/avivrab/Documents/CS-workshop/CS-Brain-Research-Workshop/filtered_dataset.csv"
    data_preprocessed = 1
    subject_features_file_path = "/Users/avivrab/Documents/CS-workshop/CS-Brain-Research-Workshop/subjects_features.csv"
    last_session_success_rates_file_path = "/Users/avivrab/Documents/CS-workshop/CS-Brain-Research-Workshop/last_session_success_rates.csv"
    last_minus_first_session_success_rates_file_path = "/Users/avivrab/Documents/CS-workshop/CS-Brain-Research-Workshop/last_minus_first_session_success_rates.csv"
    run_on_processed_data = 1

    ###-----------------LOAD DATA-----------------------------------###

    # read data set
    data = pd.read_csv(data_file_path)
    df = pd.DataFrame(data)

    ###----01------------------PREPROCESSING DATA---------------------###
    df = df.fillna(0)

    ###----------------- extract success rates -----------------------###

    df_success = df.loc[:, df.columns.isin([subject_num_col_name, session_num_col_name]+success_cols_names)]
    df_filtered = df.drop(success_cols_names, axis="columns")

    ###----------------- remove low variance columns -----------------###
    variances = df.loc[:, ~df.columns.isin([subject_num_col_name, session_num_col_name])].var()
    variance_threshold = 0.025
    low_variance_columns = variances[variances < variance_threshold].index
    df_filtered = df_filtered.drop(low_variance_columns, axis="columns")

    ###--------------- remove highly correlated columns ---------------###

    # Calculate correlation matrix
    correlation_matrix = df_filtered.loc[
        :, ~df_filtered.columns.isin([subject_num_col_name, session_num_col_name])
    ].corr()

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
    df_filtered.to_csv(filtered_data_file_path, index=False,)

    ###------------------FEATURE EXTRACTION FUNCS---------------------###

    def softCos(a, b, matrix):
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
                        softCos(table.iloc[run_a], table.iloc[run_b], correlation_matrix)
                    )
                else:  # classify == "between"
                    subject_features.append(
                        softCos(table.iloc[run_a], table.iloc[run_b], correlation_matrix)
                    )
        if classify_sessions == "within":
            subject_features.append(np.array(soft_cosine_arr).mean())


    def meanVectorForEachSession(session):
        return np.array(session).mean(axis=0)


    ###----02---PROCESSING WITHIN SUBJECT: FEATURE EXTRACTION---------###
    if run_on_processed_data:
        if not data_preprocessed:
            per_subject = df_filtered.groupby([subject_num_col_name])
            success_per_subject = df_success.groupby([subject_num_col_name])
            subjects = []
            last_session_success_rates = []
            last_minus_first_session_success_rates = []
            for subject_tuple in per_subject:
                subject_trial = subject_tuple[1]
                subject_num = subject_tuple[0][0]
                subject_trial_success = success_per_subject.get_group(subject_num)
                subject_features = []
                subject_sessions = subject_trial.groupby([session_num_col_name])
                subject_sessions_success = subject_trial_success.groupby([session_num_col_name])

                #drop subjects with less than 5 sessions
                if len(subject_sessions) < 5:
                    continue
                #drop first session if they have 6 sessions
                if len(subject_sessions) == 6:
                    subject_sessions = subject_sessions.filter(lambda x: x.name != 1)
                    subject_sessions_success = subject_sessions_success.filter(lambda x: x.name != 1)
                    subject_sessions = subject_sessions.groupby([session_num_col_name])
                    subject_sessions_success = subject_sessions_success.groupby([session_num_col_name])
                
                session_nums = subject_sessions_success.groups.keys()
                first_session_mean = subject_sessions_success.get_group(min(session_nums))["Qhat"].mean()
                last_session_mean = subject_sessions_success.get_group(max(session_nums))["Qhat"].mean()
                last_session_success_rates.append(last_session_mean)
                last_minus_first_session_success_rates.append(last_session_mean - first_session_mean)

                strategies_cosines = []
                mean_sessions = []
                # cosine for session (5 runs)
                for session_tuple in subject_sessions:
                    session = session_tuple[1].drop([session_num_col_name, subject_num_col_name], axis="columns")
                    session_num = session_tuple[0][0]
                    # session has 1 run -> don't perform cosine within session
                    if len(session) >= 2:
                        cosineWithinSubject(
                            session, subject_features, classify_sessions="within"
                        ) 
                    else:
                        subject_features.append(-1) # -1 means that the subject has less than 2 runs in this session
                    mean_sessions.append(
                        meanVectorForEachSession(session)
                    )
                cosineWithinSubject(
                    pd.DataFrame(mean_sessions), subject_features, classify_sessions = "between"
                )
                # replace -1 with mean of other sessions
                mean_cosine = 0
                for cosine_within_session in subject_features[:5]:
                    if cosine_within_session != -1:
                        mean_cosine += cosine_within_session
                mean_cosine = mean_cosine/5
                for session_idx,cosine_within_session in enumerate(subject_features[:5]):
                    if cosine_within_session == -1:
                        subject_features[session_idx] = mean_cosine

                subjects.append(subject_features)
                pd.DataFrame(subjects).to_csv(subject_features_file_path, index=False,)
                pd.DataFrame(last_session_success_rates).to_csv(last_session_success_rates_file_path, index=False,)
                pd.DataFrame(last_minus_first_session_success_rates).to_csv(last_minus_first_session_success_rates_file_path, index=False,)
                data_preprocessed = 1
        else:
            subjects = pd.DataFrame(pd.read_csv(subject_features_file_path))
            last_session_success_rates = pd.DataFrame(pd.read_csv(last_session_success_rates_file_path))
            last_minus_first_session_success_rates = pd.DataFrame(pd.read_csv(last_minus_first_session_success_rates_file_path))


    ###-------03---PROCESSING BETWEEN SUBJECT: PREDICTION MODELS---------###

    ###----------------------REGRESSION MODELS------------------------###

    def regression(model, x_train, y_train, x_test):
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        return predictions

    ###----------------------CLUSTERING MODELS------------------------###

    def clustering(model, x):
        predictions = model.fit_predict(x)
        return predictions


    ###---------------------TEST-----------------------------###

    #linear models
    linear_regression_model = sklearn.linear_model.LinearRegression()
    random_forest_model = sklearn.ensemble.RandomForestRegressor()
    ridge_model = sklearn.linear_model.Ridge()
    lasso_model = sklearn.linear_model.Lasso()
    logistic_regression_model = sklearn.linear_model.LogisticRegression()

    #clustering models
    knn_model = sklearn.neighbors.KNeighborsRegressor()
    kmeans = sklearn.cluster.KMeans(n_clusters=2, random_state=0)



    if run_on_processed_data:
    ###---------------------RUN KMEANS-----------------------------###
        model = kmeans
        predictions = clustering(model, subjects)
    ###----------------------ESTIMATE ACCURACY------------------------###
        model = logistic_regression_model

        score = 0
        max_score = 0
        np.array(last_minus_first_session_success_rates).reshape(-1, 1)
        for i in range(5):
            logistic_X_train, logistic_X_test, logistic_y_train, logistic_y_test = train_test_split(last_minus_first_session_success_rates, predictions, test_size=0.2)
            logistic_predictions = regression(model, logistic_X_train, logistic_y_train, logistic_X_test)
            model_score = model.score(logistic_y_test, logistic_predictions)
            if model_score > max_score:
                best_predictions = logistic_predictions
            score += model_score
        score = score/5

    ###----------------------PLOT RESULTS------------------------###
        plt.scatter(logistic_y_test, best_predictions)
        plt.xlabel("Actual Success Rates")
        plt.ylabel("Predicted Success Rates")
        plt.title(f"{model}: Actual vs. Predicted (Test Data), score: {score}")
        plt.show()


    else:
        #TODO: run on unprocessed data - 
        # 1. linear regression - each subject is a mean mental strategy vector and mean success rate
        # 2. 6 linear regressions - each subject is a mean mental strategy vector and success rate for each session
        i = 0

