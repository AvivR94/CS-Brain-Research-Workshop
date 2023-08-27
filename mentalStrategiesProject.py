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
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

def runApp():
    print("Start running app...")
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
    run_on_processed_data = 0
    unprocessed_subject_features_file_path = "/Users/avivrab/Documents/CS-workshop/CS-Brain-Research-Workshop/unprocessed_subject_features.csv"
    unprocessed_subjects_mean_sessions_file_path = "/Users/avivrab/Documents/CS-workshop/CS-Brain-Research-Workshop/unprocessed_subjects_mean_sessions.csv"
    unprocessed_subject_success_rates_file_path = "/Users/avivrab/Documents/CS-workshop/CS-Brain-Research-Workshop/unprocessed_subject_success_rates.csv"
    unprocessed_subjects_mean_sessions_success_file_path = "/Users/avivrab/Documents/CS-workshop/CS-Brain-Research-Workshopunprocessed_subjects_mean_sessions_success_file_path.csv"
    num_of_sessions = 6
    num_of_runs_per_session = 5
    ###-----------------LOAD DATA-----------------------------------###

    # read data set
    data = pd.read_csv(data_file_path)
    df = pd.DataFrame(data)

    ###----01------------------PREPROCESSING DATA---------------------###
    df = df.fillna(0.5)

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

    if not data_preprocessed:
        per_subject = df_filtered.groupby([subject_num_col_name])
        success_per_subject = df_success.groupby([subject_num_col_name])
        processed_subjects = []
        unprocessed_subjects_mean_sessions = [[]for _ in range(num_of_sessions)]
        unprocessed_subjects_mean_sessions_success = [[]for _ in range(num_of_sessions)]
        unprocessed_subject_success_rates = []
        unprocessed_subject_features = []
        last_session_success_rates = []
        last_minus_first_session_success_rates = []
        for subject_tuple in per_subject:
            subject_trial = subject_tuple[1]
            subject_num = subject_tuple[0][0]
            subject_trial_success = success_per_subject.get_group(subject_num)
            subject_features = []
            subject_sessions = subject_trial.groupby([session_num_col_name])
            subject_sessions_success = subject_trial_success.groupby([session_num_col_name])

            #drop subjects with more than 2 sessions missing
            if len(subject_sessions) < num_of_sessions - 1:
                continue
            #drop first session if they have all sessions
            if len(subject_sessions) == num_of_sessions:
                subject_sessions = subject_sessions.filter(lambda x: x.name != 1)
                subject_sessions_success = subject_sessions_success.filter(lambda x: x.name != 1)
                subject_sessions = subject_sessions.groupby([session_num_col_name])
                subject_sessions_success = subject_sessions_success.groupby([session_num_col_name])
            
            session_nums = subject_sessions_success.groups.keys()
            first_session_mean = subject_sessions_success.get_group(min(session_nums))["Qhat"].mean()
            last_session_mean = subject_sessions_success.get_group(max(session_nums))["Qhat"].mean()
            last_session_success_rates.append(last_session_mean)
            last_minus_first_session_success_rates.append(last_session_mean - first_session_mean)

            mean_sessions = []
            mean_session_success_rates = []
            # cosine for session (num_of_runs_per_session)
            for session_tuple in subject_sessions:
                session = session_tuple[1].drop([session_num_col_name, subject_num_col_name], axis="columns")
                session_num = session_tuple[0][0]
                session_success = subject_sessions_success.get_group(session_num)
                unprocessed_subjects_mean_sessions_success[session_num-1].append(session_success["Qhat"].mean())
                mean_session_success_rates.append(session_success["Qhat"].mean())
                # session has 1 run -> don't perform cosine within session
                if len(session) >= 2:
                    cosineWithinSubject(
                        session, subject_features, classify_sessions="within"
                    ) 
                else:
                    subject_features.append(-1) # -1 means that the subject has less than 2 runs in this session
                mean_session = meanVectorForEachSession(session)
                mean_sessions.append(mean_session)
                unprocessed_subjects_mean_sessions[session_num-1].append(mean_session)
            cosineWithinSubject(pd.DataFrame(mean_sessions), subject_features, classify_sessions = "between")
            unprocessed_subject_features.append(meanVectorForEachSession(pd.DataFrame(mean_sessions)))
            unprocessed_subject_success_rates.append(pd.DataFrame(mean_session_success_rates).mean())

            # replace -1 with mean of other sessions
            mean_cosine = 0
            for cosine_within_session in subject_features[:num_of_runs_per_session]:
                if cosine_within_session != -1:
                    mean_cosine += cosine_within_session
            mean_cosine = mean_cosine/num_of_runs_per_session
            for session_idx,cosine_within_session in enumerate(subject_features[:num_of_runs_per_session]):
                if cosine_within_session == -1:
                    subject_features[session_idx] = mean_cosine

            processed_subjects.append(subject_features)
            pd.DataFrame(processed_subjects).to_csv(subject_features_file_path, index=False,)
            pd.DataFrame(last_session_success_rates).to_csv(last_session_success_rates_file_path, index=False,)
            pd.DataFrame(last_minus_first_session_success_rates).to_csv(last_minus_first_session_success_rates_file_path, index=False,)
            pd.DataFrame(unprocessed_subject_features).to_csv(unprocessed_subject_features_file_path, index=False,)
            pd.DataFrame(unprocessed_subjects_mean_sessions).to_csv(unprocessed_subjects_mean_sessions_file_path, index=False,)
            pd.DataFrame(unprocessed_subject_success_rates).to_csv(unprocessed_subject_success_rates_file_path, index=False,)
            pd.DataFrame(unprocessed_subjects_mean_sessions_success).to_csv(unprocessed_subjects_mean_sessions_success_file_path, index=False,)
            data_preprocessed = 1
    else:
        processed_subjects = pd.DataFrame(pd.read_csv(subject_features_file_path))
        last_session_success_rates = pd.DataFrame(pd.read_csv(last_session_success_rates_file_path))
        last_minus_first_session_success_rates = pd.DataFrame(pd.read_csv(last_minus_first_session_success_rates_file_path))
        unprocessed_subject_features = pd.DataFrame(pd.read_csv(unprocessed_subject_features_file_path))
        unprocessed_subjects_mean_sessions = pd.DataFrame(pd.read_csv(unprocessed_subjects_mean_sessions_file_path))
        unprocessed_subject_success_rates = pd.DataFrame(pd.read_csv(unprocessed_subject_success_rates_file_path))
        unprocessed_subjects_mean_sessions_success = pd.DataFrame(pd.read_csv(unprocessed_subjects_mean_sessions_success_file_path))



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


    def r2_accuracy_score(y_test, predictions):
        # Calculate R-squared
        score = r2_score(y_test, predictions)
        # Scale R-squared to the range [0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        score = scaler.fit_transform([[score]])[0][0]
        return score

    ###----------------------PLOT RESULTS------------------------###
    def plot_kmeans(model, predictions, score, y_test):
        plt.scatter(y_test, predictions)
        plt.xlabel("Actual Success Rates")
        ytick_positions = [0, 1]
        ytick_labels = ['0', '1']
        plt.yticks(ytick_positions, ytick_labels)
        plt.ylabel("Kmeans Classification")
        plt.title(f"{model}, accuracy score: {score}")
        plt.show()

    def plot_linear(model, y_test, y_pred, score, session_num = -1):
        plt.scatter(y_test, y_pred, color="black")
        plt.plot(y_test, y_test, color="blue", linewidth=3)
        plt.xlabel("Actual Success Rates")
        plt.ylabel("Predicted Success Rates")
        if session_num > -1:
            plt.title(f"{model}, accuracy score: {score}, session: {session_num}")
        else:
            plt.title(f"{model}, accuracy score: {score}")
        plt.show()
        

    ###---------------------TEST-----------------------------###

    #linear models
    linear_regression_model = sklearn.linear_model.LinearRegression()
    logistic_regression_model = sklearn.linear_model.LogisticRegression()
    random_forest_model = sklearn.ensemble.RandomForestClassifier()
    xgboost_model = sklearn.ensemble.GradientBoostingClassifier()
    svm_model = sklearn.svm.SVC()

    #clustering models
    kmeans = sklearn.cluster.KMeans(n_clusters=2, random_state=0)

    if run_on_processed_data:
        #run kmeans
        model = kmeans
        kmeans_predictions = clustering(model, processed_subjects)
        #estimate accuracy
        model = random_forest_model
        score = 0
        max_score = 0
        predictions_avg = 0 
        success_avg = 0
        last_minus_first_session_success_rates = np.array(last_minus_first_session_success_rates).reshape(-1, 1)
        for i in range(5):
            logistic_X_train, logistic_X_test, logistic_y_train, logistic_y_test = \
                train_test_split(last_minus_first_session_success_rates, kmeans_predictions, test_size=0.3)
            logistic_predictions = regression(model, logistic_X_train, logistic_y_train, logistic_X_test)
            logistic_y_test = np.array(logistic_y_test).reshape(-1, 1)
            logistic_predictions = np.array(logistic_predictions).reshape(-1, 1)
            # model_score = model.score(logistic_y_test, logistic_predictions)
            model_score = accuracy_score(logistic_y_test, logistic_predictions)
            print(f"i:{i}")
            # print(f"x train:{logistic_X_train}\n y train:{logistic_y_train}\n")
            print(f"x test:{logistic_X_test}\n y test:{logistic_y_test}\n predictions:{logistic_predictions}")
            print(f"model score: {model_score}")
            if model_score > max_score:
                max_score = model_score
                best_predictions = logistic_predictions
                best_success = logistic_X_test
            score += model_score
            predictions_avg += logistic_predictions
            success_avg += logistic_X_test
        score = score/5
        print(score)
        predictions_avg = predictions_avg/5
        success_avg = success_avg/5


        last_minus_first_session_success_rates = np.array(last_minus_first_session_success_rates).reshape(-1, 1)
        plot_kmeans("Kmeans", kmeans_predictions, max_score, last_minus_first_session_success_rates)
        # plot_kmeans(model, best_predictions, max_score, best_success)
        # plot_kmeans(model, predictions_avg, score, success_avg)

    else: #run on unprocessed data
        # 1. linear regression - each subject is a mean mental strategy vector and mean success rate
        model = linear_regression_model
        unprocessed_X_train, unprocessed_X_test, unprocessed_y_train, unprocessed_y_test = \
            train_test_split(unprocessed_subject_features, unprocessed_subject_success_rates, test_size=0.3)
        unprocessed_predictions = regression(model, unprocessed_X_train, unprocessed_y_train, unprocessed_X_test)
        score = r2_accuracy_score(unprocessed_y_test, unprocessed_predictions)
        plot_linear("Linear Regression Model", unprocessed_y_test, unprocessed_predictions, score)
    
        # 2. 6 linear regressions - each subject is a mean mental strategy vector and success rate for each session
        # Convert array string cells to actual NumPy arrays
        def parse_array_string(i,s):
            return  np.fromstring(s[1:-1], sep=' ')

        def parse_table(session):
            # Apply the parsing function to each cell in the table
            parsed_table = [parse_array_string(i,subject) for i,subject in enumerate(session) if not pd.isna(subject)]
            # Convert the parsed table into a 2-dimensional NumPy array
            return np.array(parsed_table)

        model = linear_regression_model
        for session_num in range(1,num_of_sessions + 1):
            # print(unprocessed_subjects_mean_sessions.iloc[session_num-1])
            session = parse_table(unprocessed_subjects_mean_sessions.iloc[session_num-1])
            success = unprocessed_subjects_mean_sessions_success.iloc[session_num-1].dropna()
            unprocessed_X_train, unprocessed_X_test, unprocessed_y_train, unprocessed_y_test = \
                train_test_split(session, success, test_size=0.3)
            unprocessed_predictions = regression(model, \
            unprocessed_X_train, unprocessed_y_train, unprocessed_X_test)
            score = r2_accuracy_score(unprocessed_y_test, unprocessed_predictions)
            plot_linear("Linear Regression Model", unprocessed_y_test, unprocessed_predictions, score, session_num)

runApp()