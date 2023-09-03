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
# from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from GUI import saveToFile

def runApp():
    def load_variables(filename):
        script_dir = os.path.dirname(__file__)  # Directory of the script
        abs_file_path = os.path.join(script_dir, filename)
        with open(abs_file_path, 'r') as file:
            variables_dict = json.load(file)
            return variables_dict


    ###-----------------LOAD DATA-----------------------------------###
    # GUI variables
    loaded_variables = load_variables("user_variables.json")
    subject_num_col_name = loaded_variables["subject_num_col_name"]
    session_num_col_name = loaded_variables["session_num_col_name"]
    success_cols_names = loaded_variables["success_cols_names"]
    correlation_threshold = float(loaded_variables["correlation_threshold"])
    data_preprocessed = int(loaded_variables["data_preprocessed"])
    run_on_processed_data = int(loaded_variables["run_on_processed_data"])
    num_of_sessions = int(loaded_variables["num_of_sessions"])
    num_of_runs_per_session = int(loaded_variables["num_of_runs_per_session"])
    data_file_path = saveToFile('mental_strategies_data.csv')
    filtered_data_file_path = saveToFile('filtered_dataset.csv')
    subject_features_file_path = saveToFile('subjects_features.csv')
    last_session_success_rates_file_path = saveToFile('last_session_success_rates.csv')
    last_minus_first_session_success_rates_file_path = saveToFile('last_minus_first_session_success_rates.csv')
    unprocessed_subject_features_file_path = saveToFile('unprocessed_subject_features.csv')
    unprocessed_subjects_mean_sessions_file_path = saveToFile('unprocessed_subjects_mean_sessions.csv')
    unprocessed_subject_success_rates_file_path = saveToFile('unprocessed_subject_success_rates.csv')
    unprocessed_subjects_mean_sessions_success_file_path = saveToFile('unprocessed_subjects_mean_sessions_success_file_path.csv')
    kmeans_successful_file_path = saveToFile('kmeans_successful.csv')
    kmeans_unsuccessful_file_path = saveToFile('kmeans_unsuccessful.csv')
    # read data set
    data = pd.read_csv(data_file_path)
    df = pd.DataFrame(data)

    ###----01------------------PREPROCESSING DATA---------------------###
    df = df.fillna(0.5)

    ###----------------- extract success rates -----------------------###

    df_success = df.loc[:, df.columns.isin([subject_num_col_name, session_num_col_name]+[success_cols_names])]
    df_filtered = df.drop([success_cols_names], axis="columns")

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
            first_session_mean = subject_sessions_success.get_group(min(session_nums))[success_cols_names].mean()
            last_session_mean = subject_sessions_success.get_group(max(session_nums))[success_cols_names].mean()
            last_session_success_rates.append(last_session_mean)
            last_minus_first_session_success_rates.append(last_session_mean - first_session_mean)

            mean_sessions = []
            mean_session_success_rates = []
            # cosine for session (num_of_runs_per_session)
            for session_tuple in subject_sessions:
                session = session_tuple[1].drop([session_num_col_name, subject_num_col_name], axis="columns")
                session_num = session_tuple[0][0]
                session_success = subject_sessions_success.get_group(session_num)
                unprocessed_subjects_mean_sessions_success[session_num-1].append(session_success[success_cols_names].mean())
                mean_session_success_rates.append(session_success[success_cols_names].mean())
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

    ###----------------------PLOT RESULTS------------------------###
    def plot_kmeans(predictions, y_test, title, color='blue'):
        plt.scatter(y_test, predictions, color=color)
        plt.xlabel("Success Rates")
        ytick_positions = [0, 1]
        ytick_labels = ['0', '1']
        plt.ylim(-0.5, 1.5)
        plt.yticks(ytick_positions, ytick_labels)
        plt.ylabel("Kmeans Classification")
        plt.title(title)
        plt.show()

    def plot_linear(scores):
        xtick_labels = ['Mean Overall', 'Session 1', 'Session 2', 'Session 3', 'Session 4', 'Session 5', 'Session 6']
        fig, ax = plt.subplots()
        bar_container = ax.bar(xtick_labels, scores)
        ax.set(ylabel="Mean Pearson's R", title="Linear Regressions' Mean Pearson's R", ylim=(-1,1))
        ax.bar_label(bar_container, fmt='{:,.3f}')    
        plt.show()
        

###----------------------- PARSING FUNCTIONS ------------------###
    def parse_array_string(i,s):
        return  np.fromstring(s[1:-1], sep=' ')

    def parse_table(session):
        # Apply the parsing function to each cell in the table
        parsed_table = [parse_array_string(i,subject) for i,subject in enumerate(session) if not pd.isna(subject)]
        # Convert the parsed table into a 2-dimensional NumPy array
        return np.array(parsed_table)

    ###---------------------TEST-----------------------------###

    #linear models
    linear_regression_model = sklearn.linear_model.LinearRegression()
    random_forest_model = sklearn.ensemble.RandomForestClassifier()

    #clustering models
    kmeans = sklearn.cluster.KMeans(n_clusters=2, random_state=0)

    kmeans_predictions = []
    if run_on_processed_data:
        #run kmeans
        model = kmeans
        kmeans_predictions = clustering(model, processed_subjects)
        #estimate accuracy
        model = random_forest_model
        max_score = 0
        score=0
        last_minus_first_session_success_rates = np.array(last_minus_first_session_success_rates).reshape(-1, 1)
        predictions_avg = 0 
        for i in range(5):
            forest_x_train, forest_x_test, forest_y_train, forest_y_test = \
                train_test_split(last_minus_first_session_success_rates, kmeans_predictions, test_size=0.3)
            forest_predictions = regression(model, forest_x_train, forest_y_train, forest_x_test) 
            forest_y_test = np.array(forest_y_test).reshape(-1, 1)
            forest_predictions = np.array(forest_predictions).reshape(-1, 1)
            model_score = accuracy_score(forest_y_test, forest_predictions)
            if model_score > max_score:
                max_score = model_score
                best_predictions = forest_predictions
                best_x_test = forest_x_test
                best_y_test = forest_y_test
            score += model_score
            predictions_avg += forest_predictions
        score = score/5
        print(f'kmeans average score: {round(score,3)}\nkmeans best score: {round(max_score,3)}')
        last_minus_first_session_success_rates = np.array(last_minus_first_session_success_rates).reshape(-1, 1)
        plot_kmeans(kmeans_predictions, last_minus_first_session_success_rates, "Kmeans Classification")
        plot_kmeans(best_predictions, best_x_test, f'Random Forest prediction, Accuracy score = {round(max_score,3)}', color="red")
        plot_kmeans(best_y_test, best_x_test, 'Kmeans Classification')


    else: #run on unprocessed data
        # 1. linear regression - each subject is a mean mental strategy vector and mean success rate
        model = linear_regression_model
        mean_pearson_r = 0
        mean_scores = []
        for i in range(5):
            unprocessed_X_train, unprocessed_X_test, unprocessed_y_train, unprocessed_y_test = \
                train_test_split(unprocessed_subject_features, unprocessed_subject_success_rates, test_size=0.3)
            unprocessed_predictions = regression(model, unprocessed_X_train, unprocessed_y_train, unprocessed_X_test)
            score = sklearn.feature_selection.r_regression(unprocessed_y_test, unprocessed_predictions)[0]
            mean_pearson_r += score
        mean_pearson_r /= 5
        mean_scores.append(mean_pearson_r)
    
        # 2. 6 linear regressions - each subject is a mean mental strategy vector and success rate for each session

        model = linear_regression_model
        for session_num in range(1,num_of_sessions + 1):
            mean_pearson_r = 0
            scores = []
            for i in range(5):
                session = parse_table(unprocessed_subjects_mean_sessions.iloc[session_num-1])
                success = unprocessed_subjects_mean_sessions_success.iloc[session_num-1].dropna()
                unprocessed_X_train, unprocessed_X_test, unprocessed_y_train, unprocessed_y_test = \
                    train_test_split(session, success, test_size=0.3)
                unprocessed_predictions = regression(model, \
                unprocessed_X_train, unprocessed_y_train, unprocessed_X_test)
                unprocessed_y_test = np.array(unprocessed_y_test).reshape(-1,1)
                unprocessed_predictions = np.array(unprocessed_predictions).reshape(-1,1)
                score = sklearn.feature_selection.r_regression(unprocessed_y_test, unprocessed_predictions)[0]
                mean_pearson_r += score
            mean_pearson_r /= 5
            mean_scores.append(mean_pearson_r)
        plot_linear(mean_scores)

###---------------------- FEATURE IMPORTANCE -----------------###

    if run_on_processed_data:
        kmeans_successful = processed_subjects[kmeans.labels_==0]
        kmeans_unsuccessful = processed_subjects[kmeans.labels_==1]
        kmeans_successful.to_csv(kmeans_successful_file_path, index=False,)
        kmeans_unsuccessful.to_csv(kmeans_unsuccessful_file_path, index=False,)
        mean_successful_success_rate = last_minus_first_session_success_rates[kmeans.labels_==0].mean()
        mean_unsuccessful_success_rate = last_minus_first_session_success_rates[kmeans.labels_==1].mean()
        print(f"successful success rate: {mean_successful_success_rate}\nunsuccessful success rate: {mean_unsuccessful_success_rate}")
        successful_mean_vector = meanVectorForEachSession(kmeans_successful)
        unsuccessful_mean_vector = meanVectorForEachSession(kmeans_unsuccessful)
        plt.plot(successful_mean_vector, color="red")
        plt.plot(unsuccessful_mean_vector, color="blue")
        plt.xlabel('Features')
        plt.ylabel('Soft Cosine Parameter')
        plt.ylim(0,1)
        xticks_labels = [i for i in range(1,16)]
        xticks_locations = [i for i in range(15)]
        yticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        plt.xticks(xticks_locations, xticks_labels)
        plt.yticks(yticks, yticks)
        plt.legend(['Successful', 'Unsuccessful'])
        plt.title('Successful vs. Unsuccessful Mean Feature Values')
        plt.show()
        features_mean_differences = []
        features_total_variance = []
        processed_subjects = np.array(processed_subjects)
        for feature in range(processed_subjects.shape[1]):
            mean1 = processed_subjects[kmeans.labels_==0][:,feature].mean()
            mean2 = processed_subjects[kmeans.labels_==1][:,feature].mean()
            
            var = processed_subjects[:,feature].var()
        
            features_mean_differences.append(round(abs(mean1-mean2),3))
            features_total_variance.append(round(var,3))

        barWidth = 0.25
        br1 = np.arange(len(features_mean_differences))
        br2 = [x + barWidth for x in br1]
        plt.bar(br1, features_mean_differences, color='red', width=barWidth, label='Mean Difference')
        plt.bar(br2, features_total_variance, color='blue', width=barWidth, label='Total Variance')
        plt.xlabel('Features')
        plt.title('Feature Significance')
        plt.legend()
        plt.show()