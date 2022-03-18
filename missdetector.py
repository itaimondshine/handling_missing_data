import pandas as pd
import numpy as np
import scipy as sp

# statistics
from scipy import stats
import statsmodels.api as sm

# data visualizations
import seaborn as sns
import matplotlib.pyplot as plt

# Machine learning library
import sklearn

import warnings

warnings.filterwarnings("ignore")
import missingno as msno


def try_mcar(dtf, column_name):
    type = 'mcar'

    missing_values = no_correlation = False
    number_of_missing_values = dtf[column_name].isna().sum()
    percentage_of_missing_values = number_of_missing_values / dtf.shape[0]
    if percentage_of_missing_values < 0.1:
        missing_values = True

    # in mcar we want to be sure that column name is completely random. To do this, we can
    corr_matrix = coorelation_matrix(dtf)
    arr = list(corr_matrix[corr_matrix[column_name] < 1][column_name].values > 0.5)
    if not any(arr):
        no_correlation = True

    if missing_values:
        return "mcar"
    else:
        return "no mcar"


def try_mar(dtf, column_name):
    numeric_columns = find_numeric_columns(dtf)

    # here we will try to find a connection between the different features
    ls = list(dtf[column_name])  # make a list of values in column name
    ls = [str(a) for a in ls]  # convert to string
    replace_nan_values = list(map(lambda x: x.replace('nan', '-1000000'), ls))
    array_of_features = find_numeric_columns(dtf)
    for feature in array_of_features:
        if dtf[feature].isna().sum() > 0:
            number_nan_values_before = subarray(replace_nan_values, len(replace_nan_values))
            ls = list(dtf.sort_values(column_name)[feature])
            ls = [str(a) for a in ls]
            l = list(map(lambda x: x.replace('nan', '-1000000'), ls))
            number_nan_values_after = subarray(l, len(l))
            if number_nan_values_after > 3 * number_nan_values_before:
                return 'mar'
    else:
        return 'no mar'

    # return type


def try_mnar():
    pass


def detect_type_of_missingness(df, column_name):
    no_correlation = False
    corr_matrix = coorelation_matrix(df)
    negative = (corr_matrix[column_name] < -0.3) & (corr_matrix[column_name] > -1)
    positive = (corr_matrix[column_name] > 0.3) & (corr_matrix[column_name] < 1)
    if not any(positive | negative):
        no_correlation = True

    out_mcar = try_mcar(df, column_name)
    if out_mcar == 'mcar' and no_correlation:
        return "mcar"
    else:
        out_mar = try_mar(df, column_name)
        if out_mar == "mar":
            return "mar"
        else:
            return "mnar"


def subarray(arr, n):
    ans, temp = 1, 1
    # Traverse the array
    for i in range(1, n):
        # If element is same as previous
        # increment temp value
        if arr[i - 1] == '-1000000':
            if arr[i] == arr[i - 1]:
                temp = temp + 1
        else:
            ans = max(ans, temp)
            temp = 1
    ans = max(ans, temp)
    # Return the required answer
    return ans


def coorelation_matrix(dtf):
    dtf = dtf.iloc[:, [i for i, n in enumerate(np.var(dtf.isnull(), axis='rows')) if n > 0]]
    corr_mat = dtf.isnull().corr()
    return corr_mat


# def chi2_test(data, feature_selected):
#     Age_null = np.where(dtf[feature_selected].isnull(), True, False)
#     data["Age_null"] = Age_null
#     row_1 = data[data["Age_null"] == True].groupby("Survived")["Age_null"].count()
#     row_2 = data[data["Age_null"] == False].groupby("Survived")["Age_null"].count()
#     table = [row_1.values, row_2.values]
#     from scipy.stats import chi2_contingency
#     chi2, p, dof, ex = chi2_contingency(table)
#     return p

def find_numeric_columns(dtf):
    numeric_columns = dtf.dtypes[(dtf.dtypes == "float64") | (dtf.dtypes == "int64")].index.tolist()
    categorical_columns = [c for c in dtf.columns if c not in numeric_columns]
    very_numerical = [nc for nc in numeric_columns if dtf[nc].nunique() > 20]
    return categorical_columns


def load_dtf(dataset_path):
    dtf = pd.read_csv(dataset_path)
    dtf_copy = dtf.copy()
    return dtf_copy


def RSI(dtf, feature):
    dtf[feature + '_random'] = dtf[feature]  # Copy feature into new feature
    # calculate random smaple and store into random_sample_values
    random_sample_value = dtf[feature].dropna().sample(dtf[feature].isnull().sum(), random_state=0)
    # in random_sample_value all filled nan values are present now we want to put/merge this all filled values in our dataset
    # for this we want to match all nan values index in random_sample_values with df[variavle_'random]
    # Pandas need to have same index in order to merge dataset
    random_sample_value.index = dtf[dtf[feature].isnull()].index  # find index of NaN values in feature
    # now put a condition where ever it is null with loc function then replace with random_sample_values
    dtf.loc[dtf[feature].isnull(), feature + '_random'] = random_sample_value
    return dtf.loc[dtf[feature].isnull(), feature + '_random']


def dealing_with_mcar(df, feature):
    current_skew = df[feature].skew()
    # applying different methods
    mean = df[feature].mean()
    median = df[feature].median()
    mode = df[feature].mode()[0]
    new_f_mean = df[feature].fillna(mean)
    new_f_median = df[feature].fillna(median)
    new_f_mode = df[feature].fillna(mode)
    new_f_random = RSI(df, feature)
    df[feature + "_mean"] = new_f_mean
    df[feature + "_median"] = new_f_median
    df[feature + "_mode"] = new_f_mode
    df[feature + "_random"] = new_f_random
    options = ['mean', 'median', 'mode', 'random']
    opt_fetures = [new_f_mean, new_f_median, new_f_mode, new_f_random]
    opt_skew = [new_f_mean.skew() - current_skew, new_f_median.skew() - current_skew, new_f_mode.skew() - current_skew,
                new_f_random.skew() - current_skew]
    new_f_ind = np.argmin(opt_skew)
    return opt_fetures[new_f_ind], options[new_f_ind]


def plot_handling(dtf, column_feature):
    plt.figure(figsize=(12, 8))
    dtf[column_feature].plot(kind='kde', color='b')
    dtf[column_feature + '_median'].plot(kind='kde', color='y')
    dtf[column_feature + '_random'].plot(kind='kde', color='r')
    dtf[column_feature + '_mean'].plot(kind='kde', color='r')
    dtf[column_feature + '_mode'].plot(kind='kde', color='r')
    plt.plot()
    plt.legend()


