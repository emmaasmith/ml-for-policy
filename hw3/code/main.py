import numpy as np
import pandas as pd
import pylab as pl
import re
import math
import matplotlib.pyplot as plt
import seaborn as sns

# Helper functions
from explore import print_describe
from explore import print_value_counts
from explore import build_graph
from explore import print_crosstab
from clean import camel_to_snake
from clean import select_cols
from clean import see_missing_vals_long
from clean import fill_missing_vals
from predictors import bucket
from predictors import dummy
from classifier import classifier
from classifier import evaluate_classifier
from classifier import evaluate_at_k
from classifier import plot_roc
from configs import config_data
from configs import config_classifiers

# sklearn
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid

from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler


# With liberal help from https://github.com/yhat/DataGotham2013
# and https://github.com/rayidghani/magicloops


# Run cleaning, exploration, and feature generation on a dataset
# df: pandas dataframe
# remove_outliers: bool to trip outliers in place
# plot: bool to show plots
# continuous: string[] of column names
# yvar: string yvar
def run(df, remove_outliers, plot, continuous, yvar, missing_solution):
    ##############################
    ## Read / load data 
    ##############################
    df.columns = [camel_to_snake(col) for col in df.columns]
    colnames = df.columns.tolist()

    ##############################
    ## Clean data
    ##############################

    # Melt (can include id_vars=[...], value_vars=[...])
    df_long = pd.melt(df)
    missing = see_missing_vals_long(df_long)

    # Fill missing values with the mean 
    unique_missing = df_long[missing]["variable"].unique()
    fill_missing_vals (df, unique_missing, missing_solution)
    df_long = pd.melt(df)

    # Remove outliers option
    if remove_outliers==True:
        old = df
        for c in continuous:
            df = df[np.abs(df[c] - df[c].mean()) <= (3 * df[c].std())]

        print "\nRemoved outliers: from ", old.shape, " to ", df.shape

    ##############################
    ## Explore data
    ##############################

    print_value_counts(df)
    
    print "\nFor each variable:"
    for c in continuous:
        # Describe the variable
        print_describe(df, c)

        if plot==True:
            sns.distplot(df[c])

            sns.jointplot(x=c, 
                y=yvar, data=df);

            sns.plt.show()

    ##############################
    ## Generate features / predictors (hard-coded)
    ##############################

    # Discretize a continuous variable: IP or bucket
    bucket_debt = bucket(df, "debt_ratio", 10, "bucket_debt_ratio", "q")
    bucket_rev = bucket(df, "revolving_utilization_of_unsecured_lines", 5, 
        "bucket_rev", "q")

    # Create binary/dummy variables from a categorical variable 
    bucket_90 = bucket(df, "number_of_times90_days_late", 2, "bucket_90days", "b")
    df = dummy(df, "bucket_90days", "dummy_90days")

    return df


# Run a loop over all models and parameters
# data: df of data
# models: array of models
# features: string[] of features
# yvar: string of outcome variable
def run_models(data, models, features, yvar, size):
    # Split into train and test sets
    train, test, y_train, y_test = train_test_split(data[features], data[yvar], test_size=0.1, random_state=33)

    # load the classifier configuration
    classifiers, configs = config_classifiers(size)

    # create results df
    results =  pd.DataFrame(columns=('model_type','clf', 'parameters', 'auc-roc', 
        'p_5', 'a_5', 'p_10', 'a_10', 'p_20', 'a_20'))

    # For each selected classifier model, take the (index, classifier)
    for i, clf in enumerate([classifiers[m] for m in models]):
        # Grab the config settings for this indexed model
        settings = configs[models[i]]
        # try/except with ParameterGrid
        for p in ParameterGrid(settings):
            try:
                # Configure this classifier with a pointer to the setting
                clf.set_params(**p)
                # Fit classifier to training data
                if hasattr(clf, "predict_proba"):
                    y_pred_probs = clf.fit(train, y_train).predict_proba(test)[:, 1]
                else:  # use decision function
                    y_pred_probs = clf.fit(train, y_train).decision_function(test)
                    y_pred_probs = (y_pred_probs - y_pred_probs.min()) / (y_pred_probs.max() - y_pred_probs.min())
                
                # Sort data
                y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
                
                # Run precision and accuracy
                p5, a5 = evaluate_at_k(y_test_sorted, y_pred_probs_sorted, 5.0)
                p10, a10 = evaluate_at_k(y_test_sorted, y_pred_probs_sorted, 10.0)
                p20, a20 = evaluate_at_k(y_test_sorted, y_pred_probs_sorted, 20.0)

                # Add to results df, indexing by growing len:
                results.loc[len(results)] = [models[i], # model name
                                            clf, # classifier with parameters
                                            p, # parameter
                                            roc_auc_score(y_test, y_pred_probs), # AUC score
                                            p5, a5, # precision/accuracy at k=5
                                            p10, a10, # precision/accuracy at k=10
                                            p20, a20] # precision/accuracy at k=20

                # Plot
                plot_roc(y_test, str(clf), str(i)+models[i], y_pred_probs)

            except IndexError, e:
                print 'Error: ',e
                continue
    return results


# main: run all code for given config settings
def main():
    # Readin config
    filepath, yvar, continuous, models, features, show_plots, size, missing = config_data()

    # Read in data
    df = pd.read_csv(filepath)

    # Prime data sets
    all_data = run(df, False, show_plots, continuous, yvar, missing)
    all_features = features + [yvar]
    df = all_data[all_features]

    # Run the models and build a matrix
    matrix = run_models(df, models, features, yvar, size)

    # Export results
    matrix.to_csv('../data/results/results.csv', index=False)


if __name__ == '__main__':
    main()
