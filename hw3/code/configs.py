import numpy as np
import pandas as pd

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


# Define the configuration settings of the data file, variables, plotting
def config_data():
    filepath = "../data/credit-data.csv" 
    # y variable
    yvar = "serious_dlqin2yrs"
    # all continuous x vars
    continuous = ["revolving_utilization_of_unsecured_lines", "age", 
        "number_of_dependents", "number_of_time30-59_days_past_due_not_worse", 
        "number_of_time60-89_days_past_due_not_worse",
        "number_of_times90_days_late", "debt_ratio", "monthly_income", 
        "number_of_open_credit_lines_and_loans", "number_real_estate_loans_or_lines"]
    # selected models
    models = ['SVM', 'LR', 'KNN', 'DT', 'RF', 'ET', 'GB', 'BAG']
    # selected features
    features = ["bucket_debt_ratio", "dummy_90days_1", "bucket_rev"]
    # show all plots? (True/False)
    show_plots = True
    # size of configs for output grid: 0 for small test grid, 1 for normal, 2 for large
    size = 1
    # what to replace missing data with ("mean" or value)
    missing = "mean"

    return filepath, yvar, continuous, models, features, show_plots, size, missing


# Define the set of models, and the parameters being tested
# size: size of parameters we will test and output (0, 1, 2)
def config_classifiers(size=0):
    classifiers = {
        'SVM': svm.LinearSVC(penalty='l1', random_state=0, dual=False, 
                loss='squared_hinge'),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'DT': DecisionTreeClassifier(),
        'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, 
                n_estimators=10),
        'BAG': BaggingClassifier(base_estimator=None, n_estimators=10, max_samples=1, 
                max_features=5, random_state=33)
    }

    configs_test = { 
        'SVM': { 'C' :[0.01]},
        'LR':  { 'penalty': ['l1'], 'C': [0.01]},
        'KNN': { 'n_neighbors': [5],'weights': ['uniform'],'algorithm': ['auto']},
        'DT':  { 'criterion': ['gini'], 'max_depth': [1], 'max_features': ['sqrt'],
                 'min_samples_split': [10]},
        'RF':  { 'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],
                 'min_samples_split': [10]},
        'ET':  { 'n_estimators': [1], 'criterion' : ['gini'] ,'max_depth': [1], 
                 'max_features': ['sqrt'],'min_samples_split': [10]},
        'GB':  { 'n_estimators': [1], 'learning_rate' : [0.1],'subsample' : [0.5], 
                 'max_depth': [1]},
        'BAG': { 'n_estimators': [1], 'max_samples': [1], 'max_features': [1]}    
    }

    configs = { 
        'SVM': { 'penalty': ['l1','l2'], 'C' :[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]},
        'LR':  { 'penalty': ['l1','l2'], 'C': [0.00001, 0.001, 0.1, 1, 10]},
        'KNN': { 'n_neighbors': [1,5,10,25,50,100], 'weights': ['uniform','distance'],
                 'algorithm': ['auto','ball_tree','kd_tree']},
        'DT':  { 'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 
                 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
        'RF':  { 'n_estimators': [1, 10, 100], 'max_depth': [5,50], 
                 'max_features': ['sqrt','log2'],'min_samples_split': [2,10]},
        'ET':  { 'n_estimators': [1, 10, 100], 'criterion' : ['gini', 'entropy'],
                 'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10]},
        'GB':  { 'n_estimators': [1, 10, 100], 'learning_rate' : [0.001,0.1,0.5],
                 'subsample' : [0.1,0.5,1.0], 'max_depth': [5,50]},
        'BAG': { 'n_estimators': [1, 10, 100], 'max_samples': [1, 2], 'max_features': [1, 2]}
    }

    configs_large = { 
        'SVM': { 'C' :[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10],'kernel':['linear']},
        'LR':  { 'penalty': ['l1','l2'], 'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]},
        'KNN': { 'n_neighbors': [1, 5, 10, 25, 50, 100],'weights': ['uniform','distance'],
                 'algorithm': ['auto','ball_tree','kd_tree']},
        'DT':  { 'criterion': ['gini', 'entropy'], 'max_depth': [1, 5, 10, 20, 50, 100], 
                 'max_features': ['sqrt','log2'],'min_samples_split': [2, 5, 10]},
        'RF':  { 'n_estimators': [1, 10, 100, 1000, 10000], 'max_depth': [1, 5, 10, 20, 50, 100], 
                 'max_features': ['sqrt','log2'],'min_samples_split': [2, 5, 10]},
        'ET':  { 'n_estimators': [1, 10, 100, 1000, 10000], 'criterion' : ['gini', 'entropy'],
                 'max_depth': [1, 5, 10, 20, 50, 100], 'max_features': ['sqrt','log2'],
                'min_samples_split': [2, 5, 10]},
        'GB':  { 'n_estimators': [1, 10, 100, 1000, 10000], 
                 'learning_rate' : [0.001, 0.01, 0.05, 0.1, 0.5],
                 'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
        'BAG': { 'n_estimators': [1, 10, 100, 1000, 10000], 'max_samples': [1, 2, 5], 
                 'max_features': [1, 2, 5]}
    }

    if size==0:
        return classifiers, configs_test
    elif size==1:
        return classifiers, configs
    else:
        return classifiers, configs_large
