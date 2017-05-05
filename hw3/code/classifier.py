import numpy as np
import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt

# Classifiers
from sklearn.svm import SVC
from sklearn.metrics import *


# Set a cap for a single value of data
# datum: value
# val: cap value
def bool_cap (datum, val):
    if datum < val:
        return 1
    return 0


# Set and apply a capped threshold for data
# data: df
# val: cap value
def bool_threshold(data, val):
    test = np.array([bool_cap(x, val) for x in data])
    return test


# Define, fit, and use a classifier. Plot true predictions
# train: df for train
# val: df for testing
# features: string[] of features
# model: rf for random forest
    # knn for k nearest neighbor
def classifier(train, test, features, model="rf"):
    # Define classifier
    if model == "knn":
        classifier = KNeighborsClassifier(n_neighbors=13)
    else:
        classifier = RandomForestClassifier()

    # Fit classifier to training data
    classifier.fit(train[features], train.serious_dlqin2yrs)

    # Use classifier to predict outcomes for test data
    predictions = classifier.predict_proba(test[features])
    predictions_true = predictions[::,1]

    # Plot how many are predicted to be true
    pl.hist(predictions_true)
    pl.show()

    return predictions_true


# Run one evaluation
# test_ys: data
# bool_probabilities: output from bool_threshold()
def one_evaluation(test_ys, bool_probabilities):
    # 2x2 +- matrix
    ys = pd.Series(list(test_ys))
    ps = pd.Series(bool_probabilities)
    cross = pd.crosstab(ys, ps, 
        rownames=["Actual"], colnames=["Predicted"])
    if np.unique(cross).size != 4:
        return

    # Accuracy
    accuracy = cross[0][0] + cross[1][1]
    div = cross[0][0] + cross[1][1] + cross[1][0] + cross[0][1]
    if div == 0:
        accuracy = 0
    else:
        accuracy = float(accuracy) / float(div)
    print "\nAccuracy: ", accuracy

    # Print 2x2
    print "\nActual vs. predicted matrix:"
    print cross

    # Precision and recall
    print "\nMetrics:"
    print classification_report(ys, bool_probabilities, labels=[0, 1])

    return accuracy


# Run an evaluation of data and probabilities with k% cutoff
# data: df of data
# probs: output from bool_threshold()
# k: percent cutoff of data
def evaluate_at_k(data, probs, k):
    # Set up the cutoff threshhold for the top k% of scores
    threshold = int((k / 100.0) * len(probs))
    bool_probabilities = [1 if x < threshold else 0 for x in range(len(probs))]
    
    # Get precision, recall, f1, accuracy
    precision = precision_score(data, bool_probabilities)
    print "Precision: ", precision
    recall = recall_score(data, bool_probabilities)
    print "Recall: ", recall
    f1 = f1_score(data, bool_probabilities)
    print "F1: ", f1
    accuracy = one_evaluation(data, bool_probabilities)


    return precision, accuracy, recall, f1

# Plot the ROC
# The below function is taken from https://github.com/yhat/DataGotham2013/ and
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
def plot_roc(test, name, shortname, probs, plot=2):
    fpr, tpr, thresholds = roc_curve(test, probs)
    roc_auc = auc(fpr, tpr)
    if plot>0:
        pl.clf()
        pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        pl.plot([0, 1], [0, 1], 'k--')
        pl.xlim([0.0, 1.05])
        pl.ylim([0.0, 1.05])
        pl.xlabel('False Positive Rate')
        pl.ylabel('True Positive Rate')
        pl.title(name)
        pl.legend(loc="lower right")
        if plot == 1:
            pl.show()
        else:
            pl.savefig("data/results/"+shortname+".png", bbox_inches='tight')
    return roc_auc


# Plot the precision-recall curve
# test_ys: data
# bool_probabilities: output from bool_threshold()
def plot_pr(test_ys, name, shortname, bool_probabilities):
    ys = pd.Series(list(test_ys))
    ps = pd.Series(bool_probabilities)
    precisions, recalls, thresholds = precision_recall_curve(ys, ps, pos_label=1)

    plt.clf()
    plt.plot(recalls, precisions, color='navy', label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(name)
    plt.legend(loc="lower left")
    pl.savefig("data/results/"+shortname+".png", bbox_inches='tight')


# DEPRECATED:
# Test 100 thresholds and choose the one that performs best via AUC
# data: df of data
# probabilities: classifier from classifier()
def evaluate_classifier(data, probabilities):
    best_a = 0
    best_i = 0

    # Test 100 different thresholds
    for i in range(1, 100):
        i2 = 1.0 - (.01 * float(i))
        print "\n####################\nTrying ", i2
        bool_probabilities = bool_threshold(probabilities, i2)
        accuracy = one_evaluation(data, bool_probabilities)

        if accuracy > best_a:
            best_i = i2
            best_a = accuracy

    # Choose the best performing threshold, according to AUC
    print "\n####################\n####################\nBest individual accuracy:"
    print best_i, " produced ", best_a
    bool_probabilities = bool_threshold(probabilities, best_i)
    one_evaluation(data, bool_probabilities)

    print "\n####################\n####################\nROC plot:"
    roc_auc = plot_roc(data, "ROC", probabilities, True)
    print "\nAUC: ", roc_auc
