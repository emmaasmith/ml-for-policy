import numpy as np
import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt

# Classifiers
# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix


def bool_cap (datum, val):
	if datum < val:
		return 1
	return 0


def bool_threshold(data, val):
	test = np.array([bool_cap(x, val) for x in data])
	return test


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
	print predictions
	predictions_true = predictions[::,1]
	print predictions_true

	# Plot how many are predicted to be true
	pl.hist(predictions_true)
	pl.show()

	return predictions_true


def one_evaluation(test, bool_probabilities):
	# 2x2 +- matrix
	print "\nActual vs. predicted matrix:"
	cross = pd.crosstab(test['serious_dlqin2yrs'], 
		bool_probabilities, rownames=["Actual"], colnames=["Predicted"])
	print cross
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

	# Precision and recall
	print "\nMetrics:"
	print classification_report(test['serious_dlqin2yrs'], bool_probabilities, labels=[0, 1])
	return accuracy

# The below function is taken from https://github.com/yhat/DataGotham2013/ and
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
def plot_roc(test, name, probs, plot=True):
	fpr, tpr, thresholds = roc_curve(test['serious_dlqin2yrs'], probs)
	roc_auc = auc(fpr, tpr)
	if plot==True:
		pl.clf()
		pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
		pl.plot([0, 1], [0, 1], 'k--')
		pl.xlim([0.0, 1.05])
		pl.ylim([0.0, 1.05])
		pl.xlabel('False Positive Rate')
		pl.ylabel('True Positive Rate')
		pl.title(name)
		pl.legend(loc="lower right")
		pl.show()
	return roc_auc


def evaluate_classifier(test, probabilities):
	best_a = 0
	best_i = 0

	# Test 100 different thresholds
	for i in range(1, 100):
		i2 = 1.0 - (.01 * float(i))
		print "\n####################\nTrying ", i2
		bool_probabilities = bool_threshold(probabilities, i2)
		accuracy = one_evaluation(test, bool_probabilities)

		if accuracy > best_a:
			best_i = i2
			best_a = accuracy

	# Choose the best performing threshold, according to AUC
	print "\n####################\n####################\nBest individual accuracy:"
	print best_i, " produced ", best_a
	bool_probabilities = bool_threshold(probabilities, best_i)
	one_evaluation(test, bool_probabilities)

	print "\n####################\n####################\nROC plot:"
	roc_auc = plot_roc(test, "ROC", probabilities, True)
	print "\nAUC: ", roc_auc
