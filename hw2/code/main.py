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
from clean import see_missing_vals_long
from clean import fill_missing_vals
from predictors import bucket
from predictors import dummy
from classifier import classifier
from classifier import evaluate_classifier


# With liberal help from https://github.com/yhat/DataGotham2013

def run(df, remove_outliers, plot, continuous):
	##############################
	##### Read / load data 
	##############################
	df.columns = [camel_to_snake(col) for col in df.columns]
	colnames = df.columns.tolist()


	##############################
	##### Clean data
	##############################

	# Melt (can include id_vars=[...], value_vars=[...])
	df_long = pd.melt(df)
	missing = see_missing_vals_long(df_long)

	# Fill missing values with the mean
	unique_missing = df_long[missing]["variable"].unique()
	fill_missing_vals (df, unique_missing, "mean")
	df_long = pd.melt(df)

	# Remove outliers option
	if remove_outliers==True:
		old = df
		for c in continuous:
			df = df[np.abs(df[c] - df[c].mean()) <= (3 * df[c].std())]

		print "\nRemoved outliers: from ", old.shape, " to ", df.shape

	##############################
	##### Explore data
	##############################

	print_value_counts(df)
	
	print "\nFor each variable:"
	for c in continuous:
		# Describe the variable
		print_describe(df, c)

		if plot==True:
			sns.distplot(df[c])

			sns.jointplot(x=c, 
				y="serious_dlqin2yrs", data=df);

			sns.plt.show()


	##############################
	##### Generate features / predictors
	##############################

	# Discretize a continuous variable: IP or bucket
	bucket_debt = bucket(df, "debt_ratio", 10, "bucket_debt_ratio", "q")
	bucket_rev = bucket(df, "revolving_utilization_of_unsecured_lines", 5, 
		"bucket_rev", "q")

	# Create binary/dummy variables from a categorical variable 
	bucket_90 = bucket(df, "number_of_times90_days_late", 2, "bucket_90days", "b")
	df = dummy(df, "bucket_90days", "dummy_90days")
	return df


def main():
	df = pd.read_csv("../data/credit-data.csv")
	count = len(df.index)
	split = int(math.ceil(float(count) * 9.0 / 10.0))

	continuous = ["revolving_utilization_of_unsecured_lines", "age", 
		"number_of_dependents", "number_of_time30-59_days_past_due_not_worse", 
		"number_of_time60-89_days_past_due_not_worse",
		"number_of_times90_days_late", "debt_ratio", "monthly_income", 
		"number_of_open_credit_lines_and_loans", "number_real_estate_loans_or_lines"]

	train = run(df.iloc[:split,], False, True, continuous)
	test = run(df.iloc[split:,], False, False, continuous)

	features = ["bucket_debt_ratio", "dummy_90days_1", "bucket_rev"]

	probabilities = classifier(train, test, features)
	evaluate_classifier(test, probabilities)

	pl.show()


main()