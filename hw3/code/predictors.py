import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Discretize a continuous variable, return altered df
# data: df
# col: column to discretize
# n: number of bins
# name: name of the newly created column
# q:  "q"/default: take quantile, not even-sized buckets
	# "b": take even-sized buckets
def bucket(data, col, n, name, q="q"):
	if q=="b":
		x = pd.cut(data[col], bins=n, labels=False)
		print pd.value_counts(x)
		data[name] = x
	else:
		x = pd.qcut(data[col], q=n, retbins=True, labels=False)
		print pd.value_counts(x[0])
		data[name] = x[0]
	return x


# Take a categorical variable and create binary/dummy variables from it
# data: df
# col: column to discretize
# name: name of the newly created column
def dummy(data, col, name):
	dummies = pd.get_dummies(data[col], prefix=name)
	data = pd.concat([data, dummies], axis=1)
	return data
