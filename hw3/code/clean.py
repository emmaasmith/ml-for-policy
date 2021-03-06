import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re


# Select only certain columns from data
# data: the table to return
# selected_cols: array of strings, the column names we want to select
    # in order of the data
# new_col_names (optional): array of strings, the new names to give to the
    # selected columns, in order 
def select_cols (data, selected_cols, new_col_names=[]):
    data = data.loc[:, selected_cols]
    if new_col_names != []:
        data.columns = new_col_names
    return data


# Turn a column into date data
# data: the table of data
# col_name: string for the name of the column we want to date-ify
# date_format: e.g. '%m/%d/%Y'
def col_to_date (data, col_name, date_format):
    data[col_name] = pd.to_datetime(data[col_name], 
        format=date_format, errors='ignore')
    return data


# Turn a column into numeric data
# data: the table of data
# col_name: string for the name of the column we want to make numeric
def col_to_num (data, col_name):
    data[col_name] = pd.to_numeric(data[col_name])
    return data


# Find and replace fields in a column
# data: the table of data
# col_name: string for the name of the column we want to change
# old: string of old field
# new: string of new field
def find_replace (data, col_name, old, new):
    data[col_name].replace(to_replace=old, value=new, inplace=True)


# View the missing values in the daata
# data: the table of data
# col_name: string for the name of the column we want to change
# x: value (untyped) to replace missing values with
def see_missing_vals_long (data):
    nulls = data.value.isnull()
    print "\n##########################\nCrosstab of missing values"
    print pd.crosstab(data.variable, nulls)
    return nulls


# Fill in missing values in a column
# data: the table of data
# col_name: string for the name of the column we want to change
# x: either:
    # 1. the value (untyped) to replace missing values with
    # 2. "mean" to fill in the mean
def fill_missing_vals (data, col_name, x):
    for c in col_name:
        if x=="mean":
            x_mean = data[c].mean()
            data[c].fillna(x_mean, inplace=True)
        else:
            data[c].fillna(x, inplace=True)
    return data


# Turn a camelcase string into a snakecase string
# column_name: string t modify
# From https://gist.github.com/glamp/6529725/raw/e38ffd2fc4cb840be21098486ffe5df991946736/camel_to_snake.py
def camel_to_snake(column_name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', column_name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
