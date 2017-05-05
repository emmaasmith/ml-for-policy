import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Print a crosstab of the data
# data: the table of data
# col_x: string for the name of the column, to populate x values
# col_y: EITHER:
	# 1) string for the column name, to populate y values, e.g. "name"
	# 2) data to further partition by, e.g. total["name"]
# csv (optional): string for the name of the output file
def print_crosstab(data, col_x, col_y, csv=False):
	type_table = pd.crosstab(index=data[col_x], columns=col_y)
	if csv!=False:
		type_table.to_csv(csv)
	else:
		print type_table
	return type_table


# Build a plot for given data
# data: df of data
# title_lab: string for the title label
# x_lab: string for the x axis label
# y_lab: string for the y axis label
# start (optional): start, e.g. for time string for the start date pd.Timestamp('2009-01-01') 
# end (optional): end
def build_graph(data, title_lab, x_lab, y_lab, start=-1, end=-1):
	p = data.plot(
		title=title_lab,
		colormap='jet'
		)
	if start!=-1 and end !=-1:
		p.set(xlabel=x_lab, ylabel=y_lab,
			xlim=[start, end])
	else:
		p.set(xlabel=x_lab, ylabel=y_lab)
	plt.show()


# Print the description of one column (if col), or the dataset (if no col)
# data: the table of data
# col (optional): string for the name of the column, to populate x values 
def print_describe(data, col=0):
	if col == 0:
		describe = data.describe()
	else:
		describe = data[col].describe()
	print "\n", col, ": \n", describe


# Print the value counts of one column (if col), or the dataset (if no col)
# data: the table of data
# col (optional): string for the name of the column, to populate x values 
def print_value_counts(data, col=0):
	if col == 0:
		describe = [data[col].value_counts(dropna=False) for col in data.columns] 
	else:
		describe = data[col].value_counts(dropna=False)
	print "\nValue counts:\n", describe
