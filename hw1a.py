import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

###########################
# Graffiti data: import
###########################

graffiti_raw = pd.read_csv('311_Service_Requests_-_Graffiti_Removal.csv')

graffiti = graffiti_raw.loc[:, ['Service Request Number', 'Type of Service Request',
	'ZIP Code', 'Community Area', 'Latitude', 'Longitude',
	'Creation Date', 'Completion Date', 'Status'
	]]
graffiti.columns = ['service_request_number', 'service_request_type',
	'zip', 'community_area', 'lat', 'lon',
	'request_date', 'completion_date', 'status'
	]

graffiti['request_date'] = pd.to_datetime(graffiti['request_date'], 
	format='%m/%d/%Y', errors='ignore')
graffiti['completion_date'] = pd.to_datetime(graffiti['completion_date'], 
	format='%m/%d/%Y', errors='ignore')

print "GRAFFITI DTYPES"
print graffiti.dtypes
print graffiti.shape


##########################
# Vacant and abandoned building data: import

building_raw = pd.read_csv('311_Service_Requests_-_Vacant_and_Abandoned_Buildings_Reported.csv',
	low_memory=False)

building = building_raw.loc[1:, ['SERVICE REQUEST NUMBER', 'SERVICE REQUEST TYPE',
	'ZIP CODE', 'Community Area', 'LATITUDE', 'LONGITUDE',
	'DATE SERVICE REQUEST WAS RECEIVED', 'IS THE BUILDING CURRENTLY VACANT OR OCCUPIED?' 
	]]
building.columns = ['service_request_number', 'service_request_type',
	'zip', 'community_area', 'lat', 'lon',
	'request_date', 'vacant_or_occupied'
	]

building['request_date'] = pd.to_datetime(building['request_date'], 
	format='%m/%d/%Y', errors='ignore')

print "BUILDING DTYPES"
print building.dtypes
print building.shape


##########################
# Pothole data: import

pothole_raw = pd.read_csv('311_Service_Requests_-_Pot_Holes_Reported.csv')

pothole = pothole_raw.loc[:, ['SERVICE REQUEST NUMBER', 'TYPE OF SERVICE REQUEST',
	'ZIP', 'Community Area', 'LATITUDE', 'LONGITUDE',
	'CREATION DATE', 'COMPLETION DATE', 'STATUS', 'NUMBER OF POTHOLES FILLED ON BLOCK'
	]]
pothole.columns = ['service_request_number', 'service_request_type',
	'zip', 'community_area', 'lat', 'lon',
	'request_date', 'completion_date', 'status', 'pothole_num',
	]

pothole['request_date'] = pd.to_datetime(pothole['request_date'], 
	format='%m/%d/%Y', errors='ignore')
pothole['completion_date'] = pd.to_datetime(pothole['completion_date'], 
	format='%m/%d/%Y', errors='ignore')

pothole['service_request_type'].replace(to_replace='Pot Hole in Street', 
	value='Pothole in Street', inplace=True)

print "POTHOLE DTYPES"
print pothole.dtypes
print pothole.shape


###########################
# Sanitation code complaint data: import

sanitation_raw = pd.read_csv('311_Service_Requests_-_Sanitation_Code_Complaints.csv')
sanitation = sanitation_raw.loc[1:, ['Service Request Number', 'Type of Service Request',
	'ZIP Code', 'Community Area', 'Latitude', 'Longitude',
	'Creation Date', 'Completion Date', 'Status', 'What is the Nature of this Code Violation?'
	]]

sanitation.columns = ['service_request_number', 'service_request_type',
	'zip', 'community_area', 'lat', 'lon',
	'request_date', 'completion_date', 'status', 'sanitation_description',
	]

sanitation['request_date'] = pd.to_datetime(sanitation['request_date'], 
	format='%m/%d/%Y', errors='ignore')
sanitation['completion_date'] = pd.to_datetime(sanitation['completion_date'], 
	format='%m/%d/%Y', errors='ignore')

print "SANITATION DTYPES"
print sanitation.dtypes
print sanitation.shape


###########################
# Append with NaN/null for irrelevant types

total = graffiti.append(building, ignore_index=True)
total = total.append(pothole, ignore_index=True)
total = total.append(sanitation, ignore_index=True)

print "TOTAL DTYPES"
print total.dtypes
print total.shape



###########################
# Code to run for question 3B
###########################
# How much more/less likely that it came from Lawndale vs. Uptown?
# Lawndale: 60608, 60623, 60624.
# Uptown: 60613, 60640

def isclosezip(x):
	if pd.isnull(x):
		return 0
	if np.isclose([60608.0, 60623.0, 60624.0], x, atol=0.01).any():
		return 1
	if np.isclose([60613.0, 60640.0], x, atol=0.01).any():
		return 2
	return 0

lim_zips = graffiti['zip'].apply(lambda x: isclosezip(x))
print lim_zips.value_counts()



###########################
# Summary stats
###########################

print "Details per request"
print total[total['service_request_type']=='Vacant/Abandoned Building']['vacant_or_occupied'].value_counts()
print total[total['service_request_type']=='Pothole in Street']['pothole_num'].describe()
print total[total['service_request_type']=='Sanitation Code Violation']['sanitation_description'].value_counts()

# Number of requests of each type (and subtype within each of the types above) 
	# over time, 
	# by neighborhood, 
	# response time by the city.

###########################
# Total number of requests by type
print "\n\n1. Number of requests of each type over time"

print "\nA. Total"

type_table = pd.crosstab(index=total["service_request_type"], 
	columns="count")
type_table.to_csv('1_type_table.csv')


print "\nB. Over time of request, total"

count_by_date = pd.crosstab(index=total["request_date"], 
	columns="count")
# print count_by_date

p1 = count_by_date.plot(
	title='Over time of request, total',
	colormap='jet'
	)
p1.set(xlabel="Request date", ylabel="Frequency",
	xlim=[pd.Timestamp('2009-01-01'), pd.Timestamp('2017-05-01')])


print "\nC. Over time of request, by type"

count_by_date_type = pd.crosstab(index=total["request_date"], 
	columns=total["service_request_type"])
# print count_by_date_type

p2 = count_by_date_type.plot(
	title='Over time of request, by type',
	colormap='jet'
	)
p2.set(xlabel="Request date", ylabel="Frequency",
	xlim=[pd.Timestamp('2009-01-01'), pd.Timestamp('2017-05-01')])


###########################
print "\n\n2. Number of requests of each type by neighborhood"

# Total number of requests by community area
print "\nA. Community areas"

type_area_table = pd.crosstab(index=total["community_area"], 
	columns=total["service_request_type"])
type_area_table.to_csv('1_type_area_table.csv')


# Total number of requests by zip code
print "\nB. Zip codes"

type_zip_table = pd.crosstab(index=total["zip"], 
	columns=total["service_request_type"])
type_zip_table.to_csv('1_type_zip_table.csv')


###########################
print "\n\n3. Number of requests of each type by response time"

# Total number of requests by response time
print "\nA. Response time, total"

response_time = total["completion_date"] - total["request_date"]
response_time = response_time.astype('timedelta64[D]')

count_by_response_time = pd.crosstab(index=response_time, 
	columns="count")

p3 = count_by_response_time.plot(
	title='# requests by community area, total (month-long)',
	colormap='jet'
	)
p3.set(xlabel="Request date", ylabel="Frequency",
	xlim=[0.0, 30.0])


# Total number of requests by response time and by type
print "\nB. Response time, by type"

count_by_response_time_type = pd.crosstab(index=response_time, 
	columns=total["service_request_type"])

p4 = count_by_response_time_type.plot(
	title='# requests by community area, by type (2-week)',
	colormap='jet'
	)
p4.set(xlabel="Request date", ylabel="Frequency",
	xlim=[0.0, 14.0])

plt.show()
