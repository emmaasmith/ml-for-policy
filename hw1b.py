import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import requests 


###########################
# Data import
###########################
# "Vacant and Abandoned Buildings Reported" and "Sanitation Code Complaints" 
# for the past 3 months


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


###########################
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


###########################
# Reduce and merge

recent_building = building[building['request_date'] > '2017-1-4']
recent_sanitation = sanitation[sanitation['request_date'] > '2017-1-4']

total = recent_building.append(recent_sanitation, ignore_index=True)



###########################
# API helpers
###########################


###########################
# FCC API: Lat, lon -> 4-digit block code

def find_block(lat, lon):
    url = 'http://data.fcc.gov/api/block/find?'
    payload = {
        'format': 'json',
        'latitude': lat,
        'longitude': lon,
        'showall': 'true'
    }

    request = requests.get(url, params=payload).json()
    block = request['Block']['FIPS']
    return block[-4:]

def apply_block(x):
    if pd.isnull(x['lat']) or pd.isnull(x['lon']):
        return
    return find_block(x['lat'], x['lon'])


# Census API: All data by block

def get_census():
    url = ('http://api.census.gov/data/2015/acs5?'
        'get=NAME,B02001_001E,B02001_002E,B02001_003E,B02001_005E,B03002_001E,B03002_002E,B03002_012E,'
        'B19013_001E,B99162_001E,B99162_002E,B99162_003E,B11001_001E,B11001_002E,B11001_007E'
        '&for=block+group:*&in=state:17+county:031&key=0a6e7fe8ad121c07c804431e0f001be2647b4511')

    request  = requests.get(url).json()
    request_df = pd.DataFrame.from_records(request[1:])
    request_df.columns = ['name', 'race_tot', 'race_white', 'race_black', 'race_asian',
        'hispanic_tot', 'hispanic_not', 'hispanic_yes', 'median_yearly_income',
        'speak_tot', 'speak_only_eng', 'speak_other', 
        'household_tot', 'household_fam', 'household_nonfam',
        'state', 'county', 'tract', 'block']
    return request_df



###########################
# Call the helpers
###########################


###########################
# If running this function and writing to a file:
# blocks = total.apply(lambda x: apply_block(x), axis=1)
# blocks.columns = 'block'
# blocks.to_csv('blocks_1a.csv')
# blocks1b = total['service_request_number']
# blocks1b.to_csv('blocks_1b.csv')


###########################
# If not running this function; instead reading from a file:
blocks1a = pd.read_csv('blocks_1a.csv', header=None, dtype=object)
blocks1b = pd.read_csv('blocks_1b.csv', header=None)
blocks = pd.concat([blocks1a[1], blocks1b[1]], axis=1)
blocks.columns = ['block', 'service_request_number']


###########################
# Census

census = get_census()

def truncate_tract_block(x):
    y = str(x)
    y = y[-10:]
    y = y[:7]
    return y

blocks['block'] = blocks['block'].apply(truncate_tract_block)

def append_tract_block(x):
    try:
        str_floats = "%d%d" % (x['tract'], x['block'])
        return str_floats
    except:
        return

census["block"] = census["tract"].map(str) + census["block"]

merged1 = pd.merge(blocks, census, on='block')
merged = pd.merge(merged1, total, on='service_request_number')



###########################
# Analysis
###########################

# Merge

merged['race_tot'] = pd.to_numeric(merged['race_tot'])
merged['race_white'] = pd.to_numeric(merged['race_white'])
merged['race_black'] = pd.to_numeric(merged['race_black'])
merged['race_asian'] = pd.to_numeric(merged['race_asian'])

merged['hispanic_yes'] = pd.to_numeric(merged['hispanic_yes'])
merged['hispanic_tot'] = pd.to_numeric(merged['hispanic_tot'])

merged['median_yearly_income'] = pd.to_numeric(merged['median_yearly_income'])

merged['speak_tot'] = pd.to_numeric(merged['speak_tot'])
merged['speak_only_eng'] = pd.to_numeric(merged['speak_only_eng'])

merged['household_tot'] = pd.to_numeric(merged['household_tot'])
merged['household_fam'] = pd.to_numeric(merged['household_fam'])


# Print: vacant and abandoned buildings

merged['hisp_pct'] = merged['hispanic_yes'] / merged['hispanic_tot']
print merged[merged['service_request_type'] == 'Vacant/Abandoned Building']['hisp_pct'].describe()

merged['white_pct'] = merged['race_white'] / merged['race_tot']
print merged[merged['service_request_type'] == 'Vacant/Abandoned Building']['white_pct'].describe()

merged['black_pct'] = merged['race_black'] / merged['race_tot']
print merged[merged['service_request_type'] == 'Vacant/Abandoned Building']['black_pct'].describe()

merged['asian_pct'] = merged['race_asian'] / merged['race_tot']
print merged[merged['service_request_type'] == 'Vacant/Abandoned Building']['asian_pct'].describe()

print merged[merged['service_request_type'] == 'Vacant/Abandoned Building']['median_yearly_income'].describe()

merged['only_eng_pct'] = merged['speak_only_eng'] / merged['speak_tot']
print merged[merged['service_request_type'] == 'Vacant/Abandoned Building']['only_eng_pct'].describe()

merged['fam_pct'] = merged['household_fam'] / merged['household_tot']
print merged[merged['service_request_type'] == 'Vacant/Abandoned Building']['fam_pct'].describe()


# Print: sanitation

merged['hisp_pct'] = merged['hispanic_yes'] / merged['hispanic_tot']
print merged[merged['service_request_type'] == 'Sanitation Code Violation']['hisp_pct'].describe()

merged['white_pct'] = merged['race_white'] / merged['race_tot']
print merged[merged['service_request_type'] == 'Sanitation Code Violation']['white_pct'].describe()

merged['black_pct'] = merged['race_black'] / merged['race_tot']
print merged[merged['service_request_type'] == 'Sanitation Code Violation']['black_pct'].describe()

merged['asian_pct'] = merged['race_asian'] / merged['race_tot']
print merged[merged['service_request_type'] == 'Sanitation Code Violation']['asian_pct'].describe()

print merged[merged['service_request_type'] == 'Sanitation Code Violation']['median_yearly_income'].describe()

merged['only_eng_pct'] = merged['speak_only_eng'] / merged['speak_tot']
print merged[merged['service_request_type'] == 'Sanitation Code Violation']['only_eng_pct'].describe()

merged['fam_pct'] = merged['household_fam'] / merged['household_tot']
print merged[merged['service_request_type'] == 'Sanitation Code Violation']['fam_pct'].describe()



###########################
# Code to run for question 3A
###########################

test = merged[merged['service_request_type'] == 'Vacant/Abandoned Building']
print test[test['block']=='7104005'].describe()
test2 = merged[merged['service_request_type'] == 'Sanitation Code Violation']
print test2[test2['block']=='7104005'].describe()
