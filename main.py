# US Hospitalization Dataset Version : Data update for 2020-10-07 16:24 UTC
# US Search Trend Dataset Version: Data update for 2020-10-07 16:24 UTC

import numpy as np

# Load data directly into an array (2D). This includes the header.
search_data = np.genfromtxt('2020_US_weekly_symptoms_dataset.csv', delimiter=',', dtype=None, encoding='utf-8')
hospitalization_data = np.genfromtxt('aggregated_cc_by.csv', delimiter=',', dtype=None, encoding='utf-8')

search_data_no_head = np.delete(search_data, 0, axis=0)
hospitalization_data_no_head = np.delete(hospitalization_data, 0, axis=0)

# Removing all symptoms with absolutely no search data, and subregion 2 columns
search_data = np.delete(search_data, np.all(search_data_no_head == '', axis=0), axis=1)

'''
Cleaning the hospitalization data. This was harder than just removing the empty columns in the search data.
Database contained entries from several different countries, and in USA, data was available for every state. 
Data was only needed for the 13 states present in the search database with valid data. Hence each states 
data was separately parsed from the large database and appended to a US specific hospitalization array.
'''

hospital_columns = [0, 1, 2, 16, 17]
american_states = ['AK', 'HI', 'ID', 'ME', 'MT', 'ND', 'NE', 'NH', 'NM', 'RI', 'SD', 'VT', 'WY']

hospitalization_data = np.take(hospitalization_data, hospital_columns, axis=1)

us_hospitalization_data = np.empty((1, 5))

for state in american_states:
    covid_region_code = f"US-{state}"
    state_data = np.delete(hospitalization_data, ~(np.any(hospitalization_data == covid_region_code, axis=1)), axis=0)
    us_hospitalization_data = np.append(us_hospitalization_data, state_data, axis=0)

us_hospitalization_data = np.delete(us_hospitalization_data, 0, axis=0)
print(us_hospitalization_data.shape)
