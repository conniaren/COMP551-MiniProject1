# US Hospitalization Dataset Version : Data update for 2020-10-07 16:24 UTC
# US Search Trend Dataset Version: Data update for 2020-10-07 16:24 UTC

import numpy as np
import pandas as pd

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

# Remove states with no hospitalization data
search_data = np.delete(search_data, np.any(search_data == "US-WV", axis=1), axis=0)
search_data = np.delete(search_data, np.any(search_data == "US-DC", axis=1), axis=0)
search_data = np.delete(search_data, np.any(search_data == "US-DE", axis=1), axis=0)
# Remove dates with no hospitalization data
remove_dates = ['01-06', '01-13', '01-20', '01-27', '02-03', '02-10', '02-17', '02-24']
for date in remove_dates:
    full_date = f"2020-{date}"
    search_data = np.delete(search_data, np.any(search_data == full_date, axis=1), axis=0)

# Merge daily data into weekly data based on the prior Monday of the week using Dataframes
us_hospitalization_data = pd.DataFrame(us_hospitalization_data, columns=['open_covid_region_code', 'sub_region_1',
                                                                         'date', 'hospitalized_new',
                                                                         'hospitalized_cumulative'])

us_hospitalization_data['date'] = us_hospitalization_data['date'].astype('datetime64[ns]')
us_hospitalization_data['hospitalized_new'] = us_hospitalization_data['hospitalized_new'].astype('float')
us_hospitalization_data['hospitalized_cumulative'] = us_hospitalization_data['hospitalized_cumulative'].astype('float')

weekly_hospitalization_data = us_hospitalization_data.groupby(by=["open_covid_region_code", 'sub_region_1']).resample(
    "W-MON",
    label='left',
    closed='left',
    on='date').sum().reset_index()

# Delete weeks not present in the search data
weekly_hospitalization_data = weekly_hospitalization_data.drop(weekly_hospitalization_data[(weekly_hospitalization_data.
                                                                                            date == '2020-09-28') |
                                                                                           (weekly_hospitalization_data.
                                                                                            date == '2020-10-05')].index)

# Note: FIx rhode island has an extra week from starting its data entry march 1st
print(weekly_hospitalization_data)
weekly_hospitalization_data.to_csv('clean-hospital-data.csv')
print(np.append(search_data, np.array(weekly_hospitalization_data[['hospitalized_new', 'hospitalized_cumulative']]), axis=1))
