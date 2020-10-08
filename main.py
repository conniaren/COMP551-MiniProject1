# US Hospitalization Dataset Version : Data update for 2020-10-07 16:24 UTC
# US Search Trend Dataset Version: Data update for 2020-10-07 16:24 UTC

import numpy as np

# Load data directly into an array (2D). This includes the header.
search_data = np.genfromtxt('2020_US_weekly_symptoms_dataset.csv', delimiter=',', dtype=None, encoding='utf-8')
hospitalization_data = np.genfromtxt('aggregated_cc_by.csv', delimiter=',', dtype=None, encoding='utf-8')

print(search_data[1, 8:608])
