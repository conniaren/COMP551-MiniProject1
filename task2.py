import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

plt.rcParams.update({'figure.max_open_warning': 0})

final_covid_dataframe = pd.read_csv("us_covid_dataset_final.csv")
colors = ['maroon','orange','gold','y','olivedrab','springgreen','turquoise','deepskyblue','darkblue','mediumpurple','purple','hotpink','lightpink']

### PART 1 ###
for col in final_covid_dataframe.columns:
        if 'symptom:' in col:
                df = final_covid_dataframe[["sub_region_1_code", "date", col]]
                pivot_df = df.pivot(index='date', columns='sub_region_1_code', values=col).replace(np.nan, 0)
                pivot_df_percent = pivot_df.apply(lambda x: x if x.name == 'date' else x*100/sum(x), axis=1)
		
                pivot_df_percent.plot.bar(stacked=True, color=colors)
                plt.ylim(0, 100)
                plt.ylabel("Searches (%)")
                plt.title('Searches of {}'.format(col))
                plt.legend(loc="center right", bbox_to_anchor=(1.13, 0.5), ncol=1)

### PART 2 ###
features = ['symptom:Allergic conjunctivitis','symptom:Angular cheilitis','symptom:Aphonia','symptom:Auditory hallucination','symptom:Burning Chest Pain','symptom:Clouding of consciousness','symptom:Crackles','symptom:Crepitus','symptom:Depersonalization','symptom:Dysautonomia','symptom:Epiphora','symptom:Hemolysis','symptom:Laryngitis','symptom:Myoclonus','symptom:Nasal polyp','symptom:Polydipsia','symptom:Pulmonary edema','symptom:Rectal pain','symptom:Rumination','symptom:Shallow breathing','symptom:Stridor','symptom:Urinary urgency','symptom:Ventricular fibrillation','symptom:Viral pneumonia']

x = final_covid_dataframe.loc[:, features].values
y = final_covid_dataframe.loc[:, ['sub_region_1_code']].values
# x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, final_covid_dataframe[['sub_region_1_code']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('Application of 2-component PCA on Search Trends', fontsize = 20)

targets = ['US-AK','US-HI','US-ID','US-ME','US-MT','US-ND','US-NE','US-NH','US-NM','US-RI','US-SD','US-VT','US-WY']

for target, color in zip(targets,colors):
	indicesToKeep = finalDf['sub_region_1_code'] == target
	ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'], finalDf.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)
ax.legend(targets)
ax.grid()
plt.show()

