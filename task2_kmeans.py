import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

final_covid_dataframe = pd.read_csv("us_covid_dataset_final.csv")

features = ['symptom:Allergic conjunctivitis','symptom:Angular cheilitis','symptom:Aphonia','symptom:Auditory hallucination','symptom:Burning Chest Pain','symptom:Clouding of consciousness','symptom:Crackles','symptom:Crepitus','symptom:Depersonalization','symptom:Dysautonomia','symptom:Epiphora','symptom:Hemolysis','symptom:Laryngitis','symptom:Myoclonus','symptom:Nasal polyp','symptom:Polydipsia','symptom:Pulmonary edema','symptom:Rectal pain','symptom:Rumination','symptom:Shallow breathing','symptom:Stridor','symptom:Urinary urgency','symptom:Ventricular fibrillation','symptom:Viral pneumonia']

x = final_covid_dataframe.loc[:, features].values
y = final_covid_dataframe.loc[:, ['sub_region_1_code']].values

kmeans = KMeans(n_clusters=3)
y_kmeans = kmeans.fit_predict(x)

f1 = plt.figure(1)
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s=50, c='lightgreen', marker='s', edgecolor='black',label='cluster 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s=50, c='orange', marker='o', edgecolor='black',label='cluster 2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s=50, c='lightblue', marker='v', edgecolor='black',label='cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=250, marker='*', c='red', edgecolor='black', label='centroids')

plt.title('Application of K-Means clustering to raw data')
plt.legend(scatterpoints=1)
plt.grid()
f1.show()

x = StandardScaler().fit_transform(x)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)

kmeans2 = KMeans(n_clusters=3)
y_kmeans2 = kmeans2.fit_predict(principalComponents)

f2 = plt.figure(2)
plt.scatter(principalComponents[y_kmeans2 == 0, 0], principalComponents[y_kmeans2 == 0, 1], s=50, c='lightgreen', marker='s', edgecolor='black',label='cluster 1')
plt.scatter(principalComponents[y_kmeans2 == 1, 0], principalComponents[y_kmeans2 == 1, 1], s=50, c='orange', marker='o', edgecolor='black',label='cluster 2')
plt.scatter(principalComponents[y_kmeans2 == 2, 0], principalComponents[y_kmeans2 == 2, 1], s=50, c='lightblue', marker='v', edgecolor='black',label='cluster 3')
plt.scatter(kmeans2.cluster_centers_[:, 0], kmeans2.cluster_centers_[:, 1], s=250, marker='*', c='red', edgecolor='black', label='centroids')

plt.title('Application of K-Means clustering to PCA-reduced data')
plt.legend(scatterpoints=1)
plt.grid()
f2.show()

input()
