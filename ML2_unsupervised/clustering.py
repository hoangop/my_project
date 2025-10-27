import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
import time

# Read data. Do not change the variable names (data, label)
data = pd.read_csv('ML2_unsupervised/data/data.csv')
label = pd.read_csv('ML2_unsupervised/data/labels.csv')
data=data.drop('Unnamed: 0',axis=1)
label=label.drop('Unnamed: 0',axis=1)
print(data.info())
print(label.info())

# 1. K-Means Clustering
# kmeans = KMeans(n_clusters=10, random_state=42)
# kmeans.fit(data)
# predictions = kmeans.predict(data)
# print(predictions)
