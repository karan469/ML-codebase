# Statistical Concepts

## Hierarchical Clustering
When you use hierarchical clustering, be sure you define the partitioning method properly. This partitioning method is essentially how the distances between observations and clusters are calculated. I mostly use Ward's method or complete linkage, but other options might be the choice for you.

```python
from scipy.cluster import hierarchy
import scipy.spatial.distance as ssd

df = ...  # your dataframe with many features
corr = df.corr()  # we can consider this as affinity matrix
distances = 1 - corr.abs().values  # pairwise distnces

distArray = ssd.squareform(distances)  # scipy converts matrix to 1d array
hier = hierarchy.linkage(distArray, method="ward")  # you can use other methods

# You can print dendrogram with

dend = hierarchy.dendrogram(hier, truncate_mode="level", p=30, color_threshold=1.5)

# And finally, obtain cluster labels for your features

threshold = 1.5  # choose threshold using dendrogram or any other method (e.g. quantile or desired number of features)

cluster_labels = hierarchy.fcluster(hier, threshold, criterion="distance")
```

## Boxplots and Inter Quantile Range
Code to create boxplot for a particular column, for different labels
```python
sns.set(rc={'figure.figsize':(20,8)})
sns.boxplot(x='target', y=df.columns[i], data=df)
```
Code to create boxplot for all columns in dataframe
```python
fig, ax = plt.subplots(10,1,figsize=(20,20))
# Assuming last column is the target label
for i in range(len(df.columns)-1):
    sns.boxplot(x='target', y=df.columns[i], data=df, ax=ax[i])
```
![image](https://user-images.githubusercontent.com/33158202/128505131-0b5630aa-0c4d-4115-837a-475510df3e15.png)

Code snippet to remove samples with outliers in column 'AVG'
```python
Q1 = df['AVG'].quantile(0.25)
Q3 = df['AVG'].quantile(0.75)
IQR = Q3 - Q1    #IQR is interquartile range. 

filter = (df['AVG'] >= Q1 - 1.5 * IQR) & (df['AVG'] <= Q3 + 1.5 *IQR)
df.loc[filter]  
```
