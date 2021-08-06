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
