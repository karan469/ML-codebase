# Statistical Concepts

## 1. Linear Regression
There are many different methods that we can apply to our linear regression model in order to make it more efficient: <br>
- Gradient Descent - O(n^2)
- Least Square Methods / Normal Equation methods - O(n^3)
- SVD - O(n^3)
- Adam's Method

### 1.1 Ordinal Least Square Optimization
- Most basic linear least square solving algorithm for estimating unknown parameters.
- Minimizing the sum of the squares of the differences between the observed dependent variable in the given dataset and those predicted by the linear function of the independent variable.
- Under the additional assumption that the errors are normally distributed, OLS is the maximum likelihood estimator.
![image](https://user-images.githubusercontent.com/33158202/144732739-0bf8727b-d6cb-4495-a95c-4b993ae4aa24.png)


## 2. Regularization Techniques

Used to avoid overfitting by reducing the bias in the model parameters.

https://www.analyticsvidhya.com/blog/2021/05/complete-guide-to-regularization-techniques-in-machine-learning/

1. Lasso Regression - Does create a sparse model.
2. Ridge Regression - Helps to avoid the problem of multicollinearity in datasets. Doesn't create a sparse model.

## 3. Normalization v/s Standardization
1. Normalization is a rescaling of the data from the original range so that all values are within the new range of 0 and 1. <br>
Normalization requires that you know or are able to accurately estimate the minimum and maximum observable values. You may be able to estimate these values from your available data.
```
y = (x-min)/(max-min)
```
2. Standardizing a dataset involves rescaling the distribution of values so that the mean of observed values is 0 and the standard deviation is 1.
This can be thought of as subtracting the mean value or centering the data. <br>
Standardization assumes that your observations fit a Gaussian distribution (bell curve) with a well-behaved mean and standard deviation. You can still standardize your data if this expectation is not met, but you may not get reliable results.
```
y = (x-mean)/std
```

## 4. Hierarchical Clustering
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

## 5. Boxplots and Inter Quantile Range
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

## 6. K Means
To find number of clusters, we can use heuristics such as Elbow Method.
![image](https://user-images.githubusercontent.com/33158202/146512105-0408611b-b48c-46b6-820c-9ae0f0386ca5.png)

