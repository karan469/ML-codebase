# EDA (Explainatory Detailed Analysis)

### Lower Left Triangle correlation matrix
```
corr = df.corr()
plt.figure(figsize=(20,10))
mask = np.zeros_like(corr,dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr,mask=mask,annot=True)
plt.show()
```
![image](https://user-images.githubusercontent.com/33158202/118260197-9d95c400-b4cf-11eb-9cc9-608be998e462.png)

### Histogram for continuos variable
```
plt.figure(figsize=(12, 5))
plt.hist(train['windmill_generated_power(kW/h)'].values, bins=200)
plt.title('Histogram cat1 counts in train')
plt.xlabel('windmill_generated_power(kW/h)')
plt.ylabel('Count')
plt.show()
```
![image](https://user-images.githubusercontent.com/33158202/118260349-d59d0700-b4cf-11eb-8447-e95fa8444563.png)

### Box Plot for a single attribute
```
plt.figure(figsize=(10,5))
plt.xlim(0,20)
plt.ylabel('windmill_generated_power(kW/h)')
sns.boxplot(x=train['windmill_generated_power(kW/h)'])
plt.show()
```
![image](https://user-images.githubusercontent.com/33158202/118260417-ea799a80-b4cf-11eb-9756-56811e099429.png)

### Profile Report for Tabular Dataframe
```
from pandas_profiling import ProfileReport
profile = ProfileReport(train)
profile
```

### Density plots for different categorical values
```
temp = df_train.pivot(columns='turbine_status',
                     values='windmill_generated_power(kW/h)')
temp.plot.density()
```
Here, cloud_level is the categorical column and windmill_generated_power(kW/h) is the target column.

![image](https://user-images.githubusercontent.com/33158202/118375445-5565c700-b5df-11eb-9123-2508128bcbc6.png)

