#!/usr/bin/env python
# coding: utf-8

# In[5]:


import sys
import numpy
import sklearn
import pandas
import matplotlib
import seaborn
import scipy

print('Python: {}'.format(sys.version))
print('Numpy: {}'.format(numpy.__version__))
print('Sklearn: {}'.format(sklearn.__version__))
print('Pandas: {}'.format(pandas.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Seaborn: {}'.format(seaborn.__version__))
print('scipy: {}'.format(scipy.__version__))


# In[6]:


# Now we are importing the necessary Packages with a short name.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[20]:


# Now we load the dataset from the csv file  using the pandas library that we imported as pd
data=pd.read_csv('creditcard.csv')


# In[21]:


# let's see what's there in the dataset
print(data.columns)

# output: This gives you all the column attributes that are present in the dataset


# In[22]:


# Now let's see how many tuples are present and the number of attributes
print(data.shape)

# output: (284807, 31)


# In[23]:


# Below statement gives us some useful insights about the data like mean, standard deviation, count.
print(data.describe())

# In the ouput, as you can see all the count is the same for all the coloumns 
# which implies that we do not have any missing values.


# In[24]:


# Since this is a huge dataset, in order to save time and intense computations, I am going to select a fraction of the dataset at random
print(data.shape)
data=data.sample(frac=0.1,random_state=1)
print(data.shape) #after selecting a fraction(10%) at random


# In[26]:


# let's plot a histogram of each parameter
data.hist(figsize=(20,20))
plt.show()


# In[33]:


# Now, let's find out the number of valid and fraud cases in our updated dataset
valid=data[data['Class']==0]
fraud=data[data['Class']==1]

outlier_fraction= len(fraud)/float(len(valid))
print(outlier_fraction)
print('Valid Cases: {}'.format(len(valid)))
print('Fraud Cases: {}'.format(len(fraud)))


# In[28]:


# Now, let's find the correlation matrix to see if there is any strong positive or negative correction between the parameters
corrmat= data.corr()
fig= plt.figure(figsize=(12,9))
sns.heatmap(corrmat,vmax=.8, square=True)
plt.show()


# In[29]:


# let's create a list of all the columns present in the dataset
columns=data.columns.tolist()


# In[30]:


# Now, we will filter the columns to remove the data we don't want
columns=[c for c in columns if c not in ['Class']]

# we need to store the variable that we are predicting on which is class in this case
target= 'Class'

X=data[columns]
Y=data[target]


# In[32]:


# print the shapes of X and Y
print(X.shape)
print(Y.shape)
# notice that the X shape will have 30 columns instead of 31, because the other one is the Class parameter that we are predicting


# In[39]:


# now let's try to detect the outliers
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# define a random state
state=1

# define the outlier detection methods
classifiers= {
    'Isolation Forest': IsolationForest(max_samples=len(X),
                                        contamination=outlier_fraction,
                                       random_state=state,behaviour="new"),
    'Local Outlier Factor': LocalOutlierFactor(n_neighbors=20,
                                              contamination=outlier_fraction)
}


# In[51]:


# let's fit the model
n_outliers= len(fraud)

for i, (clf_name,clf) in enumerate(classifiers.items()):
    
    #fit the data and tag outliers
    if (clf_name =='Local Outlier Factory'):
        y_pred= clf.fit_predict(X)
        scores_pred=clf.negative_outlier_factor_
    
    else:
        clf.fit(X)
        scores_pred=clf.fit_predict(X)
        y_pred=clf.fit_predict(X)
    # the results of y prediction values will give us -1 for an outlier and 1 for a inlier
    
    # Reshape these prediction values to 0 for valid and 1 for fraud
    y_pred[y_pred == 1]=0
    y_pred[y_pred == -1]=1
    
    n_errors=(y_pred !=Y).sum()
    
    # Run the classification metrics
    print('{}: {}'.format(clf_name,n_errors))
    print(accuracy_score(Y,y_pred))
    print(classification_report(Y,y_pred))
    


# In[ ]:




