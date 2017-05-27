
# coding: utf-8

# In[1]:

#we will study the multi class classification of blues stpd roses ...yes blues roses


# In[2]:

import numpy as np
import pandas as pd
from pandas import Series,DataFrame

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

get_ipython().magic('matplotlib inline')


# In[3]:

from sklearn import linear_model
from sklearn.datasets import load_iris


# In[4]:

iris = load_iris()


# In[5]:

X = iris.data

Y = iris.target


# In[6]:

Y


# In[7]:

X


# In[8]:

print(iris.DESCR)


# In[9]:

iris_data = DataFrame(X,columns=['Sepal Length','Sepal Width','Petal Length','Petal Width'])
iris_target = DataFrame(Y,columns=['Species'])


# In[10]:

iris_data


# In[11]:

iris_target


# In[12]:

def flower(num):
    if num == 0:
        return 'Setosa'
    elif num == 1:
        return 'Versicolour'
    else:
        return 'Virginica'


# In[13]:

iris_target['Species'] =  iris_target['Species'].apply(flower)


# In[14]:

iris_target.head()


# In[15]:

iris_target.tail()


# In[16]:

iris = pd.concat([iris_data,iris_target],axis=1)


# In[17]:

iris.head()


# In[18]:

sns.pairplot(iris,hue='Species',size=2)


# In[22]:

sns.factorplot('Petal Length',data=iris,kind='count',hue='Species',size=10)


# In[23]:

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split


# In[24]:

logreg = LogisticRegression()

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.4,random_state=3)


# In[25]:

#Multi-Class Classification with Sci Kit Learn
#We already have X and Y defined as the Data Features and Target 
#We will split the data into Testing and Training sets.
# test_size argument to have the testing data be 40% of the total data set.
#I'll also pass a random seed number.


# In[26]:

# Create a Logistic Regression Class object
logreg = LogisticRegression()

# Split the data into Trainging and Testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4,random_state=3)

# Train the model with the training set
logreg.fit(X_train, Y_train)


# In[28]:

# Import testing metrics from SciKit Learn
from sklearn import metrics

# Prediction from X_test
Y_pred = logreg.predict(X_test)

#Check accuracy
print(metrics.accuracy_score(Y_test,Y_pred))


# In[29]:

#K-Nearest Neighbors


# In[30]:

from sklearn.neighbors import KNeighborsClassifier


# In[31]:

# We'll first start with k=6
knn = KNeighborsClassifier(n_neighbors = 6)

knn.fit(X_train,Y_train)


# In[32]:

Y_pred = knn.predict(X_test)

print (metrics.accuracy_score(Y_test,Y_pred))


# In[33]:

# K=1


# In[34]:

knn = KNeighborsClassifier(n_neighbors = 1)

# Fit the data
knn.fit(X_train,Y_train)

# Run a prediction
Y_pred = knn.predict(X_test)

# Check Accuracy against the Testing Set
print(metrics.accuracy_score(Y_test,Y_pred))


# In[35]:

# Test k values 1 through 20
k_range = range(1, 21)

# Set an empty list
accuracy = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    accuracy.append(metrics.accuracy_score(Y_test, Y_pred))


# In[36]:

plt.plot(k_range, accuracy)
plt.xlabel('K value for for kNN')
plt.ylabel('Testing Accuracy')


# In[37]:

#the previous plot show how the accrucy change with k (number of neighboors)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



