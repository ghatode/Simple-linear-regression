#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
dataset =pd.read_csv('C:/Users/lenovo/Desktop/KML/Simple_Linear_Regression/Salary_Data.csv')
dataset


# In[21]:


X= dataset.iloc[:,:-1].values
X
y= dataset.iloc[:,1].values
y


# In[22]:


# spliting dataset into train and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[23]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[24]:


print(X_train)
print(y_train)
print(X_test)
print(y_test)


# In[30]:


from sklearn.linear_model import LinearRegression
# instantiate
Regressor = LinearRegression()

# fit the model to the training data (learn the coefficients)
Regressor.fit(X_train, y_train)


# In[32]:


y_pred=Regressor.predict(X_test)
print(y_pred)


# In[35]:


#visulisation training set
plt.scatter(X_train,y_train)


# In[39]:


plt.plot(X_train,Regressor.predict(X_train))


# In[47]:


import matplotlib.pyplot as plt
plt.scatter(X_train,y_train)
plt.scatter(X_train,y_train)
plt.plot(X_train,Regressor.predict(X_train))
plt.title("salary vs Experience(Training set)")
plt.xlabel("Year of Experience")
plt.ylabel("Salary")
plt.show()


# In[ ]:




