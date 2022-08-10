#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


# In[3]:


df=pd.read_csv('iris.csv')


# In[4]:


df


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


df['Species'].value_counts()


# In[10]:


df.isnull().sum()


# In[11]:


df=df.drop(columns=["Id"])


# In[12]:


df


# # EDA

# In[13]:


df['SepalLengthCm'].hist()


# In[14]:


colors=['red','blue','green']
species=['Iris-virginica','Iris-versicolor','Iris-setosa']


# In[15]:


for i in range(3):
    x=df[df['Species']==species[i]]
    plt.scatter(x['PetalLengthCm'],x['PetalWidthCm'],c=colors[i],label=species[i])
plt.xlabel('Sepal Length')
plt.ylabel("Sepal Width")
plt.legend()


# In[16]:


df['SepalWidthCm'].hist()


# In[17]:


df['PetalLengthCm'].hist()


# In[18]:


df['PetalWidthCm'].hist()


# # Corelation Matrix

# In[19]:


df.corr()


# In[20]:


co=df.corr()
fig,ax=plt.subplots(figsize=(10,5))
sns.heatmap(co,annot=True,ax=ax,cmap="coolwarm")


# # Converting labels into numeric form

# In[21]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Species']=le.fit_transform(df['Species'])


# In[22]:


df


# # Model Training

# In[23]:


from sklearn.model_selection import train_test_split
x=df.drop(columns=['Species'])
y=df['Species']


# In[24]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)


# In[25]:


x_train


# In[26]:


x_test


# In[27]:


y_train


# In[28]:


y_test


# In[29]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)


# In[30]:


print("Accuracy: ",model.score(x_test,y_test))


# In[31]:


from sklearn.tree import DecisionTreeClassifier
model2=DecisionTreeClassifier()
model2.fit(x_train,y_train)


# In[32]:


print("Accuracy is:",model2.score(x_test,y_test))


# In[33]:


from sklearn.neighbors import KNeighborsClassifier
model1=KNeighborsClassifier()
model1.fit(x_train,y_train)


# In[34]:


print("Accuracy:",model1.score(x_test,y_test))


# In[35]:


import pickle
with open('iris.pkl','wb') as f:
    pickle.dump(model,f)


# In[36]:


df['Species'].value_counts()


# In[37]:


load_model=pickle.load(open('iris.pkl','rb'))
load_model.predict([[6.0,2.2,4.0,1.0]])[0]


# In[38]:


filename="savedmodel.sav"
pickle.dump(model1,open(filename,'wb'))


# In[39]:


load_model1=pickle.load(open(filename,'rb'))
load_model1.predict([[6.0,2.2,4.0,1.0]])[0]


# In[ ]:




