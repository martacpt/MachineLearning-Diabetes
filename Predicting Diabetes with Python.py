#!/usr/bin/env python
# coding: utf-8

# # Predicting Diabetes with Python
# ## End-to-EndMachine Learning Model

# ### Part I : Data Collection and Cleaning

# - Collect diabetes data from Repository
# - Replace Yes/Positive with 1 and No/Negative with 0
# - Uniformize column names
# - Export clean csv file

# In[1]:


# import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# stats libraries
from scipy.stats import chi2_contingency
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.weightstats import ztest


# In[2]:


database_link = "https://archive.ics.uci.edu/ml/machine-learning-databases/00529/diabetes_data_upload.csv"


# In[3]:


df = pd.read_csv(database_link)


# In[4]:


df


# In[5]:


# replace the values in the df with 0's and 1's
df = df.replace("No", 0).replace("Yes",1).replace("Positive", 1).replace("Negative",0)
# gender column will be isMale

df = df.replace("Male",1).replace("Female",0)


# In[6]:


df


# In[7]:


# check for missing values
df.isnull().sum()


# In[8]:


#check dtypes of the columns
df.dtypes


# In[9]:


#correcting column names
replace = {"Gender":"ismale"}

df.rename(columns=replace)

df.columns.str.lower()


# In[10]:


df.columns = df.columns.str.lower()
df


# In[11]:


#turn into csv
df.to_csv('diabetes_data_clean.csv', index=None)


# ## Part II - Exploratory Data Analysis

# In[12]:


#examine ages
plt.hist(df['age']);


# In[13]:


print(df['age'].mean())
print(df['age'].median())


# In[14]:


#create countplot for ismale
sns.countplot(df['gender'])
plt.title('Is Male')
sns.despine()


# In[15]:


columns = df.columns[1:]

for column in columns:
    sns.countplot(df[column])
    plt.title(column)
    sns.despine()
    plt.show();


# #### Questions:
# 1. Is obesity related to diabetes status?
# 2. Is age related to diabetes status

# In[16]:


#Obesity
obesity_diabetes_crosstab = pd.crosstab(df['class'], df['obesity'])


# In[17]:


chi2_contingency(obesity_diabetes_crosstab)


# In[18]:


# Gender
ismale_diabetes_crosstab = pd.crosstab(df['class'], df['gender'])
ismale_diabetes_crosstab


# In[19]:


chi2_contingency(ismale_diabetes_crosstab)


# In[20]:


#polyuria
polyuria_diabetes_crosstab = pd.crosstab(df['class'], df['polyuria'])
chi2_contingency(polyuria_diabetes_crosstab)


# In[21]:


# is there a relationship between age and diabetic status?
sns.boxplot(df['class'], df['age'])


# In[22]:


no_diabetes = df[df['class']==0]
print(no_diabetes['age'].mean())

diabetes = df[df['class']==1]
print(diabetes['age'].mean())


# In[23]:


plt.hist(df['age'])


# In[24]:


qqplot(df['age'], fit=True, line='s')
plt.show()


# In[25]:


# conduct z test 
ztest(diabetes['age'],no_diabetes['age'])


# With a p valor of less than 0.05 we must reject the null hypothesis we must reject the hypothesis that there is no difference between ages of people with and without diabetes

# In[26]:


sns.heatmap(df.corr())


# ### Part III : Machine Learning Model training

# In[31]:


#importing specific libraries
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, classification_report


# In[32]:


#prepare independent and dependant variables

X = df.drop('class', axis=1)
y = df['class']


# In[43]:


#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, stratify=y)


# In[44]:


# begin model training
# start with DummyClassifier to establish Baseline

dummy = DummyClassifier()
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)


# In[45]:


# assess DummyClassifier Model
confusion_matrix(y_test, dummy_pred)


# In[46]:


# use classification report

print(classification_report(y_test,dummy_pred))


# In[48]:


# Start with Logistic Regression
logr = LogisticRegression(max_iter = 10000)
logr.fit(X_train, y_train)
logr_pred = logr.predict(X_test)


# In[49]:


confusion_matrix(y_test, logr_pred)


# In[50]:


print(classification_report(y_test,logr_pred))


# In[51]:


#  DecisionTree

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)


# In[52]:


confusion_matrix(y_test, tree_pred)


# In[53]:


print(classification_report(y_test,tree_pred))


# In[54]:


# RandomForest

forest = RandomForestClassifier()
forest.fit(X_train, y_train)
forest_pred = forest.predict(X_test)


# In[55]:


confusion_matrix(y_test, forest_pred)


# In[56]:


print(classification_report(y_test,forest_pred))


# In[58]:


#getting model feature importance
forest.feature_importances_


# In[59]:


X.columns


# In[62]:


pd.DataFrame({'feature': X.columns,
             'importance': forest.feature_importances_}).sort_values('importance', ascending = False)


# Summary:
# 
# 1. Trained baseline model
# 2. Trained different models - logistic regression, decision tree, random forest
# 3. Identified the important features in the best performing model (Random Forest)

# In[ ]:




