#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import torch
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
import pydotplus

print(torch.cuda.is_available())  
print(torch.cuda.device_count())  
print(torch.cuda.get_device_name(0)) 


# In[2]:


# for a two-class tree, call this function like this:
# writegraphtofile(clf, ('F', 'T'), dirname+graphfilename)
# for a multi-class tree, call this function like this:
# @ writegraphtofile(clf, featurenames, dirname+graphfilename)
def writegraphtofile(clf, featurelabels, filename):
 dot_data = sklearn.tree.export_graphviz(clf, feature_names=featurelabels, out_file=None)
 graph=pydotplus.graph_from_dot_data(dot_data)
 graph.write_png(filename)


# In[4]:


pd.set_option('display.max_columns', None)


# ## Load Dataset 

# In[5]:


file_path = r"/home/coldnoodle/dev/ece-5464/project_2/diabetic_data.csv"


# In[6]:


df = pd.read_csv(file_path)
df.head()


# In[7]:


df.shape


# ## Check for Data Errors & Detect and correct missing values

# In[8]:


# Check for missing values
missing_values = df.isnull().sum()
print("Missing values per column:\n", missing_values)


# In[9]:


print(df["num_procedures"].median(),df["num_medications"].median())


# In[10]:


# median value fillna was helped by ChatGPT 4o
median_value = df["num_procedures"].median()
medi_median_value = df["num_medications"].median()
df.fillna({"num_procedures":median_value}, inplace=True)
df.fillna({"num_medications":median_value}, inplace=True)


# In[11]:


# check if two columns' missing values been filled.
missing_values = df.isnull().sum()
print("Missing values per column:\n", missing_values)


# In[12]:


# Check for "?" in the entire DataFrame
question_marks = df.applymap(lambda x: x == "?")

# Sum up how many "?" values are in each column
question_mark_count = question_marks.sum()

print("Number of '?' values per column:\n", question_mark_count)


# In[13]:


# consulted with ChatGPT 4o on imputing method and code. Asked for whether to use kNN or Mode, and the answer was mix of both.
from sklearn.impute import KNNImputer
import pandas as pd

# Replace '?' with NaN
columns_to_replace = ["race", "payer_code", "diag_1", "diag_2", "diag_3"]
df[columns_to_replace] = df[columns_to_replace].replace("?", pd.NA)

# Fill categorical columns with mode (updated method)
df["race"] = df["race"].fillna(df["race"].mode()[0])
df["payer_code"] = df["payer_code"].fillna(df["payer_code"].mode()[0])

# Convert diagnosis codes to numeric (for kNN), keeping NaNs
for col in ["diag_1", "diag_2", "diag_3"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Apply kNN Imputation for diagnosis columns
imputer = KNNImputer(n_neighbors=5)  # Using 5 nearest neighbors
df[["diag_1", "diag_2", "diag_3"]] = imputer.fit_transform(df[["diag_1", "diag_2", "diag_3"]])

# Confirm missing values are handled
print(df[["race", "payer_code", "diag_1", "diag_2", "diag_3"]].isna().sum())


# # Questions
# - Payer code does not seem to mean anything, and has over 30 % missing values. Okay to delete it? 

# In[14]:


# drop three columns that has over 50% missing values. "max_glu_serum","A1Cresult", "weight"
# drop meaningless column that has about 30 % missing values "payer_code"
df = df.drop(["max_glu_serum","A1Cresult", "weight","payer_code",], axis=1)


# In[15]:


df.head()


# In[16]:


# factorize age for ordinal values
df["age"]= df["age"].factorize()[0]

# separate target values before one hot encoding

x_df = df.drop(["readmitted"], axis=1)
y = df["readmitted"]
# do one hot encoding for categorical
x_df = pd.get_dummies(x_df)


# In[17]:


x_df.head()


# ### Questions
# - part 3 is data preparation with Python and pandas right? No using excel? 
# - What to do with missing values?
# - if it is empty.. num_procedures.. numeric .. - 0, max_glu_serum categorical, A1Cresult categorical..
# - depending on type of data.. lecture 6 slide 19 ..
# - 
# - if it is '?'.. race, weight, payer_code, diag_1, diag_2, diag_3
# - Is this below pd.get_dummies(df) right way to deal with categorical? 
# 

# In[18]:


# Check the data types of each column
print(df.dtypes)


# In[19]:


from sklearn.model_selection import train_test_split

# Define features (X) and target variable (y)
X = x_df
# Target variable.. y already set before

# First split: 60% training, 40% remaining
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=28)

# Second split: 50% of remaining (i.e., 20% of total) for validation, 50% for test
#X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Print dataset sizes to confirm
print(f"Training set: {len(X_train)} samples")
#print(f"Validation set: {len(X_val)} samples")
print(f"Test set: {len(X_test)} samples")


# In[20]:


clf = DecisionTreeClassifier(criterion='entropy', random_state=31,max_depth =4)


# In[21]:


clf.fit(X_train, y_train)


# In[22]:


y_pred = clf.predict(X_test)


# In[23]:


clf.score(X_test,y_test)


# In[24]:


accuracy = accuracy_score(y_test, y_pred)
print(f"Multivariate model accuracy: {accuracy:.2f}")


# In[25]:


feature_labels = x_df.columns


# In[26]:


writegraphtofile(clf,feature_labels, 'multi_decision_tree.png')


# ## Test for Binary Model

# In[27]:


# Replace all occurrences of 'B' in 'col2' with 'Y'
df['readmitted'].replace('>30', 'YES', inplace=True)
df['readmitted'].replace('<30', 'YES', inplace = True)
print(df['readmitted'].head(50))


# In[28]:


# Define features (X) and target variable (y)
X = x_df
# Target variable.. y already set before
y = df['readmitted']
# First split: 60% training, 40% remaining
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=28)
# Print dataset sizes to confirm
print(f"Training set: {len(X_train)} samples")
#print(f"Validation set: {len(X_val)} samples")
print(f"Test set: {len(X_test)} samples")


# In[29]:


clf = DecisionTreeClassifier(criterion='entropy', random_state=31,max_depth =4)


# In[30]:


clf.fit(X_train, y_train)


# In[31]:


y_pred = clf.predict(X_test)


# In[32]:


clf.score(X_test,y_test)


# In[33]:


accuracy = accuracy_score(y_test, y_pred)
print(f"Binary model accuracy: {accuracy:.2f}")


# In[34]:


writegraphtofile(clf,feature_labels, 'binary_decision_tree.png')

