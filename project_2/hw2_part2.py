#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
import pydotplus


# In[30]:


#os.environ["PATH"] += os.pathsep + r"C:\dev\course_material\ece_5464_apps_of_ml\env\Scripts"


# In[31]:


file_path = "/home/coldnoodle/dev/ece-5464/project_2/AlienMushrooms.xlsx"


# In[32]:


df = pd.read_excel(file_path)


# In[33]:


df.head()


# In[34]:


X = df[['White', 'Tall', 'Frilly']]
y = df['Edible']


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


# In[36]:


clf = DecisionTreeClassifier(criterion='entropy', random_state=42,max_depth=4)


# In[37]:


clf.fit(X_train, y_train)


# In[38]:


y_pred = clf.predict(X_test)


# In[39]:


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# In[40]:


feature_labels = ['White', 'Tall', 'Frilly']


# In[41]:


import pydotplus
# for a two-class tree, call this function like this:
# writegraphtofile(clf, ('F', 'T'), dirname+graphfilename)
# for a multi-class tree, call this function like this:
# @ writegraphtofile(clf, featurenames, dirname+graphfilename)
def writegraphtofile(clf, featurelabels, filename):
 dot_data = sklearn.tree.export_graphviz(clf, feature_names=featurelabels, out_file=None)
 graph=pydotplus.graph_from_dot_data(dot_data)
 graph.write_png(filename)


# In[42]:


writegraphtofile(clf,feature_labels, 'decision_tree.png')

