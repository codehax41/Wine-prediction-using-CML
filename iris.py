#!/usr/bin/env python
# coding: utf-8

# In[39]:


import itertools
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# In[42]:


dataset = pd.read_csv('iris.csv')
dataset.head()


# In[41]:


dataset.to_csv('iris.csv', index=False)


# In[6]:


#Process feature names
dataset.columns =[colname.strip(' (cm)').replace(" ", "_") for colname in dataset.columns.tolist()]
feature_names = dataset.columns.tolist()[:4]
feature_names


# In[7]:


#Feature Engineering
dataset['sepal_length_to_sepal_width'] = dataset['sepal_length']/dataset['sepal_width']
dataset['petal_length_to_petal_width'] = dataset['petal_length']/dataset['petal_width']
dataset = dataset[[
     'sepal_length', 'sepal_width', 'petal_length', 'petal_width',
    'sepal_length_to_sepal_width', 'petal_length_to_petal_width',
    'target']]


# In[14]:


#Train test split
test_size=0.2
train_dataset, test_dataset = train_test_split(dataset, test_size=test_size, random_state=42)
train_dataset.shape, test_dataset.shape

y_train = train_dataset.loc[:, 'target'].values.astype('int32')
X_train = train_dataset.drop( 'target', axis=1).values.astype('float32')
                            
logreg = LogisticRegression(C=0.001, solver= 'lbfgs', multi_class='multinomial',max_iter=100)
logreg.fit(X_train, y_train)

y_test = test_dataset.loc[:, 'target'].values.astype('int32')
x_test = test_dataset.drop('target', axis=1).values.astype('float32')
prediction = logreg.predict(x_test)
cm = confusion_matrix(prediction, y_test)
f1 = f1_score(y_true = y_test, y_pred = prediction, average = 'macro')


regr = RandomForestClassifier()
regr.fit(X_train, y_train)

train_score_rf = regr.score(X_train, y_train) * 100
test_score_rf = regr.score(X_test, y_test) * 100
with open("rf_metrics_rf.txt", 'w') as outfile:
        outfile.write("Train Var: %2.1f%%\n" % train_score_rf)
        outfile.write("Test Var: %2.1f%%\n" % test_score_rf)
        
train_score_lr = regr.score(X_train, y_train) * 100
test_score_lr = regr.score(X_test, y_test) * 100
with open("rf_metrics_lr.txt", 'w') as outfile:
        outfile.write("Train Var: %2.1f%%\n" % train_score_lr)
        outfile.write("Test Var: %2.1f%%\n" % test_score_lr)
        outfile.write("f1 score: %2.1f%%\n" % f1)

# In[32]:


def plot_confusion_matrix(cm, target_names, title="Confusion_matrix", cmap=None, normalize=True):
    
    
    accuracy = np.trace(cm)/float(np.sum(cm))
    misclass = 1 - accuracy
    #if cmap is None:
    cmap = plt.get_cmap('Blues')
        
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
    if normalize:
        cm =cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]), 
                     horizontalalignment="center",
                      color = "white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted Label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig("ConfusionMatrix.png",dpi=120) 
    plt.show()


# In[26]:
plot_confusion_matrix(cm, data.target_names, normalize=False)

importances = regr.feature_importances_
labels = df.columns
feature_df = pd.DataFrame(list(zip(labels, importances)), columns = ["feature","importance"])
features = feature_df.sort_values(by='importance', ascending=False,)

axis = 14
title = 22
sns.set(style="white")

ax = sns.barplot(x="importance", y="feature", data=features)
ax.set_xlabel('Importance',fontsize = axis) 
ax.set_ylabel('Feature', fontsize = axis)#ylabel
ax.set_title('Random forest\nfeature importance', fontsize = title)
plt.tight_layout()
plt.savefig("FeatureImportance.png",dpi=120) 
plt.close()




