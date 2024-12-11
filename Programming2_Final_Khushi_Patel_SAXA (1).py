#!/usr/bin/env python
# coding: utf-8

# In[117]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib


# #### Step 1
# Read in the data, call the dataframe "s"  and check the dimensions of the dataframe

# In[118]:


s = pd.read_csv("social_media_usage.csv")


# In[119]:


print("Dimensions of the dataset:", s.shape)


# #### Step 2
# Define a function called clean_sm that takes one input, x, and uses `np.where` to check whether x is equal to 1. If it is, make the value of x = 1, otherwise make it 0. Return x. 

# In[120]:


def clean_sm(x):
    return np.where(x == 1, 1, 0)


# Create a toy dataframe with three rows and two columns and test your function to make sure it works as expected

# In[121]:


toy_df = pd.DataFrame({'col1': [1, 2, 1], 'col2': [0, 1, 3]})


# In[122]:


toy_df_cleaned = toy_df.apply(lambda col: col.map(clean_sm))


# In[123]:


print("Toy DataFrame after applying clean_sm:\n", toy_df_cleaned)


# #### Step 3
# Create a new dataframe called "ss". The new dataframe should contain a target column called sm_li which should be a binary variable ( that takes the value of 1 if it is 1 and 0 otherwise (use clean_sm to create this) which indicates whether or not the individual uses LinkedIn, and the following features: income (ordered numeric from 1 to 9, above 9 considered missing), education (ordered numeric from 1 to 8, above 8 considered missing), parent (binary), married (binary), female (binary), and age (numeric, above 98 considered missing). Drop any missing values. Perform exploratory analysis to examine how the features are related to the target.

# In[124]:


ss = s[['web1h', 'income', 'educ2', 'par', 'marital', 'gender', 'age']].copy()


# In[125]:


ss.rename(columns={
    'web1h': 'sm_li',
    'educ2': 'education',
    'par': 'parent',
    'marital': 'married',
    'gender': 'female'
}, inplace=True)


# In[126]:


ss['sm_li'] = ss['sm_li'].apply(clean_sm)


# In[127]:


ss['female'] = ss['female'].apply(lambda x: 1 if x == 2 else 0)


# In[128]:


ss = ss[(ss['income'] <= 9) & (ss['education'] <= 8) & (ss['age'] <= 98)].dropna()


# In[129]:


print("Cleaned DataFrame:\n", ss.head())


# #### Step 4

# In[130]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[131]:


sns.pairplot(ss, hue="sm_li")
plt.show()


# #### 
# Create a target vector (y) and feature set (X)

# In[132]:


X = ss[['income', 'education', 'parent', 'married', 'female', 'age']]
y = ss['sm_li']


# In[133]:


print (f"Feature set shape: {X.shape}")


# In[134]:


print (f"Target vector shape: {y.shape}")


# #### Step 5
# Split the data into training and test sets. Hold out 20% of the data for testing. Explain what each new object contains and how it is used in machine learning

# In[135]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# X_train: 80% of the rows from the feature set X. It is used by the machine learning model to learn patterns and relationships between features and the target variable.

# X_test: 20% of the rows from the feature set X. It is used to evaluate the model's performance on unseen data after training.

# y_train: Corresponding target values (y) for X_train. It is used as the ground truth for the model to learn how to predict the target variable.

# y_test: Corresponding target values (y) for X_test. It is used as the ground truth to compare against the model's predictions on X_test.

# In[136]:


print("X_train shape:", X_train.shape)


# In[137]:


print("X_test shape:", X_test.shape)


# In[138]:


print("y_train shape:", y_train.shape)


# In[139]:


print("y_test shape:", y_test.shape)


# #### Step 6
# Instantiate a logistic regression model and set class_weight to balanced. Fit the model with the training data.
# 

# In[140]:


model = LogisticRegression(class_weight='balanced', random_state=42)


# In[141]:


model.fit(X_train, y_train)


# In[142]:


joblib.dump(model, 'Programming2_Final_Project')
print("Model and data saved successfully!")


# #### Step 7
# Evaluate the model using the testing data. What is the model accuracy for the model?
# 

# In[143]:


y_pred = model.predict(X_test)


# In[144]:


accuracy = accuracy_score(y_test, y_pred)


# In[145]:


print("Model Accuracy:", accuracy)


# #### Step 8
# Use the model to make predictions and then generate a confusion matrix from the model. Interpret the confusion matrix and explain what each number means. Create the confusion matrix as a dataframe and add informative column names and index names that indicate what each quadrant represents
# 

# In[146]:


cm = confusion_matrix(y_test, y_pred)


# In[147]:


print("Confusion Matrix:\n", cm)


# The model correctly identified 99 individuals as not being LinkedIn users.
# 
# The model incorrectly predicted 62 individuals as LinkedIn users when they are not.
# 
# The model incorrectly predicted 24 LinkedIn users as not being users.
# 
# The model correctly identified 67 individuals as LinkedIn users.

# In[148]:


cm_df = pd.DataFrame(cm, index=['Actual No', 'Actual Yes'], columns=['Predicted No', 'Predicted Yes'])


# In[149]:


print("Confusion Matrix DataFrame:\n", cm_df)


# #### Step 9
# Aside from accuracy, there are three other metrics used to evaluate model performance: precision, recall, and F1 score. Use the results in the confusion matrix to calculate each of these metrics by hand. Discuss each metric and give an actual example of when it might be the preferred metric of evaluation. After calculating the metrics by hand, create a classification_report using sklearn and check to ensure your metrics match those of the classification_report.
# 

# In[150]:


TP = cm[1, 1]  # True Positives


# In[151]:


TN = cm[0, 0]  # True Negatives


# In[152]:


FP = cm[0, 1]  # False Positives


# In[153]:


FN = cm[1, 0]  # False Negatives


# In[154]:


precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)


# In[155]:


print(f"Precision: {precision}")


# In[156]:


print(f"Recall: {recall}")


# In[157]:


print(f"F1 Score: {f1_score}")


# Precision shows how many of the predicted LinkedIn users are actually correct. It is useful when False Positives are costly, like in targeted ads where marketing to non-users wastes resources. For example, offering premium plans to people who do not use LinkedIn leads to unnecessary expenses.
# 
# Recall measures how many actual LinkedIn users are correctly identified. It is important when missing real users is a bigger problem, like in recruitment campaigns. For example, failing to recognize LinkedIn users might mean missing potential candidates for job ads.
# 
# F1 Score balances Precision and Recall. It is ideal for imbalanced datasets. For example, LinkedIn might use it to assess marketing efforts, ensuring they reach real users without wasting resources on non-users.

# In[158]:


print("Classification Report:\n", classification_report(y_test, y_pred))


# #### Step 10
# Use the model to make predictions. For instance, what is the probability that a high income (e.g. income=8), with a high level of education (e.g. 7), non-parent who is married female and 42 years old uses LinkedIn? How does the probability change if another person is 82 years old, but otherwise the same?
# 

# In[159]:


feature_columns = ['income', 'education', 'parent', 'married', 'female', 'age']


# In[160]:


example_42 = pd.DataFrame([[8, 7, 0, 1, 1, 42]], columns=feature_columns)
example_82 = pd.DataFrame([[8, 7, 0, 1, 1, 82]], columns=feature_columns)


# In[161]:


prob_high_income_42 = model.predict_proba(example_42)[0][1]


# In[162]:


prob_high_income_82 = model.predict_proba(example_82)[0][1]


# In[165]:


print(f"Probability of LinkedIn usage (42 years old): {prob_high_income_42}")


# In[164]:


print(f"Probability of LinkedIn usage (82 years old): {prob_high_income_82}")

