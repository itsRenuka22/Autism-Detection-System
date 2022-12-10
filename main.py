import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

st.title("Autism Detection System")
le = LabelEncoder()
df = pd.read_csv("dataset.csv")
df.drop(['Case_No', 'Who_completed_the_test', 'Qchat-10-Score'], axis = 1, inplace = True)
data = pd.DataFrame(df, columns = ['Ethnicity', 'Family_mem_with_ASD', 'ClassASD_Traits', 'Sex', 'Jaundice'] )
data['Family_mem_with_ASD_encoded'] = le.fit_transform(data['Family_mem_with_ASD'])

data['Ethnicity_encoded'] = le.fit_transform(data['Ethnicity'])

data['ClassASD_Traits_encoded'] = le.fit_transform(data['ClassASD_Traits'])

data['Sex_encoded'] = le.fit_transform(data['Sex'])

data['Jaundice_encoded'] = le.fit_transform(data['Jaundice'])

columns = ['Ethnicity', 'Family_mem_with_ASD', 'ClassASD_Traits', 'Sex', 'Jaundice']
for col in columns:
    df[col] = le.fit_transform(df[col])
x = df.drop('ClassASD_Traits', axis='columns')
y = df['ClassASD_Traits']
corr = df.corr()
fig = plt.figure(figsize = (15, 15))
sns.heatmap(data = corr, annot = True, square = True, cbar = True)
st.subheader("Heatmap of the correalation between the data")
st.pyplot(fig)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=7, stratify=y)
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
clf = clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
st.write("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)

st.subheader("Decision Tree")
st.image('entropy.png')
#Train Logistic Regression Classifier
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)

#Predict the response for test dataset
y_pred = log_reg.predict(x_test)

# Model Accuracy, how often is the classifier correct?
st.write("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)

