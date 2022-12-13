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
from sklearn.neighbors import KNeighborsClassifier
import warnings

st.set_page_config(
        page_title="Autism Detection System",
)

warnings.filterwarnings(action='ignore', category=UserWarning)
st.title("Autism Detection System")
le = LabelEncoder()
df = pd.read_csv("dataset.csv")
df.drop(['Case_No', 'Who_completed_the_test', 'Qchat-10-Score'], axis=1, inplace=True)
data = pd.DataFrame(df, columns=['Ethnicity', 'Family_mem_with_ASD', 'ClassASD_Traits', 'Sex', 'Jaundice'])
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
# fig = plt.figure(figsize = (15, 15))
# sns.heatmap(data = corr, annot = True, square = True, cbar = True)
# st.subheader("Heatmap of the correalation between the data")
# st.pyplot(fig)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=7, stratify=y)
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
clf = clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
# st.write("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)

# st.subheader("Decision Tree")
# st.image('entropy.png')
# Train Logistic Regression Classifier
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)

# Predict the response for test dataset
y_pred = log_reg.predict(x_test)

# Train K Nearest Neighbors Classifier
knn = KNeighborsClassifier(n_neighbors=13, p=2, metric='euclidean')
knn.fit(x_train, y_train)

# Predict the response for test dataset
y_pred = knn.predict(x_test)

# Model Accuracy, how often is the classifier correct?
# st.write("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)
arr = []

questions1 = ['Does your child look at you when you call his/her name?',
              'How easy is it for you to get eye-contact with your child?',
              'Does your child point to indicate that he/she wants something:?',
              'Does your child point to share interest with you:?',
              'Does your child pretend:?',
              'Does your child follow where you are looking:?',
              'If you or someone in your family is visibly upset, does your child show signs of wanting to comfort '
              'them:? '
              ]

questions2 = ['Does your child stare at nothing with no apparent purpose?']
for quest in questions1:
    a = st.selectbox(quest, ('Always', 'Usually', 'Sometimes', 'Rarely', 'Never'))
    st.write('You selected:', a)
    if a == 'Always':
        arr.append(0)
    elif a == 'Usually':
        arr.append(0)
    elif a == 'Sometimes':
        arr.append(1)
    elif a == 'Rarely':
        arr.append(1)
    else:
        arr.append(1)
a = st.selectbox('Would you describe your child first words as?',
                 ('Very Typical', 'Quite Typical', 'Slightly Unusual', 'Very Unusual', 'Child does not speak'))
st.write('You selected:', a)
if a == 'Very Typical':
    arr.append(0)
elif a == 'Quite Typical':
    arr.append(0)
elif a == 'Slightly Unusual':
    arr.append(1)
elif a == 'Very Unusual':
    arr.append(1)
else:
    arr.append(1)
a = st.selectbox('Does your child use simple gestures?', ('Always', 'Usually', 'Sometimes', 'Rarely', 'Never'))
st.write('You selected:', a)
if a == 'Always':
    arr.append(0)
elif a == 'Usually':
    arr.append(0)
elif a == 'Sometimes':
    arr.append(1)
elif a == 'Rarely':
    arr.append(1)
else:
    arr.append(1)
for quest in questions2:
    a = st.selectbox(quest, ('Always', 'Usually', 'Sometimes', 'Rarely', 'Never'))
    st.write('You selected:', a)
    if a == 'Always':
        arr.append(1)
    elif a == 'Usually':
        arr.append(1)
    elif a == 'Sometimes':
        arr.append(0)
    elif a == 'Rarely':
        arr.append(0)
    else:
        arr.append(0)

arr.append(int(st.text_input('Enter child age in months:', '10')))

a = st.selectbox('Enter Child sex:', ("Male", "Female"))
st.write('You selected:', a)
if a == 'Male':
    arr.append(1)
else:
    arr.append(0)

a = st.selectbox('Enter child Ethnicity:', (
    "Hispanic", "Latino", "Others", "Pacifica", "White European", "Asian", "Black", "Middle Eastern", "Mixed",
    "South Asian"))
st.write('You selected:', a)
if a == 'Hispanic':
    arr.append(0)
elif a == 'Latino':
    arr.append(1)
elif a == 'Others':
    arr.append(3)
elif a == 'Pacifica':
    arr.append(4)
elif a == 'White European':
    arr.append(5)
elif a == 'Asian':
    arr.append(6)
elif a == 'Black':
    arr.append(7)
elif a == 'Middle eastern':
    arr.append(8)
elif a == 'Mixed':
    arr.append(9)
else:
    arr.append(10)

a = st.selectbox('Does the child has an history of jaundice?', ("Yes", "No"))
st.write('You selected:', a)
if a == 'Yes':
    arr.append(1)
else:
    arr.append(0)

a = st.selectbox('Does the child family has an ASD history?', ("Yes", "No"))
st.write('You selected:', a)
if a == 'Yes':
    arr.append(1)
else:
    arr.append(0)

option = st.selectbox('Which Algorithm do you want to use? (For Research Purpose Only)',
                      ("Logistic Regression", "Decision Tree Classifier", "K-Nearest "
                                                                          "Neighbors "
                                                                          "Classifier"))
st.write('You selected:', option)
if option == 'Logistic Regression':
    a = log_reg.predict([arr])
elif option == 'Decision Tree Classifier':
    a = clf.predict([arr])
else:
    a = knn.predict([arr])

if st.button('Predict'):
    if a == 1:
        st.write('Child is autistic')
    else:
        st.write('Child is not autistic')
st.write("[Check out this link for more information about Autism](https://vibodhbhosure.github.io/Secondary-Site-for-Autism-System/)")