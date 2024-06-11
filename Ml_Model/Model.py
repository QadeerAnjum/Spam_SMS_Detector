import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

df=pd.read_csv('Train.tsv',sep='\t')

hamDF=df[df['label']=='ham']
spamDF=df[df['label']=='spam']

hamDF= hamDF.sample(spamDF.shape[0])

finalDF = hamDF._append(spamDF, ignore_index=True)


X_train,X_test,Y_train,Y_test=train_test_split(finalDF['message'],finalDF['label'],test_size=0.2,random_state=0,shuffle=True,stratify= finalDF['label'])

#Pipeline
model=Pipeline([('tfidf',TfidfVectorizer()),('model',RandomForestClassifier(n_estimators=100,n_jobs=-1))])
model.fit(X_train,Y_train)

Y_predict=model.predict(X_test)

joblib.dump(model,"myModel.pkl")
