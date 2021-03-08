import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler



d=pd.read_csv(r"C:\Users\loki\Downloads\Diabetes Database.csv")

print(d.isnull().sum())
print("square root", np.sqrt(769))


a=d.iloc[:,:-1].values
b=d.iloc[:,-1].values

a_train,a_test,b_train,b_test= train_test_split(a,b, test_size=0.1, random_state=0)

sc = StandardScaler()
a_train = sc.fit_transform(a_train)
a_test = sc.fit_transform(a_test)
a_train = pd.DataFrame(a_train)

knn= KNeighborsClassifier(n_neighbors=27)
knn.fit(a_train,b_train)
b_test_pred=knn.predict(a_test)
b_test_values=b_test

print("Accuracy of the Knn model is :", knn.score(a_test,b_test)*100)
b_pred = knn.predict(a_test)
print(confusion_matrix(b_test,b_pred))
classification = classification_report(b_test,b_pred)
print("CLASSIFICATION REPORT OF KNN \n",classification)

print("PREDICTING VALUES")
b_predict=pd.DataFrame(data=[b_test_pred,b_test_values])
print(knn.predict([[6,148,72,35,0,33,1,50]]))


    




