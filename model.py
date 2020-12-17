import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
data = pd.read_csv("Iris.csv")
data
data.shape
data.info()
data.describe()
data.drop('Id',axis=1,inplace=True)
data.head()
sns.heatmap(data.corr(),annot=True)
data.hist()
plt.show()
sns.pairplot(data=data,hue="Species",palette="colorblind")
x = data.iloc[:, :-1].values
y = data.iloc[:, 4].values
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=0)
dt = DecisionTreeClassifier(criterion = "entropy",random_state=0)
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn.metrics import classification_report,accuracy_score
print(classification_report(y_test,y_pred))
accuracy_score(y_test,y_pred)

pickle.dump(dt, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
