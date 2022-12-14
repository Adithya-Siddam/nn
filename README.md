## Exp-1-Data Preprocessing
```py
import pandas as pd
import numpy as np

df = pd.read_csv("./Churn_Modelling.csv")
df

df.isnull().sum()

df.duplicated()

df.describe()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df1 = df.copy()
df1["Geography"] = le.fit_transform(df1["Geography"])
df1["Gender"] = le.fit_transform(df1["Gender"])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df1[["CreditScore","Geography","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]] = pd.DataFrame(scaler.fit_transform(df1[["CreditScore","Geography","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]]))
df1

df1.describe()

X = df1[["CreditScore","Geography","Gender","Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"]].values
X

y = df1["Exited"].values
y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train

print("Size of X_train: ",len(X_train))

X_test

print("Size of X_test: ",len(X_test))

X_train.shape
```
## Exp-2-Perceptron
```py
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score

class Perceptron:
  def __init__(self, learning_rate=0.1):
    self.learning_rate = learning_rate
    self._b = 0.0 
    self._w = 0 
    self.misclassified_samples = []
  def fit(self, x: np.array, y: np.array, n_iter=10):
    self._b = 0.0
    self._w = np.zeros(x.shape[1])
    self.misclassified_samples = []
    for i in range(n_iter):
      errors = 0
      for xi,yi in zip(x,y):
        update = self.learning_rate * (yi-self.predict(xi))
        self._b += update
        self._w += update*xi
        errors += int(update !=0)
      self.misclassified_samples.append(errors)
  def f(self,x:np.array) -> float:
    return np.dot(x,self._w) + self._b
  def predict(self, x:np.array):
    return np.where(self.f(x) >= 0,1,-1)
    
df = pd.read_csv("./IRIS.csv")
df

x = df[["sepal_length","sepal_width","petal_length","petal_width"]].values
x

y = df["species"].values
y

x = x[0:100,0:2]
y = y[0:100]

plt.scatter(x[:50, 0], x[:50, 1], color='red', marker='o', label='Setosa')
plt.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='x',
            label='Versicolour')
plt.xlabel("Sepal length")
plt.ylabel("Petal length")
plt.legend(loc='upper left')
plt.show()

y = np.where(y=="Iris-setosa",1,-1)
y

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)
x

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,
                                                    random_state=0)
classifier = Perceptron(learning_rate=0.01)
classifier.fit(x_train, y_train)

plt.plot(range(1, len(classifier.misclassified_samples) + 1),
         classifier.misclassified_samples, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Errors')
plt.show()

print("accuracy = " , accuracy_score(classifier.predict(x_test), y_test)*100)
```
