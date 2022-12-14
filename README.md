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

## Exp-3-XOR
```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x=np.array([[0,0,1,1],[0,1,0,1]])
y=np.array([[0,1,1,0]])

n_x = 2 
n_y = 1 
n_h = 2 

m = x.shape[1]
lr = 0.1
np.random.seed(2)
w1 = np.random.rand(n_h,n_x) 
w2 = np.random.rand(n_y,n_h) 
losses = []

def sigmoid(z):
    z= 1/(1+np.exp(-z))
    return z
    
def forward_prop(w1,w2,x):
    z1 = np.dot(w1,x)  
    a1 = sigmoid(z1)   
    z2 = np.dot(w2,a1) 
    a2 = sigmoid(z2)
    return z1,a1,z2,a2

def back_prop(m,w1,w2,z1,a1,z2,a2,y):    
    dz2 = a2-y  
    dw2 = np.dot(dz2,a1.T)/m  
    dz1 = np.dot(w2.T,dz2) * a1*(1-a1) 
    dw1 = np.dot(dz1,x.T)/m  
    dw1 = np.reshape(dw1,w1.shape)
    dw2 = np.reshape(dw2,w2.shape)    
    return dz2,dw2,dz1,dw1

iterations = 10000
for i in range(iterations):
    z1,a1,z2,a2 = forward_prop(w1,w2,x)
    loss = -(1/m)*np.sum(y*np.log(a2)+(1-y)*np.log(1-a2))
    losses.append(loss)
    da2,dw2,dz1,dw1 = back_prop(m,w1,w2,z1,a1,z2,a2,y)
    w2 = w2-lr*dw2
    w1 = w1-lr*dw1

plt.plot(losses)
plt.xlabel("EPOCHS")
plt.ylabel("Loss value")

def predict(w1,w2,input):
    z1,a1,z2,a2 = forward_prop(w1,w2,test)
    a2 = np.squeeze(a2)
    if a2>=0.5:
        print( [i[0] for i in input], 1)
    else:
        print( [i[0] for i in input], 0)
print('Input',' Output')
test=np.array([[0],[0]])
predict(w1,w2,test)
test=np.array([[0],[1]])
predict(w1,w2,test)
test=np.array([[1],[0]])
predict(w1,w2,test)
test=np.array([[1],[1]])
predict(w1,w2,test)
```
## Exp-4-MLP
```py

```
