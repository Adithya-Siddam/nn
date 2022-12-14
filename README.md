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
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df=pd.read_csv("./IRIS.csv")
df

df.isnull().sum()

df1 = df.copy()
df1 = df[:100]
df1

le = LabelEncoder()
df1["species"] = le.fit_transform(df1["species"])

X = df1[["sepal_length","sepal_width","petal_length","petal_width"]]
X

Y = df1["species"]
Y

Y.value_counts()

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=0)

mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)  
mlp.fit(x_train, y_train)  
y_pred = mlp.predict(x_test) 

acc = accuracy_score(y_pred,y_test)
acc

mat = confusion_matrix(y_pred,y_test)
mat

classi = classification_report(y_pred,y_test)
print(classi)
```
## Exp-5-XOR-RBF
```py
import numpy as np
import matplotlib.pyplot as plt

def gaussian_rbf(x, landmark, gamma=1):
    return np.exp(-gamma * np.linalg.norm(x - landmark)**2)

def predict_matrix(point, weights):
    gaussian_rbf_0 = gaussian_rbf(np.array(point), mu1)
    gaussian_rbf_1 = gaussian_rbf(np.array(point), mu2)
    A = np.array([gaussian_rbf_0, gaussian_rbf_1, 1])
    return np.round(A.dot(weights))

x1 = np.array([0, 0, 1, 1])
x2 = np.array([0, 1, 0, 1])
ys = np.array([0, 1, 1, 0])

plt.figure(figsize=(13, 5))
plt.subplot(1, 2, 1)
plt.scatter((x1[0], x1[3]), (x2[0], x2[3]), label="Class_0")
plt.scatter((x1[1], x1[2]), (x2[1], x2[2]), label="Class_1")
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Linearly Inseparable")
plt.legend()

mu1 = np.array([0, 1])
mu2 = np.array([1, 0])

from_1 = [gaussian_rbf(i, mu1) for i in zip(x1, x2)]
from_2 = [gaussian_rbf(i, mu2) for i in zip(x1, x2)]

A = []

for i, j in zip(from_1, from_2):
    temp = []
    temp.append(i)
    temp.append(j)
    temp.append(1)
    A.append(temp)
    
A = np.array(A)
W = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(ys)
print(np.round(A.dot(W)))
print(ys)
print("Weights:",W)

plt.figure(figsize=(13, 5))
plt.subplot(1, 2, 2)
plt.scatter(from_1[0], from_2[0], label="Class_0")
plt.scatter(from_1[1], from_2[1], label="Class_1")
plt.scatter(from_1[2], from_2[2], label="Class_1")
plt.scatter(from_1[3], from_2[3], label="Class_0")
plt.plot([0, 0.95], [0.95, 0])
plt.annotate("Seperating hyperplane", xy=(0.5, 0.5), xytext=(0.5, 0.5))
plt.xlabel("µ1")
plt.ylabel("µ2")
plt.title("Transformed Inputs")
plt.legend()

print(f"Input:{np.array([0, 0])}, Predicted: {predict_matrix(np.array([0, 0]), W)}")
print(f"Input:{np.array([0, 1])}, Predicted: {predict_matrix(np.array([0, 1]), W)}")
print(f"Input:{np.array([1, 0])}, Predicted: {predict_matrix(np.array([1, 0]), W)}")
print(f"Input:{np.array([1, 1])}, Predicted: {predict_matrix(np.array([1, 1]), W)}")
```
