from numpy import *
from sklearn.datasets import load_iris

iris = load_iris()
samples = iris.data
target = iris.target
from sklearn.linear_model import LogisticRegression

clasifier = LogisticRegression()
clasifier.fit(samples, target)

x = clasifier.predict(array([[5,3,5,2]])) # two ]]
print(x)