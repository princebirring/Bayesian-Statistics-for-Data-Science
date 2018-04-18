from sklearn.model_selection import train_test_split
from sklearn import linear_model
import pandas as pd
import dill as pickle

df = pd.read_csv('insurance.csv')
df = pd.get_dummies(df)
X = df.drop('charges',axis=1)
y = df.charges

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
reg.score(X_test, y_test)


filename = 'model.pk'
with open(filename, 'wb') as file:
        pickle.dump(reg, file)


