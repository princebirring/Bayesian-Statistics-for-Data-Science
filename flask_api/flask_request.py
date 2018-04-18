from sklearn.model_selection import train_test_split
from sklearn import linear_model
import pandas as pd
import dill as pickle
import json
import requests

def main():
	df = pd.read_csv('insurance.csv')
	df = pd.get_dummies(df)
	X = df.drop('charges',axis=1)
	y = df.charges

	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

	X_test_1 = pd.DataFrame(X_test)
	data = X_test_1.to_json(orient = 'records')
	header = {'Content-Type': 'application/json','Accept': 'application/json'}
	resp = requests.post("http://127.0.0.1:8000/predict", data = json.dumps(data), headers= header)
	print(resp.json())

if __name__ == "__main__":
	main()


