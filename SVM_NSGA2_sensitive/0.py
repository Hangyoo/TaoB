import numpy as np
import geatpy as ea
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split,KFold
from multiprocessing import Pool as ProcessPool
import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool
from sklearn import metrics
from xgboost import XGBRegressor


datas = np.genfromtxt('wind_pressure.csv', delimiter=',')
X = datas[:, 0:4]
y = datas[:, 4]

ss = preprocessing.StandardScaler()
X = ss.fit_transform(X)

s = preprocessing.StandardScaler()
y = s.fit_transform(y.reshape(-1,1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=30)


xgb_model = XGBRegressor()
xgb_model.fit(X_train,y_train)
print(xgb_model.score(X_test,y_test))