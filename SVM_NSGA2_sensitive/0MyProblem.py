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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=30)
X_train_scaled=preprocessing.StandardScaler().fit_transform(X_train)
X_test_scaled=preprocessing.StandardScaler().fit_transform(X_test)


class MyProblem(ea.Problem):  #继承父类problem

    def __init__(self, PoolType):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 6  # 初始化Dim（决策变量维数）4
        #x: 待优化的参数列表[n_estimators,max_depth,learning_rate,min_child_weight,
        # subsample,reg_lambda]
        varTypes = [ 1 ,1, 0, 1, 0, 0]  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [ 0, 1, 0, 1, 0, 0]  # 决策变量下界
        ub = [ 200, 25, 1, 20, 1, 1]  # 决策变量上界
        lbin = [0,1,0,1,0,0]  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1,1,0,1,0,0]  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self,
                            name,
                            M,
                            maxormins,
                            Dim,
                            varTypes,
                            lb,
                            ub,
                            lbin,
                            ubin)

        self.data=X_train_scaled
        self.target=y_train
        # 设置用多线程还是多进程
        self.PoolType = PoolType
        if self.PoolType == 'Thread':
            self.pool = ThreadPool(2)  # 设置池的大小
        elif self.PoolType == 'Process':
            num_cores = int(mp.cpu_count())  # 获得计算机的核心数
            self.pool = ProcessPool(num_cores)  # 设置池的大小

    def evalVars(self, Vars):  # 目标函数，采用多线程加速计算
        N = Vars.shape[0]
        args = list(
            zip(list(range(N)), [Vars] * N, [self.data] * N,
                [self.target] * N))
        if self.PoolType == 'Thread':
            f = np.array(list(self.pool.map(subAimFunc, args)))
        elif self.PoolType == 'Process':
            result = self.pool.map_async(subAimFunc, args)
            result.wait()
            f = np.array(result.get())
        return f

def subAimFunc(args):
    i = args[0]
    Vars = args[1]
    data = args[2]
    target = args[3]
#k折交叉训练模型
    kfold = KFold(n_splits=10,shuffle=True,random_state=30)
    scores=[0]*50
    for i in range(len(Vars)):
        for train_index, test_index in kfold.split(data, target):
        # train_index 就是分类的训练集的下标，test_index 就是分配的验证集的下标
            train_x, train_y = data[train_index], target[train_index]  # 训练集
            test_x, test_y = data[test_index], target[test_index]  # 验证集
            xgb_model = XGBRegressor(n_estimators=np.int(Vars[i,0]),
                                     max_depth=np.int(Vars[i, 1]),
                                     learning_rate=Vars[i, 2],
                                     min_child_weight=Vars[i, 3],
                                     subsample=Vars[i, 4],
                                     reg_lambda=Vars[i, 5],).fit(train_x,train_y)
            prediction = xgb_model.predict(test_x)
            r2 = metrics.r2_score(test_y, prediction)
            mse=mean_squared_error(test_y,prediction)
            final_score = 1 - round(r2, 4) + round(mse, 4)
        scores[i]=final_score
    ObjV_i = [scores]
    return ObjV_i