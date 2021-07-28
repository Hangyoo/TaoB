
import sys
import warnings
import numpy as np
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore")

def fitness(weights_mat, x_train, y_train, activation="sigmoid"):

    Coef = weights_mat[0]

    class MLPClassifierOverride(MLPRegressor):

        def _init_coef(self, fan_in, fan_out):
            factor = 6.
            if self.activation == 'logistic':
                factor = 2.
            init_bound = np.sqrt(factor / (fan_in + fan_out))

            if fan_in == 8 and fan_out == 200:
                coef_init = Coef[0]
                intercept_init = np.random.uniform(-init_bound, init_bound,200)

            elif fan_in == 200 and fan_out == 100:
                coef_init = Coef[1]
                intercept_init = np.random.uniform(-init_bound, init_bound,100)

            elif fan_in == 100 and fan_out == 1:
                coef_init = Coef[2]
                intercept_init = np.random.uniform(-init_bound, init_bound,1)

            else:
                sys.exit("ANNet文件中神经元个数输入有误！")

            return coef_init, intercept_init


    rf = MLPClassifierOverride(hidden_layer_sizes=(200,100),learning_rate_init= 0.001,activation='logistic',solver='adam', alpha=1e-8,max_iter=30000)  # 神经网络
    rf=rf.fit(x_train, y_train)     # 训练分类器
    y_pred_train = rf.NN_predict(x_train)
    r = r2_score(y_train,y_pred_train)
    accuracy = np.array([-r]*weights_mat.shape[0])
    return accuracy