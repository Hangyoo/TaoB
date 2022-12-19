import torch
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
np.random.seed(0)
torch.manual_seed(0)

# STEP1 读取数据
data = datasets.load_breast_cancer()
# print(data.data.shape)
# print(data.target[:30])
X,y = data.data.astype(np.float32), data.target.astype(np.float32)
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.3,random_state=32)

sc = StandardScaler() # 期望是0标准差为1的正态分布
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#numpy转换为tensor
X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
Y_train = torch.from_numpy(Y_train)
Y_test = torch.from_numpy(Y_test)

# STEP2 构造模型
class MyLogisticalRegression(torch.nn.Module):
    def __init__(self,input_features):
        super(MyLogisticalRegression, self).__init__()
        self.linear = torch.nn.Linear(input_features,1)

    def forward(self,x):
        y = self.linear(x)
        return torch.sigmoid(y)

input_features = 30
model = MyLogisticalRegression(input_features)

# STEP3 创建loss和optimizer
lr = 0.2
num_epochs = 10
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=lr)

# STEP4 训练模型
for epoch in range(num_epochs):
    # forward计算loss
    # 前馈
    Y_pred = model(X_train.view(-1,input_features))
    loss = criterion(Y_pred.view(-1,1),Y_train.view(-1,1))
    # bac|kward更新parameter
    # 反馈 (计算梯度)
    loss.backward()
    # 更新 (用梯度更新参数值)
    optimizer.step()
    optimizer.zero_grad()

    with torch.no_grad():
        y_pred_test = model(X_test.view(-1,input_features))
        y_pred_test = y_pred_test.round().squeeze() # 先四舍五入，再把多余的一列去掉
        total_correct = y_pred_test.eq(Y_test).sum() # 正确的个数
        prec = total_correct.item() / len(Y_test)
        print(f"epoch {epoch}, loss {loss.item()} prec {prec}")

