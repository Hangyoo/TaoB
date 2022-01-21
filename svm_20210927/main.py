import warnings
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from datetime import datetime
import os
import copy
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

data = pd.read_csv(r"C:\Users\Hangyu\Desktop\data.csv")
data = data.iloc[:,24:]
position_list = np.unique(data["Site"].tolist())
position_dict = dict(zip(range(len(position_list)),position_list))
print("Site集合: \n", position_dict)

# 根据日期获取当前的星期
def get_week_day(y,m,d):
    y, m, d = int(y),int(m),int(d)
    date = datetime.date(datetime(year=y, month=m, day=d))
    day = int(date.weekday()+1)
    return day


def predict(year, month, day):
    # 对观测点进行遍历
    for position_index in range(len(position_list)):
        print(f"观测点{position_dict[position_index]},{year}-{month}-{day}的预测结果:")

        data_site = copy.deepcopy(data.loc[data["Site"]==position_list[position_index],:]) # 获取观测点的数据
        data_site.reset_index(drop=True,inplace=True)

        data_site['Week'] = [j for j in range(len(data_site))] # 添加一列
        data_site['Month'] = [j for j in range(len(data_site))] # 添加一列
        data_site['Day'] = [j for j in range(len(data_site))] # 添加一列

        # 对观测点的数据进行处理
        for j in range(len(data_site)):
            a = data_site['time'][j]
            y,m,d = a.split('-')
            week = get_week_day(y,m,d)
            data_site.loc[j,'Site'] = list(position_dict.values()).index(data_site.loc[j,'Site'])
            data_site.loc[j,'Week'] = week
            data_site.loc[j,'Month'] = int(m)
            data_site.loc[j,'Day'] = int(d)
            hour0, minute0 = data_site['cs0'][j].split(':')
            data_site.loc[j,'Hour_cs0'] = int(hour0)
            data_site.loc[j,'Minute_cs0'] = int(minute0)
            hour1, minute1 = data_site['cs1'][j].split(':')
            data_site.loc[j, 'Hour_cs1'] = int(hour1)
            data_site.loc[j, 'Minute_cs1'] = int(minute1)
            hour2, minute2 = data_site['cs2'][j].split(':')
            data_site.loc[j, 'Hour_cs2'] = int(hour2)
            data_site.loc[j, 'Minute_cs2'] = int(minute2)
            # # 空缺值采用插值法补齐
            if data_site['cs3'][j] is np.nan:
                # 最后的数据为空
                if j == len(data_site)-1:
                    data_site.loc[j, 'Hour_cs3'] = data_site['cs3'][j-1].split(':')[0]
                    data_site.loc[j, 'Minute_cs3'] = data_site['cs3'][j-1].split(':')[1]
                    data_site.loc[j, 'cg3'] = data_site['cg3'][j-1]
                    data_site.loc[j, 'cs3'] = data_site.loc[j-1, 'cs3']
                elif j == 0:
                    data_site.loc[j, 'Hour_cs3'] = data_site['cs3'][j + 1].split(':')[0]
                    data_site.loc[j, 'Minute_cs3'] = data_site['cs3'][j + 1].split(':')[1]
                    data_site.loc[j, 'cg3'] = data_site['cg3'][j + 1]
                    data_site.loc[j, 'cs3'] = data_site.loc[j + 1, 'cs3']
                else:
                    if data_site['cs3'][j-1] is np.nan:
                        hour3_pre, minute3_pre = data_site['cs3'][j-2].split(':')
                        cg3_pre = data_site['cg3'][j-2]
                    else:
                        hour3_pre, minute3_pre = data_site['cs3'][j-1].split(':')
                        cg3_pre = data_site['cg3'][j-1]
                    if data_site['cs3'][j+1] is np.nan:
                        hour3_aft, minute3_aft = data_site['cs3'][j+2].split(':')
                        cg3_aft = data_site['cg3'][j+2]
                    else:
                        hour3_aft, minute3_aft = data_site['cs3'][j+1].split(':')
                        cg3_aft = data_site['cg3'][j+1]
                    data_site.loc[j, 'Hour_cs3'] = (int(hour3_pre)+int(hour3_aft))/2
                    data_site.loc[j, 'Minute_cs3'] = (int(minute3_pre)+int(minute3_aft))/2
                    data_site.loc[j, 'cg3'] = (cg3_pre+cg3_aft)/2
                    data_site.loc[j, 'cs3'] = str(data_site.loc[j, 'Hour_cs3']) + ":" + str(data_site.loc[j, 'Minute_cs3'])
            else:
                hour3, minute3 = data_site['cs3'][j].split(':')
                data_site.loc[j, 'Hour_cs3'] = int(hour3)
                data_site.loc[j, 'Minute_cs3'] = int(minute3)

        # # 特征 : 位置, 星期, 月, 日
        # # 输出 : 时间, 数值
        train_data = copy.copy(data_site.loc[:,['Site','Week','Month','Day']])
        train_label1 = copy.copy(data_site.loc[:,['Hour_cs0','Minute_cs0','cg0']])  # 第一次极大值
        train_label2 = copy.copy(data_site.loc[:,['Hour_cs2','Minute_cs2','cg2']])  # 第二次极大值
        train_label3 = copy.copy(data_site.loc[:,['Hour_cs1','Minute_cs1','cg1']])  # 第一次极小值
        train_label4 = copy.copy(data_site.loc[:,['Hour_cs3','Minute_cs3','cg3']])  # 第二次极大值
        x_train, _, y_train, _ = train_test_split(train_data.values,train_label1.values,random_state=1,train_size=0.95)
        # 训练预测时间的SVM模型(第一次极大值)
        model1_hour = SVR(kernel='rbf')
        model1_hour.fit(x_train,y_train[:,0])
        model1_minute = SVR(kernel='rbf')
        model1_minute.fit(x_train,y_train[:,1])
        model1_value = SVR(kernel='rbf')
        model1_value.fit(x_train,y_train[:,2])

        x_train, _, y_train, _ = train_test_split(train_data.values,train_label2.values,random_state=1,train_size=0.95)
        # 训练预测时间的SVM模型(第二次极大值)
        model2_hour = SVR(kernel='rbf')
        model2_hour.fit(x_train,y_train[:,0])
        model2_minute = SVR(kernel='rbf')
        model2_minute.fit(x_train,y_train[:,1])
        model2_value = SVR(kernel='rbf')
        model2_value.fit(x_train,y_train[:,2])

        x_train, _, y_train, _ = train_test_split(train_data.values,train_label3.values,random_state=1,train_size=0.95)
        # 训练预测时间的SVM模型(第一次极小值)
        model3_hour = SVR(kernel='rbf')
        model3_hour.fit(x_train,y_train[:,0])
        model3_minute = SVR(kernel='rbf')
        model3_minute.fit(x_train,y_train[:,1])
        model3_value = SVR(kernel='rbf')
        model3_value.fit(x_train,y_train[:,2])

        x_train, _, y_train, _ = train_test_split(train_data.values,train_label4.values,random_state=1,train_size=0.95)
        # 训练预测时间的SVM模型(第二次极小值)
        model4_hour = SVR(kernel='rbf')
        model4_hour.fit(x_train,y_train[:,0])
        model4_minute = SVR(kernel='rbf')
        model4_minute.fit(x_train,y_train[:,1])
        model4_value = SVR(kernel='rbf')
        model4_value.fit(x_train,y_train[:,2])

        X = np.array([position_index,week,month,day]).reshape(1,-1)
        hour1,minute1,value1 = model1_hour.predict(X), model1_minute.predict(X), model1_value.predict(X)
        hour2,minute2,value2 = model2_hour.predict(X), model2_minute.predict(X), model2_value.predict(X)
        hour3,minute3,value3 = model3_hour.predict(X), model3_minute.predict(X), model3_value.predict(X)
        hour4,minute4,value4 = model2_hour.predict(X), model2_minute.predict(X), model2_value.predict(X)
        print(f'第1次极大值: {int(hour1)}:{int(minute1)}, 潮高: {int(value1)}')
        print(f'第2次极大值: {int(hour2)}:{int(minute2)}, 潮高: {int(value2)}')
        print(f'第1次极小值: {int(hour3)}:{int(minute3)}, 潮高: {int(value3)}')
        print(f'第2次极小值: {int(hour4)}:{int(minute4)}, 潮高: {int(value4)}')

if __name__ == "__main__":
    print("预测的年份(默认2021年)")
    year = 2021
    # year = int(input("\n输入希望预测的年份(默认2021年):"))
    month = int(input("\n输入希望预测的月份(1-12):"))
    day = int(input("\n输入希望预测的日期(0-31):"))
    predict(year,month,day)