import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, 0].values
Y = dataset.iloc[:, 1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state = 0)
X_train = X_train.reshape(-1, 1)
Y_train = Y_train.reshape(-1, 1)

regresor = LinearRegression()
regresor.fit(X_train, Y_train)
predict = regresor.predict(X_train.reshape(-1, 1))

plt.figure(figsize=(10,12)) #设置画布大小
figure = plt.subplot(211)  #将画布分成2行1列，当前位于第一块子画板

plt.scatter(X_train, Y_train,color = 'blue')  #描出训练集对应点
plt.plot(X_train,predict,color='red')  #画出预测的模型图
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.title('Train set')

#将模型应用于测试集，检验拟合结果
plt.subplot(212)
plt.scatter(X_test, Y_test,color = 'red')
plt.plot(X_train,predict,color='black')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.title('Test set')

plt.show()