import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

dataset = pd.read_table('ex0.txt', '\t')
X = dataset.iloc[:, 1].values
Y = dataset.iloc[:, 2].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
X_train = X_train.reshape(-1, 1)
Y_train = Y_train.reshape(-1, 1)

regresor = LinearRegression()
regresor.fit(X_train, Y_train)
predict = regresor.predict(X_train.reshape(-1, 1))

plt.figure(figsize=(10, 12))
figure = plt.subplot(211)

plt.scatter(X_train, Y_train, color='blue', alpha = 0.5)
plt.plot(X_train, predict, color='red', alpha = 0.5)
plt.xlabel = 'x'
plt.ylabel = 'y'
plt.title('Train Set')

plt.subplot(212)
plt.scatter(X_test, Y_test, color='blue')
plt.plot(X_train, predict, color='red')
plt.xlabel = 'x'
plt.ylabel = 'y'
plt.title('Test Set')

plt.show()