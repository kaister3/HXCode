import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

x = np.array([ 0,  3,  9, 14, 15, 19, 20, 21, 30, 35,
              40, 41, 42, 43, 54, 56, 67, 69, 72, 88])
y = np.array([68, 37, 34, 34, 37, 71, 37, 44, 48, 49,
              53, 49, 50, 48, 56, 60, 61, 63, 44, 34])

def squared_loss(theta, x = x, y = y):
    dy = (y - (theta[0] + theta[1] * x))
    return np.sum(dy ** 2 / 2)
theta0 = optimize.fmin(squared_loss, [0, 0], disp=False)
print('theta0:', theta0)

def huberLoss(theta, x = x, y = y, delta=3):
    residual = abs(y - (theta[0] + theta[1] * x))
    # 残差
    loss = (residual <= delta) * residual ** 2 / 2 + delta * (residual > delta) * (residual - delta / 2)
    return np.sum(loss)

theta1 = optimize.fmin(huberLoss, [0, 0], disp=False)
print('theta1:', theta1)

plt.figure(figsize=(6, 4.5))
plt.errorbar(x, y, fmt='ok', ecolor='gray', alpha=.4)

xfit = np.linspace(0, 100)
plt.plot(xfit, theta0[0] + theta0[1] * xfit, c = 'red', alpha=.7)
plt.plot(xfit, theta1[0] + theta1[1] * xfit, c = 'blue', alpha=.7)
plt.show()