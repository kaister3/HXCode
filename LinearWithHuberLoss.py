import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

x = np.array([ 0,  3,  9, 14, 15, 19, 20, 21, 30, 35,
              40, 41, 42, 43, 54, 56, 67, 69, 72, 88])
y = np.array([33, 68, 34, 34, 37, 71, 37, 44, 48, 49,
              53, 49, 50, 48, 56, 60, 61, 63, 44, 71])
e = np.array([ 3.6, 3.9, 2.6, 3.4, 3.8, 3.8, 2.2, 2.1, 2.3, 3.8,
               2.2, 2.8, 3.9, 3.1, 3.4, 2.6, 3.4, 3.7, 2.0, 3.5])

def huberLoss(theta, x = x, y = y, e = e, delta=3):
    residual = abs(y - (theta[0] + theta[1] * x))
    # 残差
    loss = (residual <= delta) * residual ** 2 / 2 + delta * (residual > delta) * (residual - delta / 2)
    return np.sum(loss)

theta = optimize.fmin(huberLoss, [0, 0], disp=False)
print('theta:', theta)

plt.figure(figsize=(6, 4.5))
plt.errorbar(x, y, e, fmt='ok', ecolor='gray', alpha=.4)

xfit = np.linspace(0, 100)
plt.plot(xfit, theta[0] + theta[1] * xfit, c = 'red')
plt.show()