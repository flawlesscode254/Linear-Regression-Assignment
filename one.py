import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy import stats

data = pd.read_csv('Franchise_Dataset.csv')
main = data["Net Profit"]
labels = data["Counter Sales"]

slope, intercept, r, p, std_err = stats.linregress(main, labels)

def lineFunc(x):
  return slope * x + intercept

lineY = list(map(lineFunc, main))

plt.scatter(main,labels)
plt.plot(main,lineY)
plt.show()

# Testing with random value to get the linear regression estimate
speedY = lineFunc(9)
print(speedY)

# Significance of model parameters
# -Used to make predictions