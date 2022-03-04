import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('Franchise_Dataset.csv')
df.head()

X =df.iloc[:, 1:2].values
y =df.iloc[:, 2].values

model = RandomForestRegressor(n_estimators = 10, random_state = 0)
model.fit(X, y)

y_pred =model.predict([[6]])
print(y_pred)

X_grid_data = np.arange(min(X), max(X), 0.01)
X_grid_data = X_grid_data.reshape((len(X_grid_data), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid_data,model.predict(X_grid_data), color = 'blue')
plt.title('Random Forest Regression')
plt.xlabel('Counter Sales')
plt.ylabel('Net Profit')
plt.show()