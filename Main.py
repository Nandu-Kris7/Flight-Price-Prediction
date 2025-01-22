import pandas as pd
df = pd.read_csv('Clean_Dataset.csv')
#Preprocessing
df = df.drop('Unnamed: 0', axis = 1)
df = df.drop('flight', axis = 1)

df['class'] = df['class'].apply(lambda x: 1 if x =='Business' else 0)
df.stops = pd.factorize(df.stops)[0]
df = df.join(pd.get_dummies(df.airline, prefix = 'airline')).drop('airline', axis = 1)
df = df.join(pd.get_dummies(df.source_city, prefix = 'source')).drop('source_city', axis = 1)
df = df.join(pd.get_dummies(df.destination_city, prefix = 'dest')).drop('destination_city', axis = 1)
df = df.join(pd.get_dummies(df.arrival_time, prefix = 'arrival')).drop('arrival_time', axis = 1)
df = df.join(pd.get_dummies(df.departure_time, prefix = 'departure')).drop('departure_time', axis = 1)

#Training Regression Model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

x, y = df.drop('price', axis = 1), df.price
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
reg = RandomForestRegressor(n_jobs=-1)
reg.fit(x_train, y_train)

print(reg.score(x_test, y_test))

import math
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = reg.predict(x_test)

print('R2 Score: ', r2_score(y_test, y_pred))
print('Mean Absolute Error: ', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error: ', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error: ', math.sqrt(mean_squared_error(y_test, y_pred)))

#plotting the predicted values against the actual values
import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual Price vs Predicted Price')
plt.show()

importances = dict(zip(reg.feature_names_in_, reg.feature_importances_))
sorted_importances = sorted(importances.items(), key = lambda x: x[1], reverse = True)
plt.figure(figsize = (10, 6))
plt.bar([x[0] for x in sorted_importances[:5]], [x[1] for x in sorted_importances[:5]])
plt.show()

#Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

reg = RandomForestRegressor(n_jobs=-1)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt']
}

grid_search = GridSearchCV(reg, param_grid, cv = 5)
grid_search.fit(x_train, y_train)
best_param = grid_search.best_params_