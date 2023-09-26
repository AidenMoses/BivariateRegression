import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

# Read the advertising data from a CSV file
df = pd.read_csv('Advertising.csv', index_col=0)

# Visualize the data
sns.pairplot(df)

# Analyze and interpret the data
# (You mentioned D) and E) comments)

# Simple linear regression using sklearn
TV_data = df['TV'].to_numpy().reshape(-1, 1)
sales_data = df['sales'].to_numpy().reshape(-1, 1)

model = LinearRegression()
model.fit(X=TV_data, y=sales_data)

y_intercept = round(model.intercept_[0], 2)
slope = round(model.coef_[0][0], 2)

# Print regression results
if y_intercept < 0:
    print(f"Therefore the equation of the line is: y = {slope}x {y_intercept}")
else:
    print(f"Therefore the equation of the line is: y = {slope}x + {y_intercept}")

new_R2 = model.score(TV_data, sales_data)
print(f"The new R^2 coefficient is: {new_R2:.3f}")

# Visualize the regression line
plt.title('TV plotted against Sales')
plt.xlabel('TV')
plt.ylabel('Sales')

plt.plot(TV_data, sales_data, 'b.')  # Dots will be blue
plt.plot(TV_data, model.predict(TV_data), color='r')  # Red regression line
plt.grid(True)
plt.show()

# Splitting data for polynomial regression (comment about training and test data)

# 1. Dense Data
train_x_dense = TV_data[:80]
train_y_dense = sales_data[:80]

test_x_dense = TV_data[80:]
test_y_dense = sales_data[80:]

mymodel_dense = np.poly1d(np.polyfit(train_x_dense.flatten(), train_y_dense.flatten(), 4))

myline_dense = np.linspace(0, 300, 100)

plt.scatter(train_x_dense, train_y_dense)
plt.plot(myline_dense, mymodel_dense(myline_dense))
plt.show()

# 2. Sparse Data
train_x_sparse = TV_data[:10]
train_y_sparse = sales_data[:10]

test_x_sparse = TV_data[10:]
test_y_sparse = sales_data[10:]

mymodel_sparse = np.poly1d(np.polyfit(train_x_sparse.flatten(), train_y_sparse.flatten(), 4))

myline_sparse = np.linspace(0, 300, 100)

plt.scatter(train_x_sparse, train_y_sparse)
plt.plot(myline_sparse, mymodel_sparse(myline_sparse))
plt.show()

# Ordinary Least Squares (OLS) regression with statsmodels

# 1. Dense Data
x_dense = sm.add_constant(train_x_dense)
model_dense = sm.OLS(train_y_dense, x_dense).fit()
print(model_dense.summary())

# 2. Sparse Data
x_sparse = sm.add_constant(train_x_sparse)
model_sparse = sm.OLS(train_y_sparse, x_sparse).fit()
print(model_sparse.summary())

# Additional analysis

# Data visualization
advertising_spend = [223, 220, 15, 45, 76, 11, 32, 0]
actual_sales = [18.5, 25.5, 5.5, 9.4, 14.1, 5.8, 10.3, 6.1]

plt.title('Advertising Spend plotted against Actual Sales')
plt.xlabel('Advertising Spend')
plt.ylabel('Actual Sales')

plt.plot(advertising_spend, actual_sales, 'b.')  # Dots will be blue

mymodel_ad = np.poly1d(np.polyfit(advertising_spend, actual_sales, 1))
myline_ad = np.linspace(0, 223, 100)

plt.scatter(advertising_spend, actual_sales)
plt.plot(myline_ad, mymodel_ad(myline_ad))
plt.grid(True)
plt.show()

x_ad = sm.add_constant(advertising_spend)
model_ad = sm.OLS(actual_sales, x_ad).fit()
print(model_ad.summary())
