# AIH2024-S1_Linear_Regression_in_Healthcare
### AIH2024-S1: Predicting Diabetes Progression through Linear Regression in Python 
**Offered in the Class of Artificial Intelligence, March 13, 2024 | Faculty of New Sciences and Technologies, University of Tehran**

<div align="center">
<img src="Python.png" alt="Alt text" width="350" height="350">
</div>


# Intruductions
Linear regression is a powerful statistical technique used to model the relationship between a dependent variable (target) and one or more independent variables (features). It's widely used in various fields, including healthcare, economics, and social sciences, to make predictions and infer relationships between variables. The overall idea is quite straightforward: linear regression attempts to model the relationship between two (simple linear regression) or more (multiple linear regression) features or variables by fitting a linear equation to observed data. At its core, linear regression involves plotting a line through a set of data points in such a way that it minimizes the distance between each point and the line itself. This line can then be used to predict the value of a dependent variable Y based on the value(s) of one or more independent variables X. The straight line equation is:  
$`Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \beta_nX_n + \epsilon`$

Where:
- Y is the dependent variable (target).
- X1, X2, ..., Xn are the independent variables (features).
- β0 is the intercept term.
- β1, β2, ..., βn are the coefficients representing the impact of each independent variable on the dependent variable.
- ϵ is the error term, representing the difference between the observed and predicted values.

<div align="center">
<img src="images/Rabbit.webp" alt="Alt text" width="350" height="350">
</div>

The goal of linear regression is to estimate the coefficients (β) that minimize the sum of squared differences between the observed and predicted values of the dependent variable. This value is a cost function value referred to as **Mean Squared Error (MSE)**:
<p align="center">
$MSE = \frac{1}{n} \sum (y_i - \hat{y}_i)^2$  
</p>

The squared term in the MSE makes it differentiable, facilitating mathematical treatment, especially for optimization using gradient descent. We typically use optimization techniques such as gradient descent to find the values of the coefficients (βs) that minimize the MSE. Linear regression aims to find the best-fitting line through a set of data points in a way that minimizes the distance between the data points and the line itself. The best-fitting line, also known as the regression line, is the line that minimizes the sum of the squared differences **(residuals)** between the observed values (data points) and the values predicted by the linear model. The algorithm calculates the distance (residuals) between the actual data points and the predicted points on the line. The best-fit line will be the one where the sum of the squared residuals is the minimum (Least Squares Method).

<div align="center">
<img src="Residuals.png" alt="Alt text" width="450" height="450">
</div>

Imagine you want to predict a patient's blood pressure, which is the dependent variable. You might consider age and weight as independent variables because you hypothesize that these factors could influence blood pressure. In simple linear regression, you would predict blood pressure from just one of these variables (say, age). The model would help you understand how blood pressure varies with age. If you use both age and weight, you'd move to multiple linear regression, allowing you to predict blood pressure based on a combination of both factors. Linear regression involves finding the "best fit" line through the data points. In simple linear regression, "best fit" means that the sum of the squared differences between the observed values and the values predicted by the model is as small as possible, a method known as least squares.

<div align="center">
<img src="Plane fitted.png" alt="Alt text" width="450" height="450">
</div>

# Python Code
### Dataset Description
In this part of the lecture, we want to predict the quantitative measure of diabetes progression one year after the baseline using the linear regression model. The dataset is an open-source diabetes dataset available on the website of [North Carolina State University](https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt). 
This dataset consists of several columns representing different medical measurements:
- Number of Instances: 442
- Number of Attributes: The first 10 columns are numeric predictive values:
- AGE: Age in years
- SEX: Biological sex (encoded as 1 for male and 2 for female)
- BMI: Body mass index
- BP: Average blood pressure
- S1, S2, S3, S4, S5, S6: Various blood serum measurements
- Target: Column 11 is a quantitative measure of disease progression one year after baseline
- Y: Quantitative measure of diabetes progression one year after the baseline

More details: http://www4.stat.ncsu.edu/~boos/var.select/diabetes.html

### Make Data Ready

### - Import Required Packages

```python
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import math
import pandas as pd
```

- [matplotlib.pyplot](https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.html): A plotting library used for creating static, interactive, and animated visualizations in Python.
-  [%matplotlib inline](https://stackoverflow.com/questions/43027980/purpose-of-matplotlib-inline): This Jupyter notebook  command ensures that plots are displayed inline within the Jupyter notebook directly below the code cell that produced it.
-  [sklearn.model_selection.train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html): Splits arrays or matrices into random train and test subsets.
-  [sklearn.linear_model.LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html): Implements linear regression, a method to predict a target variable by fitting the best linear relationship with the predictor variables.
-  [sklearn.metrics.mean_squared_error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html): Calculates the mean squared error, a measure of the difference between the predicted and actual values.
-  [sklearn.metrics.r2_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html): Computes the coefficient of determination, indicating the proportion of the variance in the dependent variable predictable from the independent variables.
-  [math](https://docs.python.org/3/library/math.html): Provides access to mathematical functions defined by the C standard.
-  [pandas](https://pandas.pydata.org/): A library providing high-performance, easy-to-use data structures, and data analysis tools.
### - Load the data

```python
# Load the diabetes dataset
df = pd.read_csv("https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt",sep="\t")
df.head()
```
- **df.head()**: Displays the first five rows of the dataframe df, giving a preview of the dataset's structure and the initial data points.

### - Exploring the dataset
```python
df.info()
```

```python
df.describe()
```

```python
df.describe()
```

- **df.info()**: Provides a concise summary of the dataframe, including the number of entries, the number of non-null entries per column, and the datatype of each column.
- **df.describe()**: Generates descriptive statistics that summarize the central tendency, dispersion, and shape of the dataset's distribution, excluding NaN values.
- **df.isna().sum()**: Calculates the number of missing (NaN) values in each column of the dataframe, which helps in identifying if any data preprocessing like filling missing values is needed.

### - Correlation

To determine which features are most relevant for linear regression with the target variable, we can perform feature selection or ranking. One common method is to use the **correlation coefficient** to assess the linear relationship between each independent variable and the dependent variable. In Python, you can use the **corr()** method from Pandas to compute correlation coefficients, and then visualize these correlations using **seaborn's heatmap** for a more intuitive understanding. Here's a code snippet to demonstrate this:

```python
import seaborn as sns

correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f")
plt.show()

# To specifically look at the correlation with the target variable 'Y'
print(correlation_matrix['Y'].sort_values(ascending=False))
```

```python
# Use only The first feature (BMI) as variable for predicting target Y
diabetes_X = df[["BMI"]]
diabetes_y = df[["Y"]]
```

### - Train-Test Split

```python
# Split the data into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(diabetes_X,diabetes_y,test_size=0.2,random_state=42 )
```

- X_train and X_test are the independent variables for training and testing, respectively, while y_train and y_test are the corresponding dependent variables.
- The test_size=0.2 specifies that 20% of the data is used for testing.
- random_state=42 ensures that the splits are reproducible; using the same random state will produce the same split each time.

### - Implement LinearRegression 

```python
# Create linear regression object
from sklearn.linear_model import LinearRegression
regr = LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)
``` 
- Imports the **LinearRegression class**.
- Creates an instance of LinearRegression, named **regr**.
- Fits the model to the training data (X_train, y_train) to learn the coefficients.
- Predicts the target (y) for the testing set (X_test) using the learned model, storing the predictions in y_pred.

### - Evaluation 

```python
# print coeff and intercept
print(regr.coef_)
print(regr.intercept_)
```

```python
BMI = 25
print(f"Your predicted sugar level is {regr.predict([[BMI]])}")
```

```python
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print(f"Mean squared error:{mean_squared_error(y_test, y_pred): .2f}")
print(f"Root Mean squared error: {math.sqrt(mean_squared_error(y_test, y_pred)) :.2f}")
# Explained variance score: 1 is perfect prediction
print(f'Variance score: {r2_score(y_test, y_pred):.2f}')
```

### - Visualization

```python
# Plot outputs
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)

plt.xticks(ticks=[10, 20, 30,40,50])
plt.yticks(ticks=[100, 200, 300])
plt.xlabel("BMI (scaled)")
plt.ylabel("Diabetes progression")
plt.show()
```

<div align="center">
<img src="Plane fitted.png" alt="Alt text" width="450" height="450">
</div>
