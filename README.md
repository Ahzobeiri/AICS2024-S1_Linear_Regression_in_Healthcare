# AIH2024-S1_Linear_Regression_in_Healthcare
### AIH2024-S1: Predicting Diabetes Progression through Linear Regression in Python 
**Offered in the Class of Artificial Intelligence, March 13, 2024 | Faculty of New Sciences and Technologies, University of Tehran**

<div align="center">
<img src="Python.png" alt="Alt text" width="256" height="256">
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

`MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2`

This value is a cost function value referred to as Mean Squared Error (MSE):  

`MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2`



$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$  
The squared term in the MSE makes it differentiable, facilitating mathematical treatment, especially for optimization using gradient descent. To find the values of the coefficients (βs) that minimize the MSE, we typically use optimization techniques such as gradient descent.
Imagine you want to predict a patient's blood pressure, which is the dependent variable. You might consider age and weight as independent variables because you hypothesize that blood pressure could be influenced by these factors.




```python
# This is a sample Python code
def hello_world():
    print("Hello, world!")

hello_world()
```


