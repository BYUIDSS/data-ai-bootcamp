# Lesson 2: Implementing Linear Regression

## Table of Contents
1. [Objective](#objective)
2. [Math Behind Linear Regression](#math-behind)
3. [Cost Function and Gradient Descent](#cost-function)
4. [Implementing Linear Regression from Scratch](#from-scratch)
5. [Using `sklearn` for Linear Regression](#sklearn)
6. [Conclusion](#conclusion)
7. [Activities](#activities)
8. [References](#references)
9. [Go Back to Main Lesson](#main-lesson)
10. [Go Back Prev Lesson](#lesson-1)
11. [Go to Next Lesson](#lesson-3)

<a name="objective"></a>
## Objective
By the end of this lesson, you should be able to implement Linear Regression both from scratch and using the `sklearn` library in Python.

<a name="math-behind"></a>
## Math Behind Linear Regression
Linear Regression is represented by the equation:
y = beta_0 + beta_1*x + epsilon
Where:
- y is the dependent variable (what we're trying to predict),
- x is the independent variable (the input),
- beta_0 is the y-intercept,
- beta_1 is the slope of the line,
- epsilon is the error term.



<a name="cost-function"></a>
## Cost Function and Gradient Descent
The cost function, often represented as J(beta), measures the difference between the predicted values and the actual values. For Linear Regression, we commonly use the Mean Squared Error (MSE) as the cost function:
J(beta) = 1/2m * sum(h(beta)^(i) - y^(i))^2

Gradient Descent is an optimization algorithm used to minimize the cost function. It adjusts the parameters iteratively to find the best-fit line for the data.

<a name="from-scratch"></a>
## Implementing Linear Regression from Scratch
Here, we'll walk through the steps to implement Linear Regression using basic Python:

1. Initialize parameters \( \beta_0 \) and \( \beta_1 \) with random values.
2. Compute the predicted values using the current parameters.
3. Calculate the cost using the MSE formula.
4. Update the parameters using Gradient Descent.
5. Repeat steps 2-4 until the cost converges to a minimum value.

<a name="sklearn"></a>
## Using `sklearn` for Linear Regression
`sklearn` is a popular machine learning library in Python that provides simple tools for data analysis. To implement Linear Regression using `sklearn`:

1. Import necessary libraries.
2. Split the dataset into training and testing sets.
3. Create a Linear Regression model and fit it to the training data.
4. Predict the output for the testing data.
5. Evaluate the model's performance using metrics like MSE.

<a name="conclusion"></a>
## Conclusion
Implementing Linear Regression, whether from scratch or using libraries, provides valuable insights into the algorithm's workings. It's essential to understand the underlying principles to effectively apply it to real-world problems.

<a name="activities"></a>
## Activities
1. Implement Linear Regression from scratch on a sample dataset.
2. Use `sklearn` to create a Linear Regression model and compare its performance with the scratch implementation.

<a name="references"></a>
## References
1. [Linear Regression from Scratch - Towards Data Science](https://towardsdatascience.com/linear-regression-from-scratch-cd0dee067f72)
2. [Linear Regression in Python - Real Python](https://realpython.com/linear-regression-in-python/)
3. [`sklearn.linear_model.LinearRegression` - Official Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)



<a name="main-lesson"></a>
## Go Back to Main Lesson
[Back to Main Lesson](../main_lesson.md)

<a name="lesson-1"></a>
## Go to Prev Lesson
[Introduction to Linear Regression: Lesson 1](lesson1.md)


<a name="lesson-3"></a>
## Go to Next Lesson
[Evaluating and Improving Linear Regression Models: Lesson 3](lesson3.md)