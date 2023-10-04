# Lesson 3: Evaluating and Improving Linear Regression Models

## Table of Contents
1. [Objective](#objective)
2. [Evaluation Metrics for Regression](#evaluation-metrics)
3. [Overfitting and Underfitting](#overfitting-underfitting)
4. [Improving Model Performance](#improving-performance)
5. [Conclusion](#conclusion)
6. [Activities](#activities)
7. [References](#references)
9. [Go Back to Main Lesson](#main-lesson)
10. [Go Back Prev Lesson](#lesson-2)
11. [Go to Next Lesson](#lesson-4)


<a name="objective"></a>
## Objective
By the end of this lesson, you should understand how to evaluate the performance of Linear Regression models and be familiar with techniques to enhance their predictions.

<a name="evaluation-metrics"></a>
## Evaluation Metrics for Regression
Evaluating the performance of a regression model is crucial to understand its effectiveness. Common metrics include:

- **Mean Absolute Error (MAE):** Represents the average of the absolute differences between predicted and actual values.
- **Mean Squared Error (MSE):** Represents the average of the squared differences between predicted and actual values.
- **Root Mean Squared Error (RMSE):** Square root of MSE, offering a more interpretable metric as it's in the same unit as the target variable.

<a name="overfitting-underfitting"></a>
## Overfitting and Underfitting
Understanding the balance between overfitting and underfitting is crucial:

- **Overfitting:** When the model performs exceptionally well on the training data but poorly on unseen data. It's too complex and captures noise.
- **Underfitting:** When the model performs poorly on both training and unseen data. It's too simple to capture the underlying patterns.

<a name="improving-performance"></a>
## Improving Model Performance
Several techniques can enhance the performance of a Linear Regression model:

- **Feature Scaling:** Standardizing or normalizing features to bring them to a similar scale.
- **Feature Selection:** Choosing only relevant features that contribute to the model's predictive power.
- **Regularization:** Techniques like Ridge and Lasso regression that add a penalty to the loss function to prevent overfitting.

<a name="conclusion"></a>
## Conclusion
Evaluating and improving the performance of a Linear Regression model is an iterative process. By understanding the metrics and techniques, you can refine your model to make more accurate predictions.

<a name="activities"></a>
## Activities
1. Calculate MAE, MSE, and RMSE for a Linear Regression model on a sample dataset.
2. Experiment with feature scaling and regularization techniques using `sklearn` and observe the changes in model performance.

<a name="references"></a>
## References
1. [Metrics to Evaluate your Machine Learning Algorithm](https://towardsdatascience.com/metrics-to-evaluate-your-machine-learning-algorithm-f10ba6e38234)
2. [Overfitting vs. Underfitting: A Complete Example](https://towardsdatascience.com/overfitting-vs-underfitting-a-complete-example-d05dd7e19765)
3. [Regularization in Machine Learning](https://www.datacamp.com/community/tutorials/tutorial-ridge-lasso-elastic-net)

<a name="main-lesson"></a>
## Go Back to Main Lesson
[Back to Main Lesson](../main_lesson.md)

<a name="lesson-2"></a>
## Go to Prev Lesson
[Implementing Linear Regression: Lesson 2](lesson2.md)


<a name="lesson-4"></a>
## Go to Next Lesson
[Introduction to KNN: Lesson 4](lesson4.md)