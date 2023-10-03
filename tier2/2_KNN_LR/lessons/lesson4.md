# Lesson 4: Introduction to KNN

## Table of Contents
1. [Objective](#objective)
2. [What is KNN?](#what-is-knn)
3. [How Does KNN Work?](#how-knn-works)
4. [Applications of KNN](#applications)
5. [Advantages and Disadvantages](#advantages-disadvantages)
6. [Conclusion](#conclusion)

<a name="objective"></a>
## Objective
By the end of this lesson, you should have a foundational understanding of the K-Nearest Neighbors (KNN) algorithm and its applications in various domains.

<a name="what-is-knn"></a>
## What is KNN?
KNN is a non-parametric, lazy learning algorithm. Its purpose is to use a dataset in which the data points are separated into several classes to predict the classification of a new sample point.

<a name="how-knn-works"></a>
## How Does KNN Work?
- **Step 1:** Choose the number of `k` and a distance metric.
- **Step 2:** Find the `k` nearest neighbors of the sample point.
- **Step 3:** Assign the data point to the class where you have the most neighbors.

<a name="applications"></a>
## Applications of KNN
- Image Recognition
- Video Recognition
- Recommender Systems
- ... and many more.

<a name="advantages-disadvantages"></a>
## Advantages and Disadvantages
**Advantages:**
- No assumptions about data
- Simple algorithm to understand and interpret
- Versatility (can be used for classification, regression, search)

**Disadvantages:**
- Computationally expensive
- Requires feature scaling
- Sensitive to irrelevant features

<a name="conclusion"></a>
## Conclusion
KNN is a fundamental algorithm for classification problems. However, its simplicity can lead to challenges in processing large datasets. In the next lesson, we'll dive into implementing KNN from scratch and using `sklearn`.


<a name="references"></a>
## References
1. [Introduction to K-Nearest Neighbors](https://towardsdatascience.com/introduction-to-k-nearest-neighbors-3b534bb11d26)
2. [KNN Algorithm: A Practical Implementation On Python](https://www.analyticsvidhya.com/blog/2018/03/introduction-k-neighbours-algorithm-clustering/)
3. [K-Nearest Neighbors Algorithm in Python and Scikit-Learn](https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/)