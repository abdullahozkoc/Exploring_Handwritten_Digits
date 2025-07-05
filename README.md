# Exploring Handwritten Digits Application

This project showcases an application of unsupervised learning (dimensionality reduction) and supervised learning (classification) algorithms using the **handwritten digits dataset** from the Scikit-learn library.

-----

## About the Project

Handwritten digit recognition is a classic problem in machine learning and serves as an excellent starting point for understanding both **dimensionality reduction** techniques and applying **classification algorithms**. This application demonstrates a step-by-step approach to visualize data, reduce its dimensions, and train a classification model.

-----

## Libraries Used

  * **`sklearn.datasets`**: To load the handwritten digits dataset.
  * **`matplotlib.pyplot`**: For data visualization.
  * **`sklearn.manifold.Isomap`**: An unsupervised learning algorithm for dimensionality reduction.
  * **`sklearn.model_selection`**: To split the data into training and test sets.
  * **`sklearn.naive_bayes.GaussianNB`**: A supervised learning algorithm for digit classification.
  * **`sklearn.metrics`**: To evaluate model performance (accuracy score and confusion matrix).
  * **`seaborn`**: To visualize the confusion matrix.

-----

## Application Steps

1.  **Data Loading and Exploration**: The handwritten digits dataset is loaded using the `load_digits()` function. The shape of the images in the dataset is examined, and the first 100 digits are visualized along with their corresponding labels.

2.  **Unsupervised Learning (Dimensionality Reduction)**:

      * **Isomap**: The Isomap algorithm is used to demonstrate how the data is reduced from a high-dimensional space (64 dimensions) to a 2-dimensional space.
      * The projected data is visualized with a scatter plot, colored by digit labels, to observe how different digits cluster in the 2D space.

3.  **Supervised Learning (Classification)**:

      * **Data Splitting**: The dataset is split into training (`Xtrain`, `ytrain`) and test (`Xtest`, `ytest`) sets for model training and evaluation.
      * **Model Training**: A `GaussianNB` (Gaussian Naive Bayes) classification model is trained on `Xtrain` and `ytrain`.
      * **Prediction and Evaluation**: The trained model makes predictions (`y_model`) on the `Xtest` set. The model's performance is measured using the **accuracy score** via `accuracy_score`.
      * **Confusion Matrix**: A **confusion matrix** is generated and visualized with `seaborn` to provide a more detailed look at which digits the model classified correctly or incorrectly.

-----

## How to Run

You can run this Jupyter Notebook in a Colab environment or on a local Jupyter Notebook server. Make sure you have the necessary libraries installed:

```bash
pip install scikit-learn matplotlib seaborn
```

Then, simply run the cells in the notebook sequentially to execute the entire application.

-----

