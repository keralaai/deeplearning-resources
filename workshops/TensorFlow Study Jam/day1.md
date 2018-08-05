# Welcome to Tensorflow Study Jam : Day 1
<p align="center">
  <img src="https://scontent.fcok1-1.fna.fbcdn.net/v/t1.0-9/37838208_1074737562684398_3500528767216910336_n.png?_nc_cat=0&oh=9069a483cf036ee6b508d92e5edc12b8&oe=5C0325F3" alt="Keralaai tf study jam"/>
</p>

1. Intro to machine learning
2. Reducing Loss
3. Training and Testing sets
4. Logistic Regression
5. First Steps with Tensorflow
6. Tasks

## Intro to machine learning

Machine learning is the art of making sense of data !

1. It does things normal code can't do and it helps to reduce the time you spend for coding.   

2. In machine learning we don't need to tell the algorithm what to do, we only need to show it some examples.

### Types of machine learning algorithms:
- Supervised Learning

<p>
  <img src="https://github.com/GopikrishnanSasikumar/Intro-to-Machine-Learning-Workshop/blob/master/images-2.png"alt="supervised learning"/>
</p>

- Unsupervised Learning

<p>
  <img src="https://github.com/GopikrishnanSasikumar/Intro-to-Machine-Learning-Workshop/blob/master/unsupervised_learning.png" height="200" width="400" alt="Un-Supervised Learning"/>
</p>

- Reinforcement Learning

<p>
  <img src="https://github.com/GopikrishnanSasikumar/Intro-to-Machine-Learning-Workshop/blob/master/1*HvoLc50Dpq1ESKuejhICHg.png" alt="Reinforcement Learning"/>
</p>

In supervised learning we create “models”, a model is basically a function that takes in simple inputs and produces useful predictions. Here we have features and labels in the dataset.

### Features:

A feature is an input variable—the x variable in simple linear regression. A simple machine learning project might use a single feature, while a more sophisticated machine learning project could use millions of features.

Features of house price predicting ML model can be,

- Total number of rooms.
- Age of house
- Locality of the house

### Labels:

A label is the thing we're predicting. It can be the price of a product, class probability in a classification.

### Regression Vs Classification

A regression model is used to predict continuous values.

For example,

- The probability of Captain America Dying in Avengers 4.
- Price of houses in california.

A classification model predicts discrete values. It can make predictions that answer questions like,

- Is this an image of a cat or dog or Wade wilson.
- Pedicting whether a movie belongs to DC or Marvel(based on the dark screen may be) 

### Linear Regression

Linear regression is a method for finding the straight line or hyperplane that best fits a set of points.

The line equation is,

```
y = mx + b

```

In machine learning we use this convention instead,

```
y' = b + w1x1
```
Where,

- y' is the label we are predicting.
- b is the bias.
- w1 is the weight of feature 1. Weight is the same concept as the "slope" 
 in the traditional equation of a line.
- x1 is a feature (a known input).

To predict, just substitute the x1 values to the trained model.

A sophisticated model can use more than one features.

```
y' = b + w1x1 + w2x2 + w3x3 + .... + wNxN
```
1. **Training** a model simply means learning (determining) good values for all the weights and the bias from labeled examples. 

2. In supervised learning, a machine learning algorithm builds a model by examining many examples and attempting to find a model that minimizes loss; this process is called **empirical risk minimization**.

3. Loss is the penalty for a bad prediction. If the model's prediction is perfect, the loss is zero; otherwise, the loss is greater. 

4. The goal of training a model is to find a set of weights and biases that have low loss, on average, across all examples.

First we have to find the loss.

**L2 Loss/square loss** is a popular loss function. It is the given as

```
= the square of the difference between the label and the prediction
= (observation - prediction(x))2
= (y - y')2
```
**Mean square error (MSE)** is the average squared loss per example over the whole dataset.

<p align="center">
  <img src="https://github.com/GopikrishnanSasikumar/Intro-to-Machine-Learning-Workshop/blob/master/mse.png"alt="MSE"/>
</p>

## Reducing Loss
Reducing the loss is similar to the **"Hot and cold game"** kids play!

A Machine Learning model is trained by starting with an initial guess for the weights and bias and iteratively adjusting those guesses until learning the weights and bias with the lowest possible loss.

### Gradient Descent

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/1600/0*rBQI7uBhBKE8KT-X.png" height="300" width="500" alt="Gradient Descent"/>
</p>

### Learning Rate

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/1600/0*QwE8M4MupSdqA3M4.png" height="300" width="500" alt="Gradient Descent"/>
</p>


## Training and Testing Sets
<br>

<p align="center">
  <img src="http://blogs-images.forbes.com/janakirammsv/files/2017/04/ML-FaaS-1.png" height="360" width="740" alt="train-test divison"/>
</p>

1. Our goal is to create a machine learning model that generalizes well to new data. 

2. We train the model using a Training set and the test set act as a proxy for new data!


<p align="center">
  <img src="https://am207.github.io/2017/wiki/images/train-test.png" height="360" width="740" alt="train-test divison"/>
</p>

- **training set** — a subset to train a model.
- **test set** — a subset to test the trained model.

## Logistic Regression

1. Many problems require a probability estimate as output.
2. Logistic regression is an extremely efficient mechanism for calculating probabilities.

For example, consider that the probability of coconut falling on someone's head while walking through a field is 0.05.
Then over the year 18 accidents will happen in that field because of coconut!
```
P(thenga|day) = 0.05
coconut falling on head =
0.05*365 
~= 18
```


3. a sigmoid function, defined as follows, produces output that always falls between 0 and 1.

<p align="center">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSw89aMI5qlmImjji48z1agmJOhIJDSJvZgrHD9WPR4q783tMEMkw" height="100" width="150" alt="sigmoid function"/>
</p>

Where,

```
y = w1x1 + w2x2 + ... wNxN
```
and p is the predicted output.



<p align="center">
  <img src="https://developers.google.com/machine-learning/crash-course/images/SigmoidFunction.png" height="300" width="600" alt="sigmoid graph"/>
</p>



### Loss function for Logistic regression is Log Loss

<p align="center">
  <img src="https://github.com/GopikrishnanSasikumar/Intro-to-Machine-Learning-Workshop/blob/master/logloss.png" alt="sigmoid graph"/>
</p>

## First Steps with Tensorflow

Tensorflow is a computational framework for building machine learning models. TensorFlow provides a variety of different toolkits that allow you to construct models at your preferred level of abstraction. You can use lower-level APIs to build models by defining a series of mathematical operations. Alternatively, you can use higher-level APIs (like tf.estimator) to specify predefined architectures, such as linear regressors or neural networks.

Tensorflow consist of,

- A graph protocol buffer

- A runtime that executes the distributed graph

### Tensorflow hierarchy

|                                 |                                       |
|---------------------------------|---------------------------------------|
| Estimator (tf.estimator)        | High-level, OOP API.                  |  
| tf.layers/tf.losses/tf.metrics  | Libraries for common model components.|
| TensorFlow	                    | Lower-level APIs                      |

## Tasks

1. Make changes and try different hyper-parameters, learning rate and number of iterations in train_model() function to get a model with better accuracy.

2. Write another function ```test_model()``` in [regression.py](https://github.com/GopikrishnanSasikumar/deeplearning-resources/blob/master/workshops/TensorFlow%20Study%20Jam/regression.py) for testing the model with a new input.


