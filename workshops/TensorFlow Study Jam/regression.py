"""
Linear Regression model using tensorflow estimators library. Using the same code from Google MLCC.
"""



import pandas as pd #importing pandas for processing dataset.
import numpy as np # importing numpy for array processing.
import tensorflow as tf # importing tensorflow
from tensorflow.data import Dataset # importing Dataset module for preparing dataset.
from sklearn import metrics # importing metrics from sklearn for calculating loss.
import math # importing math for some mathamatical operations.

insurance_dataframe = pd.read_csv('regression.csv', sep=',') #reading the csv dataset to a pandas dataframe
insurance_dataframe = insurance_dataframe.reindex(np.random.permutation(insurance_dataframe.index)) #randomizing the dataset
print(insurance_dataframe.describe()) # Printing a quick summary of our dataset, like mean, min, max..etc.


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """
    Input function for batching the dataset. This function return batches of features and targets.
    """
    features = {key:np.array(value) for key, value in dict(features).items()}
    
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs) #constructing a dataset object from our data, and then break our data into batches of batch_size, to be repeated for the specified number of epochs.
    if shuffle:
        # if shuffle is true, randomize the dataset.
        ds = ds.shuffle(buffer_size=100)

    features, labels = ds.make_one_shot_iterator().get_next() # getting the current batch of feature and targets
    return features, labels


def train_model(learning_rate, steps, batch_size, input_feature="X"):
  """
  Method for training the model. Learning rate, batch size and input features are the parameters. 
  """
  periods = 10
  steps_per_period = steps / periods

  my_feature = input_feature
  my_feature_data = insurance_dataframe[[my_feature]]

  
  my_label = "Y"
  targets = insurance_dataframe[my_label]

  # Create feature columns.
  feature_columns = [tf.feature_column.numeric_column(my_feature)]

  # Create input functions.
  training_input_fn = lambda:my_input_fn(my_feature_data, targets, batch_size=batch_size) #input function training 
  prediction_input_fn = lambda: my_input_fn(my_feature_data, targets, num_epochs=1, shuffle=False) # input function for prediction

  # Create a linear regressor object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate) # gradient descent optimiser
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_regressor = tf.estimator.LinearRegressor(feature_columns=feature_columns,optimizer=my_optimizer)


  # Train the model, but do so inside a loop so that we can periodically assess loss metrics.
  print("Training model...")
  print("RMSE (on training data):")
  root_mean_squared_errors = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    linear_regressor.train(input_fn=training_input_fn, steps=steps_per_period)
    # Take a break and compute predictions.
    predictions = linear_regressor.predict(input_fn=prediction_input_fn)
    predictions = np.array([item['predictions'][0] for item in predictions])

    # Compute loss.
    root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(predictions, targets))
    # Occasionally print the current loss.
    print("  period %02d : %0.2f" % (period, root_mean_squared_error))
    # Add the loss metrics from this period to our list.
    root_mean_squared_errors.append(root_mean_squared_error)


  print("Model training finished! ")
  print("Final RMSE (on training data): %0.2f" % root_mean_squared_error)
  

train_model(learning_rate=0.0001, steps=10000, batch_size=1) # training the model




