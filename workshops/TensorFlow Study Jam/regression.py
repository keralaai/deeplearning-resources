import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset
from sklearn import metrics
import math
#sheet = pd.ExcelFile("regression.xls")
#print(sheet.sheet_names)cl
#df = sheet.parse("sheet1")
#df.to_csv('regression.csv', index=False)
insurance_dataframe = pd.read_csv('regression.csv', sep=',')
insurance_dataframe = insurance_dataframe.reindex(np.random.permutation(insurance_dataframe.index))
print(insurance_dataframe.describe())

#my_feature = insurance_dataframe[["X"]]
#print(my_feature)
#feature_columns = [tf.feature_column.numeric_column("X")]
#targets = insurance_dataframe["Y"]
#my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
#my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

#linear_regressor = tf.estimator.LinearRegressor(feature_columns=feature_columns, optimizer=my_optimizer)

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    features = {key:np.array(value) for key, value in dict(features).items()}
    #print(features)
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)
    if shuffle:
        ds = ds.shuffle(buffer_size=100)

    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

#k = my_input_function(my_feature, targets)
#print(k)
#_ = linear_regressor.train(input_fn = lambda:my_input_fn(my_feature, targets),steps=100)

#prediction_input_fn =lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)
#predictions = linear_regressor.predict(input_fn=prediction_input_fn)
#predictions = np.array([item['predictions'][0] for item in predictions])
#print(predictions)
#mean_squared_error = metrics.mean_squared_error(predictions, targets)
#root_mean_squared_error = math.sqrt(mean_squared_error)
#print("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
#print("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)

def train_model(learning_rate, steps, batch_size, input_feature="X"):


  periods = 10
  steps_per_period = steps / periods

  my_feature = input_feature
  my_feature_data = insurance_dataframe[[my_feature]]

  #print(my_feature_data)
  my_label = "Y"
  targets = insurance_dataframe[my_label]

  # Create feature columns.
  feature_columns = [tf.feature_column.numeric_column(my_feature)]

  # Create input functions.
  training_input_fn = lambda:my_input_fn(my_feature_data, targets, batch_size=batch_size)
  prediction_input_fn = lambda: my_input_fn(my_feature_data, targets, num_epochs=1, shuffle=False)

  # Create a linear regressor object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_regressor = tf.estimator.LinearRegressor(
      feature_columns=feature_columns,
      optimizer=my_optimizer
  )


  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print("RMSE (on training data):")
  root_mean_squared_errors = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    linear_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
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


  print("Model training finished.")
  print("Final RMSE (on training data): %0.2f" % root_mean_squared_error)
  test_input_fn = lambda: my_input_fn({'X':[53]}, [0], num_epochs=1, shuffle=False)
  predictions = linear_regressor.predict(input_fn=test_input_fn)
  print(np.array([item['predictions'][0] for item in predictions]))
train_model(learning_rate=0.0001, steps=10000, batch_size=1)




