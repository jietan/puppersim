from puppersim.reacher import reacher_kinematics
import math
import tensorflow as tf
import numpy as np
import pdb

# TODO
# start RL with supervised learning weights
# if ARS goes away that'd be bad

## Experiment results
# relu: 64x3 (9k) has 2.5mm error, 128x2 has 4-6mm error, 256x2 has 2mm error, 
# tanh: 64x2 has 7mm error. 64x3 has 3mm error, 32x3 has 5mm error, 32x2 (1.2k) has 7mm error 

N_train = 2000
N_eval = 10
batch_size = 512
hidden_units = 64
num_hidden_layers = 2
learning_rate = 1e-3
activation = 'tanh'


def generate_data(N, lower_bound=-0.5 * math.pi, upper_bound=0.5 * math.pi):
  X = []
  y = []
  for i in range(N):
    # target_angles = np.random.uniform(lower_bound, upper_bound, 3)
    target_angles = np.concatenate([
        np.random.uniform(-np.radians(45), np.radians(45), 1),
        np.random.uniform(0, np.radians(90), 2)
    ])
    target = reacher_kinematics.calculate_forward_kinematics_robot(
        target_angles)
    X.append(target)
    y.append(target_angles)
    # X.append(target_angles)
    # y.append(target)
  X = np.array(X)
  y = np.array(y)
  return tf.data.Dataset.from_tensor_slices((X, y))


def cartesian_loss(model, eval_data):
  avg = 0.0
  count = 0
  for batch in eval_data:
    # pdb.set_trace()
    targets = batch[0]
    model_estimated_angles = model(targets)
    for i, target in enumerate(targets):
      target_vec = tf.squeeze(target).numpy()
      count += 1
      model_angle = model_estimated_angles[i]
      fk_model = reacher_kinematics.calculate_forward_kinematics_robot(
          model_angle)
      avg += np.linalg.norm((fk_model - target_vec))
  print("eval count", count)
  return avg / count


train_data = generate_data(N_train).batch(batch_size).shuffle(
    buffer_size=1000).prefetch(buffer_size=10000)
eval_data = generate_data(N_eval).batch(1)

# def normalize_fn(output):
#   ub = np.array([-2 * math.pi, -1.5 * math.pi, -1.0 * math.pi]),
#   lb = np.array([2 * math.pi, 1.5 * math.pi, 1.0 * math.pi]),
#   return output * (ub - lb) * 0.5 + (ub + lb) * 0.5

model_layers = []
for i in range(num_hidden_layers):
  model_layers.append(tf.keras.layers.Dense(hidden_units, activation=activation))
model_layers.append(tf.keras.layers.Dense(3, activation='linear'))
model = tf.keras.models.Sequential(model_layers)

loss = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss=loss)
try:
  model.fit(train_data, validation_data=eval_data, epochs=1000)
finally:
  model.summary()
  print("\n\n----------------------------\nCartesian loss",
        cartesian_loss(model, eval_data), "\n\n")
