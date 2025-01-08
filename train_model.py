import tensorflow as tf
import numpy as np

# Generate some dummy data
X = np.array([[1], [2], [3], [4], [5]], dtype=np.float32)
y = np.array([[2], [4], [6], [8], [10]], dtype=np.float32)

# Define the linear regression model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=200, verbose=0)

# Save the model
model.save('model.keras') # Modified line

print("Model training complete, model saved in 'model.keras'")