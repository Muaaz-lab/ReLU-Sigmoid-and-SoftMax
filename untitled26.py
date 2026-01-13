import numpy as np
def sigmoid(z): return 1 / (1 + np.exp(-z))
def softmax(z): return np.exp(z) / np.sum(np.exp(z))

z_binary = 1.5  # Ripley edge
print("Sigmoid (Ripley win prob):", sigmoid(z_binary))  # ~0.82

z_multi = np.array([2.1, 1.8, 3.2, 7.3])
print("Softmax probs:", softmax(z_multi))  # [0.21, 0.16, 0.63]

import numpy as np

# ReLU function
def relu(x):
    return np.maximum(0, x)

# Sample inputs (negative + positive)
inputs = np.array([-10, -5, -1, 0, 2, 5, 10])

# Apply ReLU
outputs = relu(inputs)

print("Input values :", inputs)
print("ReLU outputs :", outputs)

import numpy as np
import matplotlib.pyplot as plt

# Input range
x = np.linspace(-10, 10, 100)

# ReLU function
y = np.maximum(0, x)

# Plot
plt.figure()
plt.plot(x, y)
plt.xlabel("Input")
plt.ylabel("ReLU Output")
plt.title("ReLU Activation Function")
plt.show()
