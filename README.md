<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Activation Functions Explanation (Sigmoid, Softmax, ReLU)</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.7;
            margin: 30px;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        pre {
            background-color: #f4f4f4;
            padding: 12px;
            overflow-x: auto;
        }
        code {
            background-color: #f4f4f4;
            padding: 3px;
        }
        ul {
            margin-left: 20px;
        }
    </style>
</head>
<body>

<h1>Detailed Explanation of Activation Functions in Python</h1>

<p>
This document explains the Python code that demonstrates three important
activation functions used in Machine Learning and Deep Learning:
<b>Sigmoid</b>, <b>Softmax</b>, and <b>ReLU</b>.
These functions are commonly used in neural networks to transform raw model
outputs into meaningful values.
</p>

<hr>

<h2>1. Importing NumPy</h2>

<pre><code>import numpy as np</code></pre>

<p>
NumPy is a numerical computing library used for:
</p>
<ul>
    <li>Mathematical operations</li>
    <li>Working with arrays and vectors</li>
    <li>Efficient computation of exponential functions</li>
</ul>

<p>
All activation functions in this code rely on NumPy for fast calculations.
</p>

<hr>

<h2>2. Sigmoid Activation Function</h2>

<pre><code>def sigmoid(z):
    return 1 / (1 + np.exp(-z))</code></pre>

<p>
The Sigmoid function converts any real number into a value between <b>0 and 1</b>.
It is commonly used for <b>binary classification</b> problems.
</p>

<p><b>Mathematical Formula:</b></p>
<pre>σ(z) = 1 / (1 + e⁻ᶻ)</pre>

<h3>Example Usage</h3>

<pre><code>z_binary = 1.5
print("Sigmoid (Ripley win prob):", sigmoid(z_binary))</code></pre>

<p>
Here, the input value <code>z = 1.5</code> represents a positive advantage.
The output (~0.82) can be interpreted as an <b>82% probability</b>.
</p>

<p>
<b>Interpretation:</b>  
A higher sigmoid value means a higher likelihood of a positive outcome.
</p>

<hr>

<h2>3. Softmax Activation Function</h2>

<pre><code>def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))</code></pre>

<p>
The Softmax function converts a vector of raw scores into a
<b>probability distribution</b>.
All output values:
</p>

<ul>
    <li>Are between 0 and 1</li>
    <li>Sum up to exactly 1</li>
</ul>

<p>
Softmax is mainly used in <b>multi-class classification</b>.
</p>

<h3>Example Usage</h3>

<pre><code>z_multi = np.array([2.1, 1.8, 3.2, 7.3])
print("Softmax probs:", softmax(z_multi))</code></pre>

<p>
Each value represents the probability of a different class.
The class with the highest score (7.3) gets the highest probability.
</p>

<p>
<b>Interpretation:</b>  
Softmax helps the model decide which class is the most likely among many options.
</p>

<hr>

<h2>4. ReLU (Rectified Linear Unit) Function</h2>

<pre><code>def relu(x):
    return np.maximum(0, x)</code></pre>

<p>
ReLU is one of the most widely used activation functions in deep learning.
It works as follows:
</p>

<ul>
    <li>If input is negative → output is 0</li>
    <li>If input is positive → output remains unchanged</li>
</ul>

<p><b>Mathematical Representation:</b></p>
<pre>ReLU(x) = max(0, x)</pre>

<h3>Applying ReLU to Sample Inputs</h3>

<pre><code>inputs = np.array([-10, -5, -1, 0, 2, 5, 10])
outputs = relu(inputs)</code></pre>

<p>
Negative values are blocked, while positive values pass through.
This helps neural networks ignore irrelevant or harmful signals.
</p>

<hr>

<h2>5. Visualizing ReLU Using a Graph</h2>

<pre><code>import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)
y = np.maximum(0, x)

plt.figure()
plt.plot(x, y)
plt.xlabel("Input")
plt.ylabel("ReLU Output")
plt.title("ReLU Activation Function")
plt.show()</code></pre>

<p>
This section plots the ReLU function:
</p>

<ul>
    <li>The left side (negative inputs) is flat at zero</li>
    <li>The right side (positive inputs) is a straight line</li>
</ul>

<p>
The graph visually explains how ReLU activates only when the input becomes positive.
</p>

<hr>

<h2>6. Summary</h2>

<ul>
    <li><b>Sigmoid</b>: Used for binary classification, outputs probabilities between 0 and 1</li>
    <li><b>Softmax</b>: Used for multi-class classification, outputs probability distribution</li>
    <li><b>ReLU</b>: Used in hidden layers, removes negative values and improves learning speed</li>
</ul>

<p>
These activation functions are fundamental components of modern neural networks
and are essential for transforming raw model outputs into meaningful decisions.
</p>

</body>
</html>
