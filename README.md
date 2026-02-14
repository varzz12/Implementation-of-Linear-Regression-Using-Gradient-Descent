# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe. 
2.Write a function computeCost to generate the cost function. 
3.Perform iterations og gradient steps with learning rate. 
4.Plot the Cost function using Gradient Descent and generate the required graph
## Program:
```
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Data
# -----------------------
x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 6, 8, 10], dtype=float)

# -----------------------
# Parameters
# -----------------------
w = 0.0
b = 0.0
alpha = 0.01
epochs = 100
n = len(x)

losses = []

# -----------------------
# Gradient Descent
# -----------------------
for _ in range(epochs):
    y_hat = w * x + b

    # Mean Squared Error
    loss = np.mean((y_hat - y) ** 2)
    losses.append(loss)

    dw = (2/n) * np.sum((y_hat - y) * x)
    db = (2/n) * np.sum(y_hat - y)

    w -= alpha * dw
    b -= alpha * db

# -----------------------
# Plots
# -----------------------
plt.figure(figsize=(12, 5))

# 1️⃣ Loss vs Iterations
plt.subplot(1, 2, 1)
plt.plot(losses, color="blue")
plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.title("Loss vs Iterations")

# 2️⃣ Regression Line
plt.subplot(1, 2, 2)
plt.scatter(x, y, color="red", label="Data")
plt.plot(x, w * x + b, color="green", label="Regression Line")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression Fit")
plt.legend()

plt.tight_layout()
plt.show()

print("Final weight (w):", w)
print("Final bias (b):", b)

/*
Developed by: VARUNA. R
RegisterNumber:25004445
*/
```

## Output:
<img width="1334" height="565" alt="image" src="https://github.com/user-attachments/assets/92a679d9-1695-4a41-8b24-06889abf1313" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
