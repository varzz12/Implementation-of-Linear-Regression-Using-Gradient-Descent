# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize Parameters:
    theta1 and theta2
2.Compute Hypothesis:
For each training example, compute predicted profit
3.Update Parameters (Gradient Descent Rule) 
4Predict Profit:After training, use final theta1 and theta2 

## Program:
```
/*
import numpy as np
import matplotlib.pyplot as plt

x=np.array([10,20,30,40,50],dtype=float)
y=np.array([2,4,6,8,10],dtype=float)
w=0.0
b=0.0
alpha=0.01
epochs=100
n=len(x)

losses=[]

for i in range(epochs):
    y_hat=w*x+b
    loss=np.mean((y_hat-y)**2)
    losses.append(loss)
    
    dw=(2/n)*np.sum((y_hat-y)*x)
    db=(2/n)*np.sum(y_hat-y)
    w-=alpha*dw
    b-=alpha*db
    
plt.figure(figsize=(12,5))
plt.subplot(1,2,2)
plt.plot(losses,color="blue")
plt.xlabel=("Population(hundred)")
plt.ylabel=("Profit of a city(crore)")
plt.title("Population Vs Profit of a city")
    
plt.tight_layout()
plt.show()
    
print("Final Weight(w):",w)
print("Final bias(b):",b)

Developed by: VARUNA. R
RegisterNumber:25004445
*/
```

## Output:
<img width="815" height="688" alt="Screenshot 2026-01-27 094236" src="https://github.com/user-attachments/assets/8926ca6d-9da5-4a91-a1c7-20f896f6bc43" />




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
