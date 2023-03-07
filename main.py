import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Salary_Data.csv")

for i in range(len(data)):
    x = data.iloc[i].YearsExperience
    y = data.iloc[i].Salary

    plt.scatter(x, y)


def loss_function(m, b, points):
    total_loss = 0
    for i in range(len(points)):
        x = points.iloc[i].YearsExperience
        y = points.iloc[i].Salary

        total_loss += (y - (m*x+b))**2
    return total_loss / len(points)


def gradient_decent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    n = len(points)

    for i in range(n):
        x = points.iloc[i].YearsExperience
        y = points.iloc[i].Salary

        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))

    m = m_now - L * m_gradient
    b = b_now - L * b_gradient

    return m, b


m = 0
b = 0
L = 0.0001
epochs = 2500

for i in range(epochs):
    m, b = gradient_decent(m, b, data, L)
    loss = loss_function(m, b, data)

    if i % 100 == 0:
        print(f"Epoch: {i}\tLoss: {loss}")


plt.plot(list(range(15)), [m * x + b for x in range(15)], color="red")

plt.show()
