# -*- coding: utf-8 -*-
# Python 3

# Исходный код к уроку 1.
# Построение двухслойной нейронный сети для классификации цветков ириса

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from tqdm import tqdm
from sklearn.model_selection import train_test_split

warnings.simplefilter(action='ignore', category=FutureWarning)


def to_one_hot(Y):
    n_col = np.amax(Y) + 1
    binarized = np.zeros((len(Y), n_col))
    for i in range(len(Y)):
        binarized[i, Y[i]] = 1.
    return binarized


def from_one_hot(Y):
    arr = np.zeros((len(Y), 1))

    for i in range(len(Y)):
        l = layer2[i]
        for j in range(len(l)):
            if l[j] == 1:
                arr[i] = j + 1
    return arr


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def ReLU(x):
    return x * (x > 0)


def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))


def ReLU_deriv(x):
    return x > 0


def tanh_deriv(x):
    return 1 - np.square(x)


def normalize(X, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)


def main():

    iris_data = pd.read_csv("iris.csv")
    iris_data = iris_data.reset_index()

    # g = sns.pairplot(iris_data.drop("index", axis=1), hue="species")

    iris_data['species'].replace(['setosa', 'virginica', 'versicolor'], [0, 1, 2], inplace=True)

    columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    x = pd.DataFrame(iris_data, columns=columns)
    x = normalize(x.as_matrix())

    columns = ['species']
    y = pd.DataFrame(iris_data, columns=columns)
    y = y.as_matrix()
    y = y.flatten()
    y = to_one_hot(y)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

    w0 = 2 * np.random.random((4, 5)) - 1  # для входного слоя   - 4 входа, 3 выхода
    w1 = 2 * np.random.random((5, 3)) - 1  # для внутреннего слоя - 5 входов, 3 выхода

    n = 0.01
    errors = []

    for _ in tqdm(range(1000000)):
        # прямое распространение(feed forward)
        layer0 = X_train
        layer1 = sigmoid(np.dot(layer0, w0))
        layer2 = sigmoid(np.dot(layer1, w1))

        # обратное распространение(back propagation) с использованием градиентного спуска
        layer2_error = y_train - layer2
        layer2_delta = layer2_error * sigmoid_deriv(layer2)

        layer1_error = layer2_delta.dot(w1.T)
        layer1_delta = layer1_error * sigmoid_deriv(layer1)

        w1 += layer1.T.dot(layer2_delta) * n
        w0 += layer0.T.dot(layer1_delta) * n

        error = np.mean(np.abs(layer2_error))
        errors.append(error)
        accuracy = (1 - error) * 100

    plt.plot(errors)
    plt.xlabel('Обучение')
    plt.ylabel('Ошибка')
    plt.show()  # расскоментируйте, чтобы посмотреть

    print(f'Точность нейронной сети {round(accuracy, 2)}%')


if __name__ == '__main__':
    main()
