import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ppn import Perceptron
from adaline_gd import AdalineGD
from adaline_sgd import AdalineSGD
from pdr import plot_decision_regions


data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'


def main():
    data = pd.read_csv(data_url, header=None)

    # выведем последние 5 строк, чтобы убедиться, что данные были загружены правильно
    # print(data.tail())

    # выделим из 100 тренировочных образцов столбец первого признака (длина чашелистика)
    # и столбец третьего признака (длина лепестка)
    X = data.iloc[0:100, [0, 2]].values

    # выделим первые 100 меток классов и преобразуем их в целочисленные
    y = data.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)

    # визуализируем образцы при помощи диаграммы рассеяния
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='щетинистый')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='разноцветный')
    plt.xlabel('длина чашелистика')
    plt.ylabel('длина лепестка')
    plt.title('Образцы')
    plt.legend(loc='upper left')
    plt.savefig('data.pdf')
    plt.close()

    # обучим персептрон
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)

    # для обучения adaline выполним стандартизацию данных
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    # обучим 2 вида adaline
    ada_gd = AdalineGD(0.01, 15)
    ada_gd.fit(X_std, y)

    ada_sgd = AdalineSGD(0.01, 15, random_state=1)
    ada_sgd.fit(X_std, y)

    # построим графики ошибок
    fig, ax = plt.subplots(1, 3, figsize=(24, 8))
    ax[0].plot(range(1, len(ppn.errors) + 1), ppn.errors, marker='o')
    ax[0].set_xlabel('Эпохи')
    ax[0].set_ylabel('Число случаев ошибочной классификации')
    ax[0].set_title('Персептрон: график зависимости числа случаев ошибочной классификации\nот числа эпох')

    ax[1].plot(range(1, len(ada_gd.costs) + 1), np.log10(ada_gd.costs), marker='o')
    ax[1].set_xlabel('Эпохи')
    ax[1].set_ylabel('lg(Сумма квадратичных ошибок)')
    ax[1].set_title('ADALINE GD: график зависимости lg(SSE)\nот числа эпох')

    ax[2].plot(range(1, len(ada_sgd.costs) + 1), np.log10(ada_sgd.costs), marker='o')
    ax[2].set_xlabel('Эпохи')
    ax[2].set_ylabel('lg(Сумма квадратичных ошибок)')
    ax[2].set_title('ADALINE SGD: график зависимости lg(SSE)\nот числа эпох')

    plt.savefig('errors_comparison.pdf')
    plt.close(fig)

    # сравним результаты
    fig, ax = plt.subplots(1, 3, figsize=(24, 8))
    plot_decision_regions(X, y, ppn, ax=ax[0])
    ax[0].set_xlabel('длина чашелистика (см)')
    ax[0].set_ylabel('длина лепестка (см)')
    ax[0].set_title('Персептрон')
    ax[0].legend(loc='upper left')

    plot_decision_regions(X_std, y, ada_gd, ax=ax[1])
    ax[1].set_xlabel('длина чашелистика (см)')
    ax[1].set_ylabel('длина лепестка (см)')
    ax[1].set_title('AdalineGD')
    ax[1].legend(loc='upper left')

    plot_decision_regions(X_std, y, ada_sgd, ax=ax[2])
    ax[2].set_xlabel('длина чашелистика (см)')
    ax[2].set_ylabel('длина лепестка (см)')
    ax[2].set_title('AdalineSGD')
    ax[2].legend(loc='upper left')

    plt.savefig('result_comparison.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
