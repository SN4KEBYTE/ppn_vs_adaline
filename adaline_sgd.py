import numpy as np
from numpy.random import seed
from slnn import SLNN


class AdalineSGD(SLNN):
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        super().__init__(eta, n_iter)  # темп обучения, число эпох
        self.__w_initialized = False
        self.__shuffle = shuffle  # если True, перемешивает тренировочные данные в каждой эпохе для предотвращения
        # зацикливания
        self.__cost = None  # стоимость в каждой эпохе

        if random_state:
            seed(random_state)  # инициализируем генератор случайнах чисел

    def fit(self, X, y):
        """Выполнить подгонку под тренировочные данные"""
        self.__initialize_weights(X.shape[1])  # инициализируем веса
        self.__cost = []

        for i in range(self.n_iter):
            if self.__shuffle:
                X, y = self.__shuffle_data(X, y)

            cost = []

            for xi, target in zip(X, y):
                cost.append(self.__update_weights(xi, target))

            avg_cost = sum(cost) / len(y)
            self.__cost.append(avg_cost)

        return self

    def partial_fit(self, X, y):
        """Выполнить подгонку под тренировочные данные без повторной инициализации весов"""
        if not self.__w_initialized:
            self.__initialize_weights(X.shape[1])

        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self.__update_weights(xi, target)
        else:
            self.__update_weights(X, y)

        return self

    def __shuffle_data(self, X, y):
        """Перемешать тренировочные данные"""
        r = np.random.permutation(len(y))

        return X[r], y[r]

    def __initialize_weights(self, m):
        """Инициализировать веса нулями"""
        self.__w = np.zeros(1 + m)
        self.__w_initialized = True

    def __update_weights(self, xi, target):
        """Применить правило обучеия для обновления весов"""
        output = self.__net_input(xi)
        error = target - output

        self.__w[1:] += self.eta * xi.dot(error)
        self.__w[0] += self.eta * error

        cost = 0.5 * error ** 2

        return cost

    def __net_input(self, X):
        """Рассчитать чистый вход"""
        return np.dot(X, self.__w[1:]) + self.__w[0]

    def __activation(self, X):
        """Рассчитать линейную активацию"""
        return self.__net_input(X)

    def predict(self, X):
        """Вернуть метку класса после единичного скачка"""
        return np.where(self.__activation(X) >= 0.0, 1, -1)

    @property
    def costs(self):
        """Вернуть стоимость в каждой эпохе"""
        return self.__cost
