import numpy as np
from slnn import SLNN


class AdalineGD(SLNN):
    def __init__(self, eta=0.01, n_iter=10):
        super().__init__(eta, n_iter)  # темп обучения, число эпох
        self.__w = None  # веса
        self.__cost = []  # стоимость

    def fit(self, X, y):
        """Выполнить подгонку под тренировочные данные"""
        self.__w = np.zeros(1 + X.shape[1])  # инициализуруем весовой вектор

        for _ in range(self.n_iter):
            output = self.__net_input(X)  # чистый вход
            errors = y - output  # отклонения расчетных результатов от истинных меток классов
            self.__w[1:] += self.eta * X.T.dot(errors)  # обновляем веса, вычислив градиент
            self.__w[0] += self.eta * errors.sum()  # не забываем про вес с нулевым индексом
            cost = (errors ** 2).sum() / 2.0  # считыаем стоимость как SSE

            self.__cost.append(cost)

        return self

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

