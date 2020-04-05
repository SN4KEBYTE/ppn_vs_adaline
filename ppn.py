import numpy as np
from slnn import SLNN


class Perceptron(SLNN):
    def __init__(self, eta=0.01, n_iter=10):
        super().__init__(self, eta, n_iter)  # темп обучения, число эпох
        self.__w = None  # веса
        self.__errors = None  # число ошибок в каждой эпохе

    def fit(self, X, y):
        """Выполнить подгонку под тренировочные данные"""
        self.__w = np.zeros(1 + X.shape[1])  # инициализируем все веса нулями
        self.__errors = []  # число ошибок в каждой эпохе

        for _ in range(self.n_iter):
            errors_count = 0  # число ошибок в текущей эпохе

            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))  # значение для обновления весов
                self.__w[1:] += update * xi  # обновляем веса
                self.__w[0] += update  # не забываем про вес с нулевым индексом
                errors_count += int(update != 0.0)  # если персептрон правильно распознает метку класса,
                # веса остаются неизменными

            self.__errors.append(errors_count)  # запоминаем число ошибок в текущей эпохе

        return self

    def net_input(self, X):
        """Рассчитать чистый вход"""
        return np.dot(X, self.__w[1:]) + self.__w[0]  # вычисляем чистый вход как wT * x

    def predict(self, X):
        """Вернуть метку класса после единичного скачка"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)  # функция активации

    @property
    def errors(self):
        """Вернуть число ошибок в каждой эпохе"""
        return self.__errors
