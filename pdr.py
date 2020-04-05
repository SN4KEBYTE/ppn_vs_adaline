import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# вспомогательная функция, которая визуально показывает границы решения для двумерных наборов данных
def plot_decision_regions(X, y, classifier, resolution=0.02, ax=None):
    if ax is None:
        ax = plt.gca()

    markers = ('s', 'x', 'o', '^', 'v')  # определяем маркеры
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')  # определяем цвета
    cmap = ListedColormap(colors[:len(np.unique(y))])  # создаем палитру цветов

    # выводим поверхность решения
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # находим минимум и максимум для 1 признака
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # находим минимум и максимум для 2 признака
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),  # получаем матрицу координат
                           np.arange(x2_min, x2_max, resolution))

    # идентифицируем метки классов и преобразуем в матрицу с размерностями как у xx1 и xx2
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T).reshape(xx1.shape)

    # рисуем контурный график
    ax.contourf(xx1, xx2, z, alpha=0.4, cmap=cmap)
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        ax.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)