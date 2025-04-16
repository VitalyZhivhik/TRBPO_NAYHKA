import numpy as np
import pandas as pd

def read_data_from_file(file_path):
    """
    Считывает данные из файла и преобразует их в массив numpy.
    Игнорирует строки с некорректными данными.
    """
    try:
        # Чтение файла с помощью pandas
        df = pd.read_csv(file_path, delimiter=',', header=None)
        # Преобразование в числа, игнорируя ошибки
        data = df.apply(pd.to_numeric, errors='coerce').dropna().to_numpy()
        if data.size == 0:
            raise ValueError("Файл не содержит корректных числовых данных.")
        return data
    except Exception as e:
        raise ValueError(f"Ошибка при чтении файла: {e}")

def calculate_metrics(data, labels):
    """
    Вычисляет метрики для данных:
    1. Средние значения компонентов подсовокупности.
    2. Ковариации средних значений компонентов подсовокупности.
    3. Весовые коэффициенты смеси компонентов подсовокупности.
    """
    unique_labels = np.unique(labels)
    mean_values_per_cluster = np.array([
        data[labels == label].mean(axis=0) for label in unique_labels
    ])
    covariance_matrices = [
        np.cov(data[labels == label], rowvar=False) for label in unique_labels
    ]
    counts = np.array([np.sum(labels == label) for label in unique_labels])
    weights = counts / len(labels)
    return {
        "mean_values_per_cluster": mean_values_per_cluster,
        "covariance_matrices": covariance_matrices,
        "weights": weights,
        "unique_labels": unique_labels
    }

def gaussian_pdf(x, mean, cov):
    """
    Вычисляет гауссову плотность распределения вероятностей.
    
    Параметры:
        x (numpy.ndarray): Точка, в которой вычисляется плотность (вектор).
        mean (numpy.ndarray): Вектор средних значений.
        cov (numpy.ndarray): Ковариационная матрица.
    
    Возвращает:
        float: Значение гауссовой плотности.
    """
    d = len(mean)  # Размерность пространства
    diff = x - mean
    cov_inv = np.linalg.inv(cov)  # Обратная ковариационная матрица
    det_cov = np.linalg.det(cov)  # Определитель ковариационной матрицы
    
    # Проверка на положительно определенную ковариационную матрицу
    if det_cov <= 0:
        raise ValueError("Ковариационная матрица должна быть положительно определенной.")
    
    # Вычисление гауссовой плотности
    exponent = -0.5 * np.dot(diff.T, np.dot(cov_inv, diff))
    normalization = 1 / ((2 * np.pi) ** (d / 2) * np.sqrt(det_cov))
    return normalization * np.exp(exponent)

def weighted_gaussian_mixture_pdf(x, means, covariances, weights):
    """
    Вычисляет взвешенную сумму гауссовых плотностей распределения вероятностей.
    
    Параметры:
        x (numpy.ndarray): Точка, в которой вычисляется плотность (вектор).
        means (list of numpy.ndarray): Список векторов средних значений для каждого кластера.
        covariances (list of numpy.ndarray): Список ковариационных матриц для каждого кластера.
        weights (list of float): Список весовых коэффициентов для каждого кластера.
    
    Возвращает:
        float: Значение взвешенной суммы гауссовых плотностей.
    """
    pdf_value = 0.0
    for mean, cov, weight in zip(means, covariances, weights):
        pdf_value += weight * gaussian_pdf(x, mean, cov)
    return pdf_value

if __name__ == "__main__":
    file_path = r"C:\Users\Vitalik\Desktop\TRBPO_NAYHKA\large_data.txt"  # Замените на путь к вашему файлу
    try:
        data = read_data_from_file(file_path)
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        exit(1)

    try:
        # Предполагаем, что последний столбец содержит метки кластеров
        labels = data[:, -1].astype(int)
        data = data[:, :-1]
    except ValueError:
        print("Ошибка: Последний столбец должен содержать целочисленные метки кластеров.")
        exit(1)

    metrics = calculate_metrics(data, labels)

    print("\nРезультаты:")
    print("1. Средние значения компонентов подсовокупности:")
    for label, mean in zip(metrics["unique_labels"], metrics["mean_values_per_cluster"]):
        print(f"   Кластер {label}: {mean}")

    print("\n2. Ковариационные матрицы компонентов подсовокупности:")
    for label, cov in zip(metrics["unique_labels"], metrics["covariance_matrices"]):
        print(f"   Ковариационная матрица для кластера {label}:\n{cov}")

    print("\n3. Весовые коэффициенты смеси компонентов подсовокупности:")
    for label, weight in zip(metrics["unique_labels"], metrics["weights"]):
        print(f"   Кластер {label}: {weight:.4f}")
    # Пример использования функции gaussian_pdf
    print("\nВычисление гауссовой плотности:")
    x = np.array([0.5, 0.5, 0.5, 0.5, 0.5])  # Пример точки (должна иметь ту же размерность, что и данные)
    for label, mean, cov in zip(metrics["unique_labels"], metrics["mean_values_per_cluster"], metrics["covariance_matrices"]):
        pdf_value = gaussian_pdf(x, mean, cov)
        print(f"   Гауссова плотность для кластера {label} в точке {x}: {pdf_value:.6f}")
    # Пример использования функции weighted_gaussian_mixture_pdf
    print("\nВычисление взвешенной суммы гауссовых плотностей:")
    x = np.array([0.5, 0.5, 0.5, 0.5, 0.5])  # Пример точки (должна иметь ту же размерность, что и данные)
    mixture_pdf_value = weighted_gaussian_mixture_pdf(
        x,
        metrics["mean_values_per_cluster"],
        metrics["covariance_matrices"],
        metrics["weights"]
    )
    print(f"   Взвешенная сумма гауссовых плотностей для точки {x}: {mixture_pdf_value:.6f}")