import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from docx import Document  # Для работы с Word

def read_txt_file(file_path):
    """
    Считывает данные из текстового файла и преобразует их в массив numpy.
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
        raise ValueError(f"Ошибка при чтении текстового файла: {e}")

def read_docx_file(file_path):
    """
    Считывает данные из Word-документа и преобразует их в массив numpy.
    Предполагается, что данные находятся в таблице или разделены запятыми.
    """
    try:
        document = Document(file_path)
        data = []
        for table in document.tables:
            for row in table.rows:
                row_data = [cell.text.strip() for cell in row.cells]
                numeric_row = []
                for value in row_data:
                    try:
                        numeric_value = float(value)
                        numeric_row.append(numeric_value)
                    except ValueError:
                        continue  # Пропускаем некорректные значения
                if numeric_row:
                    data.append(numeric_row)
        if not data:
            raise ValueError("Word-документ не содержит корректных числовых данных.")
        return np.array(data, dtype=float)
    except Exception as e:
        raise ValueError(f"Ошибка при чтении Word-документа: {e}")

def read_excel_file(file_path):
    """
    Считывает данные из Excel-файла и преобразует их в массив numpy.
    """
    try:
        # Чтение файла с помощью pandas
        df = pd.read_excel(file_path, header=None)
        # Преобразование в числа, игнорируя ошибки
        data = df.apply(pd.to_numeric, errors='coerce').dropna().to_numpy()
        if data.size == 0:
            raise ValueError("Excel-файл не содержит корректных числовых данных.")
        return data
    except Exception as e:
        raise ValueError(f"Ошибка при чтении Excel-файла: {e}")

def read_data_from_file(file_path):
    """
    Определяет тип файла и вызывает соответствующую функцию для его чтения.
    """
    if file_path.endswith('.txt'):
        return read_txt_file(file_path)
    elif file_path.endswith('.docx'):
        return read_docx_file(file_path)
    elif file_path.endswith('.xlsx'):
        return read_excel_file(file_path)
    else:
        raise ValueError("Неподдерживаемый формат файла. Поддерживаются только .txt, .docx и .xlsx.")

def calculate_metrics(data, labels):
    """
    Вычисляет метрики для данных:
    1. Средние значения компонентов подсовокупности.
    2. Ковариации средних значений компонентов подсовокупности.
    3. Весовые коэффициенты смеси компонентов подсовокупности.
    """
    unique_labels = np.unique(labels)
    mean_values_per_cluster = np.array([
        data[labels == label].mean(axis=0) for label in unique_labels#Вычисление средних значений для каждого кластера
    ])
    covariance_matrices = [
        np.cov(data[labels == label], rowvar=False) for label in unique_labels#Вычисление ковариационных матриц для каждого кластера
    ]
    counts = np.array([np.sum(labels == label) for label in unique_labels])# Вычисление количества точек в каждом кластере
    weights = counts / len(labels)#Вычисление весовых коэффициентов для каждого кластера
    return {
        "mean_values_per_cluster": mean_values_per_cluster,
        "covariance_matrices": covariance_matrices,
        "weights": weights,
        "unique_labels": unique_labels
    }

def gaussian_pdf(x, mean, cov):
    """
    x: Точка данных (numpy.ndarray), для которой вычисляется плотность вероятности. Это вектор размерности D, где D — количество признаков.
    mean: Среднее значение (μ) кластера. Это вектор размерности D, представляющий центр кластера.
    cov: Ковариационная матрица (Σ) кластера. Это матрица размерности D×D, которая описывает взаимосвязь между признаками внутри кластера.
    Вычисляет гауссову плотность распределения вероятностей(g(x├|μ_i "," "Σ" _i ┤)=1/(("2π" )^(D/2) ├|Σ_i ├|1/2┤┤ ) "exp" {-1/2 ("x-" "μ" _i )^' Σ_i^"-1"  ("x-" "μ" _i )}).
    """
    d = len(mean)  # Размерность пространства
    diff = x - mean #Вычисление разности между точкой и средним значением
    cov_inv = np.linalg.inv(cov)  # Обратная ковариационная матрица
    det_cov = np.linalg.det(cov)  # Определитель ковариационной матрицы

    # Проверка на положительно определенную ковариационную матрицу
    if det_cov <= 0:
        raise ValueError("Ковариационная матрица должна быть положительно определенной.")
    
    exponent = -0.5 * np.dot(diff.T, np.dot(cov_inv, diff)) #Вычисление экспоненты 
    """
    np.dot(cov_inv, diff): Умножает обратную ковариационную матрицу на вектор разности diff.
    np.dot(diff.T, ...) : Умножает транспонированный вектор разности на результат предыдущего шага.
    -0.5 * ...: Масштабирует результат, так как в формуле стоит коэффициент
    """
    normalization = 1 / ((2 * np.pi) ** (d / 2) * np.sqrt(det_cov)) #Вычисление нормализующего множителя
    """
    (2 * np.pi) ** (d / 2): Вычисляет (2π) D/2, где D — размерность пространства.
    np.sqrt(det_cov): Вычисляет квадратный корень из определителя ковариационной матрицы.
    1 / (...): Берет обратное значение для нормализации.
    """
    return normalization * np.exp(exponent)#Вычисление плотности вероятности
"""
Плотность вероятности вычисляется как произведение нормализующего множителя и экспоненты:
Если normalization = 1 / (2 * np.pi) и exponent = -4, то
p(x | \mu, \Sigma) = (1 / (2 * np.pi)) * exp(-4).
"""

def weighted_gaussian_mixture_pdf(x, means, covariances, weights):
    """
    Вычисляет взвешенную сумму гауссовых плотностей распределения вероятностей(p(x├|λ┤)=∑_"i=1" ^R▒w_i  g(x├|μ_i "," "Σ" _i ┤)).
    x: Точка данных (numpy.ndarray), для которой вычисляется плотность вероятности. Это вектор размерности D, где D — количество признаков.
    means: Список средних значений (μi) для каждого кластера. Каждый элемент списка — это вектор размерности D, представляющий центр кластера.
    covariances: Список ковариационных матриц (Σi) для каждого кластера. Каждый элемент списка — это матрица размерности D×D, описывающая взаимосвязь между признаками внутри кластера.
    weights: Список весовых коэффициентов (wi) для каждого кластера. Каждый вес показывает относительный вклад кластера в общую смесь. Сумма всех весов должна быть равна 1.
    """
    pdf_value = 0.0 #Инициализация переменной для хранения результата
    for mean, cov, weight in zip(means, covariances, weights): #Вычисление взвешенной суммы гауссовых плотностей
        pdf_value += weight * gaussian_pdf(x, mean, cov)
    return pdf_value

def compute_ckx(means_list, covariances_list, weights_list, pn_list, x):
    """
    Вычисляет значение C^K(X) для заданной точки X.
    
    Параметры:
        means_list (list of np.ndarray): Список средних значений для каждого класса.
        covariances_list (list of np.ndarray): Список ковариационных матриц для каждого класса.
        weights_list (list of np.ndarray): Список весовых коэффициентов для каждого класса.
        pn_list (list of float): Априорные вероятности для каждого класса.
        x (np.ndarray): Точка данных, для которой вычисляется C^K(X).
    
    Возвращает:
        float: Значение C^K(X).
    """
    # Шаг 1: Вычисление числителя для каждого класса
    numerator = []
    for j in range(len(means_list)):
        fx_j = weighted_gaussian_mixture_pdf(x, means_list[j], covariances_list[j], weights_list[j])
        pn_j = pn_list[j]
        numerator.append(fx_j * pn_j)
    
    # Отладочная информация
    print("Числители:", numerator)
    
    # Шаг 2: Вычисление знаменателя
    denominator = sum(numerator)
    print("Знаменатель:", denominator)
    
    # Шаг 3: Вычисление C^K(X)
    ckx = [num / denominator for num in numerator]
    
    return ckx

if __name__ == "__main__":
    file_path = r"large_data.txt"  # Замените на путь к вашему файлу
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

    # Извлекаем параметры гауссовской смеси для каждого класса
    means_list = [metrics["mean_values_per_cluster"]]
    covariances_list = [metrics["covariance_matrices"]]
    weights_list = [metrics["weights"]]

    # Априорные вероятности (пример)
    pn_list = [0.5, 0.5]

    y = np.array([0.7, 0.7, 0.7, 0.7, 0.7])

    # Вычисляем C^K(X)
    ckx = compute_ckx(means_list, covariances_list, weights_list, pn_list, y)
    print("\nЗначения C^K(X) для каждого класса:")
    for j, value in enumerate(ckx):
        print(f"   Класс {j + 1}: {value:.6f}")

# Данные для двух классов
means_list = [
    [np.array([0, 0, 0, 0, 0]), np.array([1, 1, 1, 1, 1])],  # Средние значения для первого класса
    [np.array([5, 5, 5, 5, 5]), np.array([6, 6, 6, 6, 6])]   # Средние значения для второго класса
]
covariances_list = [
    [np.eye(5), np.eye(5)],  # Ковариационные матрицы для первого класса
    [np.eye(5), np.eye(5)]   # Ковариационные матрицы для второго класса
]
weights_list = [
    [0.7, 0.3],  # Весовые коэффициенты для первого класса
    [0.6, 0.4]   # Весовые коэффициенты для второго класса
]
pn_list = [0.7, 0.3]  # Априорные вероятности для каждого класса

# Пример точки для вычисления C^K(X)
x = np.array([2.5, 2.5, 2.5, 2.5, 2.5])

# Вычисляем C^K(X)
ckx = compute_ckx(means_list, covariances_list, weights_list, pn_list, x)
print("\nЗначения C^K(X) для каждого класса:")
for j, value in enumerate(ckx):
    print(f"   Класс {j + 1}: {value:.6f}")

# Параметры классов
means = [np.array([0, 0]), np.array([5, 5])]
covariances = [np.eye(2), np.eye(2)]

# Создаем сетку точек
x, y = np.mgrid[-5:10:.01, -5:10:.01]
pos = np.dstack((x, y))

# Вычисляем плотности вероятностей
rv1 = multivariate_normal(means[0], covariances[0])
rv2 = multivariate_normal(means[1], covariances[1])

# Строим графики
plt.figure(figsize=(8, 6))
plt.contour(x, y, rv1.pdf(pos), cmap='Blues', alpha=0.7)
plt.contour(x, y, rv2.pdf(pos), cmap='Reds', alpha=0.7)

# Отмечаем точку X
x_point = np.array([2.5, 2.5])
plt.scatter(x_point[0], x_point[1], color='black', label='Точка X')

plt.legend()
plt.title("Плотности вероятностей для двух классов")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()