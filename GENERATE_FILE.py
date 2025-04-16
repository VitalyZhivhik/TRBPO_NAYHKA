import numpy as np

def generate_large_file(file_path, num_samples=10000, num_features=5, num_clusters=3):
    """
    Генерирует большой файл с случайными данными и метками кластеров.
    
    Параметры:
        file_path (str): Путь для сохранения файла.
        num_samples (int): Количество строк (объектов) в файле.
        num_features (int): Количество признаков (столбцов) для каждого объекта.
        num_clusters (int): Количество кластеров (меток).
    """
    # Генерация случайных данных
    data = np.random.rand(num_samples, num_features)  # Случайные числа от 0 до 1
    labels = np.random.randint(0, num_clusters, size=num_samples)  # Метки кластеров (0, 1, ..., num_clusters-1)
    
    # Добавление меток к данным
    data_with_labels = np.hstack((data, labels.reshape(-1, 1)))
    
    # Сохранение данных в файл
    np.savetxt(file_path, data_with_labels, delimiter=',', fmt='%.6f')

if __name__ == "__main__":
    file_path = "large_data.txt"  # Путь для сохранения файла
    generate_large_file(file_path, num_samples=10000, num_features=5, num_clusters=3)
    print(f"Файл успешно создан: {file_path}")