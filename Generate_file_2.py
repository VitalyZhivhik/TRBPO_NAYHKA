import numpy as np

def generate_large_txt_file(file_path, num_points=100000, num_features=5, num_clusters=3):
    """
    Генерирует большой текстовый файл с данными для тестирования программы.
    
    Параметры:
        file_path (str): Путь к файлу, который будет создан.
        num_points (int): Количество точек данных.
        num_features (int): Количество признаков (столбцов) для каждой точки.
        num_clusters (int): Количество кластеров (меток).
    """
    # Генерация случайных данных
    data = np.random.rand(num_points, num_features)  # Случайные значения от 0 до 1
    labels = np.random.randint(0, num_clusters, size=num_points)  # Метки кластеров
    
    # Объединение данных и меток
    full_data = np.hstack((data, labels.reshape(-1, 1)))
    
    # Сохранение данных в текстовый файл
    np.savetxt(file_path, full_data, delimiter=',', fmt='%.6f')
    print(f"Файл '{file_path}' успешно создан. Размер: {num_points} строк, {num_features + 1} столбцов.")

# Пример использования
if __name__ == "__main__":
    file_path = "C:/Users/Vitalik/Desktop/TRBPO_NAYHKA-main/TRBPO_NAYHKA-main/large_data_2.txt"  # Имя файла для сохранения
    num_points = 100000  # Количество точек данных
    num_features = 5  # Количество признаков
    num_clusters = 3  # Количество кластеров
    
    generate_large_txt_file(file_path, num_points, num_features, num_clusters)