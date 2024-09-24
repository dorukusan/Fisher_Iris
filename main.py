import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt


# Загружаем набор данных Ирисы
iris = datasets.load_iris()

# Выводим названия признаков
print(iris.feature_names)
# Смотрим на данные, выводим 10 первых строк:
print(iris.data[:10])
# Смотрим на целевую переменную:
print(iris.target_names)
print(iris.target)

# Снимаем ограничения вывода таблицы
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)


# Создаем DataFrame из данных ирисов
iris_frame = pd.DataFrame(iris.data, columns=iris.feature_names)

# Добавляем столбец с целевой переменной
iris_frame['target'] = iris.target

# Для наглядности добавляем столбец с сортами
iris_frame['name'] = iris_frame.target.apply(lambda x: iris.target_names[x])

print(iris_frame)


# Разделяем данные на обучающую и тестовую выборки
(train_set, test_set, train_labels, test_labels) = train_test_split(
    iris_frame[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']],
    iris_frame['target'],
    test_size=0.3,
    # random_state=42
)


# Стандартизация данных
scaler = StandardScaler()
scaled_train_set = scaler.fit_transform(train_set)
scaled_test_set = scaler.transform(test_set)

# # Данные без стандартизации
# scaled_train_data = train_data.values  # Преобразуем в массив NumPy
# scaled_test_data = test_data.values


# Обучение модели KMeans с использованием инициализации k-means++
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=1000, n_init=10, random_state=42)
kmeans.fit(scaled_train_set)

# Предсказание на тестовых данных (KMeans)
model_predictions_kmeans = kmeans.predict(scaled_test_set)


# Обучение модели SVM
svm = SVC(kernel='linear', C=100)
svm.fit(scaled_train_set, train_labels)

# Предсказание на тестовых данных (SVM)
model_predictions_svm = svm.predict(scaled_test_set)


# Создаем DataFrame для удобства
results = pd.DataFrame({'Cluster': model_predictions_kmeans, 'TrueLabel': test_labels})

# Находим наиболее частую метку для каждого кластера
cluster_labels = results.groupby('Cluster')['TrueLabel'].agg(lambda x: x.mode()[0]).to_dict()

print("\nСоответствие кластеров и истинных меток:")
print(cluster_labels)

# Сопоставление кластеров с истинными метками
mapped_predictions_kmeans = np.array([cluster_labels[cluster] for cluster in model_predictions_kmeans])

# Оценка точности для KMeans и SVM
accuracy_kmeans = metrics.accuracy_score(test_labels, mapped_predictions_kmeans)
accuracy_svm = metrics.accuracy_score(test_labels, model_predictions_svm)

print("\nОценка точности для KMeans:", accuracy_kmeans)
print(metrics.classification_report(test_labels, mapped_predictions_kmeans))

print("\nОценка точности для SVM:", accuracy_svm)
print(metrics.classification_report(test_labels, model_predictions_svm))


# # Получение центроидов и меток из KMeans
# centroids = kmeans.cluster_centers_
# labels = kmeans.labels_
#
# print("Центроиды:", centroids)
# print("Метки:", labels)


# Визуализация результатов
plt.figure(figsize=(12, 12))

# График для сравнения истинных меток классов с предсказанными метками KMeans
plt.subplot(2, 2, 1)
plt.scatter(scaled_test_set[:, 0], scaled_test_set[:, 1], c=test_labels, cmap='viridis', marker='o', edgecolor='k', s=100)
plt.title('Истинные метки классов')
plt.xlabel('Sepal Length (scaled)')
plt.ylabel('Sepal Width (scaled)')
plt.colorbar(label='True Label')

plt.subplot(2, 2, 2)
plt.scatter(test_set['sepal length (cm)'], test_set['sepal width (cm)'],
            c=mapped_predictions_kmeans, cmap='viridis', marker='o', edgecolor='k', s=100)
plt.title('Предсказанные кластеры KMeans после сопоставления')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.colorbar(label='Mapped Label')

# График для истинных меток классов
plt.subplot(2, 2, 3)
plt.scatter(scaled_test_set[:, 0], scaled_test_set[:, 1], c=test_labels, cmap='viridis', marker='o', edgecolor='k', s=100)
plt.title('Истинные метки классов')
plt.xlabel('Sepal Length (scaled)')
plt.ylabel('Sepal Width (scaled)')
plt.colorbar(label='True Label')

# График для предсказанных меток SVM
plt.subplot(2, 2, 4)
plt.scatter(scaled_test_set[:, 0], scaled_test_set[:, 1], c=model_predictions_svm, cmap='viridis', marker='o', edgecolor='k', s=100)
plt.title('Предсказанные метки SVM')
plt.xlabel('Sepal Length (scaled)')
plt.ylabel('Sepal Width (scaled)')
plt.colorbar(label='Predicted Label')

plt.tight_layout()
plt.show()
