import numpy as np
from sklearn import preprocessing

# Определим метки.
# Предоставление меток входных данных
input_labels = ['red', 'black', 'red', 'green', 'yellow', 'white']

# Создадим объект кодирования меток и обучим его.
# Создание кодировщика и установление соответствия
# между метками и числами

encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)

# Выведем отображение слов на числа.
# Вывод отображения
print("\nLabel mapping: ")
for i, item in  enumerate(encoder.classes_):
    print(item, '-->', i)

#Преобразуем набор случайно упорядоченных меток, чтобы проверить ра­
#боту кодировщика.

test_labels = ['green', 'red', 'black']
encoded_values = encoder.transform(test_labels)
print("\nLabels = ", test_labels)
print("Encoded values = ", list(encoded_values))

#Декодируем случайный набор чисел.
encoded_values = [3, 0, 4, 1]
decoded_list = encoder.inverse_transform(encoded_values)
print("\nEncoded_values =", encoded_values)
print("Decoded labels = ", list(decoded_list))

