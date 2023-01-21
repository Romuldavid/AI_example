import numpy as np
from sklearn import preprocessing

input_data = np.array(
    [[5.1, -2.9, 3.3],
     [-1.2, 7.8, -6.1],
     [3.9, 0.4, 2.1],
     [7.3, -9.9, -4.5]]
)

# Binar data
data_binarized = preprocessing.Binarizer(threshold= 2.1).transform(input_data)
print("\nBinarized data:\n", data_binarized)

#49
#Вывод среднего значения и стандартного отклонения
print("\nBEFORE:")
print("Mean = ", input_data.mean(axis = 0))
print("Std deviation = ", input_data.std(axis = 0))

#50
#Исключение среднего
data_scaled = preprocessing.scale(input_data)
print("\nAfter: ")
print("Mean = ", data_scaled.mean(axis = 0))
print("Std deviation = ", data_scaled.std(axis = 0))

#Масштабирование MinМax
data_scaler_mimax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaler_mimax = data_scaler_mimax.fit_transform(input_data)
print("\nMin max scaled data: \n", data_scaler_mimax)

#51
#Нормализация данных
data_norma_l1 = preprocessing.normalize(input_data, norm = 'l1')
data_norma_l2 = preprocessing.normalize(input_data, norm = 'l2')
print("\nL1 normalized data:\n", data_norma_l1)
print("\nL2 normalized data:\n", data_norma_l2)
