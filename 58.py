import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split

from utilities import visualize_classifier

#Входной файл,содержащий данные
input_file = 'data_multivar_nb.txt'

#load data
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

#Создание наивного байесовского классификатора
classifier = GaussianNB()

#Тренировка классификатора
classifier.fit(X, y)

#Прогнозирование значений дпя тренировочных данных
y_pred = classifier.predict(X)

#Вычислим качество (accuracy)классификатора, сравнив предсказанные
#значения с истинными метками, а затем визуализируем результат.
#Вычисление качества классификатора
accurancy = 100.0 * (y == y_pred).sum() / X.shape[0]
print("Accuracy of Naive Bayes classifier =", round(accurancy, 2), "%")
#Визуализация результатов работы классификатора
visualize_classifier(classifier, X, y)

#Разбивка данных на обучающий и тестовый наборы
#X_train, X_test, y_train, y_test = cross_validate.train_test_split(X, y, test_size = 0.2, random_state = 3)
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state= 3)
classifier_new = GaussianNB()
classifier_new.fit(X_train, y_train)
y_test_pred = classifier_new.predict(X_test)

#Вычисление качества классификатора
accurancy = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]
print("Accuracy of the new classifier =", round(accurancy, 2), "%")

#Визуализация работы классификатора
visualize_classifier(classifier_new, X_test, y_test)

# Scoring functions

num_folds = 3
accuracy_values = cross_validate.cross_val_score(classifier, 
        X, y, scoring='accuracy', cv=num_folds)
print("Accuracy: " + str(round(100*accuracy_values.mean(), 2)) + "%")

precision_values = cross_validate.cross_val_score(classifier, 
        X, y, scoring='precision_weighted', cv=num_folds)
print("Precision: " + str(round(100*precision_values.mean(), 2)) + "%")

recall_values = cross_validate.cross_val_score(classifier, 
        X, y, scoring='recall_weighted', cv=num_folds)
print("Recall: " + str(round(100*recall_values.mean(), 2)) + "%")

f1_values = cross_validate.cross_val_score(classifier, 
        X, y, scoring='f1_weighted', cv=num_folds)
print("F1: " + str(round(100*f1_values.mean(), 2)) + "%")
