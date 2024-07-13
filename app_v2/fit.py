import pickle

from sklearn.ensemble import RandomForestClassifier
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np


def compute_accuracy(Y_true, Y_pred):
    correctly_predicted = 0
    
    for true_label, predicted in zip(Y_true, Y_pred):
        if true_label == predicted:
            correctly_predicted += 1
    accuracy_score = correctly_predicted / len(Y_true)
    return accuracy_score

data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)
y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)

print(f'Accuracy is {score * 100}%')

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
