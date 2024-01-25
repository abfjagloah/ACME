import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

def svc_classify(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True,random_state=123)
    
    accuracies = []
    # accuracies_train = []
    for train_index, test_index in kf.split(x, y):
        # test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy',n_jobs=-1, verbose=0) #balanced_

        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))

    accuracies = np.array(accuracies)
    return accuracies.mean(), accuracies.std()

def evaluate_embedding(embeddings, labels,search=True):
    labels = preprocessing.LabelEncoder().fit_transform(labels)
    x, y = np.array(embeddings), np.array(labels)

    acc = 0
    std = 0

    acc, std = svc_classify(x, y, search)
    return acc, std
