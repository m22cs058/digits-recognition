import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import argparse
from joblib import dump
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

parser = argparse.ArgumentParser(description ='Model Training')
parser.add_argument('--clf_name', type = str)

parser.add_argument('--random_state', type = int)
args = parser.parse_args()

clf_name = args.clf_name
random_state = args.random_state

gm = 0.001
c = 0.2

models = {"svm": svm.SVC(gamma=gm, C = c), 'decision_tree':DecisionTreeClassifier()}


def train(clf_name, random_state):
    X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, random_state = random_state
)
    clf = models[clf_name]
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    acc = accuracy_score(predicted, y_test)
    f1 = f1_score(predicted.reshape(-1,1), y_test.reshape(-1,1), average = "macro")
    print("test accuracy: " + str(accuracy_score(predicted, y_test)))
    print("test macro-f1: " + str(f1_score(predicted.reshape(-1,1), y_test.reshape(-1,1), average = "macro")))
    #dump(clf, "models/" + clf_name + "_" + "random_state=" + str(random_state) + ".joblib")

train(clf_name, random_state)