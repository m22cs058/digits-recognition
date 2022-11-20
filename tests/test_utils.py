import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import numpy as np

digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(
    data, digits.target, test_size=0.5, random_state = 123
)

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
    data, digits.target, test_size=0.5,  random_state = 123
)

X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(
    data, digits.target, test_size=0.5, random_state = 456
)

print(X_train_1)
print(X_train_2)
print(np.array_equal(X_train_2, X_train_3))
def test_if_same():
    #checks if the split is same
    assert (X_train_2 == X_train_1).all()
    assert (X_test_2 == X_test_1).all()

def test_if_different():
    #checks if the split is different
    assert (X_train_2 != X_train_3).any()
    assert (X_test_2 != X_test_3).any()
