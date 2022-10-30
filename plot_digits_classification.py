from sklearn import datasets, svm, metrics
import pdb
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

from utils import (
    preprocess_digits,
    train_dev_test_split,
    h_param_tuning,
    data_viz,
    pred_image_viz,
    get_all_h_param_comb,
    tune_and_save,
)
from joblib import dump, load

train_frac, dev_frac, test_frac = 0.8, 0.1, 0.1
assert train_frac + dev_frac + test_frac == 1.0

gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

params = {}
params["gamma"] = gamma_list
params["C"] = c_list

h_param_comb = get_all_h_param_comb(params)
max_depth_list = [5, 10, 20, 40, 80, 160]

# PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()
data_viz(digits)
data, label = preprocess_digits(digits)
# housekeeping
del digits

n_cv = 5
results = {}
random = [14, 56, 290, 456, 78]
metric = metrics.accuracy_score
predicted = {}
for i in range(n_cv):

    x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
        data, label, train_frac, dev_frac, random_state = random[i]
    )

    # PART: Define the model
    # Create a classifier: a support vector classifier
    models_of_choice = {'svm':svm.SVC(), 'decision_tree':DecisionTreeClassifier()}
    for clf_name in models_of_choice:
        best_model = models_of_choice[clf_name]
        
        # 2. load the best_model
        #best_model = load(actual_model_path)
        best_model.fit(x_train, y_train)
        
        predicted[clf_name] = best_model.predict(x_test)
        if not clf_name in results:
            results[clf_name] = []
        results[clf_name].append(metric(y_pred = predicted[clf_name], y_true = y_test))


        
print(results)
print("Mean and standard deviation for svm are {:.4} and {:.4}".format(np.mean(results['svm']), np.std(results['svm'])))
print("Mean and standard deviation for decision_tree are {:.4} and {:.4}".format(np.mean(results['decision_tree']), np.std(results['decision_tree'])))
print("Actual Vs Predicted Labels using SVM:")
print(list(zip(list(y_test), list(predicted['svm']))))
print("Actual Vs Predicted Labels using Decision Tree:")
print(list(zip(list(y_test), list(predicted['decision_tree']))))