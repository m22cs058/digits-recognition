import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from joblib import dump
from sklearn import svm


def get_all_h_param_comb(params):
    h_param_comb = [{"gamma": g, "C": c} for g in params['gamma'] for c in params['C']]
    return h_param_comb

def h_param_tuning(h_param_comb, clf, x_train, y_train, x_dev, y_dev, metric):
    best_metric = -1.0
    best_model = None
    best_h_params = None

    for cur_h_params in h_param_comb:

        hyper_params = cur_h_params
        clf.set_params(**hyper_params)

        clf.fit(x_train, y_train)

        predicted_dev = clf.predict(x_dev)

        cur_metric = metric(y_pred=predicted_dev, y_true=y_dev)
        if cur_metric > best_metric:
            best_metric = cur_metric
            best_model = clf
            best_h_params = cur_h_params
            print("Found new best metric with :" + str(cur_h_params))
            print("New best val metric:" + str(cur_metric))
    return best_model, best_metric, best_h_params

def tune_and_save(clf, x_train, y_train, x_dev, y_dev, metric, h_param_comb, model_path):
    best_model, best_metric, best_h_params = h_param_tuning(
        h_param_comb, clf, x_train, y_train, x_dev, y_dev, metric
    )

    best_param_config = "_".join([h + "=" + str(best_h_params[h]) for h in best_h_params])
    
    if type(clf) == svm.SVC:
        model_type = 'svm' 

    best_model_name = model_type + "_" + best_param_config + ".joblib"
    if model_path == None:
        model_path = best_model_name
    dump(best_model, model_path)

    print("Best hyperparameters were:")
    print(best_h_params)

    print("Best Metric on Dev was:{}".format(best_metric))

    return model_path