# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause


#PART: library dependencies -- sklear, torch, tensorflow, numpy, transformers

# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from skimage import transform
from sklearn.model_selection import train_test_split

# 1. set the ranges of hyper parameters 
gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [1, 2, 5, 7, 10] 

h_param_comb = [{'gamma':g, 'C':c} for g in gamma_list for c in c_list]

assert len(h_param_comb) == len(gamma_list)*len(c_list)

train_frac = 0.8
test_frac = 0.1
dev_frac = 0.1

train_acc_list = []
test_acc_list = []
dev_acc_list = []
#PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()
#PART: sanity check visualization of the data
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

print(f"\nImage size of this dataset is {digits.images[0].shape}")
#PART: data pre-processing -- to remove some noise, to normalize data, format the data to be consumed by mode
# flatten the images
images = {}
def resize_and_predict(size = 8):
    print(f"\n----------FOR IMAGE SIZE ({size},{size})-----------\n")
    n_samples = len(digits.images)
    images[str(size)] = np.zeros((n_samples,size, size))
    for i in range(0,n_samples):
        images[str(size)][i] = transform.resize(digits.images[i],(size, size),anti_aliasing=True)
    data = images[str(size)].reshape((n_samples, -1))


    #PART: define train/dev/test splits of experiment protocol
    # train to train model
    # dev to set hyperparameters of the model
    # test to evaluate the performance of the model
    dev_test_frac = 1-train_frac
    X_train, X_dev_test, y_train, y_dev_test = train_test_split(
        data, digits.target, test_size=dev_test_frac, shuffle=True
    )
    X_test, X_dev, y_test, y_dev = train_test_split(
        X_dev_test, y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True
    )


    best_acc = -1.0
    best_model = None
    best_h_params = None
    print("C          Gamma      Train_Acc  Val_Acc    Test_Acc\n")
    # 2. For every combination-of-hyper-parameter values
    for cur_h_params in h_param_comb:

        #PART: Define the model
        # Create a classifier: a support vector classifier
        clf = svm.SVC()

        #PART: setting up hyperparameter
        hyper_params = cur_h_params
        clf.set_params(**hyper_params)


        #PART: Train model
        # 2.a train the model 
        # Learn the digits on the train subset
        clf.fit(X_train, y_train)

        # print(cur_h_params)
        #PART: get dev set predictions
        predicted_train = clf.predict(X_train)
        predicted_dev = clf.predict(X_dev)
        predicted_test = clf.predict(X_test)
        # 2.b compute the accuracy on the validation set
        cur_train_acc = metrics.accuracy_score(y_pred=predicted_train, y_true=y_train)
        cur_val_acc = metrics.accuracy_score(y_pred=predicted_dev, y_true=y_dev)
        cur_test_acc = metrics.accuracy_score(y_pred=predicted_test, y_true=y_test)
        #print(f"C:{cur_h_params['C']} Gamma:{cur_h_params['gamma']}, {cur_train_acc} {cur_val_acc} {cur_test_acc}")
        print("{:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(cur_h_params['C'], cur_h_params['gamma'], cur_train_acc, cur_val_acc, cur_test_acc))
        train_acc_list.append(cur_train_acc)
        test_acc_list.append(cur_test_acc)
        dev_acc_list.append(cur_val_acc)
        # 3. identify the combination-of-hyper-parameter for which validation set accuracy is the highest. 
        if cur_test_acc > best_acc:
            best_acc = cur_test_acc
            best_model = clf
            best_h_params = cur_h_params
            #print("Found new best acc with :"+str(cur_h_params))
            #print("New best val accuracy:" + str(cur_acc))'''

    print("Best hyperparameters were:")
    print(cur_h_params)
    print("Best Accuracy = ", best_acc)
resize_and_predict(size = 16)
from statistics import mean, median
print("Mean of train, dev and test accuracies = ", mean(train_acc_list), mean(dev_acc_list), mean(test_acc_list))
print("Median of train, dev and test accuracies = ", median(train_acc_list), median(dev_acc_list), median(test_acc_list))
print("Min of train, dev and test accuracies = ", min(train_acc_list), min(dev_acc_list), min(test_acc_list))
print("Max of train, dev and test accuracies = ", max(train_acc_list), max(dev_acc_list), max(test_acc_list))

