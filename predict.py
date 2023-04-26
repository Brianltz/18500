import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def importdata():
    model_data = pd.read_csv("model_data.csv")
    print("Dataset Length: ", len(model_data))
    print("Dataset Shape: ", model_data.shape)
    print("Dataset: ", model_data.head())
    return model_data


def splitdataset(model_data):

    # Separating the target variable
    X = model_data.values[:, 0:11]
    Y = model_data.values[:, 14]

    # Splitting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.33, random_state=9
    ) 

    return X, Y, X_train, X_test, y_train, y_test


# Function to perform training with giniIndex.
def train_using_gini(X_train, X_test, y_train):

    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(
        criterion="gini", random_state=100, max_depth=10, min_samples_leaf=2
    )

    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini


# Function to perform training with entropy.
def train_using_entropy(X_train, X_test, y_train):

    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(
        criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5
    )

    # Performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy


# Function to make predictions
def prediction(X_test, clf_object):

    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    # print("Predicted values:")
    # print(y_pred)
    return y_pred


# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):

    # print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))

    print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)

    print("Report : ", classification_report(y_test, y_pred))


def categorize(firstMax, secondMax, thirdMax, val):
    if val < firstMax:
        return "almost empty"
    elif val < secondMax:
        return "almost not busy"
    elif val < thirdMax:
        return "busy"
    else:
        return "almost full"


def classification_error(y_tr, y_t):
    maxCap = 90
    firstMax = maxCap * 0.2
    secondMax = maxCap * 0.4
    thirdMax = maxCap * 0.7
    total = len(y_tr)
    count = 0
    for i in range(len(y_tr)):
        tr_cat = categorize(firstMax, secondMax, thirdMax, y_tr[i])
        t_cat = categorize(firstMax, secondMax, thirdMax, y_t[i])
        if tr_cat != t_cat:
            count += 1
    return count / total


def main():
    # Building Phase
    data = importdata()
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
    # print("X: ", X)
    # print("Y: ", Y)
    clf_gini = train_using_gini(X_train, X_test, y_train)
    clf_entropy = train_using_entropy(X_train, X_test, y_train)

    # Operational Phase
    print("Results Using Gini Index:")

    # Prediction using gini
    y_pred_gini = prediction(X_test, clf_gini)
    f = open("output.txt", "w")
    f.write("gini index results")
    for i in range(y_pred_gini.shape[0]):
        f.write(
            str(X_test[i])
            + " predicted: "
            + str(y_pred_gini[i])
            + " real output: "
            + str(y_test[i])
            + "\n"
        )
    cal_accuracy(y_test, y_pred_gini)
    gini_class_error = classification_error(y_pred_gini, y_test)
    print("gini classification error", gini_class_error)

    print("Results Using Entropy:")

    # Prediction using entropy
    y_pred_entropy = prediction(X_test, clf_entropy)
    f.write("\nentropy results:\n")
    for i in range(y_pred_entropy.shape[0]):
        f.write(
            str(X_test[i])
            + " predicted: "
            + str(y_pred_entropy[i])
            + " real output: "
            + str(y_test[i])
            + "\n"
        )
    cal_accuracy(y_test, y_pred_entropy)
    entropy_class_error = classification_error(y_pred_gini, y_test)
    print("entropy classification error", entropy_class_error)
    f.close()


# Calling main function
if __name__ == "__main__":
    main()
