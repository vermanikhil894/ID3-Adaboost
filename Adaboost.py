import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import KFold


class AdaboostClassifier:
    def __init__(self):
        self.klass = []

    def fit(self, X, y):
        number_of_samples, number_of_features = X.shape

        weight_features = np.full(number_of_samples, (1 / number_of_samples))

        self.klass = []

        for _ in range(2):
            klass = DecisionTreeFirstLevel()
            minimum_error = float("inf")
            # Loops through classifier

            for feature_index in range(number_of_features):
                X_column = X[:, feature_index]
                thresholds_array = np.unique(X_column)

                # Finding the best threshold frequency
                for t in thresholds_array:

                    polarity_level = 1  # setting the polarity by 1
                    predictions = np.ones(number_of_samples)
                    predictions[X_column < t] = -1

                    misclassified = weight_features[y != predictions]
                    weights_of_misclassfied_instance = sum(misclassified)

                    if weights_of_misclassfied_instance > 0.5:
                        weights_of_misclassfied_instance = 1 - weights_of_misclassfied_instance
                        polarity_level = -1

                    if weights_of_misclassfied_instance < minimum_error:
                        klass.polarity = polarity_level
                        klass.threshold_point = t
                        klass.feature_index = feature_index
                        minimum_error = weights_of_misclassfied_instance

            EPS = 1e-10
            klass.alpha = 0.5 * \
                np.log((1.0 - minimum_error + EPS) /
                       (minimum_error + EPS))  # calculating alpha

            predicted_values = klass.predict(X)

            weight_features *= np.exp(-klass.alpha * y * predicted_values)
            weight_features /= np.sum(weight_features)

            self.klass.append(klass)

    def predict(self, X):
        klass_preds = [klass.alpha * klass.predict(X) for klass in self.klass]
        y_pred = np.sum(klass_preds, axis=0)
        y_pred = np.sign(y_pred)

        return y_pred


class DecisionTreeFirstLevel:
    def __init__(self):
        self.polarity = 1
        self.feature_index = None
        self.threshold_point = None
        self.alpha = None

    def predict(self, X):
        number_of_samples = X.shape[0]
        X_column = X[:, self.feature_index]
        predicted_values = np.ones(number_of_samples)
        if self.polarity == 1:
            predicted_values[X_column < self.threshold_point] = -1
        else:
            predicted_values[X_column > self.threshold_point] = -1

        return predicted_values


if __name__ == "__main__":

    dataset_name = sys.argv[1]

    count = 0
    accuracy_adaboost = []

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    if dataset_name == "car":

        car_dataset = pd.read_csv("./dataset/car.data", names=[
                                  "buying", "maint", "doors", "persons", "lug_boot", "safety", "decision"])

        car_dataset["decision"].replace(
            ["unacc", "acc", "good", "vgood"], [0, 1, 2, 3], inplace=True)
        car_dataset["safety"].replace(
            ["low", "med", "high"], [0, 1, 2], inplace=True)
        car_dataset["lug_boot"].replace(
            ["small", "med", "big"], [0, 1, 2], inplace=True)
        car_dataset["persons"].replace(["more"], [4], inplace=True)
        car_dataset["doors"].replace(["5more"], [6], inplace=True)
        car_dataset["maint"].replace(["vhigh", "high", "med", "low"], [
            4, 3, 2, 1], inplace=True)
        car_dataset["buying"].replace(["vhigh", "high", "med", "low"], [
            4, 3, 2, 1], inplace=True)

        car_dataset['decision'] = car_dataset['decision'].astype(int)
        car_dataset['safety'] = car_dataset['safety'].astype(int)
        car_dataset['lug_boot'] = car_dataset['lug_boot'].astype(int)
        car_dataset['persons'] = car_dataset['persons'].astype(int)
        car_dataset['doors'] = car_dataset['doors'].astype(int)
        car_dataset['maint'] = car_dataset['maint'].astype(int)
        car_dataset['buying'] = car_dataset['buying'].astype(int)

        adaboost_model = AdaboostClassifier()

        for i in range(10):
            car_dataset = car_dataset.sample(frac=1)

            X, y = car_dataset.iloc[:, 0:6].to_numpy(
            ), car_dataset.iloc[:, 6].to_numpy()
            y[y == 0] = -1

            kfold_adaboost = KFold(n_splits=5)
            for train_index, test_index in kfold_adaboost.split(X):
                X_train, X_test = X[train_index, :], X[test_index, :]
                y_train, y_test = y[train_index], y[test_index]

                adaboost_model.fit(X_train, y_train)
                predicted_values = adaboost_model.predict(X_test)

                accuracy_dataset = accuracy(predicted_values, y_test)

            accuracy_adaboost.append(accuracy_dataset)

        mean_accuracy = np.sum(accuracy_adaboost)/float(len(accuracy_adaboost))
        std_dev = np.std(accuracy_adaboost)
        print("Average Accuracy for Car Dataset: ", mean_accuracy)
        print("Standard Deviation for Car Dataset:", std_dev, " \n ")

    elif dataset_name == "breastcancer":

        breast_cancer_dataset = pd.read_csv("./dataset/breast-cancer-wisconsin.data", names=[
                                   "column1", "column2", "column3", "column4", "column5", "column6", "column7", "column8", "column9", "column10", "decision"])

        breast_cancer_dataset.drop(["column7"], axis=1)
        breast_cancer_dataset["decision"].replace([2, 4], [-1, 1], inplace=True)

        adaboost_model = AdaboostClassifier()

        for i in range(0, 10):

            breast_cancer_dataset = breast_cancer_dataset.sample(frac=1)

            X, y = breast_cancer_dataset.iloc[:, 1:10].to_numpy(
            ), breast_cancer_dataset.iloc[:, 10].to_numpy()
            y[y == 0] = -1

            kfold_adaboost = KFold(n_splits=5)
            for train_index, test_index in kfold_adaboost.split(X):
                X_train, X_test = X[train_index, :], X[test_index, :]
                y_train, y_test = y[train_index], y[test_index]

                adaboost_model.fit(X_train, y_train)
                predicted_values = adaboost_model.predict(X_test)

                accuracy_dataset = accuracy(predicted_values, y_test)

            accuracy_adaboost.append(accuracy_dataset)

        mean_accuracy = np.sum(accuracy_adaboost)/float(len(accuracy_adaboost))
        std_dev = np.std(accuracy_adaboost)
        print("Average Accuracy for Breast Cancer Dataset:", mean_accuracy)
        print("Standard Deviation for Breast Cancer Dataset:", std_dev, " \n ")

    elif dataset_name == "mushroom":

        mushroom_dataset = pd.read_csv("./dataset/mushroom.data", names=["decision", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",  "gill-attachment",
                                                                 "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
                                                                 "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color", "ring-number", "ring-type", "spore-print-color",
                                                                 "population", "habitat"])

        mushroom_dataset["decision"].replace(["e", "p"], [0, 1], inplace=True)
        mushroom_dataset["cap-shape"].replace(["b", "c", "x", "f",
                                       "k", "s"], [0, 1, 2, 3, 4, 5], inplace=True)
        mushroom_dataset["cap-surface"].replace(["f", "g",
                                         "y", "s"], [0, 1, 2, 3], inplace=True)
        mushroom_dataset["cap-color"].replace(["n", "b", "c", "g", "r", "p", "u", "e", "w", "y"], [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9], inplace=True)
        mushroom_dataset["bruises"].replace(["t", "f"], [0, 1], inplace=True)
        mushroom_dataset["odor"].replace(["a", "l", "c", "y", "f", "m", "n", "p", "s"], [
            1, 2, 3, 4, 5, 6, 7, 8, 9], inplace=True)
        mushroom_dataset["gill-attachment"].replace(["a",
                                             "d", "f", "n"], [0, 1, 2, 3], inplace=True)
        mushroom_dataset["gill-spacing"].replace(["c",
                                          "w", "d"], [0, 1, 2], inplace=True)
        mushroom_dataset["gill-size"].replace(["b", "n"], [0, 1], inplace=True)
        mushroom_dataset["gill-color"].replace(["k", "n", "b", "h", "g", "r", "o", "p", "u", "e", "w", "y"], [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], inplace=True)
        mushroom_dataset["stalk-shape"].replace(["e", "t"], [0, 1], inplace=True)
        mushroom_dataset["stalk-root"].replace(["b", "c", "u", "e",
                                        "z", "r", "?"], [1, 2, 3, 4, 5, 6, 0], inplace=True)
        mushroom_dataset["stalk-surface-above-ring"].replace(
            ["f", "y", "k", "s"], [1, 2, 3, 4], inplace=True)
        mushroom_dataset["stalk-surface-below-ring"].replace(
            ["f", "y", "k", "s"], [1, 2, 3, 4], inplace=True)
        mushroom_dataset["stalk-color-above-ring"].replace(["n", "b", "c", "g", "o", "p", "e", "w", "y"], [
            1, 2, 3, 4, 5, 6, 7, 8, 9], inplace=True)
        mushroom_dataset["stalk-color-below-ring"].replace(["n", "b", "c", "g", "o", "p", "e", "w", "y"], [
            1, 2, 3, 4, 5, 6, 7, 8, 9], inplace=True)
        mushroom_dataset["veil-type"].replace(["p", "u"], [1, 2], inplace=True)
        mushroom_dataset["veil-color"].replace(["n", "o",
                                        "w", "y"], [1, 2, 3, 4], inplace=True)
        mushroom_dataset["ring-number"].replace(["n", "o", "t"],
                                        [1, 2, 3], inplace=True)
        mushroom_dataset["ring-type"].replace(["c", "e", "f", "l", "n", "p", "s", "z"], [
            1, 2, 3, 4, 5, 6, 7, 8], inplace=True)
        mushroom_dataset["spore-print-color"].replace(["k", "n", "b", "h", "r", "o", "u", "w", "y"], [
            1, 2, 3, 4, 5, 6, 7, 8, 9], inplace=True)
        mushroom_dataset["population"].replace(["a", "c", "n", "s", "v", "y"], [
            1, 2, 3, 4, 5, 6], inplace=True)
        mushroom_dataset["habitat"].replace(["g", "l", "m", "p", "u", "w", "d"], [
            1, 2, 3, 4, 5, 6, 7], inplace=True)

        adaboost_model = AdaboostClassifier()

        for i in range(1, 11):

            mushroom_dataset = mushroom_dataset.sample(frac=1)
            X, y = mushroom_dataset.iloc[:, 1:23].to_numpy(
            ), mushroom_dataset.iloc[:, 0].to_numpy()
            y[y == 0] = -1

            kfold_adaboost = KFold(n_splits=5)
            for train_index, test_index in kfold_adaboost.split(X):
                X_train, X_test = X[train_index, :], X[test_index, :]
                y_train, y_test = y[train_index], y[test_index]

                adaboost_model.fit(X_train, y_train)
                predicted_values = adaboost_model.predict(X_test)

                accuracy_dataset = accuracy(predicted_values, y_test)

            accuracy_adaboost.append(accuracy_dataset)

        mean_accuracy = np.sum(accuracy_adaboost)/float(len(accuracy_adaboost))
        std_dev = np.std(accuracy_adaboost)
        print('Average Accuracy for Mushroom Dataset:', mean_accuracy)
        print("Standard Deviation for Mushroom Dataset:", std_dev, " \n ")

    elif dataset_name == "ecoli":

        ecoli_dataset = pd.read_csv("./dataset/ecoli.data", names=["column1", "column2", "column3", "column4",
                            "column5", "column6", "column7", "column8", "decision"], delim_whitespace=True)

        # Shuffle the dataset
        ecoli_dataset = ecoli_dataset.sample(frac=1)

        ecoli_dataset = ecoli_dataset.drop("column1", axis="columns")
        ecoli_dataset["decision"].replace(["cp", "im", "imU", "imS", "imL", "om", "omL", "pp"], [
                                  0, 1, 2, 3, 4, 5, 6, 7], inplace=True)

        ecoli_dataset['column2'] = ecoli_dataset['column2'].astype(float)
        ecoli_dataset['column3'] = ecoli_dataset['column3'].astype(float)
        ecoli_dataset['column4'] = ecoli_dataset['column4'].astype(float)
        ecoli_dataset['column5'] = ecoli_dataset['column5'].astype(float)
        ecoli_dataset['column6'] = ecoli_dataset['column6'].astype(float)
        ecoli_dataset['column7'] = ecoli_dataset['column7'].astype(float)
        ecoli_dataset['column8'] = ecoli_dataset['column8'].astype(float)
        ecoli_dataset['decision'] = ecoli_dataset['decision'].astype(int)

        adaboost_model = AdaboostClassifier()
        for i in range(0, 10):

            ecoli_dataset = ecoli_dataset.sample(frac=1)
            X, y = ecoli_dataset.iloc[:, 1:17].to_numpy(), ecoli_dataset.iloc[:, 0].to_numpy()
            y[y == 0] = -1

            kfold_adaboost = KFold(n_splits=5)
            for train_index, test_index in kfold_adaboost.split(X):
                X_train, X_test = X[train_index, :], X[test_index, :]
                y_train, y_test = y[train_index], y[test_index]

                adaboost_model.fit(X_train, y_train)
                predicted_values = adaboost_model.predict(X_test)

                accuracy_dataset = accuracy(predicted_values, y_test)

            accuracy_adaboost.append(accuracy_dataset)

        mean_accuracy = np.sum(accuracy_adaboost)/float(len(accuracy_adaboost))
        std_dev = np.std(accuracy_adaboost)
        print('Average Accuracy for Ecoli Dataset', mean_accuracy)
        print("Standard Deviation for Ecoli Dataset:", std_dev, " \n ")

    elif dataset_name == "letterrecognition":

        letterrecognition = pd.read_csv("./dataset/letter-recognition.data", names=["decision", "column2", "column3", "column4", "column5", "column6",
                                        "column7", "column8", "column9", "column10", "column11", "column12", "column13", "column14", "column15", "column16", "column17"])

        letterrecognition["decision"].replace(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"], [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], inplace=True)

        adaboost_model = AdaboostClassifier()
        for i in range(0, 10):

            letterrecognition = letterrecognition.sample(frac=1)
            X, y = letterrecognition.iloc[:, 1:17].to_numpy(
            ), letterrecognition.iloc[:, 0].to_numpy()
            y[y == 0] = -1

            kfold_adaboost = KFold(n_splits=5)
            for train_index, test_index in kfold_adaboost.split(X):
                X_train, X_test = X[train_index, :], X[test_index, :]
                y_train, y_test = y[train_index], y[test_index]

                adaboost_model.fit(X_train, y_train)
                predicted_values = adaboost_model.predict(X_test)

                accuracy_dataset = accuracy(predicted_values, y_test)

            accuracy_adaboost.append(accuracy_dataset)

        mean_accuracy = np.sum(accuracy_adaboost)/float(len(accuracy_adaboost))
        std_dev = np.std(accuracy_adaboost)
        print('Average Accuracy for Letter Recognition dataset', mean_accuracy)
        print("Standard Deviation for Letter Recognition Dataset:", std_dev, " \n ")

    else:
        print("Give proper dataset")