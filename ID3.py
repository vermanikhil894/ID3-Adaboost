from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import sys
import pandas as pd
from collections import Counter
from math import log
import numpy as np
from sklearn.base import BaseEstimator as estimator, ClassifierMixin as mix


# decision_tree_classifier implementation
class decision_tree_classifier(estimator, mix):

    def __init__(self, features):
        self.target_class = "labels"
        self.features = features

    def fit(self, X, y):
        target_class = self.target_class
        corpus = X.assign(labels=y)
        self.id3tree = {}
        decision_tree_classifier.search_node(
            corpus, self.id3tree, target_class)
        return self

    @staticmethod
    def calculate_score(attributes, entropy, total):
        # calculating entropy for each node
        updated_entro_set = [calculate_entropy(*i) for i in attributes]

        def f(x, y): return (sum(x) / total) * y

        result = [f(i, j) for i, j in zip(attributes, updated_entro_set)]
        return entropy - sum(result)

    @staticmethod
    def construct_branch(header, corpus, target_class):
        # Treat each attributes as branch
        df = pd.DataFrame(corpus.groupby(
            [header, target_class])[target_class].count())
        result = []
        for i in Counter(corpus[header]).keys():
            result.append(df.loc[i].values)
        return result

    def predict(self, X_test):
        result = []
        for i in X_test.itertuples():
            result.append(decision_tree_classifier.recurring(
                i, self.id3tree, self.features))
        return pd.Series(result)

    @classmethod
    def search_node(klass, corpus, id3, target_class):
        current_entropy = calculate_entropy(
            *[i for i in Counter(corpus[target_class]).values()])
        result = {}
        for record in corpus.columns:
            if record != target_class:
                attribute = klass.construct_branch(
                    record, corpus, target_class)
                tree_score = klass.calculate_score(attribute, current_entropy, total=len(
                    corpus))
                result[record] = tree_score
            value = max(result, key=result.__getitem__)
        child_nodes = [i for i in Counter(corpus[value])]

        id3[value] = {}

        # Creating nodes using iteration process
        for node in child_nodes:
            child_data = corpus[corpus[value] == node]
            if calculate_entropy(*[i for i in Counter(child_data[target_class]).values()]) != 0:
                id3[value][node] = {}
                klass.search_node(child_data, id3[value][node], target_class)
            else:
                r = Counter(child_data[target_class])
                id3[value][node] = max(r, key=r.__getitem__)
        return

    @classmethod
    def recurring(klass, tree, key, features):
        if type(key) is not dict:
            return key
        for k in key.keys():
            if k in features.keys():
                xyz = tree[features[k]]
                abc = key[k].get(tree[features[k]])
                result = klass.recurring(
                    tree, key[k].get(tree[features[k]], 0), features)
        return result


class DecisionTreeClassifierCar(estimator, mix):

    def __init__(self, features):
        self.target_class = "labels"
        self.features = features

    def fit(self, X, y):
        target_class = self.target_class
        corpus = X.assign(labels=y)
        self.id3tree = {}
        DecisionTreeClassifierCar.search_node(
            corpus, self.id3tree, target_class)
        return self

    @staticmethod
    def calculate_score(attributes, entropy, total):
        # calculating entropy for each node
        updated_entro_set = [calculate_entropy(*i) for i in attributes]

        def f(x, y): return (sum(x) / total) * y

        result = [f(i, j) for i, j in zip(attributes, updated_entro_set)]
        return entropy - sum(result)

    @staticmethod
    def construct_branch(header, corpus, target_class):
        # Treat each attributes as branch
        df = pd.DataFrame(corpus.groupby(
            [header, target_class])[target_class].count())
        result = []
        for i in Counter(corpus[header]).keys():
            result.append(df.loc[i].values)
        return result

    def predict(self, X_test):
        result = []
        for i in X_test.itertuples():
            result.append(DecisionTreeClassifierCar.recurring(
                i, self.id3tree, self.features))
        return pd.Series(result)

    @classmethod
    def search_node(klass, corpus, id3, target_class):
        current_entropy = calculate_entropy(
            *[i for i in Counter(corpus[target_class]).values()])
        result = {}
        for record in corpus.columns:
            if record != target_class:
                attribute = klass.construct_branch(
                    record, corpus, target_class)
                tree_score = klass.calculate_score(attribute, current_entropy, total=len(
                    corpus))
                result[record] = tree_score
            value = max(result, key=result.__getitem__)
        child_nodes = [i for i in Counter(corpus[value])]

        id3[value] = {}

        # Creating nodes using iteration process
        for node in child_nodes:
            child_data = corpus[corpus[value] == node]
            if calculate_entropy(*[i for i in Counter(child_data[target_class]).values()]) != 0:
                id3[value][node] = {}
                klass.search_node(child_data, id3[value][node], target_class)
            else:
                r = Counter(child_data[target_class])
                id3[value][node] = max(r, key=r.__getitem__)
        return

    @classmethod
    def recurring(klass, tree, key, features):
        if type(key) is int:
            return "NaN"
        elif type(key) is not dict:
            return key
        for k in key.keys():
            if k in features.keys():
                xyz = tree[features[k]]
                abc = key[k].get(tree[features[k]])
                result = klass.recurring(
                    tree, key[k].get(tree[features[k]], 0), features)
        return result


if __name__ == '__main__':

    dataset_name = sys.argv[1]

    count = 0
    accuracy_id3 = []
    std_dev = 0.0

    if dataset_name == "car":

        def calculate_entropy(class1=0, class2=0, class3=0, class4=0):
            toal_number_of_class = [class1, class2, class3, class4]
            final_entropy = 0
            for c in toal_number_of_class:
                if c != 0:
                    final_entropy += -((c / sum(toal_number_of_class))
                                       * log(c / sum(toal_number_of_class), 4))
            return final_entropy


        filename = './dataset/car.data'
        car_dataset = pd.read_csv(r'./dataset/car.data',
                                  names=(["buying", "maint", "doors", "persons", "lug_boot", "safety", "labels"]))

        X = car_dataset.drop(["labels"], axis=1)
        y = car_dataset["labels"]

        for i in range(10):
            car_dataset = car_dataset.sample(frac=1)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.55)

            final_entropy = calculate_entropy(
                *[i for i in Counter(y_train).values()])

            features = {
                'buying': 1,
                'maint': 2,
                'doors': 3,
                'persons': 4,
                'lug_boot': 5,
                'safety': 6
            }

            id3 = DecisionTreeClassifierCar(features)

            # main function trigger
            id3.fit(X_train, y_train)

            accuracy_score(y_test, id3.predict(X_test))

            temp_accuracy = cross_val_score(
                id3, X, y, cv=5, scoring='accuracy')

            for i in range(0, len(temp_accuracy)):
                accuracy_id3.append(temp_accuracy[i])
            count += 1

        mean_accuracy = np.sum(accuracy_id3) / len(accuracy_id3)
        std_dev = np.std(accuracy_id3)

        print("Average Accuracy for Car Dataset: ", mean_accuracy)
        print("Standard Deviation for Car Dataset:", std_dev, " \n ")


    elif dataset_name == "breastcancer":

        breast_cancer_dataset = pd.read_csv("./dataset/breast-cancer-wisconsin.data", names=[
            "column1", "column2", "column3", "column4", "column5", "column6", "column7", "column8", "column9",
            "column10", "labels"])

        breast_cancer_dataset["column7"].replace('?', '0', inplace=True)
        breast_cancer_dataset['column7'] = breast_cancer_dataset['column7'].astype(int)
        average = breast_cancer_dataset["column7"].mean()
        breast_cancer_dataset["column7"].replace('?', average, inplace=True)

        breast_cancer_dataset['column2'] = breast_cancer_dataset['column2'].astype(int)
        breast_cancer_dataset['column3'] = breast_cancer_dataset['column3'].astype(int)
        breast_cancer_dataset['column4'] = breast_cancer_dataset['column4'].astype(int)
        breast_cancer_dataset['column5'] = breast_cancer_dataset['column5'].astype(int)
        breast_cancer_dataset['column6'] = breast_cancer_dataset['column6'].astype(int)
        breast_cancer_dataset['column7'] = breast_cancer_dataset['column7'].astype(int)
        breast_cancer_dataset['column8'] = breast_cancer_dataset['column8'].astype(int)
        breast_cancer_dataset['column9'] = breast_cancer_dataset['column9'].astype(int)
        breast_cancer_dataset['column10'] = breast_cancer_dataset['column10'].astype(int)
        breast_cancer_dataset['labels'] = breast_cancer_dataset['labels'].astype(int)

        X = breast_cancer_dataset.drop(["column1", "labels"], axis=1)
        y = breast_cancer_dataset["labels"]


        def calculate_entropy(class1=0, class2=0, class3=0, class4=0):
            total_number_of_class = [class1, class2, class3, class4]
            final_entropy = 0
            for c in total_number_of_class:
                if c != 0:
                    final_entropy += -((c / sum(total_number_of_class))
                                       * log(c / sum(total_number_of_class), 4))
            return final_entropy

        for i in range(10):
            breast_cancer_dataset = breast_cancer_dataset.sample(frac=1)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.55)

            final_entropy = calculate_entropy(
                *[i for i in Counter(y_train).values()])

            features = {'column2': 1, 'column3': 2, 'column4': 3, 'column5': 4,
                        'column6': 5, 'column7': 6, 'column8': 7, 'column9': 8, 'column10': 9}

            id3 = decision_tree_classifier(features)

            # main function trigger
            id3.fit(X_train, y_train)
            accuracy_score(y_test, id3.predict(X_test))

            temp_accuracy = cross_val_score(
                id3, X, y, cv=5, scoring='accuracy')

            for i in range(0, len(temp_accuracy)):
                accuracy_id3.append(temp_accuracy[i])
            count += 1

        mean_accuracy = np.sum(accuracy_id3) / len(accuracy_id3)
        std_dev = np.std(accuracy_id3)

        print("Average Accuracy for Breast Dataset Cancer: ", mean_accuracy)
        print("Standard Deviation for Breast Dataset Cancer:", std_dev, " \n ")

    elif dataset_name == "ecoli":

        ecoli_dataset = pd.read_csv("./dataset/ecoli.data", names=["column1", "column2", "column3", "column4",
                                                                   "column5", "column6", "column7", "column8",
                                                                   "labels"], delim_whitespace=True)

        ecoli_dataset = ecoli_dataset.drop("column1", axis="columns")
        ecoli_dataset["labels"].replace(["cp", "im", "imU", "imS", "imL", "om", "omL", "pp"], [
            0, 1, 2, 3, 4, 5, 6, 7], inplace=True)

        ecoli_dataset['column2'] = ecoli_dataset['column2'].astype(float)
        ecoli_dataset['column3'] = ecoli_dataset['column3'].astype(float)
        ecoli_dataset['column4'] = ecoli_dataset['column4'].astype(float)
        ecoli_dataset['column5'] = ecoli_dataset['column5'].astype(float)
        ecoli_dataset['column6'] = ecoli_dataset['column6'].astype(float)
        ecoli_dataset['column7'] = ecoli_dataset['column7'].astype(float)
        ecoli_dataset['column8'] = ecoli_dataset['column8'].astype(float)
        ecoli_dataset['labels'] = ecoli_dataset['labels'].astype(int)

        X = ecoli_dataset.drop(["labels"], axis=1)
        y = ecoli_dataset["labels"]


        def calculate_entropy(class1=0, class2=0, class3=0, class4=0, class5=0, class6=0, class7=0, class8=0):
            total_number_of_class = [class1, class2, class3,
                                     class4, class5, class6, class7, class8]
            final_entropy = 0
            for c in total_number_of_class:
                if c != 0:
                    final_entropy += -((c / sum(total_number_of_class))
                                       * log(c / sum(total_number_of_class), 4))
            return final_entropy


        for i in range(10):
            ecoli_dataset = ecoli_dataset.sample(frac=1)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.55)

            final_entropy = calculate_entropy(
                *[i for i in Counter(y_train).values()])

            features = {'column2': 1, 'column3': 2, 'column4': 3,
                        'column5': 4, 'column6': 5, 'column7': 6, 'column8': 7}

            id3 = decision_tree_classifier(features)

            # main function trigger
            id3.fit(X_train, y_train)

            accuracy_score(y_test, id3.predict(X_test))

            temp_accuracy = cross_val_score(
                id3, X, y, cv=2, scoring='accuracy')

            for i in range(0, len(temp_accuracy)):
                accuracy_id3.append(temp_accuracy[i])
            count += 1

        mean_accuracy = np.sum(accuracy_id3) / len(accuracy_id3)
        std_dev = np.std(accuracy_id3)

        print("Average Accuracy for Ecoli Cancer: ", mean_accuracy)
        print("Standard Deviation for Ecoli Cancer:", std_dev, " \n ")

    elif dataset_name == "mushroom":

        mushroom_dataset = pd.read_csv("./dataset/mushroom.data",
                                       names=["labels", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
                                              "gill-attachment",
                                              "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root",
                                              "stalk-surface-above-ring", "stalk-surface-below-ring",
                                              "stalk-color-above-ring", "stalk-color-below-ring", "veil-type",
                                              "veil-color", "ring-number", "ring-type", "spore-print-color",
                                              "population", "habitat"])

        mushroom_dataset["labels"].replace(["e", "p"], [0, 1], inplace=True)
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

        # Shuffle the dataset
        mushroom_dataset = mushroom_dataset.sample(frac=1)

        mushroom_dataset['labels'] = mushroom_dataset['labels'].astype(int)
        mushroom_dataset['cap-shape'] = mushroom_dataset['cap-shape'].astype(int)
        mushroom_dataset['cap-surface'] = mushroom_dataset['cap-surface'].astype(int)
        mushroom_dataset['cap-color'] = mushroom_dataset['cap-color'].astype(int)
        mushroom_dataset['bruises'] = mushroom_dataset['bruises'].astype(int)
        mushroom_dataset['odor'] = mushroom_dataset['odor'].astype(int)
        mushroom_dataset['gill-attachment'] = mushroom_dataset['gill-attachment'].astype(int)
        mushroom_dataset['gill-spacing'] = mushroom_dataset['gill-spacing'].astype(int)
        mushroom_dataset['gill-size'] = mushroom_dataset['gill-size'].astype(int)
        mushroom_dataset['gill-color'] = mushroom_dataset['gill-color'].astype(int)
        mushroom_dataset['stalk-shape'] = mushroom_dataset['stalk-shape'].astype(int)
        mushroom_dataset['stalk-root'] = mushroom_dataset['stalk-root'].astype(int)
        mushroom_dataset['stalk-surface-above-ring'] = mushroom_dataset['stalk-surface-above-ring'].astype(
            int)
        mushroom_dataset['stalk-surface-below-ring'] = mushroom_dataset['stalk-surface-below-ring'].astype(
            int)
        mushroom_dataset['stalk-color-above-ring'] = mushroom_dataset['stalk-color-above-ring'].astype(
            int)
        mushroom_dataset['stalk-color-below-ring'] = mushroom_dataset['stalk-color-below-ring'].astype(
            int)
        mushroom_dataset['veil-type'] = mushroom_dataset['veil-type'].astype(int)
        mushroom_dataset['veil-color'] = mushroom_dataset['veil-color'].astype(int)
        mushroom_dataset['ring-number'] = mushroom_dataset['ring-number'].astype(int)
        mushroom_dataset['ring-type'] = mushroom_dataset['ring-type'].astype(int)
        mushroom_dataset['spore-print-color'] = mushroom_dataset['spore-print-color'].astype(
            int)
        mushroom_dataset['population'] = mushroom_dataset['population'].astype(int)
        mushroom_dataset['habitat'] = mushroom_dataset['habitat'].astype(int)

        X = mushroom_dataset.drop(["labels"], axis=1)
        y = mushroom_dataset["labels"]


        def calculate_entropy(class1=0, class2=0):
            total_number_of_class = [class1, class2]
            final_entropy = 0
            for c in total_number_of_class:
                if c != 0:
                    final_entropy += -((c / sum(total_number_of_class))
                                       * log(c / sum(total_number_of_class), 4))
            return final_entropy

        for i in range(10):
            mushroom_dataset = mushroom_dataset.sample(frac=1)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.55)

            final_entropy = calculate_entropy(
                *[i for i in Counter(y_train).values()])

            features = {'cap-shape': 1, 'cap-surface': 2, 'cap-color': 3, 'bruises': 4, 'odor': 5, 'gill-attachment': 6,
                        'gill-spacing': 7, 'gill-size': 8, 'gill-color': 9, 'stalk-shape': 10, 'stalk-root': 11,
                        'stalk-surface-above-ring': 12, 'stalk-surface-below-ring': 13,
                        'stalk-color-above-ring': 14, 'stalk-color-below-ring': 15, 'veil-type': 16, 'veil-color': 17,
                        'ring-number': 18, 'ring-type': 19, 'spore-print-color': 20,
                        'population': 21, 'habitat': 22}

            id3 = decision_tree_classifier(features)

            # main function trigger
            id3.fit(X_train, y_train)

            accuracy_score(y_test, id3.predict(X_test))

            temp_accuracy = cross_val_score(
                id3, X, y, cv=2, scoring='accuracy')

            for i in range(0, len(temp_accuracy)):
                accuracy_id3.append(temp_accuracy[i])
            count += 1

        mean_accuracy = np.sum(accuracy_id3) / len(accuracy_id3)
        std_dev = np.std(accuracy_id3)

        print("Average Accuracy for Mushroom: ", mean_accuracy)
        print("Standard Deviation for Mushroom:", std_dev, " \n ")


    elif dataset_name == "letterrecognition":

        letterrecognition = pd.read_csv("./dataset/letter-recognition.data",
                                        names=["labels", "column2", "column3", "column4", "column5", "column6",
                                               "column7", "column8", "column9", "column10", "column11", "column12",
                                               "column13", "column14", "column15", "column16", "column17"])

        letterrecognition["labels"].replace(
            ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
             "V", "W", "X", "Y", "Z"], [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
            inplace=True)

        letterrecognition['labels'] = letterrecognition['labels'].astype(
            int)
        letterrecognition['column2'] = letterrecognition['column2'].astype(int)
        letterrecognition['column3'] = letterrecognition['column3'].astype(int)
        letterrecognition['column4'] = letterrecognition['column4'].astype(int)
        letterrecognition['column5'] = letterrecognition['column5'].astype(int)
        letterrecognition['column6'] = letterrecognition['column6'].astype(int)
        letterrecognition['column7'] = letterrecognition['column7'].astype(int)
        letterrecognition['column8'] = letterrecognition['column8'].astype(int)
        letterrecognition['column9'] = letterrecognition['column9'].astype(int)
        letterrecognition['column10'] = letterrecognition['column10'].astype(
            int)
        letterrecognition['column11'] = letterrecognition['column11'].astype(
            int)
        letterrecognition['column12'] = letterrecognition['column12'].astype(
            int)
        letterrecognition['column13'] = letterrecognition['column13'].astype(
            int)
        letterrecognition['column14'] = letterrecognition['column14'].astype(
            int)
        letterrecognition['column15'] = letterrecognition['column15'].astype(
            int)
        letterrecognition['column16'] = letterrecognition['column16'].astype(
            int)
        letterrecognition['column17'] = letterrecognition['column17'].astype(
            int)

        X = letterrecognition.drop(["labels"], axis=1)
        y = letterrecognition["labels"]
        print("Data set uploaded")


        def calculate_entropy(class1=0, class2=0, class3=0, class4=0, class5=0, class6=0, class7=0, class8=0, class9=0,
                              class10=0,
                              class11=0, class12=0, class13=0, class14=0, class15=0, class16=0, class17=0, class18=0,
                              class19=0, class20=0,
                              class21=0, class22=0, class23=0, class24=0, class25=0, class26=0):
            total_number_of_class = [class1, class2, class3, class4, class5, class6, class7, class8, class9, class10,
                                     class11, class12, class13, class14, class15, class16, class17, class18, class19,
                                     class20,
                                     class21, class22, class23, class24, class25, class26]
            final_entropy = 0
            for c in total_number_of_class:
                if c != 0:
                    final_entropy += -((c / sum(total_number_of_class))
                                       * log(c / sum(total_number_of_class), 4))
            return final_entropy


        print("Data set cleaned")
        for i in range(10):
            letterrecognition = letterrecognition.sample(frac=1)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.55)

            final_entropy = calculate_entropy(
                *[i for i in Counter(y_train).values()])

            features = {'column2': 1, 'column3': 2, 'column4': 3, 'column5': 4, 'column6': 5, 'column7': 6,
                        'column8': 7, 'column9': 8,
                        'column10': 9, 'column11': 10, 'column12': 11, 'column13': 12, 'column14': 13, 'column15': 14,
                        'column16': 15, 'column17': 16}

            id3 = decision_tree_classifier(features)
            # main function trigger
            id3.fit(X_train, y_train)

            accuracy_score(y_test, id3.predict(X_test))

            temp_accuracy = cross_val_score(
                id3, X, y, cv=2, scoring='accuracy')
            for i in range(0, len(temp_accuracy)):
                accuracy_id3.append(temp_accuracy[i])
            count += 1

        mean_accuracy = np.sum(accuracy_id3) / len(accuracy_id3)
        std_dev = np.std(accuracy_id3)

        print("Average Accuracy for Letter Recognition: ", mean_accuracy)
        print("Standard Deviation for Letter Recognition:", std_dev, " \n ")

    else:
        print("Give proper dataset")
