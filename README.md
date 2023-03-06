# ID3-Adaboost
ID3 and Adaboost algorithm

Decision Tree: A decision tree is a structure that contains nodes (rectangular boxes) and edges(arrows) and is built from a dataset (table of columns representing features/attributes and rows corresponds to records). Each node is either used to make a decision (known as decision node) or represent an outcome (known as leaf node).

ID3 Algorithm: ID3 stands for Iterative Dichotomiser 3 because the algorithm iteratively (repeatedly) dichotomizes(divides) features into two or more groups at each step. Invented by Ross Quinlan, ID3 uses a top-down greedy approach to build a decision tree. The top-down approach means that we start building the tree from the top and the greedy approach means that at each iteration we select the best feature at the present moment to create a node.

Adaboost algorithm: AdaBoost, also called Adaptive Boosting, is a technique in Machine Learning used as an Ensemble Method. The most common estimator used with AdaBoost is decision trees with one level which means Decision trees with only 1 split.


We have used the following datasets. The description of datasets can be found at:
Car.data -> https://archive.ics.uci.edu/ml/datasets/Car+Evaluation
ecoli.data -> https://archive.ics.uci.edu/ml/datasets/Ecoli
mushroom.data -> https://archive.ics.uci.edu/ml/datasets/Mushroom
letter-recognition.data -> https://archive.ics.uci.edu/ml/datasets/Letter+Recognition
breast-cancer-wisconsin.data -> https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)

Run below command to perform classification on the data:
