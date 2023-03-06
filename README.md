# ID3-Adaboost
ID3 and Adaboost algorithm

Decision Tree: A decision tree is a structure that contains nodes (rectangular boxes) and edges(arrows) and is built from a dataset (table of columns representing features/attributes and rows corresponds to records). Each node is either used to make a decision (known as decision node) or represent an outcome (known as leaf node).

ID3 Algorithm: ID3 stands for Iterative Dichotomiser 3 because the algorithm iteratively (repeatedly) dichotomizes(divides) features into two or more groups at each step. Invented by Ross Quinlan, ID3 uses a top-down greedy approach to build a decision tree. The top-down approach means that we start building the tree from the top and the greedy approach means that at each iteration we select the best feature at the present moment to create a node.

Adaboost algorithm: AdaBoost, also called Adaptive Boosting, is a technique in Machine Learning used as an Ensemble Method. The most common estimator used with AdaBoost is decision trees with one level which means Decision trees with only 1 split.


We have used the following datasets. The description of datasets can be found at:
1. Car.data -> https://archive.ics.uci.edu/ml/datasets/Car+Evaluation
2. ecoli.data -> https://archive.ics.uci.edu/ml/datasets/Ecoli
3. mushroom.data -> https://archive.ics.uci.edu/ml/datasets/Mushroom
4. letter-recognition.data -> https://archive.ics.uci.edu/ml/datasets/Letter+Recognition
5. breast-cancer-wisconsin.data -> https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)

Run below command to perform classification on the data:
1. python3 ID3.py car
2. python3 ID3.py beastcancer
3. python3 ID3.py ecoli
4. python3 ID3.py letterrecognition
5. python3 ID3.py mushroom

6. python3 Adaboost.py car
7. python3 Adaboost.py beastcancer
8. python3 Adaboost.py ecoli
9. python3 Adaboost.py letterrecognition
10. python3 Adaboost.py mushroom
