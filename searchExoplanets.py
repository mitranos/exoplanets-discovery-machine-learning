# Import Generic libraries
import numpy as np
import pandas as pd
from time import time
from datetime import datetime
import matplotlib.pyplot as plt

# Import SkLearn libraries
# Import Metrics
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer

# Import Machine learning Algorithms
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Import Cross Validation and Grid Searcch Libraries
from sklearn.cross_validation import train_test_split, KFold
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import ExtraTreesClassifier

# Read Exoplanet data
exoplanets_data_raw = pd.read_csv("data/kepler_pre_ml.csv")
print "Exoplanets data read successfully!"

# Remove NAN raws => Corrupted data
exoplanets_data = exoplanets_data_raw.dropna(axis=0, how='any')

# Calculate number of exoplanets
n_exoplanets = exoplanets_data.shape[0]

# Calculate number of features
n_features = exoplanets_data.shape[1] - 3  # minus index, koi_disposition and kepoi_name

# Calculate confirmed exoplanets
n_confirmed = (exoplanets_data['koi_disposition'] == 'CONFIRMED').sum()

# Calculate false positive exoplanets
n_false_positive = (exoplanets_data['koi_disposition'] == 'FALSE POSITIVE').sum()

# Calculate candidate exoplanets
n_candidate = (exoplanets_data['koi_disposition'] == 'CANDIDATE').sum()

# Calculate dropped exoplanets
n_dropped = exoplanets_data_raw.shape[0] - exoplanets_data.shape[0]

# Calculate dropped confirmed exoplanets
n_dropped_confirmed = ((exoplanets_data_raw['dtw_sap_flux'].isnull())
                       & (exoplanets_data_raw['koi_disposition'] == 'CONFIRMED')).sum()

# Calculate dropped false positive exoplanets
n_dropped_false_positive = ((exoplanets_data_raw['dtw_sap_flux'].isnull())
                            & (exoplanets_data_raw['koi_disposition'] == 'FALSE POSITIVE')).sum()

# Calculate dropped candidate exoplanets
n_dropped_candidate = ((exoplanets_data_raw['dtw_sap_flux'].isnull())
                       & (exoplanets_data_raw['koi_disposition'] == 'CANDIDATE')).sum()

# Calculate Dropping Rate
dropping_rate = (float(n_dropped) / exoplanets_data_raw.shape[0]) * 100

# Calculate discovery rate
discovery_rate = (float(n_confirmed) / (n_exoplanets) * 100)

# Print the results
print "\nNumber of exoplanetes dropped: {}".format(n_dropped)
print "Number of confirmed exoplanetes dropped: {}".format(n_dropped_confirmed)
print "Number of false positive exoplanetes dropped: {}".format(n_dropped_false_positive)
print "Number of candidate exoplanetes dropped: {}".format(n_dropped_candidate)
print "Dropping Rate: {:.2f}%".format(dropping_rate)

print "\nTotal number of exoplanets: {}".format(n_exoplanets)
print "Number of features: {}".format(n_features)
print "Number of confirmed exoplanets: {}".format(n_confirmed)
print "Number of false positive exoplanetes: {}".format(n_false_positive)
print "Number of candidate exoplanetes: {}".format(n_candidate)
print "Number of confirmed plus false positive exoplanetes: {}".format(n_confirmed + n_false_positive)
print "Discovery rate of the data: {:.2f}%".format(discovery_rate)

# Divide candidate planets on their respective dataframe to perform final exoplanetary search
candidates = exoplanets_data[exoplanets_data['koi_disposition'] == 'CANDIDATE']
exoplanets = exoplanets_data[exoplanets_data['koi_disposition'] != 'CANDIDATE']

# Extract feature columns
feature_cols = list(exoplanets_data.columns[3:])

# Extract target column 'koi_disposition'
target_col = exoplanets_data.columns[2]

# Show the list of columns
print "Target column: {}".format(target_col)

# Separate the data into feature data and target data (X_all and y_all, respectively) also for candidates
X_all = exoplanets[feature_cols]
y_all = exoplanets[target_col]
X_candidates = candidates[feature_cols]

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X_all, y_all)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Plot the feature importances of the forest
plt.figure(figsize=(18, 6), dpi=80)
plt.title("Feature importances")
plt.bar(range(X_all.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(X_all.shape[1]), indices)
plt.xlim([-1, X_all.shape[1]])
plt.show()

# Remove last 25% of features
n_features_75 = int(n_features * 0.75)

# Extract feature columns
feature_cols = list(X_all.columns[indices[0:27]])

print feature_cols

X_all = X_all[feature_cols]
X_candidates = candidates[feature_cols]

print "\nNumber of features after feature selection: {}".format(X_all.shape[1])


def train_classifier(clf, X_train, y_train):
    # Fits a classifier to the training data.
    start = time()
    clf.fit(X_train, y_train)
    end = time()

    # Print the results
    # print "Trained model in {:.4f} seconds".format(end - start) [Debug]


def predict_labels(clf, features, target, confusion):
    # Makes predictions using a fit classifier based on Precision score.
    start = time()
    y_pred = clf.predict(features)
    end = time()

    # Print and return results
    # print "Made predictions in {:.4f} seconds.".format(end - start)

    if confusion:
        print "Confusion Matrix:"
        print confusion_matrix(target.values, y_pred, labels=['CONFIRMED', 'FALSE POSITIVE'])

    return precision_score(target.values, y_pred, pos_label='CONFIRMED')


def predict(clf, features):
    # Makes predictions using a fit classifier based on Precision score.
    start = time()
    y_pred = clf.predict(features)
    end = time()

    # Print and return results
    # print "\nMade predictions in {:.4f} seconds.".format(end - start)  [Debug]
    return y_pred


def train_predict(clf, X_train, y_train, X_test, y_test, confusion=False):
    # Train and predict using a classifer based on F1 score.

    # Train the classifier
    train_classifier(clf, X_train, y_train)

    train_predict = predict_labels(clf, X_train, y_train, confusion)
    test_predict = predict_labels(clf, X_test, y_test, confusion)

    # Print the results of prediction for both training and testing
    # print "Precision score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train))
    # print "Precision score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test))

    return train_predict, test_predict


def k_fold_train_predict(clf, kf, X_all):
    # Indicate the classifier
    print "\nTraining a {} ".format(clf.__class__.__name__)

    results = pd.DataFrame(columns=['score_train', 'score_test'])
    for train_index, test_index in kf:
        X_train, X_test = X_all.iloc[train_index], X_all.iloc[test_index]
        y_train, y_test = y_all.iloc[train_index], y_all.iloc[test_index]

        # Training and predicting the results for the model
        predict_train, predict_test = train_predict(clf, X_train, y_train, X_test, y_test)
        results.loc[len(results)] = [predict_train, predict_test]

    print "The average precision score for traing is {:.4f}" \
        .format(results['score_train'].describe()['mean'])
    print "The average precision score for traing is {:.4f}" \
        .format(results['score_test'].describe()['mean'])


# Initialize the models
clf_A = GaussianNB()
clf_B = LogisticRegression()
clf_C = SVC()
clf_D = DecisionTreeClassifier()

kf = KFold(len(X_all), n_folds=10, shuffle=True)
# print kf [Debug]

k_fold_train_predict(clf_A, kf, X_all)
k_fold_train_predict(clf_B, kf, X_all)
k_fold_train_predict(clf_C, kf, X_all)
k_fold_train_predict(clf_D, kf, X_all)

# Set the number of training points to 75% of data set
num_train = int(np.floor(X_all.shape[0] * 0.75))

# Set the number of testing points
num_test = X_all.shape[0] - num_train

# Shuffle and split the dataset into the number of training and testing points above
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=69)

# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])
print "Candidates set has {} samples".format(X_candidates.shape[0])

# Training and predicting the results for each model on the entire dataset
print "\nTraining a {} ".format(clf_A.__class__.__name__)
predict_train, predict_test = train_predict(clf_A, X_train, y_train, X_test, y_test, True)
print "Precision score for training set: {:.4f}.".format(predict_train)
print "Precision score for test set: {:.4f}.".format(predict_test)

print "\nTraining a {} ".format(clf_B.__class__.__name__)
predict_train, predict_test = train_predict(clf_B, X_train, y_train, X_test, y_test, True)
print "Precision score for training set: {:.4f}.".format(predict_train)
print "Precision score for test set: {:.4f}.".format(predict_test)

print "\nTraining a {} ".format(clf_C.__class__.__name__)
predict_train, predict_test = train_predict(clf_C, X_train, y_train, X_test, y_test, True)
print "Precision score for training set: {:.4f}.".format(predict_train)
print "Precision score for test set: {:.4f}.".format(predict_test)

print "\nTraining a {} ".format(clf_D.__class__.__name__)
predict_train, predict_test = train_predict(clf_D, X_train, y_train, X_test, y_test, True)
print "Precision score for training set: {:.4f}.".format(predict_train)
print "Precision score for test set: {:.4f}.".format(predict_test)

# Create the parameters list to tune
C_range = np.logspace(-2, 10, 5)
gamma_range = np.logspace(-5, 5, 5)
degree_range = np.logspace(1, 6, 6)
parameters = dict(gamma=gamma_range, C=C_range, degree=degree_range)

# Initialize the classifier
clf = SVC()

# Make an Precision scoring function using 'make_scorer'
precision_scorer = make_scorer(precision_score, pos_label="CONFIRMED")

# Perform grid search on the classifier using the f1_scorer as the scoring method
grid_obj = GridSearchCV(clf, parameters, scoring=precision_scorer)

# Fit the grid search object to the training data and find the optimal parameters
grid_obj = grid_obj.fit(X_train, y_train)

# Get the estimator
clf = grid_obj.best_estimator_

# Report the final F1 score for training and testing after parameter tuning
print "\nTuned model has a training precision score of {:.4f}." \
    .format(predict_labels(clf, X_train, y_train, True))
print "Tuned model has a testing precision score of {:.4f}." \
    .format(predict_labels(clf, X_test, y_test, True))


def discover_possible_exoplanets(clf, X_candidates, candidates):
    # Setting up confirmed and false positive array
    confirmed = []
    false_positive = []

    # Predicting the candidate exoplanets
    results = pd.Series(predict(clf, X_candidates), index=candidates.index)
    candidates.insert(3, 'koi_disposition_pred', results)

    # Looping trough all the planets predictions
    for index, planet in candidates.iterrows():
        if planet['koi_disposition_pred'] == 'CONFIRMED':
            confirmed.append({'name': planet['kepoi_name'], 'index': planet['index']})
        else:
            false_positive.append({'name': planet['kepoi_name'], 'index': planet['index']})

    discovery_rate = float(len(confirmed)) / candidates.shape[0] * 100

    print "\nExoplanets: {} have been Confirmed!".format(len(confirmed))
    print "\nExoplanets: {} are False Positive!".format(len(false_positive))
    print "\nDiscovery Rate: {:.4f}%".format(discovery_rate)


discover_possible_exoplanets(clf, X_candidates, candidates)