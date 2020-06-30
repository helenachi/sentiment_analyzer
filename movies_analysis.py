"""
Helena Chi, helenac2
LING406 SP19
Term Project: Sentiment Classifier

movies_analysis.py:
    -   data retrieved from Cornell Movie Reviews dataset release June 2004:
        http://www.cs.cornell.edu/people/pabo/movie-review-data (Bo Pang and Lillian Lee, 2004)
    -   pre-processes data and formatted into list of features
    -   sentiment analysis on features by Machine Learning Models/Classifiers:
        Naive Bayes, Decision Tree, and Support Vector Machine

Note: the following warning occurs when this file is run:
    -   /Users/helenachi/PycharmProjects/checkr_exercise/venv/lib/python3.6/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
        "the number of iterations.", ConvergenceWarning)

"""

# preprocess imports
import re
import glob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# classifier imports
import numpy
import nltk.classify.util
from collections import defaultdict
from nltk import precision
from nltk import recall
from nltk.classify import NaiveBayesClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

StopWords = set(stopwords.words('english'))


"""
Pre-processes the data in a given file. Converts to lowercase, 
filters out punctuation and numbers.
@:param     filename                name of text file to pre-process
# @:param     level                   number indicating what kind of pre-processing to go through
@:return    lemmatized_result       pre-processed array of individual words from filename

levels:
    0 = full pre-processing (lowercase, remove punctuation and numbers, remove stop words, lemmatize
    1 = bag_of_words baseline model (only lowercase and remove punctuation and numbers
    2 = full pre-processing without removing stop words
    3 = full pre-processing without elm
"""
def preprocess(filename, level):
    file = open(filename, "r+")

    # lowercase only
    text = file.read().lower()
    file.close()

    # keeps all lowercase letters, a spaces, and apostrophes; no punctuation or numbers
    bag_of_words = re.sub('[^a-z\ \']+', " ", text).split()

    # removing stopwords
    words = [word for word in list(bag_of_words) if word not in StopWords]

    # lemmatizing
    lemmatizer = WordNetLemmatizer()
    lemmatized_result = []
    for word in words:
        lemmatized_version = lemmatizer.lemmatize(word)
        if lemmatized_version not in lemmatized_result:
            lemmatized_result.append(lemmatized_version)

    # returns an array of preprocessed words given a filename
    # return lemmatized_result

    if level == 0:
        return lemmatized_result
    elif level == 1:
        return bag_of_words
    elif level == 2:
        no_stop_words = []
        for word in bag_of_words:
            lemmed = lemmatizer.lemmatize(word)
            if lemmed not in no_stop_words:
                no_stop_words.append(lemmed)
        return no_stop_words
    elif level == 3:
        return words
    else:
        return lemmatized_result



"""
Converts words in array to a feature set.
@:param     word_list               list of words to convert to feature set
@:return                            set of features 
"""
def extract_features(word_list):
    return dict([(word, True) for word in word_list])


"""
Retrieve positive and negative feature sets from Cornell Movie Reviews.
@:param     none
@:return    features    list of features from all movie reviews, separated by polarity

features = [features_pos, features_neg]
    total feature_set = positive + negative feature sets
    [0] = positive features
    [1] = negative features
"""
def get_movie_data():
    print("Retrieving movie data...")
    # get cornell's movie data
    filenames_pos = glob.glob("cornell_movie_reviews/pos/*.txt")
    filenames_neg = glob.glob("cornell_movie_reviews/neg/*.txt")

    # positive and negative reviews
    print("Feature Set: fully pre-processed")
    features_pos = [(extract_features(preprocess(f, 0)), 'Positive') for f in filenames_pos]
    features_neg = [(extract_features(preprocess(f, 0)), 'Negative') for f in filenames_neg]

    # print("Feature Set: Bag of Words")
    # features_pos = [(extract_features(preprocess(f, 1)), 'Positive') for f in filenames_pos]
    # features_neg = [(extract_features(preprocess(f, 1)), 'Negative') for f in filenames_neg]

    # print("Feature Set: stop words unfiltered")
    # features_pos = [(extract_features(preprocess(f, 2)), 'Positive') for f in filenames_pos]
    # features_neg = [(extract_features(preprocess(f, 2)), 'Negative') for f in filenames_neg]

    # print("Feature Set: unlemmatized")
    # features_pos = [(extract_features(preprocess(f, 3)), 'Positive') for f in filenames_pos]
    # features_neg = [(extract_features(preprocess(f, 3)), 'Negative') for f in filenames_neg]

    features = [features_pos, features_neg]

    return features


"""
Divides a given feature set into training and testing sets based on 
cross-validation arguments.
@:param     feature_set         set to be separated into testing and training sets
@:param     cur_fold            number indication current iteration of cross-validation
@:param     num_folds           number of folds specified for cross-validation
"""
def divide_feature_set(feature_set, cur_fold, num_folds):
    sub_size = int(len(feature_set) / num_folds)

    # indicies for cross validation
    start = int(cur_fold * sub_size)
    # mid = sub_size
    end = int((cur_fold + 1) * sub_size)

    train = feature_set[:start] + feature_set[end:]
    test = feature_set[start:] + feature_set[:sub_size]

    return [train, test]


"""
Returns precision and recall values for a classifier trained by a training set and 
tested on a testing set.
@:param     classifer           classifier to test
@:param     features_train      training set
@:param     features_test       testing set
@:return    [p,r]               list containing positive and negative precision and recall values

[p, r] = 
    [[
            pos_precision,
            neg_precision
     ],
     [
            pos_recall,
            neg_recall
     ]]
"""
def get_pr(classifier, features_train, features_test):
    refsets = defaultdict(set)
    testsets = defaultdict(set)
    for i, (feats, label) in enumerate(features_test):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)

    # [0] = positive precision
    # [1] = negative precision
    p = [precision(refsets['Positive'], testsets['Positive']), precision(refsets['Negative'], testsets['Negative'])]

    # [0] = positive recall
    # [1] = negative recall
    r = [recall(refsets['Positive'], testsets['Positive']), recall(refsets['Negative'], testsets['Negative'])]

    return [p, r]


"""
Runs cross-validation on a given feature set with specified classifier.
@:param     classifier          classifier to test
@:param     features            feature set to divide into testing and training sets for classifier
@:return    accuracies          list of accuracy values

accuracies:
    [0] = average accuracy
    [1] = average positive precision
    [2] = average negative precision
    [3] = average positive recall
    [4] = average negative recall
"""
def cross_validation(classifer, features):
    num_folds = 5
    accuracy = []
    pp = []  # positive precision
    np = []  # negative precision
    pr = []  # positive recall
    nr = []  # negative recall

    for i in range(num_folds):
        training = divide_feature_set(features[0], i, num_folds)[0] + divide_feature_set(features[1], i, num_folds)[0]
        testing = divide_feature_set(features[0], i, num_folds)[1] + divide_feature_set(features[1], i, num_folds)[1]

        trained_classifier = classifer.train(training)

        a = nltk.classify.util.accuracy(trained_classifier, testing)
        accuracy.append(a)

        precision_recall = get_pr(trained_classifier, training, testing)
        pp.append(precision_recall[0][0])
        np.append(precision_recall[0][1])
        pr.append(precision_recall[1][0])
        nr.append(precision_recall[1][1])

    accuracies = [numpy.mean(accuracy), numpy.mean(pp), numpy.mean(np), numpy.mean(pr), numpy.mean(nr)]
    return accuracies


"""
Prints formatted accuracies data given a name and list of accuracies.
@:param     classifier_name             name of specified classifier as string
@:param     accuracies                  list of accuracies from specified classifier
@return     none
"""
def print_metrics(classifer_name, accuracies):
    print(classifer_name, "Model metrics:")
    print("accuracy: ", accuracies[0])
    print("positive precision: ", accuracies[1])
    print("negative precision: ", accuracies[2])
    print("positive recall: ", accuracies[3])
    print("negative recall: ", accuracies[4], "\n")


"""
Naive Bayes Model
"""
def NaiveBayesModel(feature_set):
    c = NaiveBayesClassifier
    accuracies = cross_validation(c, feature_set)
    print_metrics("Naive Bayes", accuracies)


"""
Decision Tree Model
"""
def DecisionTreeModel(feature_set):
    c = SklearnClassifier(DecisionTreeClassifier())
    accuracies = cross_validation(c, feature_set)
    print_metrics("Decision Tree", accuracies)



"""
Support Vector Machine Linear Mpdel)
"""
def SVMModel(feature_set):
    c = SklearnClassifier(LinearSVC())
    accuracies = cross_validation(c, feature_set)
    print_metrics("Support Vector Machine Linear", accuracies)


# main
features = get_movie_data()
print("Running Classifier Models...")
NaiveBayesModel(features)
DecisionTreeModel(features)
SVMModel(features)
