import nltk.classify.util
from nltk import collections
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk import precision



def extract_features(word_list):
    return dict([(word, True) for word in word_list])


if __name__ == '__main__':
    # using nltk's movie reviews as training data
    positive_file_ids = movie_reviews.fileids('pos')
    negative_file_ids = movie_reviews.fileids('neg')

    # positive and negative reviews
    features_pos = [(extract_features(movie_reviews.words(fileids=[f])), 'Positive') for f in positive_file_ids]
    features_neg = [(extract_features(movie_reviews.words(fileids=[f])), 'Negative') for f in negative_file_ids]

    # divide data into training and testing datasets
    threshold_factor = 0.8  # split data into train and test (80/20)
    threshold_pos = int(threshold_factor * len(features_pos))
    threshold_neg = int(threshold_factor * len(features_neg))

    # extract the features
    features_train = features_pos[:threshold_pos] + features_neg[:threshold_neg]
    features_test = features_pos[threshold_pos:] + features_neg[threshold_neg:]
    print("\nNumber of training datapoints: ", len(features_train))
    print("Number of test datapoints: ", len(features_test))

    # define classifer object and train it
    NBClassifier = NaiveBayesClassifier.train(features_train)
    print("\nAccuracy of NBClassifer: ", nltk.classify.util.accuracy(NBClassifier, features_test))

    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for i, (features, label) in enumerate(features_test):
        refsets[label].add(i)
        observed = NBClassifier.classify(features)
        testsets[observed].add(i)

    print("refsets: ", refsets)
    print("testsets: ", testsets)

    # print top 10 most informative words
    for item in NBClassifier.most_informative_features()[:10]:
        print(item[0])

    # sample input sentences
    input_reviews = [
        "It is an amazing movie",
        "This is a dull movie. I would never recommend it to anyone.",
        "The cinematography is pretty great in this movie",
        "The direction was terrible and the story was all over the place"
    ]

    # run classifier on those input sentences and obtain the predictions
    print("\nPredictions: ")
    for review in input_reviews:
        print("\nReview: ", review)
        probability_dist = NBClassifier.prob_classify(extract_features(review.split()))
        pred_sentiment = probability_dist.max()

        # print output
        print("Predicted Sentiment: ", pred_sentiment)
        print("Probability: ", round(probability_dist.prob(pred_sentiment), 2))
