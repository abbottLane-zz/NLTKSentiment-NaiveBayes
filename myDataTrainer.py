__author__ = 'wlane'

import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.tokenize import *

def word_feats(words):
    return dict([(word, True) for word in words])

def get_review_class(data, name):

    data_clean = data.decode('utf-8')
    # tokenize sentences
    sents = nltk.tokenize.sent_tokenize(data_clean)

    words = []
    for sent in sents:
        words += nltk.tokenize.wordpunct_tokenize(sent)

    test1_features = word_feats(words)

    # Print results of classification
    #print "\nReview Excerpt: " + data
    return name + " : " + classifier.classify(test1_features)


negative_ids = movie_reviews.fileids('neg')
positive_ids = movie_reviews.fileids('pos')

negative_features = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negative_ids]
positive_features = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in positive_ids]

# Split our total data set (2000 reviews) into 3rds: 2/3rds we will use to train,
# the remaining 1/3rd we will use to test
# note that the positive set and the negative set are both cut into 2/3 : 1/3 divisions
negcutoff = len(negative_features)*2/3
poscutoff = len(positive_features)*2/3

# Assemble the  training and testing features into their own list
train_features = negative_features[:negcutoff] + positive_features[:poscutoff]
test_features = negative_features[negcutoff:] + positive_features[poscutoff:]
print 'Out of the %d total reviews, we will train our classifier on %d instances, ' \
      'and test how well our classifier works with the remaining  %d instances' % (len(train_features) + len(test_features), len(train_features), len(test_features))

# Train your naive bayes classifier with the data
classifier = NaiveBayesClassifier.train(train_features)

# Run your test set through the trained classifier, print the accuracy your classifier had in guessing the sentiment
print 'accuracy:', nltk.classify.util.accuracy(classifier, test_features)

# Print out the most informative features
classifier.show_most_informative_features()

#####################################################################
###################  Now test with real data               ##########
#####################################################################
# all_training_data = train_features + test_features
#
# # Train your naive bayes classifier with the data
# print "\nRetraining ALL data ..."
# classifier = NaiveBayesClassifier.train(all_training_data)

print "Now testing recent movie reviews from online ..."

#######################################
# Cinderella
########################################
with open("data/Cinderella.txt", "r") as my_file:
    data = my_file.read().replace('\n',' ')

print get_review_class(data, "Cinderella")

#######################################
# The Gunman
########################################
with open("data/TheGunman.txt", "r") as my_file:
    data = my_file.read().replace('\n',' ')

print get_review_class(data, "The Gunman")

#######################################
# Insurgent
########################################
with open("data/Insurgent.txt", "r") as my_file:
    data = my_file.read().replace('\n',' ')

print get_review_class(data, "Insurgent")

#######################################
# Spongebob
########################################
with open("data/SpongeBob.txt", "r") as my_file:
    data = my_file.read().replace('\n',' ')

print get_review_class(data, "SpongeBob Movie")