import nltk
nltk.download('all')

from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
from nltk import word_tokenize


from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn import metrics

import pprint as pp

############################################
stemmer = EnglishStemmer()


def stemming_tokenizer(text):
    stemmed_text = [stemmer.stem(word) for word in word_tokenize(text, language='english')]
    return stemmed_text


######################################################################



## Dataset containing Positive and negative sentences on Ham-Spam comments on Youtube
data_folder_training_set = "./datasets/Ham_Spam_comments/Training"
data_folder_test_set = "./datasets/Ham_Spam_comments/Test"

training_dataset = load_files(data_folder_training_set)
test_dataset = load_files(data_folder_test_set)
print
print "----------------------"
print(training_dataset.target_names)
print "----------------------"
print

# Load Training-Set
X_train, X_test_DUMMY_to_ignore, Y_train, Y_test_DUMMY_to_ignore = train_test_split(training_dataset.data,
                                                                                    training_dataset.target,
                                                                                    test_size=0.0)
target_names = training_dataset.target_names

# Load Test-Set
X_train_DUMMY_to_ignore, X_test, Y_train_DUMMY_to_ignore, Y_test = train_test_split(test_dataset.data,
                                                                                    test_dataset.target, train_size=0.0)

print
print "----------------------"
print "Creating Training Set and Test Set"
print
print "Training Set Size"
print(Y_train.shape)
print
print "Test Set Size"
print(Y_test.shape)
print
print("Classes:")
print(target_names)
print "----------------------"

## Vectorization object
vectorizer = TfidfVectorizer(strip_accents=None, preprocessor=None)

## classifiers
kNN_classifier = KNeighborsClassifier()

## With a Pipeline object we can assemble several steps
## that can be cross-validated together while setting different parameters.

pipeline = Pipeline([
    ('vect', vectorizer),
    ('kNN', kNN_classifier),
])

## Setting parameters.
## Dictionary in which:
##  Keys are parameters of objects in the pipeline.
##  Values are set of values to try for a particular parameter.
parameters = {
    'vect__tokenizer': [None, stemming_tokenizer],
    'vect__ngram_range': [(1, 1), (1, 2), ],
    'kNN__n_neighbors': [3, 5]
}

## Create a Grid-Search-Cross-Validation object
## to find in an automated fashion the best combination of parameters.
grid_search = GridSearchCV(pipeline,
                           parameters,
                           scoring=metrics.make_scorer(metrics.average_precision_score, average='weighted'),
                           cv=10,
                           n_jobs=2,
                           verbose=10)

## Start an exhaustive search to find the best combination of parameters
## according to the selected scoring-function.
print
grid_search.fit(X_train, Y_train)
print

## Print results for each combination of parameters.
number_of_candidates = len(grid_search.cv_results_['params'])
print("Results:")
for i in range(number_of_candidates):
    print(i, 'params - %s; mean - %0.3f; std - %0.3f' %
          (grid_search.cv_results_['params'][i],
           grid_search.cv_results_['mean_test_score'][i],
           grid_search.cv_results_['std_test_score'][i]))

print
print("Best Estimator:")
pp.pprint(grid_search.best_estimator_)
print
print("Best Parameters:")
pp.pprint(grid_search.best_params_)
print
print("Used Scorer Function:")
pp.pprint(grid_search.scorer_)
print
print("Number of Folds:")
pp.pprint(grid_search.n_splits_)
print

# Let's train the classifier that achieved the best performance,
# considering the select scoring-function,
# on the entire original TRAINING-Set
Y_predicted = grid_search.predict(X_test)

# Evaluate the performance of the classifier on the original Test-Set
output_classification_report = metrics.classification_report(
    Y_test,
    Y_predicted,
    target_names=target_names)
print
print "----------------------------------------------------"
print(output_classification_report)
print "----------------------------------------------------"
print

# Compute the confusion matrix
confusion_matrix = metrics.confusion_matrix(Y_test, Y_predicted)
print
print("Confusion Matrix: True-Classes X Predicted-Classes")
print(confusion_matrix)
print
print


# Compute the Normalized-accuracy
normalized_accuracy = metrics.accuracy_score(Y_train,Y_predicted)
print
print("Normalized Accuracy: ")
print(normalized_accuracy)
print



# Compute the Matthews Corrcoef value
matthews_corr_coef = metrics.matthews_corrcoef(Y_train,Y_predicted)
print
print("Matthews Corr Coef: ")
print matthews_corr_coef
print
