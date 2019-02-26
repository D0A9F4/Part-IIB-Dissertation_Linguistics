from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from os import listdir
from sklearn import svm
import numpy as np
from os import path

directory = r"C:\Users\narci\Dissertation\nli-shared-task-2017\nli-shared-task-2017\data\essays\train\test"

essays = [("Chinese",path.join(directory + r"\chinese",f)) for f in listdir(r"C:\Users\narci\Dissertation\nli-shared-task-2017\nli-shared-task-2017\data\essays\train\test\chinese")] + \
             [("Spanish",path.join(directory + r"\spanish",f)) for f in listdir(r"C:\Users\narci\Dissertation\nli-shared-task-2017\nli-shared-task-2017\data\essays\train\test\spanish")] + \
    			[("Japanese",path.join(directory + r"\japanese",f)) for f in listdir(r"C:\Users\narci\Dissertation\nli-shared-task-2017\nli-shared-task-2017\data\essays\train\test\japanese")] + \
        			[("Tagalog",path.join(directory + r"\tagalog",f)) for f in listdir(r"C:\Users\narci\Dissertation\nli-shared-task-2017\nli-shared-task-2017\data\essays\train\test\tagalog")] + \
            			[("German",path.join(directory + r"\german",f)) for f in listdir(r"C:\Users\narci\Dissertation\nli-shared-task-2017\nli-shared-task-2017\data\essays\train\test\german")]
			
essays = [(a, open(b).read()) for (a,b) in essays]

vectorizer = TfidfVectorizer(ngram_range=(1,1),
                             analyzer="word")

vectors = vectorizer.fit_transform([f for (label,f) in essays])
labels = [label for (label, f) in essays]

X_train, X_test, y_train, y_test = train_test_split(vectors, labels, random_state=27)  

#print(X_train[0:5])
#data = {}
#for lab in y_test:
#    if lab in data.keys():
#        data[lab] = data[lab] + 1
#    else:
#        data[lab] = 1
#print(data)


clf = svm.SVC(gamma="scale", kernel="linear", class_weight = "balanced", random_state=27)
clf.fit(X_train, y_train)


results = clf.predict(X_test)
accuracy = np.sum(results == np.array(y_test))/len(results)
print(accuracy)

from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import sem

def evaluate_cross_validation(clf, X, y, K):
    k_cv = KFold(n_splits=K, shuffle = True, random_state = 27)
    scores = cross_val_score(clf, X, y, cv=k_cv)
    print(scores)
    print(("Mean score: {0: .3f} (+/-{1: .3f})").format(
            np.mean(scores), sem(scores)))

        
evaluate_cross_validation(clf, vectors, labels, 5)

from sklearn import metrics
def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
    
    #clf.fit(X_train, y_train)
    
    print ("Accuracy on training set:")
    print (clf.score(X_train, y_train))
    print ("Accuracy on testing set:")
    print (clf.score(X_test, y_test))
    
#    y_pred = clf.predict(X_test)
#    
#    print ("Classification Report:")
#    print (metrics.classification_report(y_test, y_pred))
#    print ("Confusion Matrix:")
#    print (metrics.confusion_matrix(y_test, y_pred))
    
train_and_evaluate(clf, X_train, X_test, y_train, y_test)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig

#print_confusion_matrix(metrics.confusion_matrix(y_test, clf.predict(X_test)), ["Chinese", "German", "Japanese", "Spanish", "Tagalog"], figsize = (10,7), fontsize=14)

#import pickle
#with open('NLIwTAG_classifier1-2balanced', 'wb') as picklefile:  
#    pickle.dump(clf,picklefile)

def most_informative_features(vectorizer, clf, n=10):
    labelid = 10
    feature_names = vectorizer.get_feature_names()
    svm_coef = clf.coef_.toarray()
    print(svm_coef)
    topn = sorted(zip(svm_coef[labelid], feature_names))[-n: ]
    
    for coef, feat in topn:
        print(feat, coef)
        
most_informative_features(vectorizer, clf)
        
