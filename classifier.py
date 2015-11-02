# -*- coding: utf-8 -*-
import json
import re
import numpy as np
from sklearn import grid_search
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import cross_validation
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.lda import LDA
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import matplotlib
import pickle
import os
import sys

def generateFeatures(headings, cities, sections):
	special_chars = [';', '-', '?', '#', '*', '/', '_', '(', ')', '&', ':', '<', '>',
                '{', '}', '.', '!', '@', '\\', '%', '~', '`', '^', '-', '+'
                , '=', '[', ']', '\'', '"', ',']
	X = []
	for datapoint in zip(headings, cities, sections):
		upperCharsCount = sum(1 for c in datapoint[0] if c.isupper())
		specialCharsCount = sum(1 for c in datapoint[0] if c in special_chars)
		text = datapoint[0]

		text = re.sub(r'£|\$', ' denonimation', text)

		for sc in special_chars:
			text = text.replace(sc, '')

		for sym in re.findall(r'\d+|\d+[.,]\d+', text):
			text = text.replace(sym, ' numbr ')
		text = text.replace('  ', ' ')

		l = [text, str(upperCharsCount), str(specialCharsCount), datapoint[2]]
		X.append(" ".join(ele for ele in l)) 
	return X

def generateDefaultFeatures(headings):
	X = []
	for data in headings:
		X.append(data)
	return X

def readFile(file):
	data = []
	f = open(file)
	for line in f:
		data.append(json.loads(line))
	return data

def getData(data):
	headings = []
	cities = []
	sections = []
	y = []
	for x in data:
		headings.append(x['heading'])
		cities.append(x['city'])
		sections.append(x['section'])
		try:
			y.append(x['category'])
		except:
			continue
	return headings, cities, sections, y

def trainClassifier(classifier, X, y):
	vectorizer = TfidfVectorizer(analyzer='char', use_idf=True, sublinear_tf=True, stop_words='english', ngram_range=(1,3), lowercase=True)
	# vectorizer = TfidfVectorizer()
	X = vectorizer.fit_transform(X).toarray()
	if classifier == "SVC":
		clf = LinearSVC()
		# parameters = {'kernel':['linear', 'rbf'], 'C':[0.1, 1, 10]}
		# clf = grid_search.GridSearchCV(clf, parameters)
		clf.fit(X, y)
		# print clf.best_params_
		clf_vect = [clf, vectorizer]
		f = open('svm.pkl', 'wb')
		pickle.dump(clf_vect, f)
		return clf, vectorizer
	elif classifier == "RF":
		clf = RandomForestClassifier()
		clf.fit(X, y)
		return clf, vectorizer
	elif classifier == "MNB":
		clf = MultinomialNB()
		clf.fit(X, y)
		return clf, vectorizer
	elif classifier == "LDA":
		clf = LDA()
		clf.fit(X, y)
		return clf, vectorizer
	elif classifier == "KNN":
		clf = KNeighborsClassifier()
		clf.fit(X, y)
		return clf, vectorizer

def confusionMatrix(clf, XTest, yTrue, cmap=plt.cm.Blues):
	matplotlib.rc('xtick', labelsize=6)
	matplotlib.rc('ytick', labelsize=6)
	yPred = clf.predict(XTest.toarray()).tolist()

	cm = confusion_matrix(yTrue, yPred)
	plt.figure()
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title("Confusion Matrix")
	plt.colorbar()
	mylist =  list(set(yTrue + yPred))
	mylist.sort()
	tick_marks = np.arange(len((mylist)))
	plt.xticks(tick_marks, mylist, rotation=38)
	plt.yticks(tick_marks, mylist)
	plt.tight_layout()
	plt.savefig('cm2.eps', format='eps', dpi=300)

def main():
	data = readFile('training.json')
	headings, cities, sections, y = getData(data)
	# X = generateDefaultFeatures(headings)
	X = generateFeatures(headings, cities, sections)

	if not os.path.exists("svm.pkl"):
		clf, vectorizer = trainClassifier("SVC", X, y)
	else:
		f = open("svm.pkl")
		clf, vectorizer = pickle.load(f)
		print "loading complete"

	# print clf.predict(vectorizer.transform(["iPhone"]))
	sys.exit()
	test = readFile('sample-test.in.json')
	headings_test, cities_test, sections_test, y_test = getData(test)
	X_test = generateFeatures(headings_test, cities_test, sections_test)

	y_test = []
	fytest = open('sample-test.out.json')
	for line in fytest:
	    y_test.append(line.strip('\n'))

	X_test = vectorizer.transform(X_test)
	# confusionMatrix(clf, X_test, y_test)
	print "Accuracy: ", cross_validation.cross_val_score(clf, X_test.toarray(), y_test).mean()
	print "F1: ", sklearn.metrics.f1_score(y_test, clf.predict(X_test.toarray()).tolist()).mean()
	print "Precision: ", sklearn.metrics.precision_score(y_test, clf.predict(X_test.toarray()).tolist()).mean()
	print "Recall: ", sklearn.metrics.recall_score(y_test, clf.predict(X_test.toarray()).tolist()).mean()

if __name__ == "__main__":
	main()