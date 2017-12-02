import csv
import math
import sys
import nltk
from nltk.corpus import stopwords
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans

import time 

class TweetCluster(object):

	"""This section sanitizes the tweets and removes unwanted characters, putting them in list format. 
	Creates a doc_words dict {word: doc_count of word} and removes all instances below a certain frequency."""
	def __init__(self, tweets_csv):
		csvfile = open(tweets_csv, 'r')
		reader = csv.reader(csvfile, delimiter=',')
		stop_words = set(stopwords.words('english'))
		for word in ['amp', 'today', 'tomorrow']:
			stop_words.add(word)
		rep = ['https', '.', '!', '?', ',', ';', '&', '(', ')', '\\', ':', '/', 
		'\n', '\'', '-', '\"', '#', '`', '\xe2\x80\x99', '\xe2\x80\xa6', '\xe2\x9e\xa1\xef\xb8\x8f', 
		'\xf0\x9f\x87\xba\xf0\x9f\x87\xb8', '\xe2\x80\x93', '\xe2\x80\x94', '\xe2\x80\x9d']
		self.sanitized_tweets = []
		self.raw_tweets = []
		self.doc_words = {}
		for i, row in enumerate(reader): 
			if i > 0: 
				self.raw_tweets.append(row[6])
				tweet_list = self.sanitize(row[6], stop_words, rep)
				self.sanitized_tweets.append(tweet_list)
				unique = set()
				for word in tweet_list:
					if word not in unique:
						self.doc_words[word] = self.doc_words.get(word, 0) + 1
						unique.add(word)
		csvfile.close()
		for word in self.doc_words.keys():
			if self.doc_words[word] < 3:
				del(self.doc_words[word])
		self.tfIdf()
		print 'unique words in tweets:', len(self.doc_words), '\n# of tweets:', len(self.sanitized_tweets)

	""" Function for preprocessing tweets. Should include removing stop words as well. Removes all spaces,
		words under a certain lenth, and unwanted punctuation. DOES NOT removes words that appear 
		infreuqently. """
	def sanitize(self, tweet, stop_words, rep):
		for punc in rep:
			tweet = tweet.replace(punc, ' ')
		return [word for word in tweet.lower().split(' ') if len(word) > 2 and word not in stop_words]

	""" Function for building a tf_idf statistic over the tweets. """
	def tfIdf(self):
		docs = len(self.sanitized_tweets)
		unique_word_count = len(self.doc_words)
		np_word_index = {}
		inverse_np_word_index = {}
		idf = np.zeros(unique_word_count)
		tf_idf = np.zeros((docs, unique_word_count))

		for i,word in enumerate(self.doc_words):
			idf[i] = math.log(docs / self.doc_words[word])
			np_word_index[word] = i
			inverse_np_word_index[i] = word

		for i, tweet in enumerate(self.sanitized_tweets): 
			unique = {}
			for word in tweet:
					unique[word] = unique.get(word, 0) + 1
			for word in unique: 
				if word in self.doc_words: 
					tf_idf[i][np_word_index[word]] = idf[np_word_index[word]]*unique[word]
		self.tf_idf = tf_idf
		self.np_word_index = np_word_index
		self.inverse_np_word_index = inverse_np_word_index

	""" Perform k means clustering on a tf_idf statistic."""
	def kMeans(self, k, tf_idf):
		km = KMeans(k)
		km.fit(tf_idf)
		return km.cluster_centers_, km.labels_


	""" Perform iterative clustering to track the reduction in squared error as k increases to better select
		a k value. """
	def cluster(self): 
		squared_error = []
		tf_idf = self.getTfIdf()
		for i in range(25, 55, 5):
			cluster_means, tweet_labels = self.kMeans(i, tf_idf)
			clusters = []
			error = 0
			for j in range(i):
				clusters.append([k for k, label in enumerate(tweet_labels) if label == j])
			for j, cluster in enumerate(clusters):
				error += self.calcError(tf_idf, cluster, cluster_means[j])
			squared_error.append(error)
		return np.array(squared_error)

	""" Perform recursive clustering: cluster on k = 2, select cluster with largest squared error, cluster this
		cluster on k = 2, rejoin with original clusters, and choose the cluster with largest squared error again. """
	def recursiveKMeans(self, k, tf_idf):
		next_to_cluster = range(tf_idf.shape[0])
		largest_error = -1
		clusters = []
		cluster_means = []
		errors = []
		invalid_labels = set()
		for i in range(1, k):
			# print len(next_to_cluster)
			invalid_labels.add(largest_error)
			next_means, next_labels = self.kMeans(2, tf_idf[next_to_cluster])
			for j in range(2):
				cluster = [next_to_cluster[ind] for ind, label in enumerate(next_labels) if label == j]
				clusters.append(cluster)
				cluster_means.append(next_means[j])
				errors.append(self.calcError(tf_idf, cluster, next_means[j]))
			largest_error = self.largestError(errors, invalid_labels)
			next_to_cluster = clusters[largest_error]
		
		labels = np.zeros(tf_idf.shape[0])
		means = np.zeros((k, tf_idf.shape[1]))
		label = 0
		for i, cluster in enumerate(clusters):
			if i not in invalid_labels:
				labels[cluster] = label
				means[label] = cluster_means[i]
				label += 1
		return means, labels

	def largestError(self, errors, invalid_labels):
		largest_error, ind = 0, 0
		for i, error in enumerate(errors):
			if i not in invalid_labels and error > largest_error:
				largest_error = error
				ind = i
		return ind

	def PCA(self, tf_idf, axes):
		mu = np.mean(tf_idf, axis=0)
		reduced = np.zeros((tf_idf.shape[1], axes))
		tf_idf_p = tf_idf - mu
		cov = np.dot(tf_idf_p.T, tf_idf_p)
		w, v = np.linalg.eig(cov)
		sort = np.argsort(w)
		for i in range(axes):
			reduced[:,i] = v[:, sort[i]]
		return reduced


	def calcError(self, tf_idf, cluster, mean):
		samp_to_mean = tf_idf[cluster] - mean
		return (samp_to_mean * samp_to_mean).sum(axis=1).sum()

	def searchTweets(self, word):
		return [i for i, stat in enumerate(self.tf_idf[:,self.np_word_index[word]]) if stat > 0]

	def getTweets(self):
		return self.sanitized_tweets

	def getTfIdf(self):
		return self.tf_idf

	def getIndices(self):
		return self.np_word_index, self.inverse_np_word_index

	def getRawTweets(self):
		return self.raw_tweets

def eval(cluster_means, tweet_labels, cluster):
	tf_idf = cluster.getTfIdf()
	np_word_index, inverse_np_word_index = cluster.getIndices()
	clusters = []
	total_error = 0
	for i in range(k):
		cluster_i = [j for j, label in enumerate(tweet_labels) if label == i]
		clusters.append(cluster_i)
		samp_to_mean = tf_idf[cluster_i] - cluster_means[i]
		error = (samp_to_mean * samp_to_mean).sum(axis=1).sum()
		total_error += error
		print 'tweets in cluster', str(i) +':', str(len(cluster_i)) +';', 'error:', error
	print 'total error:', total_error

	n = 8
	for i, cluster_mean in enumerate(cluster_means):
		sig_words_indices = cluster_mean.argsort()[-n:][::-1]
		sig_words = [inverse_np_word_index[ind] for ind in sig_words_indices]
		print str(i) + ':', sig_words
	return clusters

def printTweetsInCluster(tweets, cluster):
	for ind in cluster:
		print str(ind) + ':  ', tweets[ind]

def graphTweets(tf_idf): 
	eig_basis = cluster.PCA(tf_idf, 2)
	transformed_tf_idf = np.dot(tf_idf, eig_basis)
	plt.plot(transformed_tf_idf[:,0], transformed_tf_idf[:,1], 'ro')
	plt.show()

cluster = TweetCluster(sys.argv[1])
tf_idf = cluster.getTfIdf()
raw_tweets = cluster.getRawTweets()
k = 10

t0 = time.time()
print 'No PCA Non Recursive Approach'
cluster_means, tweet_labels = cluster.kMeans(k, tf_idf)
clusters_n = eval(cluster_means, tweet_labels, cluster)
t1 = time.time()
print 'time: ', t1 - t0

t0 = time.time()
print 'No PCA Recursive Approach'
cluster_means, tweet_labels = cluster.recursiveKMeans(k, tf_idf)
clusters_r = eval(cluster_means, tweet_labels, cluster)
t1 = time.time()
print 'time: ', t1 - t0

t0 = time.time()
print 'PCA Non Recursive Approach'


t1 = time.time()
print 'time: ', t1 - t0

t0 = time.time()
print 'PCA Recursive Approach'


t1 = time.time()
print 'time: ', t1 - t0

t0 = time.time()
print 'PCA Recursive Approach with automatic K selection'


t1 = time.time()
print 'time: ', t1 - t0


