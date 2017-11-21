import csv
import math
import sys
import numpy as np
import time


t0 = time.time()
csvfile = open(sys.argv[1], 'r')
reader = csv.reader(csvfile, delimiter=',')

docs = -1
words = {}

def sanitize(str):
	rep = ['.', '!', '?', ',', ';', '&', '(', ')', '\\', ':', '/', 
	'\n', '\'', '-', '\"', '#', '`', '\xe2\x80\x99', '\xe2\x80\xa6', '\xe2\x9e\xa1\xef\xb8\x8f', 
	'\xf0\x9f\x87\xba\xf0\x9f\x87\xb8', '\xe2\x80\x93', '\xe2\x80\x94', '\xe2\x80\x9d']
	for punc in rep:
		str = str.replace(punc, ' ')
	return str.lower()

"""Section sanitizes the tweets. Removes unwanted characters."""
san_tweets = open('sanitized_tweets.csv', 'w')
for row in reader: 
	if docs >= 0: 
		tweet = sanitize(row[6])
		san_tweets.write(str(docs) + ', ' + tweet + '\n')
		unique = set()
		for word in tweet.split(' '):
			if word not in unique:
				if word in words: words[word] = words[word] + 1
				else: words[word] = 1
				unique.add(word)
	docs += 1
san_tweets.close()

"""Section removes words that appear a limited number of times. Done to reduce dimensionality."""
del_keys = ['']
for word in words:
	if words[word] < 3 or len(word) == 1: del_keys.append(word)
for key in del_keys: del(words[key])

print 'unique words:', len(words), '\n# of docs: ', docs

"""Building the idf, and tf dicts. Using a indexed dict to store the index into each row for a word
   and a _indexed dict to store the word associated with each index for cluster analysis later on."""
indexed = {}
_indexed = {}
idf = np.zeros(len(words))
tf_idf = np.zeros((docs, len(words)))

i = 0
for word in words:
	idf[i] = math.log(docs / words[word])
	indexed[word] = i
	_indexed[i] = word
	i += 1

# Testing the idf terms for accuracy.
print 'clinton appears in ' + str(words['clinton']) + ' docs'
print 'clintons index is ' + str(indexed['clinton'])
print 'log of docs/clintons count is ' + str(math.log(docs / words['clinton']))
print 'stored value at clintons index is ' + str(idf[indexed['clinton']])

csvfile = open('sanitized_tweets.csv', 'r')
reader = csv.reader(csvfile, delimiter=',')

i = 0
for row in reader: 
	tweet = row[1]
	unique = {}
	for word in tweet.split(' '):
		if word != '':
			if word not in unique: unique[word] = 1
			else: unique[word] = unique[word] + 1
	for word in unique: 
		if word in words: tf_idf[i][words[word]] = idf[words[word]]*unique[word]
	i += 1

t1 = time.time()
print 'time: ' + str(t1 - t0)