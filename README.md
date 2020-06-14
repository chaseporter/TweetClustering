## To run ##
Run on python3   

Make sure necesssary python libraries are installed:   
  - numpy  
  - sklearn  
  - matplotlib  
  - nltk   
  - after intalling nltk, execute 'python -m nltk.downloader stopwords'  

execute 'python TweetCluster.py csv_path'  
Two instances will run over the dataset using two different approaches: a non-recursive approach to the k-means clustering and a recursive approach that subdvides the largest clusters recursively. The results will print in the console. Results include how many tweets are in each group, what the most significant values of a tweet are, and estimates of the error in each group. 

# Implementation Details # 

First steps: 
vectorized the words in the tweets:
	sanitized the tweets: removing punctuation, removing stop words, lowering the case
created a tf-idf statistic over the words in the tweets: 
	followed wikipedia entries for the formula of a tf-idf statistic 
	Put the stastics into a nxd matrix T, one row for each document and a feature for each unique word used in the set of all documents.
	for word w' within document d' (for all d in S) took a tf-idf statistic for that word and put it in T[d'][w']. 

# Clustering the tweets #
Tweets were clustered using k-means clustering over the generated tf-idf statistic. The basics of a k-mean clustering algorithm are to randomly generate k vectors in the feature space that act as "centroids". Then, each data point is assigned to the centroid that it is closest to. Next, the centroid values themselves are updated based on the average value of the data in its cluster. Finally the data is reassigned based on the new value of the centroids until the sorting stabilizes.   


# Recursive K-Means implementation #
I noticed a problem of initializing k clusters for k means: 
		unstable solutions: everytime ran, could get dramatically different results
		because of the large feature space (d of up to 2000 in the case of tweets alone which will have a limited word space)
		also could have multiple clusters effectively representing the same cluster when two means that are "pulled" towards a larger cluster
	problem was made worse as k increased:
		more likely to have this pulling effect on several initialized vectors 
	solution: recursively compute a k means clustering on the cluster with the greatest error
		cluster on k = 2, compute the error of each cluster (squared distance to the mean) 
		iterate on cluster with largest error, always looking at the set of all cluster for deciding which to iterate on.

	made solution much more stable: impossible to create a perfectly stable solution for arbitrary data because of the nature of the clustering problem formulation (what is a cluster exactly? when does one stop and one begin? clustering is already an abstract solution: what about making a spectrum for clustering possibilities, can analyze these distributions and draw varying degrees of cluster (almost certainly, maybe, not likely, unlikely) and draw the cluster according to need (maybe an algorithm works best if the certainty is at a certain level)).
	
	results are more indicative of "what are the top k topics"  


# PCA and graphing #
	want to see if it would be possible to visually see clusters by looking at only the top 2 or 3 most significant axes to the variance of the data. There isn't enough differentiation it seems at only 2 features to really set the tweets apart. points all just on top of each other. 
	Thought PCA might still be useful for getting more stable results: 
		do a PCA first and then remove dimensions that do not meet a certain threshold of significance to the variance of the data as a whole. 
	

# TO DO / IDEAS #
	read about other ways to create stable solutions for k means
	how to algorithmically determine the best k to use:
		track the derivative of the loss, once it stops dropping at a certain rate of its steepest descent seen, stop iterating k. 
	"gravity clustering": points close to each other exert a certain gravity on each other, pulling them in closer towards each other, leading to more distinct clusters (hopefully)
