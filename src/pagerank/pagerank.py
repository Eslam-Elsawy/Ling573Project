import os
import sys
sys.path.append('../..')
import numpy as np
import logging
logging.basicConfig(level = logging.INFO)

logging.info('Importing similarity')
from src.similarity_measure import similarity
logging.info('Finished importing similarity')

def pagerank_algorithm(g, d = 0.85, epsilon = 0.00001, max_iterations = 1000):
	g = np.matrix(g)
	N = g.shape[0]
	logging.info('N={}'.format(N))

	# Force sparsity and binarize
	low_values = g < 0.2
	g[low_values] = 0
	g[~low_values] = 1
	
	# Initialize values
	deg = np.zeros(N)
	M = np.zeros([N, N])
	for i in range(0, N):
		deg[i] = np.sum(g[i])
		M[i] = g[i] / deg[i]

	t = 0
	delta = 1
	p = np.array([1/N]*N)
	while delta >= epsilon:
		t += 1
		p_new = np.dot(np.transpose(M), p)
		delta = np.linalg.norm(p_new - p)
		logging.info('t={}, delta={}'.format(t, delta))
		p = p_new	
		
	return p

def main():
	TOPICS_DIRECTORY = "../../input/topics/"
	TOPICS_TRAINING_DIRECTORY = TOPICS_DIRECTORY + "/training/"
	TOPICS_DEVTEST_DIRECTORY = TOPICS_DIRECTORY + "/devtest/"

	topic_dirs = os.listdir(TOPICS_TRAINING_DIRECTORY)
	#get a list of all sentences from a topic directory and their adjacency matrix:
	for topic_dir in topic_dirs:
		logging.info(topic_dir)
		path = TOPICS_TRAINING_DIRECTORY + "/" + topic_dir
		if os.path.isdir(path):
			extractor = similarity.Sent_Extractor(path)
			all_sents = extractor.extract_sentences()
			sm = similarity.Similarity_Matrix(all_sents)
			sm.fill_adj_matrix()
			matrix = sm.adj_matrix
			ranks = pagerank_algorithm(matrix)	

if __name__ == "__main__":
    main()
