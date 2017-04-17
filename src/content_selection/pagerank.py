import os
#import sys
#sys.path.append('../..')
import numpy as np
import logging
logging.basicConfig(level = logging.INFO)

logging.info('Importing similarity')
#from src.similarity_measure import similarity
import similarity
logging.info('Finished importing similarity')

def pagerank_algorithm(g, d = 0.15, epsilon = 0.00001):
	g = np.matrix(g)
	N = g.shape[0]
	logging.info('N={}'.format(N))

	# Force sparsity and binarize
	low_values = g < d
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

	output_dir = '../../outputs/pagerank/training'	

	topic_dirs = os.listdir(TOPICS_TRAINING_DIRECTORY)
	#get a list of all sentences from a topic directory and their adjacency matrix:
	for topic_dir in topic_dirs:
		topic_id = topic_dir.split('_')[0]
		logging.info(topic_id)
		path = TOPICS_TRAINING_DIRECTORY + "/" + topic_dir
		if os.path.isdir(path):
			logging.info('Extracting sentences')
			extractor = similarity.Sent_Extractor(path)
			all_sents = extractor.extract_sentences()
			logging.info('Calculating similarity')
			sim_matrix = similarity.build_sim_matrix(all_sents)
			ranks = pagerank_algorithm(sim_matrix)
			ranks = np.argsort(ranks)[::-1]
			with open(os.path.join(output_dir, topic_id), 'w') as f:
				sentences = [all_sents[ix] for ix in ranks]
				sentences = '\n'.join(sentences)
				f.write(sentences)

if __name__ == "__main__":
    main()
