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
	g[low_values] = 0.0
	g[~low_values] = 1.0
	
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

def rank(dataset = 'training'):
	'''
	dataset should be one of the following:
		- 'training'
		- 'devtest'
	'''

	input_dir = os.path.join('../../input/topics', dataset)
	output_dir = os.path.join('../../outputs/pagerank', dataset)

	topic_dirs = os.listdir(input_dir)
	for topic_dir in topic_dirs:
		topic_id = topic_dir.split('_')[0]
		logging.info(topic_id)
		topic_path = os.path.join(input_dir, topic_dir)
		if os.path.isdir(topic_path):
			logging.info('Extracting sentences')
			extractor = similarity.Sent_Extractor(topic_path)
			all_sents = extractor.extract_sentences()
			all_sents = [x for x in all_sents if len(x) >= 35]
			logging.info('Calculating similarity')
			sim_matrix = similarity.build_sim_matrix(all_sents)
			ranks = pagerank_algorithm(sim_matrix)
			ranks = np.argsort(ranks)[::-1]
			with open(os.path.join(output_dir, topic_id), 'w') as f:
				sentences = [all_sents[ix] for ix in ranks]
				sentences = '\n'.join(sentences)
				f.write(sentences)

def main():
	logging.info('Ranking training data')
	rank('training')

	logging.info('Ranking devtest data')
	rank('devtest')


if __name__ == "__main__":
    main()
