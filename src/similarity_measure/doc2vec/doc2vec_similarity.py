import os
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import nltk
from collections import defaultdict
import math
import logging
from gensim import models
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

class Sent_Extractor(object):

	def __init__(self, directory_path):
		self.dir_path = directory_path

	def extract_sentences(self):
		""" Returns a list of all sentences from a topic directory """
		all_sentences = []
		files = os.listdir(self.dir_path)
		for file_name in files:
			file_path = self.dir_path + '/' + file_name
			with open(file_path, 'r') as f:
				data = f.read()
				soup = BeautifulSoup(data,'html.parser')
				paragraphs = soup.findAll('p')
				for par in paragraphs:
					paragraph_text = par.text.strip().replace('\n', ' ').replace('  ', ' ')
					paragraph_sents = nltk.sent_tokenize(paragraph_text)
					all_sentences.extend(paragraph_sents)
		return all_sentences



def build_matrix(sentences, model_file):
	""" Fills the adjacency matrix """
	model = models.Doc2Vec.load(model_file)
	sent_len = len(sentences)
	adj_matrix = []
	tokenized_sents = [[w for w in nltk.word_tokenize(s) if w not in stop] for s in sentences]
	for i in range(sent_len):
		adj_matrix.append([0]* sent_len)
	for i in range(sent_len):
		for j in range(sent_len):
			sentence1 = tokenized_sents[i]
			sentence2 = tokenized_sents[j]
			similarity = model.docvecs.similarity_unseen_docs(model, sentence1, sentence2, alpha=0.1, min_alpha=0.0001, steps=5)
			adj_matrix[i][j] = similarity
	return adj_matrix


def main():
	TOPICS_DIRECTORY = "../../input/topics/"
	TOPICS_TRAINING_DIRECTORY = TOPICS_DIRECTORY + "/training/"
	TOPICS_DEVTEST_DIRECTORY = TOPICS_DIRECTORY + "/devtest/"

	topic_dirs = os.listdir(TOPICS_TRAINING_DIRECTORY)
	doc2vec_model = 'reuters_model.doc2vec'
	#get a list of all sentences from a topic directory and their adjacency matrix:
	for topic_dir in topic_dirs:
		logging.info(topic_dir)
		path = TOPICS_TRAINING_DIRECTORY + "/" + topic_dir
		if os.path.isdir(path):
			extractor = Sent_Extractor(path)
			all_sents = extractor.extract_sentences()
			sm = Similarity_Matrix(all_sents, doc2vec_model)
			matrix = sm.fill_adj_matrix()
			print(matrix)


if __name__ == "__main__":
    main()



				



