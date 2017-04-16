import os
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import nltk
from collections import defaultdict
import math
import logging
logging.basicConfig(level = logging.INFO)

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

class Similarity_Matrix(object):

	def __init__(self, sentences):
		""" initiates the adjacency matrix """
		self.sentences = sentences
		self.matrix_size = len(sentences)
		self.adj_matrix = []
		for i in range(self.matrix_size):
			self.adj_matrix.append([0]* self.matrix_size)

	def vectorize_sent(self, sent):
		""" Vectorizes each sentence into a hash of words to word counts """
		vector = defaultdict(int)
		words = nltk.word_tokenize(sent)
		for word in words:
			vector[word]+=1
		return vector

	def cosine_sim(self, sent1, sent2):
		""" Calculates cosine similarity between 2 sentences """
		vec1 = self.vectorize_sent(sent1)
		vec2 = self.vectorize_sent(sent2)
		all_words = set(vec1.keys()).union(set(vec2.keys()))
		dot_product = 0
		mag1 = 0
		mag2 = 0
		for word in all_words:
			v1 = vec1[word]
			v2 = vec2[word]
			dot_product += v1*v2
			mag1 += v1**2
			mag2 += v2**2
		mag1 = math.sqrt(mag1)
		mag2 = math.sqrt(mag2)
		cosine = float(dot_product)/(mag1*mag2)
		return cosine

	def fill_adj_matrix(self):
		""" Fills the adjacency matrix """
		for i in range(self.matrix_size):
			for j in range(self.matrix_size):
				sentence1 = self.sentences[i]
				sentence2 = self.sentences[j]
				similarity = self.cosine_sim(sentence1, sentence2)
				self.adj_matrix[i][j] = similarity


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
			extractor = Sent_Extractor(path)
			all_sents = extractor.extract_sentences()
			sm = Similarity_Matrix(all_sents)
			sm.fill_adj_matrix()
			matrix = sm.adj_matrix


if __name__ == "__main__":
    main()



				



