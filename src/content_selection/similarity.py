import os
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import numpy as np
import nltk
from collections import defaultdict
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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


def build_sim_matrix(sentences):
	vectorizer = CountVectorizer()
	dtf_matrix = vectorizer.fit_transform(sentences)
	return cosine_similarity(dtf_matrix)
	
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
			sim_matrix = build_sim_matrix(all_sents)
			#sm.fill_adj_matrix()
			#matrix = sm.adj_matrix


if __name__ == "__main__":
    main()



				



