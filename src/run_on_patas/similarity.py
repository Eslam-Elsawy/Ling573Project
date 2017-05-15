import os
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import numpy as np
import nltk
from collections import defaultdict
import datetime
import math
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import wordpunct_tokenize
import logging
logging.basicConfig(level = logging.INFO)


class Sentence(object):

	def __init__(self, sentence, timestamp, order):
		self.original_sent = sentence
		self.time_stamp = timestamp
		self.order = order
		self.clean_sent = ''
	

class Sent_Extractor(object):
	def __init__(self, directory_path):
		self.dir_path = directory_path

	def extract_sentences(self):
		""" Returns a list of all sentences from a topic directory """
		all_sentences = []
		files = os.listdir(self.dir_path)
		topic_id = self.dir_path.split('/')[-1].split('_')[0]

		time_regex = re.compile(r'([\d]{2}):([\d]{2})')
		meta_regex = re.compile(r'^([A-Z]{2,}.{,25}\(.{,25}\))|^([A-Z\s]{2,}(\_|\-))')
		web_regex = re.compile(r'www.')

		for file_name in files:
			file_path = self.dir_path + '/' + file_name
			try:
				date = file_name.split('_')[2].split('.')[0]
			except:
				date = file_name.split('.')[0][3:]
			timestamp = datetime.datetime(int(date[:4]), int(date[4:6]), int(date[6:]))
			with open(file_path, 'r') as f:
				data = f.read()
				soup = BeautifulSoup(data,'html.parser')
				dt = soup.date_time
				if dt:
					match = re.search(time_regex, dt.text)
					if match:
						timestamp = timestamp.replace(hour = int(match.group(1)), minute = int(match.group(2)))
		
				paragraphs = soup.findAll('p')
				for par in paragraphs:
					paragraph_text = par.text.strip().replace('\n', ' ').replace('  ', ' ')
					paragraph_sents = nltk.sent_tokenize(paragraph_text)
					for i, sent in enumerate(paragraph_sents):
						if len(sent) > 35:
							phone_number_regex = re.compile('[0-9][0-9][0-9]-[0-9][0-9][0-9][0-9]')
							if re.search(phone_number_regex, sent) == None and re.search(web_regex, sent) == None:
								sent = re.sub(meta_regex, '', sent).replace('--', '')
								all_sentences.append(Sentence(sent, timestamp, i))

		return all_sentences


def build_sim_matrix(sentences):
	stemmer = SnowballStemmer("english")
	stemmed_sentences = [wordpunct_tokenize(sent) for sent in sentences]
	stemmed_sentences = [[stemmer.stem(token) for token in sent] for sent in stemmed_sentences]
	stemmed_sentences = [' '.join(sent) for sent in stemmed_sentences]
	vectorizer = TfidfVectorizer(binary = True)
	dtf_matrix = vectorizer.fit_transform(stemmed_sentences)
	return cosine_similarity(dtf_matrix)
	
def main():
	TOPICS_DIRECTORY = "input/topics/"
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
			sim_matrix = build_sim_matrix(sent.original_sent for sent in all_sents)



if __name__ == "__main__":
    main()



				



