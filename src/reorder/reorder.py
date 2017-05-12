__author__ = 'mashaivenskaya'

import io
import os
import math
import logging
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import defaultdict
import datetime
from itertools import permutations
import sys
sys.path.append('../reranker/')
import reranker

def getFilePath(fileName):
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    file_path = os.path.join(__location__, fileName)
    return file_path

def getDirectoryPath(relativePath):
    return os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(relativePath)))


def chron_order(dataset = 'training'):

    top_sentences = reranker.select_top(dataset)
    input_directoryPath = getDirectoryPath("outputs/reranker/devtest/")
    output_directoryPath = getDirectoryPath("../../outputs/reorder/devtest/")
    chron_sents = {}
    for topic_id in top_sentences.keys():
        sentences = top_sentences[topic_id]
        id_part1 = topic_id[:-1]
        id_part2 = topic_id[-1:]
        output_file_name = id_part1 + "-A.M.100." + id_part2 + ".1"
        output_file_path = output_directoryPath + "/" + output_file_name

        chron_list = []
        date_index = defaultdict(list)

        for sentence in sentences:
            date = sentence.time_stamp
            date_index[date].append(sentence)

        for date in sorted(date_index):
            date_sents = date_index[date]
            date_sents.sort(key = lambda x: x.order)
            chron_list.extend(date_sents)

        
        with io.open(output_file_path,'w', encoding='utf8') as outputFile:
            for sentence in chron_list:
                outputFile.write(sentence.clean_sent)
                outputFile.write(' ')
            outputFile.flush()
        outputFile.close()

        chron_sents[topic_id] = chron_list
    return chron_sents


def cohesion_order(dataset = 'training'):
    top_sentences = reranker.select_top(dataset)
    input_directoryPath = getDirectoryPath("outputs/reranker/devtest/")
    output_directoryPath = getDirectoryPath("../../outputs/reorder/devtest/")
    cohesion_sents = {}
    
    for topic_id in top_sentences.keys():
        sentences = top_sentences[topic_id]
        id_part1 = topic_id[:-1]
        id_part2 = topic_id[-1:]
        output_file_name = id_part1 + "-A.M.100." + id_part2 + ".1"
        output_file_path = output_directoryPath + "/" + output_file_name

        num_sents = len(sentences)
        clean_sents = [sentence.clean_sent for sentence in sentences]
        vectorizer = TfidfVectorizer()
        cosine_matrix = cosine_similarity(vectorizer.fit_transform(clean_sents))
        ids = list(range(num_sents))
        perms = list(permutations(ids, num_sents))
        max_score = 0
        for perm in perms:
            perm_score = 0
            for i in range(num_sents -1):
                sent1_id = perm[i]
                sent2_id = perm[i+1]
                adj_sim = cosine_matrix[sent1_id][sent2_id]
                perm_score += adj_sim
            if perm_score > max_score:
                max_score = perm_score
                winning_perm = perm

        cohesion_list = [sentences[i] for i in winning_perm]

        with io.open(output_file_path,'w', encoding='utf8') as outputFile:
            for sentence in cohesion_list:
                outputFile.write(sentence.clean_sent)
                outputFile.write(' ')
            outputFile.flush()
        outputFile.close()

        cohesion_sents[topic_id] = cohesion_list
    return cohesion_sent


def main():
    #logging.info('Ranking training data')
    #select_top('training')

    logging.info('Reordering devtest data')
    cohesion_order('devtest')


if __name__ == "__main__":
    main()


