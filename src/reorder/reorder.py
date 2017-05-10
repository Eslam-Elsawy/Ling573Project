__author__ = 'mashaivenskaya'

import io
import os
import math
import logging
import re
from collections import defaultdict
import datetime
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


def main():
    #logging.info('Ranking training data')
    #select_top('training')

    logging.info('Reordering devtest data')
    chron_order('devtest')


if __name__ == "__main__":
    main()


