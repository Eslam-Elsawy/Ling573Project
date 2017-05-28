__author__ = 'eslamelsawy'

import io
import os
from nltk.tag.stanford import StanfordNERTagger
from nltk.parse.stanford import StanfordDependencyParser
from random import randint
from bs4 import BeautifulSoup
import nltk
from sklearn.neighbors import KNeighborsClassifier
import reranker
import logging
import entitybasedreranker


def main():
    number_of_random_orders = 1
    number_of_files_per_topic = 2
    TOPICS_DEVTEST_DIRECTORY = "input/topics/eval/"
    output_file_path = entitybasedreranker.getFilePath("model2")

    training_samples_vectors = []
    training_samples_labels = []

    ner_tagger = entitybasedreranker.loadStanfordNERTagger()
    stanford_dependency_parser = entitybasedreranker.loadStanfordDependencyParser()

    topic_dirs = os.listdir(TOPICS_DEVTEST_DIRECTORY)

    print(topic_dirs)
    # get a list of all sentences from a topic directory and their adjacency matrix:
    for topic_dir in topic_dirs:
        if(topic_dir.startswith(".")):
            continue
        print("Topic: " + topic_dir)
        path = TOPICS_DEVTEST_DIRECTORY + "/" + topic_dir
        if os.path.isdir(path):
            print(path)
            extractor = entitybasedreranker.Sent_Extractor(path)
            all_sents = extractor.extract_sentences()

            for i in range(0,number_of_files_per_topic):
                random_file_index = randint(0, len(all_sents) - 1)
                print("Random file index = " + str(random_file_index))
                sentences = all_sents[random_file_index]
                sent_ent_matrix = entitybasedreranker.generateMatrixForSummary(sentences, ner_tagger, stanford_dependency_parser)
                if sent_ent_matrix == None:
                    continue

                # original order
                original_order = []
                for key in sent_ent_matrix:
                    original_order.append(key)

                # generate random ordering
                print("\n3: random ordering ..")
                random_orders = entitybasedreranker.generateRandomOrders(original_order, number_of_random_orders)

                # generate vectors for random orders
                for random_order in random_orders:
                    print(random_order)
                    feature_vector = entitybasedreranker.createFeatureVector(sent_ent_matrix, random_order)
                    training_samples_vectors.append(feature_vector)
                    training_samples_labels.append(0)
                    print(feature_vector)
                    print("\n")

                # generate vector for original order
                print(original_order)
                feature_vector = entitybasedreranker.createFeatureVector(sent_ent_matrix, original_order)
                training_samples_vectors.append(feature_vector)
                training_samples_labels.append(1)
                print(feature_vector)

                # print to model file
                with io.open(output_file_path, 'w', encoding='utf8') as outputFile:
                    for i in range(0, len(training_samples_vectors)):
                        vector = training_samples_vectors[i]
                        label = training_samples_labels[i]

                        outputFile.write(','.join(str(p) for p in vector) +"\n")
                        outputFile.write(str(label)+"\n")
                    outputFile.flush()
                outputFile.close()

        print(training_samples_vectors)


if __name__ == "__main__":
    main()