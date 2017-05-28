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

def entitygrid_reorder(dataset = 'training'):
    output_directoryPath = getDirectoryPath("outputs/D3/")
    model_file_path = getFilePath("model")
    KNN = 11
    number_of_random_orders = 20

    # reading model
    vectors, labels = readModel(model_file_path)

    # building calssifier
    neigh = KNeighborsClassifier(n_neighbors=KNN)
    neigh.fit(vectors, labels)

    # NER + Dep parser
    ner_tagger = loadStanfordNERTagger()
    stanford_dependency_parser = loadStanfordDependencyParser()

    # page rank + cosine reordering
    top_sentences = reranker.select_top(dataset)

    for topic_id in top_sentences.keys():
        sentences = top_sentences[topic_id]
        id_part1 = topic_id[:-1]
        id_part2 = topic_id[-1:]
        output_file_name = id_part1 + "-A.M.100." + id_part2 + ".1"
        output_file_path = output_directoryPath + "/" + output_file_name

        print("summary ....")
        sentences = [sentence.clean_sent.strip() for sentence in sentences]
        for s in sentences:
            print(s)

        sent_ent_matrix = generateMatrixForSummary(sentences, ner_tagger, stanford_dependency_parser)

        if sent_ent_matrix == None:
            continue

        # original order
        original_order = []
        for key in sent_ent_matrix:
            original_order.append(key)

        # generate random ordering
        print("\n3: random ordering ..")
        random_orders = generateRandomOrders(original_order, number_of_random_orders)

        max_prob = -1
        best_order = []

        # generate vectors for random orders
        for random_order in random_orders:
            print(random_order)
            feature_vector = createFeatureVector(sent_ent_matrix, random_order)
            print(feature_vector)
            scores = neigh.predict_proba(feature_vector)
            print("scores: " + str(scores))
            if scores[0][1] > max_prob:
                max_prob = scores[0][1]
                best_order = random_order
            print("\n")

        # generate vector for original order
        print(original_order)
        feature_vector = createFeatureVector(sent_ent_matrix, original_order)
        print(feature_vector)
        scores = neigh.predict_proba(feature_vector)
        print("scores: " + str(scores))
        if scores[0][1] > max_prob:
            max_prob = scores[0][1]
            best_order = random_order

        print("Best score: " + str(max_prob))
        print(best_order)

        # print best order to the output file
        with io.open(output_file_path, 'w', encoding='utf8') as outputFile:
            for order in best_order:
                outputFile.write(sentences[order]+"\n")
            outputFile.flush()
        outputFile.close()


def getFilePath(fileName):
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    file_path = os.path.join(__location__, fileName)
    return file_path

def getDirectoryPath(relativePath):
    return os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(relativePath)))

def loadStanfordNERTagger():
    tagger_file_path = getFilePath("english.all.3class.nodistsim.crf.ser.gz")
    jar_file_path = getFilePath("stanford-ner.jar")
    return StanfordNERTagger(tagger_file_path, jar_file_path)

def nerSentence(tagger, sentence):
    nes = tagger.tag(sentence.split())
    parsed = []
    index = 0
    while index < len(nes):
        current_word = nes[index][0]
        current_tag = nes[index][1]
        if current_tag == "PERSON" or current_tag == "ORGANIZATION" or current_tag == "LOCATION":
            ne = current_word

            index += 1
            while index < len(nes) and nes[index][1] == current_tag:
                ne = ne + ' ' + nes[index][0]
                index+=1

            index -=1

            parsed.append(ne)
        index +=1

    print(nes)
    print(parsed)
    print('---')
    return parsed

def clusterNER(nes):
    clusters = []
    for ne in nes:
        found = False
        for cluster in clusters:
            for item in cluster:
                if ne in item or item in ne:
                    found = True

            if found:
                cluster.add(ne)
                break

        if not found:
            new_cluster = set()
            new_cluster.add(ne)
            clusters.append(new_cluster)

    return clusters


def loadStanfordDependencyParser():
    path_to_jar = getFilePath('stanford-parser.jar')
    path_to_models_jar = getFilePath('stanford-parser-3.7.0-models.jar')
    return StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)

def dependencyParse(dependency_parser, sentence):
    subjects = set()
    objects = set()
    others = set()

    print(sentence)
    sentences_without_numbers  = ''.join([i for i in sentence if not i.isdigit()])
    result = dependency_parser.raw_parse(sentences_without_numbers)
    dep = next(result)

    for item in list(dep.triples()):
        #print(item)
        if "subj" in item[1]:
            subjects.add(item[2][0])
        elif "obj" in item[1]:
            objects.add((item[2][0]))

        others.add(item[0][0])
        others.add(item[2][0])
    #print("===")
    return subjects, objects, others


def generateRandomOrders(originalOrder, number_of_random_orders):
    randoms = []

    for i in range(0, number_of_random_orders):
        copied_original = originalOrder[:]

        random_order = []
        while len(copied_original) > 0:
            random_index = randint(0, len(copied_original) - 1)
            random_value = copied_original[random_index]
            random_order.append(random_value)
            copied_original.remove(random_value)

        randoms.append(random_order)

    return randoms



def createFeatureVector(sent_ent_matrix, order):
    transitions = ['ss', 'so', 'sx', 's-',
                   'os', 'oo', 'ox', 'o-',
                   'xs', 'xo', 'xx', 'x-',
                   '-s', '-o', '-x', '--']

    print(transitions)

    transition_counts = {}
    for tran in transitions:
        transition_counts[tran] = 0

    total_trans_count = 0
    for i in range(0, len(order)-1):
        first = sent_ent_matrix[order[i]]
        second = sent_ent_matrix[order[i+1]]

        for j in range(0, len(first)):
            s = first[j]
            d = second[j]

            transition_counts[s+d] = transition_counts[s+d] + 1
            total_trans_count += 1


    #for tran in transitions:
    #    print(tran + ":" + str(transition_counts[tran] / total_trans_count))

    feature_vector = []
    for tran in transitions:
        feature_vector.append(transition_counts[tran] / total_trans_count)
    return feature_vector


def generateMatrixForSummary(sentences, ner_tagger, stanford_dep_parser ):
    # mod1: ner of sentence
    print("\n1: Finding NE ....")
    all_nes = []
    for sentence in sentences:
        all_nes = all_nes + nerSentence(ner_tagger, sentence)

    if len(all_nes) == 0:
        return None

    # mod 1.1: cluster ner
    print('\n1.1: clustering ...')
    clusters = clusterNER(all_nes)
    for cluster in clusters:
        print(cluster)

    # mod 2: dependency parsing
    print('\n2: dep parsing ...')
    sent_ent_matrix = {}
    for i in range(0, len(sentences)):
        tags = []
        sentence = sentences[i]
        subjects, objects, others = dependencyParse(stanford_dep_parser, sentence)

        for j in range(0, len(clusters)):
            cluster = clusters[j]
            curr_tag = tag(cluster, subjects, objects, others)
            tags.append(curr_tag)

        sent_ent_matrix[i] = tags

        # print(subjects)
        # print(objects)
        # print(others)
        # print(tags)
        # print("---")

    print(clusters)
    for key in sent_ent_matrix:
        # print(sentences[key])
        print(sent_ent_matrix[key])

    return sent_ent_matrix

def tag(cluster, subjects, objects, others):
    for sub in subjects:
        for item in cluster:
            if sub in item:
                return 's'
    for obj in objects:
        for item in cluster:
            if obj in item:
                return 'o'
    for o in others:
        for item in cluster:
            if o in item:
                return 'x'
    return '-'

def readModel(filePath):
    vectors = []
    labels = []
    with io.open(filePath, 'r', encoding='utf8') as inputFile:
        isVector = True
        for line in inputFile:
            if isVector:
                vector_str = line.strip().split(",")
                vector = []
                for v in vector_str:
                    vector.append(float(v))
                vectors.append(vector)
            else:
                label = int(line.strip())
                labels.append(label)
            isVector = not isVector

    return vectors, labels



class Sent_Extractor(object):

    def __init__(self, directory_path):
        self.dir_path = directory_path

    def extract_sentences(self):
        """ Returns a list of all sentences from a topic directory """
        all_sentences = []
        files = os.listdir(self.dir_path)

        for file_name in files:
            file_sentences = []
            file_path = self.dir_path + '/' + file_name
            with open(file_path, 'r') as f:
                data = f.read()
                soup = BeautifulSoup(data, 'html.parser')
                paragraphs = soup.findAll('p')
                for par in paragraphs:
                    paragraph_text = par.text.strip().replace('\n', ' ').replace('  ', ' ')
                    paragraph_sents = nltk.sent_tokenize(paragraph_text)
                    file_sentences.extend(paragraph_sents)
            all_sentences.append(file_sentences)
            print(len(all_sentences))
        return all_sentences


#train()

def main():
    logging.info('Reordering using entity grid')
    entitygrid_reorder('devtest')


if __name__ == "__main__":
    main()

