__author__ = 'eslamelsawy'

import io
import os
from nltk.tag.stanford import StanfordNERTagger
from nltk.parse.stanford import StanfordDependencyParser
from random import randint
from sklearn import svm
from bs4 import BeautifulSoup
import nltk
from sklearn.neighbors import KNeighborsClassifier
import logging
logging.basicConfig(level = logging.INFO)

def entity_reranker(dataset = 'training'):
    logging.info('Running {} entity reranker'.format(dataset))
    input_directoryPath = os.path.join('outputs/reranker_D4', dataset)
    output_directoryPath = os.path.join('outputs/entity_reranker_D4', dataset)
    model_file_path = '../model'

    # reading model
    logging.info('Reading model')
    vectors, labels = readModel(model_file_path)

    # building calssifier
    logging.info('Building KNN Classifier')
    neigh = KNeighborsClassifier(n_neighbors=11)
    neigh.fit(vectors, labels)

    logging.info('Loading NER tagger')
    ner_tagger = loadStanfordNERTagger()
    logging.info('Loading Stanford dependency parser')
    stanford_dependency_parser = loadStanfordDependencyParser()
       
    logging.info('Reading content selection output') 
    for filename in os.listdir(input_directoryPath):

        if filename.startswith(".") or filename.endswith(".reordered"):
            continue

        summary_file_path = os.path.join(input_directoryPath, filename)
        logging.info(summary_file_path)
        output_file_path = os.path.join(output_directoryPath, filename)
        number_of_random_orders = 20

        # reading content selection output
        sentences = []
        with io.open(summary_file_path, 'r', encoding='utf8') as inputFile:
            for line in inputFile:
                sublines = line.split(". ")
                for sub in sublines:
                    if sub.strip():
                        #sentences.append(line)
                        sentences.append(sub.strip())
        inputFile.close()
    
        sent_ent_matrix = generateMatrixForSummary(sentences, ner_tagger, stanford_dependency_parser)

        if sent_ent_matrix == None:
            continue

        # original order
        original_order = []
        for key in sent_ent_matrix:
            original_order.append(key)

        # generate random ordering
        logging.info('Generating random ordering')
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
            first = True
            for order in best_order:
                if first:
                    first = False
                    summary = sentences[order]
                    continue
                summary = summary + "\n" + sentences[order]
            outputFile.write(summary)
            outputFile.flush()
        outputFile.close()

def getFilePath(fileName):
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    file_path = os.path.join(__location__, fileName)
    return file_path

def getDirectoryPath(relativePath):
    return os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(relativePath)))

def loadStanfordNERTagger():
    tagger_file_path = "../english.all.3class.nodistsim.crf.ser.gz"
    jar_file_path = "../stanford-ner.jar"
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
    path_to_jar = '../stanford-parser.jar'
    path_to_models_jar = '../stanford-parser-3.7.0-models.jar'
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
    logging.info('Finding NE ....')
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


def buildPermutations(sentences):
    sent_len_dic = {}
    index = 0
    for s in sentences:
        sent_len_dic[index] = len(s.split())
        index += 1

    return buildPermutationsRec(sent_len_dic, 0, [], "")


def buildPermutationsRec(sent_len_dic, current_len, history, history_str):

    found_longer = False
    perms = []
    for key in sent_len_dic:
        if key not in history and current_len + sent_len_dic[key] <= 100:
            found_longer = True
            new_history = list(history)
            new_history.append(key)
            new_history_str = ','.join(str(x) for x in new_history)
            perms = perms + buildPermutationsRec(sent_len_dic, current_len + sent_len_dic[key], new_history, new_history_str)

    if found_longer:
        return perms
    else:
        return [history_str]


def train():
    number_of_random_orders = 1
    number_of_files_per_topic = 2
    TOPICS_DIRECTORY = "input/topics/"
    #TOPICS_TRAINING_DIRECTORY = TOPICS_DIRECTORY + "/training/"
    TOPICS_DEVTEST_DIRECTORY = TOPICS_DIRECTORY + "/devtest/"

    #input_directoryPath = getDirectoryPath("../../input/entitygridtraining/gold/training/")
    output_directoryPath = "outputs/D3_entitybasedreranker/"
    output_file_path = output_directoryPath + "/model"

    training_samples_vectors = []
    training_samples_labels = []

    ner_tagger = loadStanfordNERTagger()
    stanford_dependency_parser = loadStanfordDependencyParser()

    topic_dirs = os.listdir(TOPICS_DEVTEST_DIRECTORY)
    # get a list of all sentences from a topic directory and their adjacency matrix:
    for topic_dir in topic_dirs:
        if(topic_dir.startswith(".")):
            continue
        print("Topic: " + topic_dir)
        path = TOPICS_DEVTEST_DIRECTORY + "/" + topic_dir
        if os.path.isdir(path):
            print(path)
            extractor = Sent_Extractor(path)
            all_sents = extractor.extract_sentences()

            for i in range(0,number_of_files_per_topic):
                random_file_index = randint(0, len(all_sents) - 1)
                print("Random file index = " + str(random_file_index))
                sentences = all_sents[random_file_index]
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

                # generate vectors for random orders
                for random_order in random_orders:
                    print(random_order)
                    feature_vector = createFeatureVector(sent_ent_matrix, random_order)
                    training_samples_vectors.append(feature_vector)
                    training_samples_labels.append(0)
                    print(feature_vector)
                    print("\n")

                # generate vector for original order
                print(original_order)
                feature_vector = createFeatureVector(sent_ent_matrix, original_order)
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


def main():
    logging.info('Running entity reranker')
    entity_reranker('devtest')

    logging.info('Running entity reranker on eval data')
    entity_reranker('eval')

if __name__ == '__main__':
    main()
#train()


# for filename in os.listdir(input_directoryPath):

# if filename.startswith("."):
#    continue
# print("Reranking sentences in file: " +filename)
# input_file_path = input_directoryPath + "/" + filename

# get the output file name
# id_part1 = filename[:-1]
# id_part2 = filename[-1:]
# output_file_name = id_part1 + "-A.M.100." + id_part2 + ".1"
# output_file_path = output_directoryPath + "/" + output_file_name


# reading content selection output
# print("summary ....")
# sentences = []
# with io.open(input_file_path, 'r', encoding='utf8') as inputFile:
#    for line in inputFile:
#        sublines = line.split(".")
#        for sub in sublines:
#            if sub.strip():
#               sentences.append(sub)
# inputFile.close()
