__author__ = 'eslamelsawy'

import io
import os
import math
import logging
import re
import nltk
import sys
sys.path.append('../content_selection/')
import pagerank

THRESHOLD = 0.2

def getFilePath(fileName):
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    file_path = os.path.join(__location__, fileName)
    return file_path

def getDirectoryPath(relativePath):
    return os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(relativePath)))

def normalizedTermFrequency(term, document):
    normalizeDocument = document.lower().split()
    return float(normalizeDocument.count(term.lower())) / float(len(normalizeDocument))

def createVector(document, idf):
    vec = {}
    for term in document.lower().split():
        vec[term] = normalizedTermFrequency(term, document) * idf[term]
    return vec

def vectorLength(vector):
    sum = 0.0
    for dim in vector:
        sum += vector[dim] * vector[dim]
    return math.sqrt(sum)

def cosine(document1, document2, idf):
    vec1 = createVector(document1, idf)
    vec2 = createVector(document2, idf)

    dot_product = 0
    for term in document1.lower().split():
        if term in document2.lower().split():
            dot_product = dot_product + (vec1[term] * vec2[term])

    return float(dot_product) / float(vectorLength(vec1) * vectorLength(vec2))

def compress(sentence):

    tokenized = nltk.word_tokenize(sentence.lower())

    tagged = nltk.pos_tag(tokenized)
    initial_pos = tagged[0][1]
    if initial_pos == 'CC' or initial_pos == 'RB':
        first_space = sentence.index(' ')
        sentence = sentence[first_space+1:]
        sentence = sentence[0].upper() + sentence[1:]
    


    said_regex = re.compile(r'(,[^,]*(said|says|according).*?(,(( who).*?(,|\.))?|\.))')

    match = re.search(said_regex, sentence)
    if match:
        if match.group(1).endswith('.'):
            sentence = re.sub(said_regex, '.', sentence)
                    
        else:
            sentence = re.sub(said_regex, ' ', sentence)

    parens_reg = re.compile(r'\(.*?\)')
    sentence = re.sub(parens_reg, ' ', sentence)

    return sentence

def select_top(dataset = 'training'):
    print ('selecting top...')
    meta_regex = re.compile(r'^([A-Z]{2,}.{,25}\(.{,25}\))|^([A-Z\s]{2,}(\_|\-))')
    ranked_sentences = pagerank.rank(dataset)
    input_directoryPath = getDirectoryPath("outputs/pagerank_D3/devtest/")
    output_directoryPath = getDirectoryPath("../../outputs/reranker/devtest/")
    top_sents = {}
    for topic_id in ranked_sentences.keys():
        sentences = ranked_sentences[topic_id]
        id_part1 = topic_id[:-1]
        id_part2 = topic_id[-1:]
        output_file_name = id_part1 + "-A.M.100." + id_part2 + ".1"
        output_file_path = output_directoryPath + "/" + output_file_name

        vocab = set()
        for sentence in sentences:


            original = sentence.original_sent
            match = re.search(meta_regex, original)
            clean = re.sub(meta_regex, '', original).replace('--', '').lower()
            clean = compress(clean)
            sentence.clean_sent = clean
            splitted = clean.split()
            for word in splitted:
                vocab.add(word.lower())

        vocab_sentences_count = {}
        for word in vocab:
            vocab_sentences_count[word] = 0

        for sentence in sentences:
            unique_terms = set()
            for word in sentence.clean_sent.lower().split():
                unique_terms.add(word)

            for word in unique_terms:
                vocab_sentences_count[word] = vocab_sentences_count[word] + 1

        idf = {}
        for word in vocab:
            idf[word] = 1.0 + math.log(float(len(sentences) / float(vocab_sentences_count[word])))

        chosen_sentences = []
        total_word_count = 0
        for sent_obj in sentences:
            sentence = sent_obj.clean_sent
            if len(sentence.strip()) > 0:
                if total_word_count + len(sentence.split()) > 100:
                    continue

                # decide whether or not to include the sentence based on the cosine similarity
                include_sentence = True
                for chosen_sentence_obj in chosen_sentences:
                    chosen_sentence = chosen_sentence_obj.clean_sent
                    cosine_score = cosine(sentence.lower(), chosen_sentence.lower(), idf)
                    if cosine_score > THRESHOLD:
                        include_sentence = False
                        break

                if include_sentence:
                    #exclude quotes:
                    if not sent_obj.clean_sent.startswith('\'\'') and not sent_obj.clean_sent.startswith('``') and not sent_obj.clean_sent.startswith('"'):
                        chosen_sentences.append(sent_obj)
                        total_word_count += len(sentence.split())

                with io.open(output_file_path,'w', encoding='utf8') as outputFile:
                    for sentence in chosen_sentences:
                        outputFile.write(sentence.clean_sent)
                        outputFile.write(' ')
                    outputFile.flush()
                outputFile.close()

                top_sents[topic_id] = chosen_sentences

    return top_sents





def main():
    #logging.info('Ranking training data')
    #select_top('training')

    logging.info('Ranking devtest data')
    select_top('devtest')


if __name__ == "__main__":
    main()


