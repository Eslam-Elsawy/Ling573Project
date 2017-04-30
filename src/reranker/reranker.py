__author__ = 'eslamelsawy'

import io
import os
import math

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

def main():
    input_directoryPath = getDirectoryPath("outputs/pagerank_D3/devtest/")
    output_directoryPath = getDirectoryPath("outputs/D3/")
    for filename in os.listdir(input_directoryPath):
        print("Reranking sentences in file: " +filename)
        input_file_path = input_directoryPath + "/" + filename

        # get the output file name
        id_part1 = filename[:-1]
        id_part2 = filename[-1:]
        output_file_name = id_part1 + "-A.M.100." + id_part2 + ".1"
        output_file_path = output_directoryPath + "/" + output_file_name

        # calculate idf of all terms in all sentences in the document
        # count the number of sentences and get the unique set of words
        vocab = set()
        sentences = []
        with io.open(input_file_path, 'r', encoding='utf8') as inputFile:
            for line in inputFile:
                sentences.append(line)
                splitted = line.split()
                for word in splitted:
                    vocab.add(word.lower())
        inputFile.close()

        # calculate idf
        vocab_sentences_count = {}
        for word in vocab:
            vocab_sentences_count[word] = 0

        for sentence in sentences:
            unique_terms = set()
            for word in sentence.lower().split():
                unique_terms.add(word)

            for word in unique_terms:
                vocab_sentences_count[word] = vocab_sentences_count[word] + 1

        idf = {}
        for word in vocab:
            idf[word] = 1.0 + math.log(float(len(sentences) / float(vocab_sentences_count[word])))

        chosen_sentences = []
        total_word_count = 0
        for sentence in sentences:
            if total_word_count + len(sentence.split()) > 100:
                continue

            # decide whether or not to include the sentence based on the cosine similarity
            include_sentence = True
            for chosen_sentence in chosen_sentences:
                cosine_score = cosine(sentence.lower(), chosen_sentence.lower(), idf)
                if cosine_score > THRESHOLD:
                    include_sentence = False
                    break

            if include_sentence:
                chosen_sentences.append(sentence)
                total_word_count += len(sentence.split())

        with io.open(output_file_path,'w', encoding='utf8') as outputFile:
            for sentence in chosen_sentences:
                outputFile.write(sentence)
            outputFile.flush()
        outputFile.close()

main()


