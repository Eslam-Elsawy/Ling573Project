__author__ = 'eslamelsawy'

import io
import os
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import html


def getFilePath(fileName):
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    file_path = os.path.join(__location__, fileName)
    return file_path

def getDirectoryPath(relativePath):
    return os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(relativePath)))

def main():
    input_directoryPath = getDirectoryPath("../../outputs/pagerank/devtest/")
    output_directoryPath = getDirectoryPath("../../outputs/reranker/devtest/")
    for filename in os.listdir(input_directoryPath):
        print(filename)
        input_file_path = input_directoryPath + "/" + filename
        output_file_path = output_directoryPath + "/" + filename
        with io.open(output_file_path,'w', encoding='utf8') as outputFile:
            with io.open(input_file_path,'r', encoding='utf8') as inputFile:
                word_count = 0
                for line in inputFile:
                    array = line.split()
                    if word_count + len(array) <= 100:
                        outputFile.write(line)
                        word_count += len(array)
                    else:
                        break
                inputFile.close()

            outputFile.flush()
            outputFile.close()

main()


