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
    directoryPath = getDirectoryPath("../../input/documents/")
    for filename in os.listdir(directoryPath):
        if "_" not in filename and ".xml" not in filename:
            print(filename)
            input_file_path = directoryPath + "/" + filename
            output_file_path = directoryPath + "/" + filename + ".xml"
            with io.open(output_file_path,'w', encoding='utf8') as outputFile:
                memory_line = "<DOCSTREAM>"

                with io.open(input_file_path,'r', encoding='utf8') as inputFile:
                    for line in inputFile:
                        #memory_line += html.unescape(line)
                        unescaped_line = html.unescape(line)

                        array = unescaped_line.split()
                        for word in array:
                            if not word.startswith("&") and not word.endswith(";"):
                                memory_line += " " + word.replace("&", "n")

                    inputFile.close()

                memory_line = memory_line + "</DOCSTREAM>"
                soup = BeautifulSoup(memory_line)
                outputFile.write(soup.prettify(formatter=None))
                outputFile.flush()

            outputFile.close()

main()


