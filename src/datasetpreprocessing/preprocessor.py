import io
import os
import xml.etree.ElementTree as ET

debug = True

def getFilePath(fileName):
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    file_path = os.path.join(__location__, fileName)
    return file_path

def getDirectoryPath(relativePath):
    return os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(relativePath)))

SPECS_DIRECTORY = "../../input/specs/"
TOPICS_DIRECTORY = "../../input/topics/"
DOCUMENTS_DIRECTORY = "../../input/documents/"

SPECS_TRAINING_FILEPATH = SPECS_DIRECTORY + "training/2009/UpdateSumm09_test_topics.xml"
SPECS_DEVTEST_FILEPATH = SPECS_DIRECTORY + "devtest/GuidedSumm10_test_topics.xml"
TOPICS_TRAINING_DIRECTORY = TOPICS_DIRECTORY + "/training/"
TOPICS_DEVTEST_DIRECTORY = TOPICS_DIRECTORY + "/devtest/"

class DatasetType:
    training, devtest = range(2)

def main():
    global debug
    dataset_type = DatasetType.training
    specs_file_path = ""
    topics_directory = ""

    if dataset_type == DatasetType.training:
        specs_file_path = SPECS_TRAINING_FILEPATH
        topics_directory = TOPICS_TRAINING_DIRECTORY
    elif dataset_type == DatasetType.devtest:
        specs_file_path = SPECS_DEVTEST_FILEPATH
        topics_directory = TOPICS_DEVTEST_DIRECTORY

    tree = ET.parse(getFilePath(specs_file_path))
    root = tree.getroot()
    print(root.tag)

    for topic in root.findall('topic'):
        topic_id = topic.get("id")
        topic_directory_name = None
        for child in topic:
            if child.tag == "title":
                if debug:
                    print(child.text.strip())
                topic_name = child.text.strip()
                topic_directory_name = topics_directory + topic_id + "_" + topic_name
                if not os.path.exists(topic_directory_name):
                    os.makedirs(topic_directory_name)

            elif child.tag == "docsetA":
                for doc in child:
                    if debug:
                        print("\t" + doc.get("id"))
                    document_name = doc.get("id")
                    document_file_name = doc.get("id").split(".")[0]
                    if "_" in document_file_name:
                        document_file_name = document_file_name[:-2] + ".xml"

                    tree = ET.parse(getFilePath(DOCUMENTS_DIRECTORY + document_file_name))

                    found_doc = None
                    for xmldoc in tree.getroot():
                        if xmldoc.get("id") == document_name:
                            found_doc = xmldoc
                            break

                    if found_doc is None:
                        raise Exception('can not find document ' + document_name + " in file "+ document_file_name)
                    else:

                        output_file_path = topic_directory_name + "/" + document_name + ".xml"
                        open(output_file_path, 'w+')

                        with io.open(output_file_path,'w', encoding='utf8') as outputFile:
                            outputFile.write(ET.tostring(found_doc, encoding="unicode"))
                            outputFile.flush()
                        outputFile.close()

main()