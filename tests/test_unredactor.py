import os
from project2 import redactor
from project2 import unredactor
import nltk
nltk.download('averaged_perceptron_tagger_eng')


def testRedactNames():
    expected = "I couldn't image ██████ ███████ in a serious role, but his performance truly "
    file_loc = 'project_docs/package_test/test.txt'  

    redacted_doc_loc = redactor.redactNames(file_loc)

    redacted_data = open(redacted_doc_loc).read().splitlines()

    assert redacted_data[1] == expected


def testExtractTrain():

    file_loc = 'project_docs/package_test/test.txt'

    train_xy = unredactor.extractTrain(file_loc)

    assert type(train_xy) == list
    assert type(train_xy[0]) == tuple


def testGetTrainFeatures():

    expected = [({'name_len': 13,
                  'name_len_s': 14,
                  'w1_len': 6,
                  'w2_len': 7,
                  'w3_len': 0,
                  'w4_len': 0,
                  'white_space': 1,
                  'word_cnt': 2},
                 'Ashton Kutcher')]  # Extected features to be extracted

    test_data = "I couldn't image Ashton Kutcher in a serious role, but his performance truly exemplified his character."

    extracted_features = unredactor.getTrainFeatures(test_data)

    assert type(extracted_features) == list
    assert type(extracted_features[0]) == tuple
    assert type(extracted_features[0][0]) == dict
    assert extracted_features == expected


def testExtractRedacted():

    file_loc = 'project_docs/package_test/test.redacted'
    train_xy = unredactor.extractRedacted(file_loc)
    assert type(train_xy) == list
    assert type(train_xy[0]) == tuple


def testGetRedactedFeatures():

    expected = [({'name_len': 13,
                  'name_len_s': 14,
                  'w1_len': 6,
                  'w2_len': 7,
                  'w3_len': 0,
                  'w4_len': 0,
                  'white_space': 1,
                  'word_cnt': 2},
                 '██████ ███████')] 
    test_data = "I couldn't image ██████ ███████ in a serious role, but his performance truly exemplified his character."

    extracted_features = unredactor.getRedactedFeatures(test_data)

    assert type(extracted_features) == list
    assert type(extracted_features[0]) == tuple
    assert type(extracted_features[0][0]) == dict
    assert extracted_features == expected
