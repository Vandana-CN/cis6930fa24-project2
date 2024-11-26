#### Author: Vandana Cendrollu Nagesh

---

## Introduction

When confidential information such as names, locations, or other private data is shared with external parties like the media, it must undergo a redaction process. This involves removing or hiding sensitive details to protect privacy. However, the redaction process for sensitive documents—such as police reports, medical records, or court documents—can be time-consuming and costly.

This project explores how to automate the redaction and prediction of sensitive information using machine learning. By utilizing libraries such as Sci-Kit Learn and NLTK, we build a model that predicts the unredacted data (such as names) from a redacted document, assisting in faster and more efficient redaction. The dataset used for training the model is based on the Large Movie Review Dataset, containing movie reviews.


## Assumptions

In this project, I assume that:

Names to be redacted are in a maximum of four words.
The training data and redacted files are stored as text files. Other formats might cause errors.
The redacted and unredacted names are handled by the script by extracting name features such as word length and count.
All related project files are stored in the project_docs directory.
Training data is located in project_docs/train, and the redacted text files are found in project_docs/redact. The processed redacted files are saved with a .redacted extension, and the predicted names are stored in files with the .predicted extension.

## Description

The core functionality of this project is based on reading redacted text, predicting possible unredacted names using a machine learning model, and writing the predictions to a file.

Key Files
main.py – This file drives the entire project by calling methods from redactor.py and unredactor.py to handle the redaction and prediction tasks. It accepts parameters like --tdata for training data and --input for redacted text files.

unredactor.py – This file is responsible for processing redacted text to extract features for training and prediction. It defines methods like extractTrain() and extractRedacted() to build feature sets for the model.

redactor.py – This file is used to perform the actual redaction, replacing sensitive names with block characters (█) to hide the information.

test_unredactor.py – This file contains unit tests for the methods in unredactor.py and redactor.py, ensuring the functionality works as expected.
## Process Flow
Redaction Process: The method redactor.redactNames() reads a text file, tokenizes it, identifies named entities (like people’s names), and replaces them with block characters (█).

Training the Model: unredactor.extractTrain() processes the training data, extracting features from text, such as name length, word count, and space count, using NLTK’s named entity recognition.

Prediction: After training the model using a KNN classifier, unredactor.extractRedacted() identifies the redacted names in a document and uses the trained model to predict the most likely unredacted names.

Output: The predicted names are written to a .predicted file, where the top 4 most probable unredacted names are listed.

## Unit Tests
The test_unredactor.py file tests all the key methods in redactor.py and unredactor.py. These tests ensure the correct functionality of redaction and prediction processes.

testRedactNames(): Verifies that the redacted data is correctly written to a file.
testExtractTrain(): Ensures that the training data is correctly extracted and returned as a list of tuples.
testGetTrainFeatures(): Verifies that the training features are properly extracted from the data.
testExtractRedacted(): Tests the extraction of redacted names and their features.
testGetRedactedFeatures(): Validates the extraction of features from redacted data.

