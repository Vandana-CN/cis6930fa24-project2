import argparse
import os
import re
from sklearn.feature_extraction import DictVectorizer
from sklearn.neighbors import KNeighborsClassifier

import redactor
import unredactor


def main(input_parameters):
    print('started')
    redacted_doc = redactor.redactNames(input_parameters.input)

    training_data = unredactor.extractTrain(input_parameters.tdata)

    redacted_data = unredactor.extractRedacted(redacted_doc)
    print(unredactor)

    v = DictVectorizer(sparse=False)


    X_train = v.fit_transform([x for (x, y) in training_data])

    y_train = [y for (x, y) in training_data]


    X_redacted = v.fit_transform([x for (x, y) in redacted_data])

    y_redacted = [y for (x, y) in redacted_data]

  

    knnModel = KNeighborsClassifier(
        n_neighbors=5, weights='uniform', algorithm='auto') 
    knnModel.fit(X_train, y_train)  

    indx_KNN = knnModel.kneighbors(
        X_redacted, n_neighbors=4, return_distance=False)  

    predicted_doc = open(redacted_doc.replace('.redacted', '.predicted'), 'w')

    predicted_doc.write(
        '*****  Predicting 4 most likely unredacted names for the redacted names present in document  *****\n')

    predicted_doc.close()  

    count = 1
    for x, y in zip(y_redacted, indx_KNN):
        print(count)
        doc = open(redacted_doc.replace('.redacted', '.predicted'),
                   'a')  

        message = '\n{}. The top 4 likely names for {} :: {}, {}, {}, {}\n'.format(
            count, x, y_train[y[0]], y_train[y[1]], y_train[y[2]], y_train[y[3]])
        count += 1
        doc.write(message)

        doc.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument("--input", type=str, required=True,
                        help="location of the file to be redacted and unredacted")  

    parser.add_argument("--tdata", type=str, required=True,
                        help="path and pattern to match training files")  

    args = parser.parse_args()

    main(args)
