from preprocessing.normalization import Normalizer
from preprocessing import feature_extraction as fe
from classification.data import train_test_data as data
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer
from classification.classification import create_classification_model_and_evaluate
from classification.data.train_test_data import prepare_datasets
from collections import Counter
import copy
import urllib.request
import json

from classification_perform import *

def read_file_and_process(path):
    try:
        with open(path, 'r') as f:
            fnquads = list()
            for line in f:
                split = line.split(' ')
                fnquad = list()
                fnquad.append(split[0])
                fnquad.append(split[1])
                text = " ".join(split[2:len(split) - 2])
                fnquad.append(text)
                fnquad.append(split[len(split) - 2])
                fnquad.append(split[len(split) - 1])
                fnquads.append(fnquad)
    
        labels = [fnquad[3] for fnquad in fnquads]
    
        cmn = Counter(labels)
    
        new_fnquads = list()
        for fnquad in fnquads:
            if cmn[fnquad[3]] > 2:
                new_fnquads.append(fnquad)
    
        fnquads = new_fnquads
        descriptions = [fnquad[2] for fnquad in fnquads]
        descriptions_raw = copy.deepcopy(descriptions)
        normalizer = Normalizer()
        descriptions = [normalizer.normalize_text(description) for description in descriptions]
    
        descriptions = [' '.join(description) for description in descriptions]
    
        labels = [fnquad[3] for fnquad in fnquads]
        
        sorted_labels = list(reversed([lab[0] for lab in cmn.most_common()]))
        sorted_labels = [lab for lab in sorted_labels if cmn[lab] > 2]
        
        return descriptions_raw, descriptions, labels, sorted_labels
    except Exception as e:
        raise Exception("Something went wrong in processing file on location " + path)

def create_label_mappings():
    d = json.loads(urllib.request.urlopen(r"http://lov.okfn.org/dataset/lov/api/v2/vocabulary/list").read())
    label_mappings = dict()
    for j in d:
        label_mappings[j['uri']] = j["titles"][0]["value"]
    return label_mappings
    
def train(file, type_):
    print("Training...")
    
    if type_ == "path":
        descriptions_raw, descriptions, labels, sorted_labels = read_file_and_process(r'C:\Users\aleks\Desktop\lov_filtered.nq')
    elif type_ == "bin":
        pass
    elif type_ == "uri":
        pass
    else:
        return
    
    label_mappings = create_label_mappings()
    train_data, test_data, train_labels, test_labels = prepare_datasets(descriptions, labels, 0.1)    
    
    dictionary = dict()
    
    for i in range(len(descriptions)):
        dictionary[descriptions[i]] = descriptions_raw[i]
    
    with open(r'train_data.txt', 'w') as f:
        print('Creating test data file...')
        for i in range(len(test_data)):
            f.write(dictionary[test_data[i]] + '\t' + test_labels[i] + '\n')
        print('Done creating test data file.')
    
    train_data_r, test_data_r, vectorizer = prepare_data(train_data, test_data, type_='bow', binary=False, ngram_range=(1, 3))    
    main_classifier, test_labels, predictions  = create_classification_model_and_evaluate(MultinomialNB(alpha=0.0001), train_data_r, train_labels, test_data_r, test_labels)
    
    pipeline = list()
    errors = list()
    
    index = -1
    for lab in sorted_labels:
        index += 1
        class_labels_train = list()
        for label in train_labels:
            if label == lab:
                class_labels_train.append(label)
            else:
                class_labels_train.append('other')
        
        class_labels_test = list()
        for label in test_labels:
            if label == lab:
                class_labels_test.append(label)
            else:
                class_labels_test.append('other')
        
        if len(set(class_labels_train)) != 2 or len(set(class_labels_test)) != 2:
            errors.append((lab, index))
            pipeline.append('error')
            continue
            
        #train_data_r, test_data_r, vectorizer = prepare_data(train_data, test_data, type_='bow', binary=False, ngram_range=(1, 3))
        classifier, real_labels, predictions  = create_classification_model_and_evaluate(MultinomialNB(alpha=0.0001), train_data_r, class_labels_train, test_data_r, class_labels_test)
        pipeline.append(classifier)
    
    modified_vectorizers = dict()
    for lab, index in errors:
        train_data, test_data, train_labels, test_labels = prepare_datasets(descriptions, labels, 0.1)
        
        labs = list()
        for label in train_labels:
            if label != lab:
                labs.append('other')
            else:
                labs.append(label)
        train_labels = copy.deepcopy(labs)
        
        labs = list()
        for label in test_labels:
            if label != lab:
                labs.append('other')
            else:
                labs.append(label)
        test_labels = copy.deepcopy(labs)
    
        if len(set(train_labels)) == 2:
            full = train_labels
            full_data = train_data
            empty = test_labels
            empty_data = test_data
            treshold = 1/3
        else:
            full = test_labels
            full_data = test_data
            empty = train_labels
            empty_data = train_data
            treshold = 2/3
            
        cnt = 0
        for i in range(len(full)):
            if full[i] != 'other':
                cnt += 1
            
        to_remove = list()
        to_remove_data = list()
        new_cnt = 0
        for i in range(len(full)):
            if new_cnt > cnt * treshold:
                break
            if full[i] != 'other':
                empty.append(full[i])
                empty_data.append(full_data[i])
                to_remove.append(full[i])
                to_remove_data.append(full_data[i])
                new_cnt += 1
            
        for lab in to_remove:
            full.remove(lab)
            
        for data in to_remove_data:
            full_data.remove(data)
        
        train_data_r, test_data_r, vectorizer_modified = prepare_data(train_data, test_data, type_='bow', binary=False, ngram_range=(1, 3))
        classifier, real_labels, predictions  = create_classification_model_and_evaluate(MultinomialNB(alpha=0.0001), train_data_r, train_labels, test_data_r, test_labels)
        modified_vectorizers[lab] = vectorizer_modified
        pipeline[index] = classifier
        
        print("Training done.")
        return normalizer, pipeline, sorted_labels, vectorizer, modified_vectorizers, main_classifier, label_mappings
    
def classify(text, normalizer, pipeline, labels, vectorizer, modified_vectorizers, bulk_classifier, label_mappings):
    normalized_text = [' '.join(normalizer.normalize_text(text))]
    predicted_labels = list()
    
    for i in range(len(pipeline)):
        if labels[i] in modified_vectorizers.keys():
            vector = modified_vectorizers[labels[i]].transform(normalized_text)
        else:
            vector = vectorizer.transform(normalized_text)
        
        prediction = pipeline[i].predict(vector).tolist()[0]
        if prediction != 'other':
            predicted_labels.append(prediction)
        
    bulk_vector = vectorizer.transform(normalized_text)
    bulk_prediction = bulk_classifier.predict(bulk_vector).tolist()[0]
    
    if bulk_prediction in set(predicted_labels):
        predicted_labels.remove(bulk_prediction)
        predicted_labels = [bulk_prediction] + predicted_labels
    else:
        predicted_labels.append(bulk_prediction)
        
    result = list()
    for predicted_label in predicted_labels:
        result.append(label_mappings[predicted_label[1:len(predicted_label) - 1]])
    
    return result

def decode(prediction, label_mappings):
    for key in label_mappings.keys():
        if label_mappings[key] == prediction:
            return key