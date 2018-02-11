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
import pickle
import copy
import gc

from classification_performance import *

from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2

with open(r'C:\Users\aleks\Desktop\lov_filtered.nq', 'r') as f:
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
normalizer = Normalizer()
descriptions = [normalizer.normalize_text(description) for description in descriptions]
#normalized_descriptions = copy.deepcopy(descriptions)
#with open(r'C:\Users\aleks\Desktop\descriptions.pkl', 'rb') as f:
    #descriptions = pickle.load(f)

descriptions = [' '.join(description) for description in descriptions]

labels = [fnquad[3] for fnquad in fnquads]


train_data, test_data, train_labels, test_labels = prepare_datasets(descriptions, labels, 0.1)

train_data_r, test_data_r, main_vectorizer = prepare_data(train_data, test_data, type_='bow', binary=False, ngram_range=(1, 3))

main_classifier, test_labels, predictions  = create_classification_model_and_evaluate(MultinomialNB(alpha=0.0001), train_data_r, train_labels, test_data_r, test_labels)

sorted_labels = list(reversed([lab[0] for lab in cmn.most_common()]))

print('\n\n')

pipeline = list()
vectorizers = list()
errors = list()

sorted_labels = [lab for lab in sorted_labels if cmn[lab] > 2]

train_data, test_data, train_labels, test_labels = prepare_datasets(descriptions, labels, 0.3)
train_data_r, test_data_r, vectorizer = prepare_data(train_data, test_data, type_='bow', binary=False, ngram_range=(1, 3))

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
    
    print(lab + ' [' + str(cmn[lab]) + '] ' + ':\n')
    #train_data, test_data, train_labels, test_labels = prepare_datasets(descriptions, class_labels, 0.25)
    if len(set(class_labels_train)) != 2 or len(set(test_labels)) != 2:
        errors.append((lab, index))
        pipeline.append('error')
        vectorizers.append('error')
        print('Error\n\n')
        continue
        """
        if len(set(class_labels_train)) == 2:
            full = class_labels_train
            full_data = train_data
            empty = test_labels
            empty_data = test_data
        else:
            full = test_labels
            full_data = test_data
            empty = class_labels_train
            empty_data = train_data
        
        cnt = 0
        for i in range(len(full)):
            if full[i] != 'other':
                cnt += 1
        
        to_remove = list()
        to_remove_data = list()
        new_cnt = 0
        for i in range(len(full)):
            if new_cnt > cnt / 2:
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
        
        if len(set(train_labels)) != 2 or len(set(test_labels)) != 2:
            something_wrong = True
            print("Something is terribly wrong...\n")
        
        del full, full_data, empty, empty_data, cnt, new_cnt
        """
    #train_data_r, test_data_r, vectorizer = prepare_data(train_data, test_data, type_='bow', binary=False, ngram_range=(1, 3))
    classifier, test_labels, predictions  = create_classification_model_and_evaluate(MultinomialNB(alpha=0.0001), train_data_r, class_labels_train, test_data_r, class_labels_test)
    pipeline.append(classifier)
    vectorizers.append(vectorizer)
    print('\n\n')
        
"""
print("30% test set")

train_data_r, test_data_r = prepare_data(train_data, test_data, type_='bow', binary=True, ngram_range=(1, 1))

print("BOW binary, (1, 1)")

print("Naive Bayes")
test_labels, predictions  = create_classification_model_and_evaluate(MultinomialNB(), train_data_r, train_labels, test_data_r, test_labels)
#print("Support Vector Machine")
#test_labels, predictions  = create_classification_model_and_evaluate(SVC(), train_data_r, train_labels, test_data_r, test_labels)
print("DecisionTree")
test_labels, predictions  = create_classification_model_and_evaluate(DecisionTreeClassifier(), train_data_r, train_labels, test_data_r, test_labels)
print("KNN")
test_labels, predictions  = create_classification_model_and_evaluate(KNeighborsClassifier(), train_data_r, train_labels, test_data_r, test_labels)

train_data, test_data, train_labels, test_labels = prepare_datasets(descriptions, labels, 0.2)

print("20% test set")

train_data_r, test_data_r = prepare_data(train_data, test_data, type_='bow', binary=True, ngram_range=(1, 1))

print("BOW binary, (1, 1)")

print("Naive Bayes")
test_labels, predictions  = create_classification_model_and_evaluate(MultinomialNB(), train_data_r, train_labels, test_data_r, test_labels)
#print("Support Vector Machine")
#test_labels, predictions  = create_classification_model_and_evaluate(SVC(), train_data_r, train_labels, test_data_r, test_labels)
print("DecisionTree")
test_labels, predictions  = create_classification_model_and_evaluate(DecisionTreeClassifier(), train_data_r, train_labels, test_data_r, test_labels)
print("KNN")
test_labels, predictions  = create_classification_model_and_evaluate(KNeighborsClassifier(), train_data_r, train_labels, test_data_r, test_labels)

train_data, test_data, train_labels, test_labels = prepare_datasets(descriptions, labels, 0.1)

print("10% test set")

train_data_r, test_data_r = prepare_data(train_data, test_data, type_='bow', binary=True, ngram_range=(1, 1))

print("BOW binary, (1, 1)")

print("Naive Bayes")
test_labels, predictions  = create_classification_model_and_evaluate(MultinomialNB(), train_data_r, train_labels, test_data_r, test_labels)
#print("Support Vector Machine")
#test_labels, predictions  = create_classification_model_and_evaluate(SVC(), train_data_r, train_labels, test_data_r, test_labels)
print("DecisionTree")
test_labels, predictions  = create_classification_model_and_evaluate(DecisionTreeClassifier(), train_data_r, train_labels, test_data_r, test_labels)
print("KNN")
test_labels, predictions  = create_classification_model_and_evaluate(KNeighborsClassifier(), train_data_r, train_labels, test_data_r, test_labels)


train_data, test_data, train_labels, test_labels = prepare_datasets(descriptions, labels, 0.3)

print("30% test set")

train_data_r, test_data_r = prepare_data(train_data, test_data, type_='bow', binary=True, ngram_range=(1, 2))

print("BOW binary, (1, 2)")

print("Naive Bayes")
test_labels, predictions  = create_classification_model_and_evaluate(MultinomialNB(), train_data_r, train_labels, test_data_r, test_labels)
#print("Support Vector Machine")
#test_labels, predictions  = create_classification_model_and_evaluate(SVC(), train_data_r, train_labels, test_data_r, test_labels)
print("DecisionTree")
test_labels, predictions  = create_classification_model_and_evaluate(DecisionTreeClassifier(), train_data_r, train_labels, test_data_r, test_labels)
print("KNN")
test_labels, predictions  = create_classification_model_and_evaluate(KNeighborsClassifier(), train_data_r, train_labels, test_data_r, test_labels)

train_data, test_data, train_labels, test_labels = prepare_datasets(descriptions, labels, 0.2)

print("20% test set")

train_data_r, test_data_r = prepare_data(train_data, test_data, type_='bow', binary=True, ngram_range=(1, 2))

print("BOW binary, (1, 2)")

print("Naive Bayes")
test_labels, predictions  = create_classification_model_and_evaluate(MultinomialNB(), train_data_r, train_labels, test_data_r, test_labels)
#print("Support Vector Machine")
#test_labels, predictions  = create_classification_model_and_evaluate(SVC(), train_data_r, train_labels, test_data_r, test_labels)
print("DecisionTree")
test_labels, predictions  = create_classification_model_and_evaluate(DecisionTreeClassifier(), train_data_r, train_labels, test_data_r, test_labels)
print("KNN")
test_labels, predictions  = create_classification_model_and_evaluate(KNeighborsClassifier(), train_data_r, train_labels, test_data_r, test_labels)

train_data, test_data, train_labels, test_labels = prepare_datasets(descriptions, labels, 0.1)

print("10% test set")

train_data_r, test_data_r = prepare_data(train_data, test_data, type_='bow', binary=True, ngram_range=(1, 2))

print("BOW binary, (1, 2)")

print("Naive Bayes")
test_labels, predictions  = create_classification_model_and_evaluate(MultinomialNB(), train_data_r, train_labels, test_data_r, test_labels)
#print("Support Vector Machine")
#test_labels, predictions  = create_classification_model_and_evaluate(SVC(), train_data_r, train_labels, test_data_r, test_labels)
print("DecisionTree")
test_labels, predictions  = create_classification_model_and_evaluate(DecisionTreeClassifier(), train_data_r, train_labels, test_data_r, test_labels)
print("KNN")
test_labels, predictions  = create_classification_model_and_evaluate(KNeighborsClassifier(), train_data_r, train_labels, test_data_r, test_labels)

train_data, test_data, train_labels, test_labels = prepare_datasets(descriptions, labels, 0.3)

print("30% test set")

train_data_r, test_data_r = prepare_data(train_data, test_data, type_='bow', binary=True, ngram_range=(2, 3))

print("BOW binary, (2, 3)")

print("Naive Bayes")
test_labels, predictions  = create_classification_model_and_evaluate(MultinomialNB(), train_data_r, train_labels, test_data_r, test_labels)
#print("Support Vector Machine")
#test_labels, predictions  = create_classification_model_and_evaluate(SVC(), train_data_r, train_labels, test_data_r, test_labels)
print("DecisionTree")
test_labels, predictions  = create_classification_model_and_evaluate(DecisionTreeClassifier(), train_data_r, train_labels, test_data_r, test_labels)
print("KNN")
test_labels, predictions  = create_classification_model_and_evaluate(KNeighborsClassifier(), train_data_r, train_labels, test_data_r, test_labels)

train_data, test_data, train_labels, test_labels = prepare_datasets(descriptions, labels, 0.2)

print("20% test set")

train_data_r, test_data_r = prepare_data(train_data, test_data, type_='bow', binary=True, ngram_range=(2, 3))

print("BOW binary, (2, 3)")

print("Naive Bayes")
test_labels, predictions  = create_classification_model_and_evaluate(MultinomialNB(), train_data_r, train_labels, test_data_r, test_labels)
#print("Support Vector Machine")
#test_labels, predictions  = create_classification_model_and_evaluate(SVC(), train_data_r, train_labels, test_data_r, test_labels)
print("DecisionTree")
test_labels, predictions  = create_classification_model_and_evaluate(DecisionTreeClassifier(), train_data_r, train_labels, test_data_r, test_labels)
print("KNN")
test_labels, predictions  = create_classification_model_and_evaluate(KNeighborsClassifier(), train_data_r, train_labels, test_data_r, test_labels)

train_data, test_data, train_labels, test_labels = prepare_datasets(descriptions, labels, 0.1)

print("10% test set")

train_data_r, test_data_r = prepare_data(train_data, test_data, type_='bow', binary=True, ngram_range=(2, 3))

print("BOW binary, (2, 3)")

print("Naive Bayes")
test_labels, predictions  = create_classification_model_and_evaluate(MultinomialNB(), train_data_r, train_labels, test_data_r, test_labels)
#print("Support Vector Machine")
#test_labels, predictions  = create_classification_model_and_evaluate(SVC(), train_data_r, train_labels, test_data_r, test_labels)
print("DecisionTree")
test_labels, predictions  = create_classification_model_and_evaluate(DecisionTreeClassifier(), train_data_r, train_labels, test_data_r, test_labels)
print("KNN")
test_labels, predictions  = create_classification_model_and_evaluate(KNeighborsClassifier(), train_data_r, train_labels, test_data_r, test_labels)


train_data, test_data, train_labels, test_labels = prepare_datasets(descriptions, labels, 0.3)

print("30% test set")

train_data_r, test_data_r = prepare_data(train_data, test_data, type_='bow', binary=False, ngram_range=(1, 1))

print("BOW, (1, 1)")

print("Naive Bayes")
test_labels, predictions  = create_classification_model_and_evaluate(MultinomialNB(), train_data_r, train_labels, test_data_r, test_labels)
#print("Support Vector Machine")
#test_labels, predictions  = create_classification_model_and_evaluate(SVC(), train_data_r, train_labels, test_data_r, test_labels)
print("DecisionTree")
test_labels, predictions  = create_classification_model_and_evaluate(DecisionTreeClassifier(), train_data_r, train_labels, test_data_r, test_labels)
print("KNN")
test_labels, predictions  = create_classification_model_and_evaluate(KNeighborsClassifier(), train_data_r, train_labels, test_data_r, test_labels)

train_data, test_data, train_labels, test_labels = prepare_datasets(descriptions, labels, 0.2)

print("20% test set")

train_data_r, test_data_r = prepare_data(train_data, test_data, type_='bow', binary=False, ngram_range=(1, 1))

print("BOW, (1, 1)")

print("Naive Bayes")
test_labels, predictions  = create_classification_model_and_evaluate(MultinomialNB(), train_data_r, train_labels, test_data_r, test_labels)
#print("Support Vector Machine")
#test_labels, predictions  = create_classification_model_and_evaluate(SVC(), train_data_r, train_labels, test_data_r, test_labels)
print("DecisionTree")
test_labels, predictions  = create_classification_model_and_evaluate(DecisionTreeClassifier(), train_data_r, train_labels, test_data_r, test_labels)
print("KNN")
test_labels, predictions  = create_classification_model_and_evaluate(KNeighborsClassifier(), train_data_r, train_labels, test_data_r, test_labels)

train_data, test_data, train_labels, test_labels = prepare_datasets(descriptions, labels, 0.1)

print("10% test set")

train_data_r, test_data_r = prepare_data(train_data, test_data, type_='bow', binary=False, ngram_range=(1, 1))

print("BOW, (1, 1)")

print("Naive Bayes")
test_labels, predictions  = create_classification_model_and_evaluate(MultinomialNB(), train_data_r, train_labels, test_data_r, test_labels)
#print("Support Vector Machine")
#test_labels, predictions  = create_classification_model_and_evaluate(SVC(), train_data_r, train_labels, test_data_r, test_labels)
print("DecisionTree")
test_labels, predictions  = create_classification_model_and_evaluate(DecisionTreeClassifier(), train_data_r, train_labels, test_data_r, test_labels)
print("KNN")
test_labels, predictions  = create_classification_model_and_evaluate(KNeighborsClassifier(), train_data_r, train_labels, test_data_r, test_labels)


train_data, test_data, train_labels, test_labels = prepare_datasets(descriptions, labels, 0.3)

print("30% test set")

train_data_r, test_data_r = prepare_data(train_data, test_data, type_='bow', binary=False, ngram_range=(1, 2))

print("BOW, (1, 2)")

print("Naive Bayes")
test_labels, predictions  = create_classification_model_and_evaluate(MultinomialNB(), train_data_r, train_labels, test_data_r, test_labels)
#print("Support Vector Machine")
#test_labels, predictions  = create_classification_model_and_evaluate(SVC(), train_data_r, train_labels, test_data_r, test_labels)
print("DecisionTree")
test_labels, predictions  = create_classification_model_and_evaluate(DecisionTreeClassifier(), train_data_r, train_labels, test_data_r, test_labels)
print("KNN")
test_labels, predictions  = create_classification_model_and_evaluate(KNeighborsClassifier(), train_data_r, train_labels, test_data_r, test_labels)

train_data, test_data, train_labels, test_labels = prepare_datasets(descriptions, labels, 0.2)

print("20% test set")

train_data_r, test_data_r = prepare_data(train_data, test_data, type_='bow', binary=False, ngram_range=(1, 2))

print("BOW, (1, 2)")

print("Naive Bayes")
test_labels, predictions  = create_classification_model_and_evaluate(MultinomialNB(), train_data_r, train_labels, test_data_r, test_labels)
#print("Support Vector Machine")
#test_labels, predictions  = create_classification_model_and_evaluate(SVC(), train_data_r, train_labels, test_data_r, test_labels)
print("DecisionTree")
test_labels, predictions  = create_classification_model_and_evaluate(DecisionTreeClassifier(), train_data_r, train_labels, test_data_r, test_labels)
print("KNN")
test_labels, predictions  = create_classification_model_and_evaluate(KNeighborsClassifier(), train_data_r, train_labels, test_data_r, test_labels)

train_data, test_data, train_labels, test_labels = prepare_datasets(descriptions, labels, 0.1)

print("10% test set")

train_data_r, test_data_r = prepare_data(train_data, test_data, type_='bow', binary=False, ngram_range=(1, 2))

print("BOW, (1, 2)")

print("Naive Bayes")
test_labels, predictions  = create_classification_model_and_evaluate(MultinomialNB(), train_data_r, train_labels, test_data_r, test_labels)
#print("Support Vector Machine")
#test_labels, predictions  = create_classification_model_and_evaluate(SVC(), train_data_r, train_labels, test_data_r, test_labels)
print("DecisionTree")
test_labels, predictions  = create_classification_model_and_evaluate(DecisionTreeClassifier(), train_data_r, train_labels, test_data_r, test_labels)
print("KNN")
test_labels, predictions  = create_classification_model_and_evaluate(KNeighborsClassifier(), train_data_r, train_labels, test_data_r, test_labels)

train_data, test_data, train_labels, test_labels = prepare_datasets(descriptions, labels, 0.3)

print("30% test set")

train_data_r, test_data_r = prepare_data(train_data, test_data, type_='bow', binary=False, ngram_range=(2, 3))

print("BOW, (2, 3)")

print("Naive Bayes")
test_labels, predictions  = create_classification_model_and_evaluate(MultinomialNB(), train_data_r, train_labels, test_data_r, test_labels)
#print("Support Vector Machine")
#test_labels, predictions  = create_classification_model_and_evaluate(SVC(), train_data_r, train_labels, test_data_r, test_labels)
print("DecisionTree")
test_labels, predictions  = create_classification_model_and_evaluate(DecisionTreeClassifier(), train_data_r, train_labels, test_data_r, test_labels)
print("KNN")
test_labels, predictions  = create_classification_model_and_evaluate(KNeighborsClassifier(), train_data_r, train_labels, test_data_r, test_labels)

train_data, test_data, train_labels, test_labels = prepare_datasets(descriptions, labels, 0.2)

print("20% test set")

train_data_r, test_data_r = prepare_data(train_data, test_data, type_='bow', binary=False, ngram_range=(2, 3))

print("BOW, (2, 3)")

print("Naive Bayes")
test_labels, predictions  = create_classification_model_and_evaluate(MultinomialNB(), train_data_r, train_labels, test_data_r, test_labels)
#print("Support Vector Machine")
#test_labels, predictions  = create_classification_model_and_evaluate(SVC(), train_data_r, train_labels, test_data_r, test_labels)
print("DecisionTree")
test_labels, predictions  = create_classification_model_and_evaluate(DecisionTreeClassifier(), train_data_r, train_labels, test_data_r, test_labels)
print("KNN")
test_labels, predictions  = create_classification_model_and_evaluate(KNeighborsClassifier(), train_data_r, train_labels, test_data_r, test_labels)

train_data, test_data, train_labels, test_labels = prepare_datasets(descriptions, labels, 0.1)

print("10% test set")

train_data_r, test_data_r = prepare_data(train_data, test_data, type_='bow', binary=False, ngram_range=(2, 3))

print("BOW, (2, 3)")

print("Naive Bayes")
test_labels, predictions  = create_classification_model_and_evaluate(MultinomialNB(), train_data_r, train_labels, test_data_r, test_labels)
#print("Support Vector Machine")
#test_labels, predictions  = create_classification_model_and_evaluate(SVC(), train_data_r, train_labels, test_data_r, test_labels)
print("DecisionTree")
test_labels, predictions  = create_classification_model_and_evaluate(DecisionTreeClassifier(), train_data_r, train_labels, test_data_r, test_labels)
print("KNN")
test_labels, predictions  = create_classification_model_and_evaluate(KNeighborsClassifier(), train_data_r, train_labels, test_data_r, test_labels)


train_data, test_data, train_labels, test_labels = prepare_datasets(descriptions, labels, 0.3)

print("30% test set")

train_data_r, test_data_r = prepare_data(train_data, test_data, type_='tfidf', binary=False, ngram_range=(1, 1))

print("tfidf, (1, 1)")

print("Naive Bayes")
test_labels, predictions  = create_classification_model_and_evaluate(MultinomialNB(), train_data_r, train_labels, test_data_r, test_labels)
#print("Support Vector Machine")
#test_labels, predictions  = create_classification_model_and_evaluate(SVC(), train_data_r, train_labels, test_data_r, test_labels)
print("DecisionTree")
test_labels, predictions  = create_classification_model_and_evaluate(DecisionTreeClassifier(), train_data_r, train_labels, test_data_r, test_labels)
print("KNN")
test_labels, predictions  = create_classification_model_and_evaluate(KNeighborsClassifier(), train_data_r, train_labels, test_data_r, test_labels)

train_data, test_data, train_labels, test_labels = prepare_datasets(descriptions, labels, 0.2)

print("20% test set")

train_data_r, test_data_r = prepare_data(train_data, test_data, type_='tfidf', binary=False, ngram_range=(1, 1))

print("tfidf, (1, 1)")

print("Naive Bayes")
test_labels, predictions  = create_classification_model_and_evaluate(MultinomialNB(), train_data_r, train_labels, test_data_r, test_labels)
#print("Support Vector Machine")
#test_labels, predictions  = create_classification_model_and_evaluate(SVC(), train_data_r, train_labels, test_data_r, test_labels)
print("DecisionTree")
test_labels, predictions  = create_classification_model_and_evaluate(DecisionTreeClassifier(), train_data_r, train_labels, test_data_r, test_labels)
print("KNN")
test_labels, predictions  = create_classification_model_and_evaluate(KNeighborsClassifier(), train_data_r, train_labels, test_data_r, test_labels)

train_data, test_data, train_labels, test_labels = prepare_datasets(descriptions, labels, 0.1)

print("10% test set")

train_data_r, test_data_r = prepare_data(train_data, test_data, type_='tfidf', binary=False, ngram_range=(1, 1))

print("tfidf, (1, 1)")

print("Naive Bayes")
test_labels, predictions  = create_classification_model_and_evaluate(MultinomialNB(), train_data_r, train_labels, test_data_r, test_labels)
#print("Support Vector Machine")
#test_labels, predictions  = create_classification_model_and_evaluate(SVC(), train_data_r, train_labels, test_data_r, test_labels)
print("DecisionTree")
test_labels, predictions  = create_classification_model_and_evaluate(DecisionTreeClassifier(), train_data_r, train_labels, test_data_r, test_labels)
print("KNN")
test_labels, predictions  = create_classification_model_and_evaluate(KNeighborsClassifier(), train_data_r, train_labels, test_data_r, test_labels)


train_data, test_data, train_labels, test_labels = prepare_datasets(descriptions, labels, 0.3)

print("30% test set")

train_data_r, test_data_r = prepare_data(train_data, test_data, type_='tfidf', binary=False, ngram_range=(1, 2))

print("tfidf, (1, 2)")

print("Naive Bayes")
test_labels, predictions  = create_classification_model_and_evaluate(MultinomialNB(), train_data_r, train_labels, test_data_r, test_labels)
#print("Support Vector Machine")
#test_labels, predictions  = create_classification_model_and_evaluate(SVC(), train_data_r, train_labels, test_data_r, test_labels)
print("DecisionTree")
test_labels, predictions  = create_classification_model_and_evaluate(DecisionTreeClassifier(), train_data_r, train_labels, test_data_r, test_labels)
print("KNN")
test_labels, predictions  = create_classification_model_and_evaluate(KNeighborsClassifier(), train_data_r, train_labels, test_data_r, test_labels)

train_data, test_data, train_labels, test_labels = prepare_datasets(descriptions, labels, 0.2)

print("20% test set")

train_data_r, test_data_r = prepare_data(train_data, test_data, type_='tfidf', binary=False, ngram_range=(1, 2))

print("tfidf, (1, 2)")

print("Naive Bayes")
test_labels, predictions  = create_classification_model_and_evaluate(MultinomialNB(), train_data_r, train_labels, test_data_r, test_labels)
#print("Support Vector Machine")
#test_labels, predictions  = create_classification_model_and_evaluate(SVC(), train_data_r, train_labels, test_data_r, test_labels)
print("DecisionTree")
test_labels, predictions  = create_classification_model_and_evaluate(DecisionTreeClassifier(), train_data_r, train_labels, test_data_r, test_labels)
print("KNN")
test_labels, predictions  = create_classification_model_and_evaluate(KNeighborsClassifier(), train_data_r, train_labels, test_data_r, test_labels)

train_data, test_data, train_labels, test_labels = prepare_datasets(descriptions, labels, 0.1)

print("10% test set")

train_data_r, test_data_r = prepare_data(train_data, test_data, type_='tfidf', binary=False, ngram_range=(1, 2))

print("tfidf, (1, 2)")

print("Naive Bayes")
test_labels, predictions  = create_classification_model_and_evaluate(MultinomialNB(), train_data_r, train_labels, test_data_r, test_labels)
#print("Support Vector Machine")
#test_labels, predictions  = create_classification_model_and_evaluate(SVC(), train_data_r, train_labels, test_data_r, test_labels)
print("DecisionTree")
test_labels, predictions  = create_classification_model_and_evaluate(DecisionTreeClassifier(), train_data_r, train_labels, test_data_r, test_labels)
print("KNN")
test_labels, predictions  = create_classification_model_and_evaluate(KNeighborsClassifier(), train_data_r, train_labels, test_data_r, test_labels)

train_data, test_data, train_labels, test_labels = prepare_datasets(descriptions, labels, 0.3)

print("30% test set")

train_data_r, test_data_r = prepare_data(train_data, test_data, type_='tfidf', binary=False, ngram_range=(2, 3))

print("tfidf, (2, 3)")

print("Naive Bayes")
test_labels, predictions  = create_classification_model_and_evaluate(MultinomialNB(), train_data_r, train_labels, test_data_r, test_labels)
#print("Support Vector Machine")
#test_labels, predictions  = create_classification_model_and_evaluate(SVC(), train_data_r, train_labels, test_data_r, test_labels)
print("DecisionTree")
test_labels, predictions  = create_classification_model_and_evaluate(DecisionTreeClassifier(), train_data_r, train_labels, test_data_r, test_labels)
print("KNN")
test_labels, predictions  = create_classification_model_and_evaluate(KNeighborsClassifier(), train_data_r, train_labels, test_data_r, test_labels)

train_data, test_data, train_labels, test_labels = prepare_datasets(descriptions, labels, 0.2)

print("20% test set")

train_data_r, test_data_r = prepare_data(train_data, test_data, type_='tfidf', binary=False, ngram_range=(2, 3))

print("tfidf, (2, 3)")

print("Naive Bayes")
test_labels, predictions  = create_classification_model_and_evaluate(MultinomialNB(), train_data_r, train_labels, test_data_r, test_labels)
#print("Support Vector Machine")
#test_labels, predictions  = create_classification_model_and_evaluate(SVC(), train_data_r, train_labels, test_data_r, test_labels)
print("DecisionTree")
test_labels, predictions  = create_classification_model_and_evaluate(DecisionTreeClassifier(), train_data_r, train_labels, test_data_r, test_labels)
print("KNN")
test_labels, predictions  = create_classification_model_and_evaluate(KNeighborsClassifier(), train_data_r, train_labels, test_data_r, test_labels)

train_data, test_data, train_labels, test_labels = prepare_datasets(descriptions, labels, 0.1)

print("10% test set")

train_data_r, test_data_r = prepare_data(train_data, test_data, type_='tfidf', binary=False, ngram_range=(2, 3))

print("tfidf, (2, 3)")

print("Naive Bayes")
test_labels, predictions  = create_classification_model_and_evaluate(MultinomialNB(), train_data_r, train_labels, test_data_r, test_labels)
#print("Support Vector Machine")
#test_labels, predictions  = create_classification_model_and_evaluate(SVC(), train_data_r, train_labels, test_data_r, test_labels)
print("DecisionTree")
test_labels, predictions  = create_classification_model_and_evaluate(DecisionTreeClassifier(), train_data_r, train_labels, test_data_r, test_labels)
print("KNN")
test_labels, predictions  = create_classification_model_and_evaluate(KNeighborsClassifier(), train_data_r, train_labels, test_data_r, test_labels)
"""