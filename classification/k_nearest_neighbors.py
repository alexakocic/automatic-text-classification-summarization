"""
Created on Fri Dec  8 13:50:19 2017

@author: Aleksa Kocić

K-Nearest neighbors classifier.
"""

import sys
from scipy.spatial import distance

class KNearestNeighbors:
    def train(self, training_vectors):
        """
        Train classification model.
        
        Args:
            training_vectors (list of tuple of dict of str:bool pairs, str): 
                List of classified bag of words
        """
        # Fill vector space with points
        self.vectors = training_vectors
    
    def distance(self, vector1, vector2, type_="euclidean"):
        """
        Calculate distance between two vectors.
        
        Args:
            vector1 (list of float): Vector in vector space
            vector2 (list of float): Vector in vector space
            type_ (str): Type of distance calculation. Allowed types are:
                * For numeric vectors *
                - braycurtis: Computes the Bray-Curtis distance between two arrays.
                - canberra: Computes the Canberra distance between two arrays.
                - chebyshev: 	Computes the Chebyshev distance.
                - cityblock: Computes the City Block (Manhattan) distance.
                - correlation: Computes the correlation distance between two arrays.
                - cosine: Computes the Cosine distance between arrays.
                - euclidean: Computes the Euclidean distance between two arrays.
                - sqeuclidean: Computes the squared Euclidean distance between two arrays.
                
                * For boolean vectors *
                - dice: Computes the Dice dissimilarity between two boolean arrays.
                - hamming: Computes the Hamming distance between two arrays.
                - jaccard: Computes the Jaccard-Needham dissimilarity between two boolean arrays.
                - kulsinski: Computes the Kulsinski dissimilarity between two boolean arrays.
                - rogerstanimoto: Computes the Rogers-Tanimoto dissimilarity between two boolean arrays.
                - russellrao: Computes the Russell-Rao dissimilarity between two boolean arrays.
                - sokalmichener: Computes the Sokal-Michener dissimilarity between two boolean arrays.
                - sokalsneath: Computes the Sokal-Sneath dissimilarity between two boolean arrays.
                - yule: Computes the Yule dissimilarity between two boolean arrays.
        Returns:
            float: Distance between vectors.
        """
        if type_ == "braycurtis":
            return distance.braycurtis(vector1, vector2)
        elif type_ == "canberra":
            return distance.canberra(vector1, vector2)
        elif type_ == "chebyshev":
            return distance.chebyshev(vector1, vector2)
        elif type_ == "cityblock":
            return distance.cityblock(vector1, vector2)
        elif type_ == "correlation":
            return distance.correlation(vector1, vector2)
        elif type_ == "cosine":
            return distance.cosine(vector1, vector2)
        elif type_ == "euclidean":
            return distance.euclidean(vector1, vector2)
        elif type_ == "sqeuclidean":
            return distance.sqeuclidean(vector1, vector2)
        elif type_ == "dice":
            return distance.dice(vector1, vector2)
        elif type_ == "hamming":
            return distance.hamming(vector1, vector2)
        elif type_ == "jaccard":
            return distance.jaccard(vector1, vector2)
        elif type_ == "kulsinski":
            return distance.kulsinski(vector1, vector2)
        elif type_ == "kulsinski":
            return distance.kulsinski(vector1, vector2)
        elif type_ == "rogerstanimoto":
            return distance.rogerstanimoto(vector1, vector2)
        elif type_ == "russellrao":
            return distance.russellrao(vector1, vector2)
        elif type_ == "sokalmichener":
            return distance.sokalmichener(vector1, vector2)
        elif type_ == "sokalsneath":
            return distance.sokalsneath(vector1, vector2)
        elif type_ == "yule":
            return distance.yule(vector1, vector2)
        else:
            raise ValueError("""Wrong value for type_. Please enter one of supported values.
                             Type help(distance) to see supported values.""")
    
    def classify(self, document, k, distance_type):
        """
        Use trained model to classify a document.
        
        Args:
            document (dict of str:float pairs): Bag of words containing information
                about which word from vocabulary is present in a document
            k (int): Number of nearest neighbors to choose to classify from
            distance_type: Type of distance calculation. Allowed types are:
                * For numeric vectors *
                - braycurtis: Computes the Bray-Curtis distance between two arrays.
                - canberra: Computes the Canberra distance between two arrays.
                - chebyshev: 	Computes the Chebyshev distance.
                - cityblock: Computes the City Block (Manhattan) distance.
                - correlation: Computes the correlation distance between two arrays.
                - cosine: Computes the Cosine distance between arrays.
                - euclidean: Computes the Euclidean distance between two arrays.
                - sqeuclidean: Computes the squared Euclidean distance between two arrays.
                
                * For boolean vectors *
                - dice: Computes the Dice dissimilarity between two boolean arrays.
                - hamming: Computes the Hamming distance between two arrays.
                - jaccard: Computes the Jaccard-Needham dissimilarity between two boolean arrays.
                - kulsinski: Computes the Kulsinski dissimilarity between two boolean arrays.
                - rogerstanimoto: Computes the Rogers-Tanimoto dissimilarity between two boolean arrays.
                - russellrao: Computes the Russell-Rao dissimilarity between two boolean arrays.
                - sokalmichener: Computes the Sokal-Michener dissimilarity between two boolean arrays.
                - sokalsneath: Computes the Sokal-Sneath dissimilarity between two boolean arrays.
                - yule: Computes the Yule dissimilarity between two boolean arrays.
                
        Returns:
            str: Class label of document
        """
        if k == 0:
            raise ValueError("Must enter positive value for k parameter.")
            
        # If only one neighbor, do more optimal calculation
        if k == 1:
            return self.__classify_nearest_neighbor(document, distance_type)
            
        # List of distance - class tuples
        nearest_neighbors = list()
        
        for vector in self.vectors:
            distance = self.distance(document, vector[0], distance_type)
            n = len(nearest_neighbors)
            
            if  n < k:
                nearest_neighbors = sorted(nearest_neighbors.append((distance, vector[1])))
            else:    
                for i in range(n):
                    if distance < nearest_neighbors[i][0]:
                        j = n - 1
                        while j > i:
                            nearest_neighbors[j] = nearest_neighbors[j - 1]
                            j -= 1
                        nearest_neighbors[i] = (distance, vector[1])
                        break
        
        occurrences = dict()
        for neighbor in nearest_neighbors:
            if neighbor[1] not in nearest_neighbors.keys():
                occurrences[neighbor[1]] = 1
            else:
                occurrences[neighbor[1]] += 1
        
        class_count = [(ocurrence, class_) for class_, ocurrence in occurrences]
        return class_count[max(class_count)]
    
    def __classify_nearest_neighbor(self, document, distance_type):
        """
        Special case of k-nearest neighbors where k= = 1.
        
        Args:
            document (dict of str:float pairs): Bag of words containing information
                about which word from vocabulary is present in a document
            distance_type: Type of distance calculation. Allowed types are:
                * For numeric vectors *
                - braycurtis: Computes the Bray-Curtis distance between two arrays.
                - canberra: Computes the Canberra distance between two arrays.
                - chebyshev: 	Computes the Chebyshev distance.
                - cityblock: Computes the City Block (Manhattan) distance.
                - correlation: Computes the correlation distance between two arrays.
                - cosine: Computes the Cosine distance between arrays.
                - euclidean: Computes the Euclidean distance between two arrays.
                - sqeuclidean: Computes the squared Euclidean distance between two arrays.
                
                * For boolean vectors *
                - dice: Computes the Dice dissimilarity between two boolean arrays.
                - hamming: Computes the Hamming distance between two arrays.
                - jaccard: Computes the Jaccard-Needham dissimilarity between two boolean arrays.
                - kulsinski: Computes the Kulsinski dissimilarity between two boolean arrays.
                - rogerstanimoto: Computes the Rogers-Tanimoto dissimilarity between two boolean arrays.
                - russellrao: Computes the Russell-Rao dissimilarity between two boolean arrays.
                - sokalmichener: Computes the Sokal-Michener dissimilarity between two boolean arrays.
                - sokalsneath: Computes the Sokal-Sneath dissimilarity between two boolean arrays.
                - yule: Computes the Yule dissimilarity between two boolean arrays.
                
        Returns:
            str: Class label of document
        """
        min_distance = sys.float_info.max
        min_class = ""
        
        for vector in self.vectors:
            distance = self.distance(document, vector[0], distance_type)
            if distance < min_distance:
                min_distance = distance
                min_class = vector[1]
        
        return min_class