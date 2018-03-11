from collections import defaultdict
from copy import copy
import math
import sys

from timeit import default_timer as timer

class NaiveBayes:
    
    def __init__(self, train_inp_file="", train_out_file="", test_inp_file="", test_out_file="", predict_only=False, model_name=""):
        self.predict_only = predict_only
        self.model_name = model_name
        
        if predict_only:
            self.loadTrainingData(model_name)
            self.out_filename = test_out_file
        else:
            self.train_x = self.ReadTextFile(train_inp_file)
            self.train_y = self.ReadClassFile(train_out_file)
            self.test_y = self.ReadClassFile(test_out_file)
            self.m = len(self.train_x)                  #number of examples
            self.classes = sorted(list(set(self.train_y)))
            self.num_classes = len(self.classes)   #number of classes
            self.vocab = defaultdict(int)
        
        self.test_x = self.ReadTextFile(test_inp_file)
        self.use_bigrams = True
        # self.discarded_words = defaultdict(int)
        
    

def calculate_time_elapsed():
    global start
    time_elapsed = timer()-start
    start = timer()
    return time_elapsed

def random_prediction(test_y, classes):
    correct_count = 0
    for c in test_y:
        if(random.choice(classes) == c):
            correct_count += 1
    return correct_count/len(test_y)


def majority_prediction(test_y, classes):
    majority_class = max(classes, key=test_y.count)
    return test_y.count(majority_class)/len(test_y)

if __name__=='__main__':
    
    start = timer()
    
    #create a Naive Bayes Classifier object
    predict_only = False
    if(len(sys.argv)==1):
        nb = NaiveBayes("imdb_train_text.txt","imdb_train_labels.txt","imdb_test_text.txt","imdb_test_labels.txt", model_name="model0")
    elif(len(sys.argv)==5 and sys.argv[1]=="-p"):
        predict_only = True
        nb = NaiveBayes(model_name=sys.argv[4], test_inp_file=sys.argv[2], test_out_file=sys.argv[3], predict_only=True)
    elif(len(sys.argv)==6):
        nb = NaiveBayes(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4], model_name=sys.argv[5])
    else:
        print("Invalid arguments.")
        print("Usage: NaiveBayes.py [-p] [training_file] [training_labels] testing_file testing_labels model_name")
        print("options:")
        print("-p: Prediction only.")
        sys.exit(1)
        
    