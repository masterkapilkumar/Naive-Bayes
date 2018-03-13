from collections import defaultdict
from copy import copy
import pickle
import operator
import string
import random
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
        
    def storeTrainingData(self, filename):
        f = open(filename, 'wb')
        pickle.dump(self.classes, f)
        pickle.dump(self.class_probs, f)
        pickle.dump(self.vocab, f)
        pickle.dump(self.word_probs, f)
        f.close()
    
    def loadTrainingData(self, filename):
        f = open(filename, 'rb')
        self.classes = pickle.load(f)
        self.class_probs = pickle.load(f)
        self.vocab = pickle.load(f)
        self.word_probs = pickle.load(f)
        f.close()
    
    #function to read file having input text
    def ReadTextFile(self, file_name):
        fin = open(file_name, 'r', encoding='ascii', errors='ignore')
        translator = str.maketrans('', '', string.punctuation)
        data = [inp.rstrip().lower().translate(translator) for inp in fin]
        fin.close()
        return data
    
    #function to read file having corresponding classes of input text
    def ReadClassFile(self, file_name):
        fin = open(file_name, 'r')
        data = [inp.rstrip() for inp in fin]
        fin.close()
        return data
    
    #function to compute vocabulary from training dataset
    # def customFunc2(self):
        # return {c:0 for c in self.classes}
        
    def CalculateVocab(self):
        self.words_per_class = {c:0 for c in self.classes}
        # self.vocab_per_class = defaultdict(self.customFunc2)
        for c,ex in zip(self.train_y, self.train_x):
            words = ex.split()
            bigrams = [' '.join(b) for b in zip(words[:-1], words[1:])]
            if self.use_bigrams:
                words = bigrams
            for word in words:
                self.vocab[word] = 1
                # self.vocab_per_class[word][c] += 1
            self.words_per_class[c]+=len(words)
        
        # for word in self.vocab_per_class:
            # word_class_count=0
            # total_count = sum(self.vocab_per_class[word].values())
            # flag = True
            # for v in self.vocab_per_class[word].values():
                # if v!=0:
                    # word_class_count+=1
                # if(v<total_count*0.07):
                    # flag=False
                    # break
            # if word_class_count>=7 and flag:
                # self.discarded_words[word]=1
        # print(len(self.discarded_words))
    
    #function to calculate the fraction of text belonging to each class
    def CalculateClassesProb(self):
        self.class_probs = {}
        for c in self.classes:
            self.class_probs[c] = self.train_y.count(c)/self.m
    
    #function to calculate,for each word, the fraction of each class in which that word appear (likelihood table)
    def customFunc(self):
        return {c:1.0/(self.words_per_class[c]+len(self.vocab)) for c in self.classes}
        
    def CalculateWordsProb(self):
        vocab_size = len(self.vocab)
        self.word_probs = defaultdict(self.customFunc)   #Laplace smoothing in numerator
        for c,ex in zip(self.train_y,self.train_x):
            c_words = self.words_per_class[c]+vocab_size              #Laplace smoothing in denominator
            prob = 1.0/c_words
            words = ex.split()
            bigrams = [' '.join(b) for b in zip(words[:-1], words[1:])]
            if self.use_bigrams:
                words = bigrams
            for word in words:
                # if self.discarded_words[word]==0:
                    self.word_probs[word][c] += prob
        #calculate logarithm of probabilities to avoid underflow
        for word in self.vocab:
            for c in self.classes:
                self.word_probs[word][c] = math.log(self.word_probs[word][c])
    
    def Testing(self, data_type="test"):
        if(self.predict_only):
            data_x = self.test_x
        elif(data_type=='test'):
            data_x = self.test_x
            data_y = self.test_y
        elif(data_type=='train'):
            data_x = self.train_x
            data_y = self.train_y
        
        correct_count = 0            #variable to count correct predictions
        
        self.confusion_matrix = {c:{c1:0 for c1 in self.classes} for c in self.classes}
        # a=0
        #calculate argmax [ p(x|y)*p(y) ]
        post_probs_bck = {c:math.log(self.class_probs[c]) for c in self.classes}
        if(self.predict_only):
            f = open(self.out_filename, 'w')
            for ex in data_x:
                post_probs = copy(post_probs_bck)
                words = ex.split()
                bigrams = [' '.join(b) for b in zip(words[:-1], words[1:])]
                if self.use_bigrams:
                    words = bigrams
                for word in words:
                    # if self.discarded_words[word]==0:
                        if(self.vocab[word]==1):
                            for c in self.classes:
                                post_probs[c] += self.word_probs[word][c]
                        else:
                            for c in self.classes:
                                post_probs[c] += math.log(self.word_probs[word][c])
                
                predicted_class = max(post_probs.items(), key=operator.itemgetter(1))[0]
                f.write(predicted_class)
                f.write('\n')
            f.close()
        else:
            for actual_class,ex in zip(data_y,data_x):
                post_probs = copy(post_probs_bck)
                words = ex.split()
                bigrams = [' '.join(b) for b in zip(words[:-1], words[1:])]
                if self.use_bigrams:
                    words = bigrams
                for word in words:
                    # if self.discarded_words[word]==0:
                        if(self.vocab[word]==1):
                            for c in self.classes:
                                post_probs[c] += self.word_probs[word][c]
                        else:
                            for c in self.classes:
                                post_probs[c] += math.log(self.word_probs[word][c])
                
                predicted_class = max(post_probs.items(), key=operator.itemgetter(1))[0]
                
                self.confusion_matrix[actual_class][predicted_class] += 1
                if(predicted_class == actual_class):
                    correct_count += 1
            
            accuracy = 1.0*correct_count/len(data_x)
            
            return accuracy

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
        
    print("Loading data complete...")
    print("Time taken: %.2fs\n"%(calculate_time_elapsed()))
    
    if predict_only:
        nb.Testing()
        print("Prediction complete...")
        print("Time taken: %.2fs\n"%(calculate_time_elapsed()))
    else:
        nb.CalculateVocab()
        print("Vocabulary computed...")
        print("Vocabulary size: %d" %len(nb.vocab))
        print("Time taken: %.2fs\n"%(calculate_time_elapsed()))
        
        #training phase
        nb.CalculateClassesProb()
        nb.CalculateWordsProb()
        # print_word_probs()
        print("Number of classes: %d" % nb.num_classes)
        print("Parameters calculation complete...")
        print("Time taken: %.2fs\n"%(calculate_time_elapsed()))
        
        nb.storeTrainingData(nb.model_name)
        print("Saved training data...")
        print("Time taken: %.2fs\n"%(calculate_time_elapsed()))
    
        #testing phase
        accuracy = nb.Testing('train')
        print("\nTrain data accuracy: %.2f%%"%(accuracy*100))
        print("Time taken: %.2fs\n"%(calculate_time_elapsed()))
        accuracy = nb.Testing('test')
        print("Test data accuracy: %.2f%%"%(accuracy*100))
        print("Time taken: %.2fs\n"%(calculate_time_elapsed()))
        print("Testing complete...")
        
        accuracy = random_prediction(nb.test_y, nb.classes)
        print("\nRANDOM GUESSING:")
        print("Test data accuracy: %.2f%%"%(accuracy*100))
        print("Time taken: %.2fs\n"%(calculate_time_elapsed()))
        
        accuracy = majority_prediction(nb.train_y, nb.classes)
        print("\nMAJORITY PREDICTION:")
        print("Test data accuracy: %.2f%%"%(accuracy*100))
        print("Time taken: %.2fs\n"%(calculate_time_elapsed()))
    
        #confusion matrix
        print("\nCONFUSION MATRIX:")
        s = [['']*(nb.num_classes+1) for i in range(nb.num_classes+1)]
        s[0] = ['']+nb.classes
        for i in range(1,nb.num_classes+1):
            s[i][0]=nb.classes[i-1]
        for i,c0 in enumerate(nb.classes):
            for j,c1 in enumerate(nb.classes):
                s[i+1][j+1] = str(nb.confusion_matrix[c1][c0])
        col_lengths = [max(map(len, col)) for col in zip(*s)]
        fmt = '  '.join('{{:{}}}'.format(x) for x in col_lengths)
        table = [fmt.format(*row) for row in s]
        print('\n'.join(table))