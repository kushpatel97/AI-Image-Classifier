# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    
  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    # might be useful in your code later...
    # this is a list of all features in the training set.
    self.features = list(set([f for datum in trainingData for f in datum.keys()]));
    
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """

    count = util.Counter()
    for i in trainingLabels:
      count[i] += 1
    count.normalize()
    self.count = count



    n = {}
    tot = {}
    for i in self.features:
        n[i] = {0: util.Counter(), 1: util.Counter()}
        tot[i] = util.Counter()


    for i, datum in enumerate(trainingData):
        y = trainingLabels[i]
        for j, v in datum.items():
            n[j][v][y] += 1.0
            tot[j][y] += 1.0

    bc = {}
    acc = None

    for k in kgrid or [0.0]:
        right = 0
        conditionals = {}
        for i in self.features:
            conditionals[i] = {0: util.Counter(), 1: util.Counter()}


        for i in self.features:
            for v in [0, 1]:
                for y in self.legalLabels:
                    conditionals[i][v][y] = (n[i][v][y] + k) / (tot[i][y] + k * 2)


        self.conditionals = conditionals
        guess = self.classify(validationData)
        for i, g in enumerate(guess):
            right += (validationLabels[i] == g and 1.0 or 0.0)
        accuracy = right / len(guess)


        if accuracy > acc or acc is None:
            acc = accuracy
            bc = conditionals
            self.k = k

    self.conditionals = bc


        
  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses
      
  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    logJoint = util.Counter()

    for i in self.legalLabels:
        logJoint[i] = math.log(self.count[i])
        for j in self.conditionals:
            prob = self.conditionals[j][datum[j]][i]
            logJoint[i] += (prob and math.log(prob) or 0.0)

    return logJoint

  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2) 
    
    Note: you may find 'self.features' a useful way to loop through all possible features
    """
    featuresOdds = []


    for i in self.features:
        top = self.conditionals[i][1][label1]
        bottom = self.conditionals[i][1][label2]
        ratio = top / bottom
        featuresOdds.append((i, ratio))

    featuresOdds = [f for f, odds in sorted(featuresOdds, key=lambda t: -t[1])[:100]]

    return featuresOdds
    

    
      
