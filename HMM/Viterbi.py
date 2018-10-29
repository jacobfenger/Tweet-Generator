# -*- coding: utf-8 -*-
import numpy as np
import random as rm
import nltk
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
import os
import re
import sys
import getopt
import time
import pprint
import csv

# Normal HMM:   argmax P(t | w) = argmax P(t)*P(w | t)
# Reversed HMM: argmax P(w | t) = argmax P(w)*P(t | w)
	
class Markov:
  def __init__(self):
    self.vocab_states = {} # map vocab to index in transitions and  emissions
    self.tags_states = {}  # map tag to index in emissions
    self.transitions = []  # 2D array - prob that word follows a word
    self.emissions = []    # 2D array - prob of tag given word
    self.vocab_counts = [] # 1D array - counts of each word occurence
    self.vocab_size = 0
    self.vocab = []
	# NLTK tags
    self.tags = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD",
	             "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR",
                 "RBS", "RP", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
                 "WDT", "WP", "WP$", "WRB"]
    self.tags_size = 0
	
  def initialize(self, filename):
    self.vocab = ["_START_", "_END_"] #assumption that no tweet has these words
    ifs = open(filename, "r")
    lines = ifs.read()
    lines = lines.split("\n")
  
    for line in lines:
      tokens = getTags(line)
      #line = line.split(" ")
      words = getWords(line)
      for ele in words:
        if ele not in self.vocab and ele != '':
          self.vocab += [ele]
      for tag in tokens:
        if tag not in self.tags:
          self.tags += [tag]
		  
    for w in self.vocab:
      self.vocab_states[w] = self.vocab_size
      self.vocab_size += 1
    for t in self.tags:
      self.tags_states[t] = self.tags_size
      self.tags_size += 1
  
    # Transitions: Word_i | Word_(i-1)
    self.transitions = np.zeros((self.vocab_size, self.vocab_size))
    # Emissions: Tag | Word
    self.emissions = np.zeros((self.tags_size, self.vocab_size))
    # Counts of words
    self.vocab_counts = np.zeros(self.vocab_size)
    pass
	
  def recoverWord(self, index):
    for w in self.vocab:
      if(self.vocab_states[w] == index):
        return w
    return None

  
def getTags(line):
  raw_tokens = nltk.word_tokenize(line)
  raw_tokens = nltk.pos_tag(raw_tokens)
  tokens = []
  for pair in raw_tokens:
    tokens.append(pair[1])
  return tokens
  
def getWords(line):
  raw_tokens = nltk.word_tokenize(line)
  raw_tokens = nltk.pos_tag(raw_tokens)
  tokens = []
  for pair in raw_tokens:
    tokens.append(pair[0])
  return tokens
  
def buildMarkov(markov, filename):
  #Initialize Markov
  markov.initialize(filename)
  
  # Fill in transitions and emissions
  ifs = open(filename, "r")
  lines = ifs.read()
  lines = lines.split("\n")
  ifs.close()
  
  for line in lines:
    if line != "":
      tokens = getTags(line)
      words = getWords(line)
      prev = markov.vocab_states["_START_"] #index of "start" symbol
      markov.vocab_counts[prev] += 1
      for i in range(len(words)): #looping through tags in tokens as well
        curr = markov.vocab_states[words[i]]      # index of current word
        tag = markov.tags_states[tokens[i]]   # index of current tag
        markov.transitions[prev][curr] += 1
        markov.emissions[tag][curr] += 1
        markov.vocab_counts[curr] += 1
        markov.tags_counts[tag] += 1
        prev = curr
      curr = markov.vocab_states["_END_"] #index of "end" symbol
      markov.transitions[prev][curr] += 1
      markov.vocab_counts[curr] += 1
	

  for i in range(markov.vocab_size):
    markov.transitions[i] /= markov.vocab_counts[i]
	
  for j in range(markov.tags_size):
    for i in range(markov.vocab_size):
      markov.emissions[j][i] /= markov.vocab_counts[i]


  return markov
  
    
# Reversed Viterbi
def generate_tweet(sequence, markov):
  n = len(sequence)
  trellis = np.zeros(( n , markov.vocab_size))
  bp = [[None for j in range(markov.vocab_size)] for i in range( n )]
  start = markov.vocab_states["_START_"] #index of "start" symbol
  first_tag = markov.tags_states[sequence[0]]
  
  #Initialize
  for word in range(markov.vocab_size):
    trellis[0][word] = markov.transitions[start][word] * markov.emissions[first_tag][word]
	
  #"Recursion"
  for i in range(1, n):
    tag = markov.tags_states[sequence[i]]
    for word in range(markov.vocab_size):
      for j in range(markov.vocab_size):
        temp = trellis[i-1][j] * markov.transitions[j][word]
        if(temp > trellis[i][word]):
          trellis[i][word] = temp
          bp[i][word] = j
      trellis[i][word] *= markov.emissions[tag][word]

  w_max = ""
  vit_max = 0
  end = markov.vocab_states["_END_"]
  # Find best final word in order to go backwards
  for word in range(markov.vocab_size):
    if(trellis[n-1][word]*markov.transitions[word][end] > vit_max):
      w_max = word
      vit_max = trellis[n-1][word]*markov.transitions[word][end]
	  
  result = [None for k in range(n)]
  i = n-1
  w = w_max
  while i >= 0:
    result[i] = w
    w = bp[i][w]
    i -= 1
	  
  return result

def main():
  time_start = time.time()
  tweet_file = "processed1000.txt"
  
  print("DEBUGGING - all files are hardcoded")
  
  # Random phrases
  #debug_tweet = "where is my cat"
  #debug_tweet = "looking forward to the weekend"
  #debug_tweet = "what am i doing here?"
  
  # Actual tweets from corpus
  #debug_tweet = "need a hug"   
  debug_tweet = "damn... i don't have any chalk! my chalkboard is useless"
  #debug_tweet = "michigan state you make me sad"
  #debug_tweet = "i don't have a garage. but you can park in my driveway!"
  
  tag_sequence = getTags(debug_tweet)
  #print(tag_sequence)
  
  markov = Markov()
  markov = buildMarkov(markov, tweet_file)
  print("\nBuild Time = ", (time.time() - time_start), " s")
  tweet = generate_tweet(tag_sequence, markov)
  for word in tweet:
    print(markov.recoverWord(word), " ", end="")
  print()
	
  time_end = time.time()
  print("\nTotal Time: ", (time_end-time_start), " s")
  
  
if __name__ == "__main__":
    main()
