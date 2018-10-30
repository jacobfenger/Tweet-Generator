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
    self.transitions = []  # 3D array - q(yi | yi-2, yi-1), weighted with lamdas
    self.transitions3 = []  # 3D array - prob that word follows a word pair [yi-2][yi-1][yi]
    self.transitions2 = []  # 2D array - prob that word follows a word pair [yi-1][yi]
    self.counts2 = []       # 2D array - counts of [yi-2][yi-1]
    self.counts1 = []       # 1D array - counts of [yi-1]
    self.vocab_counts = []  # 1D array - counts of each word occurence
    self.word_count = 0
    self.lambdas = [0.33, 0.33, 0.33]
    self.emissions = []    # 2D array - prob of tag given word
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
  
    # Transitions: Word_i | Word_(i-2) Word_(i-1)
	# Indices given by [w-2][w-1][w0]
    self.transitions = np.zeros((self.vocab_size, self.vocab_size, self.vocab_size))
    self.transitions3 = np.zeros((self.vocab_size, self.vocab_size, self.vocab_size))
    self.transitions2 = np.zeros((self.vocab_size, self.vocab_size))
	
	# Counts of two previous words
    self.counts2 = np.zeros((self.vocab_size, self.vocab_size))
    # Counts of previous word
    self.counts1 = np.zeros(self.vocab_size)
    # Counts of each word
    self.vocab_counts = np.zeros(self.vocab_size)
	
    # Emissions: Tag | Word
    self.emissions = np.zeros((self.tags_size, self.vocab_size))

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
      markov.word_count += len(words)
      prev1 = markov.vocab_states["_START_"] #index of "start" symbol
      prev2 = markov.vocab_states["_START_"]
      markov.vocab_counts[prev1] += 1
      markov.vocab_counts[prev2] += 1
      for i in range(len(words)): #looping through tags in tokens as well
        curr = markov.vocab_states[words[i]]      # index of current word
        tag = markov.tags_states[tokens[i]]   # index of current tag
        markov.transitions3[prev1][prev2][curr] += 1
        markov.transitions2[prev2][curr] += 1
        markov.emissions[tag][curr] += 1
        markov.counts2[prev1][prev2] += 1
        markov.counts1[prev2] += 1
        markov.vocab_counts[curr] += 1
        prev1 = prev2
        prev2 = curr
      curr = markov.vocab_states["_END_"] #index of "end" symbol
      markov.transitions3[prev1][prev2][curr] += 1
      markov.transitions2[prev2][curr] += 1
      markov.counts2[prev1][prev2] += 1
      markov.counts1[prev2] += 1
      markov.vocab_counts[curr] += 1
	
  for i in range(markov.vocab_size):
    #markov.transitions[i] /= markov.vocab_counts[i]
    #markov.transitions2[i] /= markov.counts1[i]
    for j in range(markov.vocab_size):
      #markov.transitions3[i][j] /= markov.counts2[i][j]
      markov.transitions[i][j] = (markov.lambdas[0]*markov.transitions3[i][j]/markov.counts2[i][j] + 
                                 markov.lambdas[1]*markov.transitions2[i]/markov.counts1[j] +
                                 markov.lambdas[2]*markov.vocab_counts[i]/markov.word_count)
  
  for j in range(markov.tags_size):
    for i in range(markov.vocab_size):
      markov.emissions[j][i] /= markov.vocab_counts[i]

  return markov
  
    
# Reversed Viterbi
def generate_tweet(sequence, markov):
  n = len(sequence)
  trellis = np.zeros(( n + 1 , markov.vocab_size, markov.vocab_size))
  bp = [[[None for k in range(markov.vocab_size)] for j in range(markov.vocab_size)] for i in range( n + 1 )]
  start = markov.vocab_states["_START_"] #index of "start" symbol
  
  #Initialize
  trellis[0][start][start] = 1
	
  #"Recursion"
  for k in range(1, n + 1):
    tag = markov.tags_states[sequence[k-1]]
    # Find max for trellis[k][u][v]
    for u in range(markov.vocab_size):
      for v in range(markov.vocab_size):
        for w in range(markov.vocab_size):
          temp = trellis[k-1][w][u] * markov.transitions[w][u][v]
          if(temp > trellis[k][u][v]):
            trellis[k][u][v] = temp
            bp[k][u][v] = w
        trellis[k][u][v] *= markov.emissions[tag][v]
	  
  u_max = 0
  v_max = 0
  vit_max = 0
  end = markov.vocab_states["_END_"]
  for u in range(markov.vocab_size):
    for v in range(markov.vocab_size):
      if(trellis[n][u][v]*markov.transitions[u][v][end] > vit_max):
        u_max = u
        v_max = v
        vit_max = trellis[n][u][v]*markov.transitions3[u][v][end]
		
  result = [None for k in range(n)]
  result[n-1] = v_max
  result[n-2] = u_max
  w1 = u_max
  w2 = v_max
  k = n-2
  while k > 0:
    result[k-1] = bp[k+2][w1][w2]
    w2 = w1
    w1 = result[k]
    k -= 1
	
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
  debug_tweet = "need a hug"   
  
  #debug_tweet = "damn... i don't have any chalk! my chalkboard is useless"
  #debug_tweet = "michigan state you make me sad"
  #debug_tweet = "i don't have a garage. but you can park in my driveway!"
  
  tag_sequence = getTags(debug_tweet)
  #print(tag_sequence)
  
  markov = Markov()
  markov = buildMarkov(markov, tweet_file)
  print("\nBuild Time = ", (time.time() - time_start), " s")
  print("vocab size = ", markov.vocab_size)
  tweet = generate_tweet(tag_sequence, markov)
  for i in range(len(tweet)):
    print(markov.recoverWord(tweet[i]), sep="", end="")
    if i < len(tweet)-1:
      if tag_sequence[i+1] != ":" and tag_sequence[i+1] != "." and markov.recoverWord(tweet[i+1]) != "n't":
        print(" ", sep="", end="")
  print()
	
  time_end = time.time()
  print("\nTotal Time: ", (time_end-time_start), " s")
  
  
if __name__ == "__main__":
    main()
