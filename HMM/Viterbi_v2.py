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
    self.vocab_states = {} # map vocab to index in transitions and emissions
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
	
class Viterbi:
  def __init__(self):
    self.trellis = [] #2D array
    self.bp = []      #2D array
  
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
def generate_viterbi(n, markov):
  trellis = np.zeros(( n + 1, markov.vocab_size))
  #bp = [[None for j in range(markov.vocab_size)] for i in range( n + 1 )]
  bp = np.zeros(( n + 1, markov.vocab_size))
  start = markov.vocab_states["_START_"] #index of "start" symbol

  #Initalize
  trellis[0][start] = 1
	
  #"Recursion"
  for i in range(1, n + 1):
    for word in range(markov.vocab_size):
      for j in range(markov.vocab_size):
        temp = trellis[i-1][j] * markov.transitions[j][word]
        if(temp > trellis[i][word]):
          trellis[i][word] = temp
          bp[i][word] = j
		  
  viterbi = Viterbi()
  viterbi.trellis = trellis
  viterbi.bp = bp

  return viterbi
  
def generate_tweet(sequence, trellis, bp, markov):
  n = len(sequence)
  start = markov.vocab_states["_START_"] #index of "start" symbol
  
  for i in range(1, n+1):
    tag = markov.tags_states[sequence[i-1]] # -1 because including _START_ at front
    for word in range(markov.vocab_size):
      trellis[i][word] *= markov.emissions[tag][word]
  
  w_max = ""
  vit_max = 0
  end = markov.vocab_states["_END_"]
  # Find best final word in order to go backwards
  for word in range(markov.vocab_size):
    if(trellis[n][word]*markov.transitions[word][end] > vit_max):
      w_max = word
      vit_max = trellis[n][word]*markov.transitions[word][end]
	
  w_max = ""
  vit_max = 0
  end = markov.vocab_states["_END_"]
  # Find best final word in order to go backwards
  for word in range(markov.vocab_size):
    if(trellis[n][word]*markov.transitions[word][end] > vit_max):
      w_max = word
      vit_max = trellis[n][word]*markov.transitions[word][end]
	  
  result = [None for k in range(n)]
  i = n
  w = w_max
  while i > 0:
    result[i-1] = w
    w = bp[i][w]
    i -= 1
	
  return result

def main():
  time_start = time.time()
  (options, args) = getopt.getopt(sys.argv[1:], '')
  
  # File naming convention: [data_file]_sequence[length]
  # ex. processed1000_sequence5_trellis generates tweets of length 5 using tweets from processed1000.txt
  
  # Run command is [build/load] [data_file] [sequence length]
  
  if len(args) == 3:
    tweet_file = args[1]
    length = int(args[2])
    filename = tweet_file.split(".")[0]
    trellis_file = filename + "_sequence" + str(length) + "_trellis.txt"
    bp_file = filename + "_sequence" + str(length) + "_bp.txt"
	
    markov = Markov()
    markov = buildMarkov(markov, tweet_file)
	# Rebuilding Markov is better than reading from file
    # -->Building Markov for 5000 tweets takes around 30 s
	# -->Loading Markov for 5000 tweets takes 160 s
	# ---> Transmissions File is 2.1G; it took 1422 s to generate

    # Build markov, trellis, and bp and write to file
    if args[0] == "build":
      viterbi = generate_viterbi(length, markov)
    
      np.savetxt(trellis_file, viterbi.trellis)
      np.savetxt(bp_file, viterbi.bp)
	  
    # Load trellis and bp from files and generate tweet based on keyboard input
    if args[0] == "load":
      trellis = np.loadtxt(trellis_file).astype(float)
      bp = np.loadtxt(bp_file).astype(int)
      
      word_sequence = input("Enter a word sequence: ")
      tag_sequence = getTags(word_sequence)
      tweet = generate_tweet(tag_sequence, trellis, bp, markov)
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
