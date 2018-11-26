# -*- coding: utf-8 -*-
import numpy as np
import random as rm
import os
import re
import sys
import getopt
import time
import pprint
import copy

class Markov:
  def __init__(self):
    self.vocab_states = {} # map vocab to index in transitions and  emissions
    self.tags_states = {}  # map tag to index in emissions
    self.transitions = []  # 2D array - prob that word follows a word
    self.emissions = []    # 2D array - prob of tag given word
    self.vocab_counts = [] # 1D array - counts of each word occurence
    self.vocab = []        # Words based on NLTK parsing
    self.tags = []         # NLTK tags
    self.vocab_size = 0
    self.tags_size = 0
    self.start_symbol = "_START_"
    self.end_symbol = "_END_"
	
  def initialize(self, tweet_lines, tag_lines):
    self.vocab = [self.start_symbol, self.end_symbol] #assumption that no tweet has these words
    self.vocab_states[self.start_symbol] = self.vocab_size
    self.vocab_states[self.end_symbol] = self.vocab_size + 1
    self.vocab_size += 2

    for line in tweet_lines:
      line = line.strip().split(" ")
      for ele in line:
        if ele not in self.vocab:
          self.vocab += [ele]
          self.vocab_states[ele] = self.vocab_size
          self.vocab_size += 1
    for line in tag_lines:
      line = line.strip().split(" ")
      for ele in line:
        if ele not in self.tags:
          self.tags += [ele]
          self.tags_states[ele] = self.tags_size
          self.tags_size += 1
    
    # Transitions: Word_i | Word_(i-1)
    self.transitions = np.zeros((self.vocab_size, self.vocab_size))
    # Emissions: Tag | Word
    self.emissions = np.zeros((self.tags_size, self.vocab_size))
    # Counts of words
    self.vocab_counts = np.zeros(self.vocab_size)
    pass
	
  def buildMarkov(self, tweet_file, tag_file):
    #Initialize Markov
    ifs = open(tweet_file, "r", encoding="ISO-8859-1")#, encoding='utf-8')
    tweet_lines = ifs.read().split("\n")
    ifs.close()
    ifs = open(tag_file, "r", encoding="ISO-8859-1")#, encoding='utf-8')
    tag_lines = ifs.read().split("\n")
    ifs.close()
    self.initialize(tweet_lines, tag_lines)
  
    # Fill in transitions and emissions
    for index in range(len(tweet_lines)):
      if tweet_lines[index] != "":
        tokens = tag_lines[index].strip().split(" ")
        words = tweet_lines[index].strip().split(" ")
        prev = self.vocab_states["_START_"] #index of "start" symbol
        self.vocab_counts[prev] += 1
        for i in range(len(words)): #looping through tags in tokens as well
          curr = self.vocab_states[words[i]]      # index of current word
          tag = self.tags_states[tokens[i]]   # index of current tag
          self.transitions[prev][curr] += 1
          self.emissions[tag][curr] += 1
          self.vocab_counts[curr] += 1
          prev = curr
        curr = self.vocab_states["_END_"] #index of "end" symbol
        self.transitions[prev][curr] += 1
        self.vocab_counts[curr] += 1
    
    for i in range(self.vocab_size):
      self.transitions[i] /= self.vocab_counts[i]
	
    for j in range(self.tags_size):
      for i in range(self.vocab_size):
        self.emissions[j][i] /= self.vocab_counts[i]
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
  
# Reversed Viterbi
def generate_viterbi(n, markov):
  trellis = np.zeros(( n + 1, markov.vocab_size))
  bp = np.zeros(( n + 1, markov.vocab_size))
  start = markov.vocab_states[markov.start_symbol] #index of "start" symbol

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
  
def generate_tweet(sequence, t, bp, markov):
  trellis = copy.deepcopy(t) # To prevent writing over the trellis object in main
  n = len(sequence)
  
  for i in range(1, n+1):
    tag = markov.tags_states[sequence[i-1]] # -1 because including _START_ at front
    for word in range(markov.vocab_size):
      trellis[i][word] *= markov.emissions[tag][word]
        
  w_max = ""
  vit_max = 0
  end = markov.vocab_states[markov.end_symbol]
    
  # Find best final word in order to go backwards
  for word in range(markov.vocab_size):
    if(trellis[n][word]*markov.transitions[word][end] > vit_max):
      w_max = word
      vit_max = trellis[n][word]*markov.transitions[word][end]
  
  if w_max == "":
    return ["NONE"]
 
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
  
  if len(args) == 2: # build/load, length
    #tweet_file = args[1]
    #filename = tweet_file.split(".")[0]
    #sequence_file = filename + "_sequences.txt"
    #length = int(args[2])
    #trellis_file = filename + "_sequence" + str(length) + "_trellis.txt"
    #bp_file = filename + "_sequence" + str(length) + "_bp.txt"
    #output_file = filename + "_generalizedOutput.txt"
    
    length = int(args[1])
    words_file = "words_length" + str(length) + ".txt"
    tags_file = "tags_length" + str(length) + ".txt"
    trellis_file = "trellis_length" + str(length) + ".txt"
    bp_file = "bp_length" + str(length) + ".txt"
    output_file = "generalOut_length" + str(length) + ".txt"
	
    markov = Markov()
    markov.buildMarkov(words_file, tags_file)
    print("\nBuild Time = ", (time.time() - time_start), " s")

    # Build markov, trellis, and bp and write to file
    if args[0] == "build":
      viterbi = generate_viterbi(length, markov)
    
      np.savetxt(trellis_file, viterbi.trellis)
      np.savetxt(bp_file, viterbi.bp)
	  
    # Load trellis and bp from files and generate tweet based on keyboard input
    if args[0] == "load":
      trellis = np.loadtxt(trellis_file).astype(float)
      bp = np.loadtxt(bp_file).astype(int)
      
      ifs = open(tags_file, "r", encoding="ISO-8859-1")#, encoding='utf-8')
      lines = ifs.read().split("\n")
      ifs.close()
      tag_sequences = [None for x in range(len(lines))]
      for i in range(len(lines)):
        tag_sequences[i] = lines[i].strip().split(" ")
	
      ofs = open(output_file, "w")
      for tag_sequence in tag_sequences:
        tag_sequence = tag_sequence[0:length] # truncate to n-length sequence
        tweet = generate_tweet(tag_sequence, trellis, bp, markov)
        for i in range(len(tweet)):
          ofs.write(markov.recoverWord(tweet[i]) + " ")
        ofs.write("\n")
      ofs.close()
	
  time_end = time.time()
  print("\nTotal Time: ", (time_end-time_start), " s")
  
  
if __name__ == "__main__":
    main()
