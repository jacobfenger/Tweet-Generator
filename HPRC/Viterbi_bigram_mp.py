# -*- coding: utf-8 -*-
import numpy as np
import random as rm
import multiprocessing as mp
from multiprocessing import Pool
import os
import re
import sys
import getopt
import time
import pprint
import gc
	
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
        if ele not in self.vocab and ele != "":
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
    ifs = open(tweet_file, "r", encoding='utf-8')
    tweet_lines = ifs.read().split("\n")
    ifs.close()
    ifs = open(tag_file, "r", encoding='utf-8')
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
		
    for i in range(len(self.vocab_counts)):
      if self.vocab_counts[i] == 0:
        print("Found no counts for ||", self.recoverWord(i), "||")
    
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
	


  # Reversed Viterbi
  def generate_tweet(self, sequence):
      time0 = time.time()
      n = len(sequence)
      trellis = np.zeros(( n + 1, self.vocab_size))
      bp = np.zeros(( n + 1, self.vocab_size))
      start = self.vocab_states["_START_"] #index of "start" symbol
  
      #Initialize
      trellis[0][start] = 1
	
      #"Recursion"
      for i in range(1, n + 1):
        tag = self.tags_states[sequence[i-1]] # -1 because including _START_ at front
        for word in range(self.vocab_size):
          for j in range(self.vocab_size):
            temp = trellis[i-1][j] * self.transitions[j][word]
            if(temp > trellis[i][word]):
              trellis[i][word] = temp
              bp[i][word] = j
          trellis[i][word] *= self.emissions[tag][word]

      w_max = 0
      vit_max = 0
      end = self.vocab_states["_END_"]
      # Find best final word in order to go backwards
      for word in range(self.vocab_size):
        if(trellis[n][word]*self.transitions[word][end] > vit_max):
          w_max = word
          vit_max = trellis[n][word]*self.transitions[word][end]
	  
      result = [None for k in range(n)]
      i = n
      w = w_max
      while i > 0:
        result[i-1] = self.recoverWord(w)
        w = int(bp[i][w])
        i -= 1
      #output.put(result)
      for word in result:
        print(word.encode('utf-8'), " ", end="")
      print()
      print("Process finished after ", (time0-time.time()))

def main():
  time_start = time.time()
  (options, args) = getopt.getopt(sys.argv[1:], '')
    
  if len(args) == 1: # data file (tweets)
    tweet_file = args[0]
    sequence_file = tweet_file.split(".")[0] + "_sequences.txt" #associated sequence file
    output_file = tweet_file.split(".")[0] + "_bigramOutput.txt"
	
    markov = Markov()
    markov.buildMarkov(tweet_file, sequence_file)
    print("\nBuild Time = ", (time.time() - time_start), " s")
	
    ifs = open(sequence_file, "r", encoding='utf-8')
    lines = ifs.read().split("\n")
    ifs.close()
    tag_sequences = [None for x in range(10)]#len(lines))]
    for i in range(10):#len(lines)):
      tag_sequences[i] = lines[i].strip().split(" ")
	  
    #output = mp.Queue()
    pool = Pool(processes = 8)
    pool.map(markov.generate_tweet, tag_sequences)
    #processes = []
    #for seq in tag_sequences:
    #  p = mp.Process(target=markov.generate_tweet, args=seq)
    #  processes.append(p)
    #  p.start()
      #count += 1
      #if count == 10:
      #  break

    #for p in processes:
    #  p.join()
	  
    # Get process results from the output queue
    #results = [output.get() for seq in tag_sequences]
    #for r in results:
    #  for word in r:
    #    print(word, " ", end="")
    #  print()
	  	
  time_end = time.time()
  print("\nTotal Time: ", (time_end-time_start), " s")
  
  
if __name__ == "__main__":
    main()
