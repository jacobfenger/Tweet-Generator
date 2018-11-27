# -*- coding: utf-8 -*-
import numpy as np
import random as rm
import os
import re
import sys
import getopt
import time
import pprint
import multiprocessing as mp

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
    
    self.length = 0
    self.trellis = []
    self.bp = []
	
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
	
  def buildMarkov(self, tweet_file, tag_file, num_tweets):
    #Initialize Markov
    ifs = open(tweet_file, "r", encoding="ISO-8859-1")#, encoding='utf-8')
    tweet_lines = [None for x in range(num_tweets)]
    for i in range(num_tweets):
      tweet_lines[i] = ifs.readline()
    #tweet_lines = ifs.read().split("\n")
    ifs.close()
    ifs = open(tag_file, "r", encoding="ISO-8859-1")#, encoding='utf-8')
    tag_lines = [None for x in range(num_tweets)]
    for i in range(num_tweets):
      tag_lines[i] = ifs.readline()
    #tag_lines = ifs.read().split("\n")
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
    
  # Reversed Viterbi
  def generate_viterbi(self, n):
    trellis = np.zeros(( n + 1, self.vocab_size))
    bp = np.zeros(( n + 1, self.vocab_size))
    start = self.vocab_states[self.start_symbol] #index of "start" symbol

    #Initalize
    trellis[0][start] = 1
	
    #"Recursion"
    for i in range(1, n + 1):
      for word in range(self.vocab_size):
        for j in range(self.vocab_size):
          temp = trellis[i-1][j] * self.transitions[j][word]
          if(temp > trellis[i][word]):
            trellis[i][word] = temp
            bp[i][word] = j
		  
    self.trellis = trellis
    self.bp = bp
    pass
    
  def generate_tweet(self, final_tag):
    n = self.length
  
    for word in range(self.vocab_size):
      self.trellis[n][word] *= self.emissions[self.tags_states[final_tag]][word]
        
    w_max = ""
    vit_max = 0
    end = self.vocab_states[self.end_symbol]
    
    # Find best final word in order to go backwards
    for word in range(self.vocab_size):
      if(self.trellis[n][word]*self.transitions[word][end] > vit_max):
        w_max = word
        vit_max = self.trellis[n][word]*self.transitions[word][end]
  
    if w_max == "":
      return ["NONE"]
 
    result = [None for k in range(n)]
    i = n
    w = w_max
    while i > 0:
      result[i-1] = self.recoverWord(w)
      w = self.bp[i][w]
      i -= 1
    tweet = ""
    for word in result:
      tweet += word + " "
    return tweet


def main():
  time_start = time.time()
  (options, args) = getopt.getopt(sys.argv[1:], '')
  
  if len(args) == 3: # build/load, sequence length, num tweets
    length = int(args[1])
    num_tweets = int(args[2])
    words_file = "words_length" + str(length) + ".txt"
    tags_file = "tags_length" + str(length) + ".txt"
    trellis_file = "trellis_mp_length" + str(length) + "_size" + str(num_tweets) + ".txt"
    bp_file = "bp_mp_length" + str(length) + "_size" + str(num_tweets) + ".txt"
    output_file = "generalOut_mp_length" + str(length) + "_size" + str(num_tweets) + ".txt"
	
    markov = Markov()
    markov.buildMarkov(words_file, tags_file, num_tweets)
    markov.length = length
    print("\nBuild Time = ", (time.time() - time_start), " s")

    # Build markov, trellis, and bp and write to file
    if args[0] == "build":
      markov.generate_viterbi(length)
    
      np.savetxt(trellis_file, markov.trellis)
      np.savetxt(bp_file, markov.bp)
	  
    # Load trellis and bp from files and generate tweet based on keyboard input
    if args[0] == "load":
      markov.trellis = np.loadtxt(trellis_file).astype(float)
      markov.bp = np.loadtxt(bp_file).astype(int)
      
      print("Object = ", sys.getsizeof(markov))
      print("Transitions = ", sys.getsizeof(markov.transitions))
      print("Emissions = ", sys.getsizeof(markov.emissions))
      print("Trellis = ", sys.getsizeof(markov.trellis))
      print("BP = ", sys.getsizeof(markov.bp))
      print("Function = ", sys.getsizeof(markov.generate_tweet))
      
      pool = mp.Pool(processes = 8)
      with open(output_file, "w", encoding='utf-8') as ofs:
        for result in pool.imap(markov.generate_tweet, markov.tags):
          ofs.write(str(result) + "\n")
	
  time_end = time.time()
  print("\nTotal Time: ", (time_end-time_start), " s")
  
  
if __name__ == "__main__":
    main()
