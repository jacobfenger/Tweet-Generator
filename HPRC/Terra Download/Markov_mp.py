# -*- coding: utf-8 -*-
import numpy as np
import random as rm
import multiprocessing as mp
import os
import re
import sys
import getopt
import time
import pprint
	
class Markov:
  def __init__(self):
    self.vocab_states = {} # map vocab to index in transitions and  emissions
    self.transitions = []  # 2D array - prob that word follows a word
    self.vocab_counts = [] # 1D array - counts of each word occurence
    self.vocab = []        # Words based on NLTK parsing
    self.vocab_size = 0
    self.start_symbol = "_START_"
    self.end_symbol = "_END_"
    self.length = 0 # FOR MP
	
  def initialize(self, tweet_lines):
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
    
    # Transitions: Word_i | Word_(i-1)
    self.transitions = np.zeros((self.vocab_size, self.vocab_size))
    # Counts of words
    self.vocab_counts = np.zeros(self.vocab_size)
    pass
	
  def buildMarkov(self, tweet_file, num_tweets, length):
    #Initialize Markov
    ifs = open(tweet_file, "r", encoding="ISO-8859-1")#, encoding='utf-8')
    tweet_lines = [None for x in range(num_tweets)]
    for i in range(num_tweets):
      tweet_lines[i] = ifs.readline()
    ifs.close()
    self.initialize(tweet_lines)
    
    self.length = length # FOR MP
  
    # Fill in transitions and emissions
    for index in range(len(tweet_lines)):
      if tweet_lines[index] != "":
        words = tweet_lines[index].strip().split(" ")
        prev = self.vocab_states["_START_"] #index of "start" symbol
        self.vocab_counts[prev] += 1
        for i in range(len(words)): #looping through tags in tokens as well
          curr = self.vocab_states[words[i]]      # index of current word
          self.transitions[prev][curr] += 1
          self.vocab_counts[curr] += 1
          prev = curr
        curr = self.vocab_states["_END_"] #index of "end" symbol
        self.transitions[prev][curr] += 1
        self.vocab_counts[curr] += 1
	  
    # +1 Smoothing
    self.transitions += 1
    for i in range(self.vocab_size):
      self.transitions[i] /= (self.vocab_counts[i] + self.vocab_size)
	
    pass
	
  def recoverWord(self, index):
    for w in self.vocab:
      if(self.vocab_states[w] == index):
        return w
    return None

  # Markov Chain
  def generate_tweet(self, first):
    result = [None for k in range(self.length)]
    result[0] = first
    curr = first
    for i in range(1, self.length):
      result[i] = np.random.choice(self.vocab_size, replace=True, p=self.transitions[curr])
      curr = result[i]
    tweet = ""
    for num in result:
      tweet += self.recoverWord(num) + " "
    return tweet
    
    
def main():
  time_start = time.time()
  (options, args) = getopt.getopt(sys.argv[1:], '')
    
  if len(args) == 2: # sequence length, num tweets
    length = int(args[0])
    num_tweets = int(args[1])
    words_file = "words_length" + str(length) + ".txt"
    output_file = "MarkovOut_length" + str(length) + "_size" + str(num_tweets) + ".txt"
	
    markov = Markov()
    markov.buildMarkov(words_file, num_tweets, length)
    print("\nBuild Time = ", (time.time() - time_start), " s")
    
    num_output = 1000
	
    #ifs = open(tags_file, "r", encoding="ISO-8859-1")# encoding='utf-8')
    #tag_sequences = [None for x in range(num_output)]
    #for i in range(num_output):
    #  tag_sequences[i] = ifs.readline().strip().split(" ")
    #ifs.close()
	
	# Randomly choose num_output words to be the first word
    word_sequences = np.random.randint(low=2, high=markov.vocab_size, size=num_output)
    
    with open(output_file, "w", encoding='utf-8') as ofs:
      for ele in word_sequences:
        ofs.write(str(markov.generate_tweet(ele)) + "\n")
	  
    #pool = mp.Pool(processes = 8)
    #with open(output_file, "w", encoding='utf-8') as ofs:
    #  for result in pool.imap(markov.generate_tweet, word_sequences):
    #    ofs.write(str(result) + "\n")
    	  	
  time_end = time.time()
  print("\nTotal Time: ", (time_end-time_start), " s")
  
  
if __name__ == "__main__":
    main()
