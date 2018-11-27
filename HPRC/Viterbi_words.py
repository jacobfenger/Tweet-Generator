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
import itertools
	
class Markov:
  def __init__(self, k):
    self.transitions = []  # 2D array - prob that sequence of chars follows a sequence of chars ( each length k )
    self.seq = []          # vocab of sequences
    self.vocab = []        # vocab of words
    self.seq_size = 0      # size of sequences
    self.vocab_size = 0    # size of chars
    self.k = k             # length of sequences for transition states
    self.start_symbol = "_START_"
    self.end_symbol = "_END_"
	
  def initialize(self, tweet_lines):
    self.vocab = [self.start_symbol, self.end_symbol]

	# Build char vocab
    for line in tweet_lines:
      for word in line:
        if word not in self.vocab:
          self.vocab.append(word)
    self.vocab_size = len(self.vocab)
    
    # Build sequence vocab
    product = list(itertools.product(self.vocab, repeat=self.k))
    for tuple in product:
      self.seq.append(list(tuple))
    self.seq_size = len(self.seq)
    
    # Transitions: D | ABC
    self.transitions = np.zeros((self.seq_size, self.vocab_size))
    pass
	
  def buildMarkov(self, tweet_file, num_tweets):
    ifs = open(tweet_file, "r", encoding="ISO-8859-1")#, encoding='utf-8')
    tweet_lines = [None for x in range(num_tweets)]
    for i in range(num_tweets):
      tweet_lines[i] = ifs.readline().strip().split(" ")
    ifs.close()

    self.initialize(tweet_lines)
	
    # Fill in transitions and emissions
    for line in tweet_lines:
      if len(line) != 0:
        # Build initial sequence
        prev_seq = [self.start_symbol for x in range(self.k-1)]
        prev_seq += [line[0]]
        prev = self.seq.index(prev_seq)
        
        for i in range(1, len(line)):
          # Move window over to next sequence
          curr_word = line[i]
          next_seq = prev_seq + [curr_word]
          prev_word = next_seq.pop(0)
          next = self.seq.index(next_seq)
          curr = self.vocab.index(curr_word)
          
          self.transitions[prev][curr] += 1		  
          prev_seq = next_seq
          prev = next

        # Ending sequence
        curr_word = self.end_symbol
        curr = self.vocab.index(curr_word)
          
        self.transitions[prev][curr] += 1
		
    for i in range(self.seq_size):
      sum = np.sum(self.transitions[i])
      if sum != 0:
        self.transitions[i] /= sum		
    pass
	
  # Reversed Viterbi
  def generate_tweet(self, sequence, n):
      #n = 20 # setting constant for mp
      trellis = np.zeros(( n + 1, self.seq_size))
      bp = np.zeros(( n + 1, self.seq_size))
      
      # Build start sequence
      start_seq = [self.start_symbol for x in range(self.k-1)] + sequence
      start = self.seq.index(start_seq)
      trellis[0][start] = 1 # Initialize
	
      #"Recursion"
      for i in range(1, n + 1):
        for word in range(self.vocab_size):
          for j in range(self.seq_size):
            seq = self.seq[j][1:] + [self.vocab[word]]
            s = self.seq.index(seq)
            temp = trellis[i-1][j] * self.transitions[j][word]
            if(temp > trellis[i][s] and self.end_symbol not in seq):
              trellis[i][s] = temp
              bp[i][s] = j

      # Find best final seq in order to go backwards
      seq_max = -1
      vit_max = 0
      end = self.vocab.index(self.end_symbol)
      for seq in range(self.seq_size):
          if(trellis[n][seq]*self.transitions[seq][end] > vit_max):
            seq_max = seq
            vit_max = trellis[n][seq]*self.transitions[seq][end]
		  
      result = [" " for x in range(n)]
      i = n
      s = seq_max 
      #if s == -1: # 0 probability of end_char -> end_symbol
      #  s = np.random.randint(low=2, high=self.seq_size)
      while i > 0:
        result[i-1] = self.seq[s][self.k-1]
        s = int(bp[i][s])
        i -= 1
      
      tweet = str(start_seq[len(start_seq)-1])
      for word in result:
        tweet += " " + word
      return tweet
      

def main():
  time_start = time.time()
  (options, args) = getopt.getopt(sys.argv[1:], '')
    
  if len(args) == 3: # sequence length, num tweets, k
    length = int(args[0])
    num_tweets = int(args[1])
    k = int(args[2])
    words_file = "words_length" + str(length) + ".txt"
    output_file = "VocabOut_length" + str(length) + "_size" + str(num_tweets) + "_k" + str(k) + ".txt"
	
    markov = Markov(k)
    markov.buildMarkov(words_file, num_tweets)
    print("\nBuild Time = ", (time.time() - time_start), " s")
    
    #result = markov.generate_tweet(["i", " ", "w", "a", "n"], 5)
    #print(result)
    #result = markov.generate_tweet(["i"],"\\", 9)
    #print(result)
    #result = markov.generate_tweet(["i"],"\\", 10)
    #print(result)
    result = markov.generate_tweet(["i"], 5)
    print(result)
    #for char in markov.chars:
    #for ascii in range(97, 122+1): #a - z
    #  result = markov.generate_tweet([chr(ascii)], 5)
    #  print(result)
    #result = markov.generate_tweet(["a", "b", "c"], 5)
    #print(result)
    #result = markov.generate_tweet(["a", "b", "c"], 6)
    #print(result)
    #result = markov.generate_tweet(["a", "b", "c"], 7)
    #print(result)
    #starting_chars = []
    #for ascii in range(97, 122+1): #a - z
    #  sequence = [chr(ascii)]
    #  starting_chars.append(sequence)
		  
    #pool = mp.Pool(processes = 8)
    #with open(output_file, "w", encoding='utf-8') as ofs:
    #  for result in pool.imap(markov.generate_tweet, starting_chars):
    #    ofs.write(str(result) + "\n")
    	  	
  time_end = time.time()
  print("\nTotal Time: ", (time_end-time_start), " s")
  
  
if __name__ == "__main__":
    main()
