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
    self.chars = []        # vocab of chars
    self.seq_size = 0      # size of sequences
    self.chars_size = 0    # size of chars
    self.k = k             # length of sequences for transition states
    self.start_symbol = "_START_"
    self.end_symbol = "_END_"
	
  def initialize(self, tweet_lines):
    self.chars = [self.start_symbol, self.end_symbol]

	# Build char vocab
    for line in tweet_lines:
      for char in line:
        if char not in self.chars: #and char != "":
          self.chars.append(char)
    self.chars_size = len(self.chars)
    
    # Build sequence vocab
    product = list(itertools.product(self.chars, repeat=self.k))
    for tuple in product:
      self.seq.append(list(tuple))
    self.seq_size = len(self.seq)
    
    # Transitions: D | ABC
    self.transitions = np.zeros((self.seq_size, self.chars_size))
    pass
	
  def buildMarkov(self, tweet_file, num_tweets):
    ifs = open(tweet_file, "r", encoding="ISO-8859-1")#, encoding='utf-8')
    tweet_lines = [None for x in range(num_tweets)]
    for i in range(num_tweets):
      tweet_lines[i] = list(ifs.readline().strip())
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
          curr_char = line[i]
          next_seq = prev_seq + [curr_char]
          prev_char = next_seq.pop(0)
          next = self.seq.index(next_seq)
          curr = self.chars.index(curr_char)
          
          self.transitions[prev][curr] += 1		  
          prev_seq = next_seq
          prev = next

        # Ending sequence
        curr_char = self.end_symbol
        curr = self.chars.index(curr_char)
          
        self.transitions[prev][curr] += 1
		
    for i in range(self.seq_size):
      sum = np.sum(self.transitions[i])
      if sum != 0:
        self.transitions[i] /= sum		
    pass
	
  # Reversed Viterbi
  def generate_tweet(self, sequence, n, random):
      #n = 8 # setting constant for mp
      trellis = np.zeros(( n + 1, self.seq_size))
      bp = np.zeros(( n + 1, self.seq_size))
      
      # Build start sequence
      start_seq = [self.start_symbol for x in range(self.k-1)] + sequence
      start = self.seq.index(start_seq)
      trellis[0][start] = 1 # Initialize
	
      #"Recursion"
      for i in range(1, n + 1):
        for char in range(self.chars_size):
          for j in range(self.seq_size):
            seq = self.seq[j][1:] + [self.chars[char]]
            s = self.seq.index(seq)
            temp = trellis[i-1][j] * self.transitions[j][char]
            if(temp > trellis[i][s] and self.end_symbol not in seq):
              trellis[i][s] = temp
              bp[i][s] = j

      # Find best final seq in order to go backwards
      seq_max = -1
      vit_max = 0
      #end = self.chars.index(self.end_symbol)#self.chars.index(end_char)
      end = self.chars.index(self.end_symbol)
      if random == True:
        end = np.random.randint(low=2, high=self.chars_size)
      for seq in range(self.seq_size):
          if(trellis[n][seq]*self.transitions[seq][end] > vit_max):
            seq_max = seq
            vit_max = trellis[n][seq]*self.transitions[seq][end]
		  
      result = [" " for x in range(n)]
      i = n
      s = seq_max 
      while i > 0:
        result[i-1] = self.seq[s][self.k-1]
        s = int(bp[i][s])
        i -= 1
      
      tweet = str(start_seq[len(start_seq)-1])
      for char in result:
        tweet += char
      #tweet += end_char
      return tweet
      

def main():
  time_start = time.time()
  (options, args) = getopt.getopt(sys.argv[1:], '')
    
  if len(args) == 5 or len(args) == 6: # sequence length, num tweets, k, start_char, output length, random (optional)
    length = int(args[0])
    num_tweets = int(args[1])
    k = int(args[2])
    start_char = args[3]
    output_length = int(args[4])
    random = False
    if len(args) == 6:
      random = True
    words_file = "words_length" + str(length) + ".txt"
    #output_file = "CharOut_length" + str(length) + "_size" + str(num_tweets) + "_k" + str(k) + ".txt"
	
    markov = Markov(k)
    markov.buildMarkov(words_file, num_tweets)
    print("\nBuild Time = ", (time.time() - time_start), " s")
   
    result = markov.generate_tweet([start_char], output_length, random)
    print(result)
		  
    #pool = mp.Pool(processes = 8)
    #with open(output_file, "w", encoding='utf-8') as ofs:
    #  for result in pool.imap(markov.generate_tweet, starting_chars):
    #    ofs.write(str(result) + "\n")
    	  	
  time_end = time.time()
  print("\nTotal Time: ", (time_end-time_start), " s")
  
  
if __name__ == "__main__":
    main()
