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
    self.emissions = []    # 2D array - prob of char given preceding sequence
    self.seq_counts = []   # 1D array - counts of each sequence occurence
    self.char_counts = []  # 1D array - counts of each char occurence
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
    
    # Transitions: BCD | ABC
    self.transitions = np.zeros((self.seq_size, self.seq_size))
    # Emissions: D | ABC
    self.emissions = np.zeros((self.chars_size, self.seq_size))
    # Counts of chars
    self.chars_counts = np.zeros(self.chars_size)
    # Counts of sequences
    self.seq_counts = np.zeros(self.seq_size)
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
        
        self.seq_counts[prev] += 1
        self.chars_counts[self.chars.index(self.start_symbol)] += self.k - 1
        self.chars_counts[self.chars.index(line[0])] += 1
        
        for i in range(1, len(line)):
          # Move window over to next sequence
          curr_char = line[i]
          curr_seq = prev_seq + [curr_char]
          prev_char = curr_seq.pop(0)
          curr = self.seq.index(curr_seq)
          
          self.transitions[prev][curr] += 1
          self.emissions[self.chars.index(curr_char)][curr] += 1
          self.seq_counts[curr] += 1
          self.chars_counts[self.chars.index(curr_char)] += 1
		  
          prev_seq = curr_seq
          prev = curr

        # Ending sequence
        curr_char = self.end_symbol
        curr_seq = prev_seq + [curr_char]
        prev_char = curr_seq.pop(0)
        curr = self.seq.index(curr_seq)
          
        self.transitions[prev][curr] += 1
        self.emissions[self.chars.index(curr_char)][curr] += 1
        self.seq_counts[curr] += 1
        self.chars_counts[self.chars.index(curr_char)] += 1
		
    for i in range(self.seq_size):
      if self.seq_counts[i] != 0:
        self.transitions[i] /= self.seq_counts[i]
	
    for i in range(self.chars_size):
      for j in range(self.seq_size):
        if self.seq_counts[j] != 0:
          self.emissions[i][j] /= self.seq_counts[j]
		
    pass
	
  # Reversed Viterbi
  def generate_tweet(self, sequence, n):
      #n = 20 # setting constant for mp
      trellis = np.zeros(( n + 1, self.seq_size))
      bp = np.zeros(( n + 1, self.seq_size))
      
      # Build start sequence
      start_seq = [self.start_symbol] + sequence
      special = 1 # how many special symbols are in the starting sequence
      if len(sequence) >= self.k: # trim down to k-1 length
        start_seq = [self.start_symbol] + sequence[-(self.k-1):]
      elif len(sequence) < self.k-1:
        start_seq = [self.start_symbol for x in range(self.k - len(sequence))] + sequence
        special = self.k - len(sequence)
      start = self.seq.index(start_seq)
      
      trellis[0][start] = 1 # Initialize
	
      #"Recursion"
      end_seq = -1
      for i in range(1, n + 1):
        for seq in range(self.seq_size):
          for j in range(self.seq_size):
            temp = trellis[i-1][j] * self.transitions[j][seq]
            if(temp > trellis[i][seq] and self.end_symbol not in self.seq[seq]):
              trellis[i][seq] = temp
              bp[i][seq] = j
              end_seq = seq
          #char = self.seq[end_seq][len(self.seq[end_seq])-1] # emission is last character in next sequence
          char = self.seq[seq][self.k-1] # emission is last character in next sequence
          c = self.chars.index(char)
          trellis[i][seq] *= self.emissions[c][seq]

      seq_max = -1
      vit_max = 0
      #recursion_end = self.seq[end_seq]
      #end_seq = recursion_end + [self.end_symbol]
      #end_seq.pop(0) # move window over
      #end = self.seq.index(end_seq)
      # Find best final seq in order to go backwards
      for seq in range(self.seq_size):
        end_seq = self.seq[seq][1:] + [self.end_symbol]
        end = self.seq.index(end_seq)
        if(trellis[n][seq]*self.transitions[seq][end] > vit_max):
          seq_max = seq
          vit_max = trellis[n][seq]*self.transitions[seq][end]
		  
      result = [" " for x in range(n)]
      i = n
      s = np.random.choice(self.seq)#seq_max
      while self.start_symbol in self.seq[s]:
        s = np.random.choice(self.seq)#seq_max
      #if s == -1: # 0 probability of end_seq -> end_symbol
        # Just go back from the last character generated
      #  s = self.seq.index(recursion_end)
      while i > 0:
        result[i-1] = self.seq[s][self.k-1]
        s = int(bp[i][s])
        i -= 1
      
      tweet = str(start_seq[len(start_seq)-1])
      for char in result:
        tweet += char
      return tweet
      

def main():
  time_start = time.time()
  (options, args) = getopt.getopt(sys.argv[1:], '')
    
  if len(args) == 3: # sequence length, num tweets, k
    length = int(args[0])
    num_tweets = int(args[1])
    k = int(args[2])
    words_file = "words_length" + str(length) + ".txt"
    output_file = "CharOut_length" + str(length) + "_size" + str(num_tweets) + "_k" + str(k) + ".txt"
	
    markov = Markov(k)
    markov.buildMarkov(words_file, num_tweets)
    print("\nBuild Time = ", (time.time() - time_start), " s")
    
    #result = markov.generate_tweet(["i", " ", "w", "a", "n"], 5)
    #print(result)
    result = markov.generate_tweet(["i"], 10)
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
    starting_chars = []
    for ascii in range(97, 122+1): #a - z
      sequence = [chr(ascii)]
      starting_chars.append(sequence)
		  
    #pool = mp.Pool(processes = 8)
    #with open(output_file, "w", encoding='utf-8') as ofs:
    #  for result in pool.imap(markov.generate_tweet, starting_chars):
    #    ofs.write(str(result) + "\n")
    	  	
  time_end = time.time()
  print("\nTotal Time: ", (time_end-time_start), " s")
  
  
if __name__ == "__main__":
    main()
