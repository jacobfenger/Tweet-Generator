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
import string
	
class Markov:
  def __init__(self, k):
    self.transitions = []  # 2D array - prob that sequence of chars follows a sequence of chars ( each length k )
    #self.emissions = []    # 2D array - prob of char given preceding sequence
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
    self.chars = []#[self.start_symbol, self.end_symbol]

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
    self.transitions = np.zeros((self.seq_size, self.chars_size))#self.seq_size))
    # Emissions: D | ABC
    #self.emissions = np.zeros((self.chars_size, self.seq_size))
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
        #prev_seq = [self.start_symbol for x in range(self.k-1)]
        #prev_seq += [line[0]]
        #prev = self.seq.index(prev_seq)
        
        #self.seq_counts[prev] += 1
        #self.chars_counts[self.chars.index(self.start_symbol)] += self.k - 1
        #self.chars_counts[self.chars.index(line[0])] += 1
		
        prev_seq = line[0:self.k]
        prev = self.seq.index(prev_seq)
        
        for i in range(self.k + 1, len(line)): #(1, len(line))
          # Move window over to next sequence
          curr_char = line[i]
          next_seq = prev_seq + [curr_char]
          prev_char = next_seq.pop(0)
          next = self.seq.index(next_seq)
          curr = self.chars.index(curr_char)
          
          self.transitions[prev][curr] += 1
          #self.emissions[self.chars.index(curr_char)][curr] += 1
          self.seq_counts[next] += 1#[curr] += 1
          self.chars_counts[self.chars.index(curr_char)] += 1
		  
          prev_seq = next_seq
          prev = next

        # Ending sequence
        #curr_char = self.end_symbol
        #curr_seq = prev_seq + [curr_char]
        #prev_char = curr_seq.pop(0)
        #curr = self.seq.index(curr_seq)
          
        #self.transitions[prev][curr] += 1
        #self.emissions[self.chars.index(curr_char)][curr] += 1
        #self.seq_counts[curr] += 1
        #self.chars_counts[self.chars.index(curr_char)] += 1
		
    for i in range(self.seq_size):
      if self.seq_counts[i] != 0:
        self.transitions[i] /= self.seq_counts[i]
	
    #for i in range(self.chars_size):
    #  for j in range(self.seq_size):
    #    if self.seq_counts[j] != 0:
    #      self.emissions[i][j] /= self.seq_counts[j]
		
    pass
	
  def topThree(self, transitions):
    max = [transitions[0], transitions[1], transitions[2]]
    max_i = [0, 1, 2]
    for i in range(3, len(transitions)):
      if transitions[i] > max[0]:
        max[2] = max[1]
        max_i[2] = max_i[1]
        max[1] = max[0]
        max_i[1] = max_i[0]
        max[0] = transitions[i]
        max_i[0] = i
      elif transitions[i] > max[1]:
        max[2] = max[1]
        max_i[2] = max_i[1]
        max[1] = transitions[i]
        max_i[1] = i
      elif transitions[i] > max[2]:
        max[2] = transitions[i]
        max_i[2] = i
    return max_i
	
  # Reversed Viterbi
  def generate_tweet(self, sequence, n):
      #n = 20 # setting constant for mp
      trellis = np.zeros(( n + 1, self.seq_size))
      bp = np.zeros(( n + 1, self.seq_size))
      
      # Build start sequence
      #start_seq = [self.start_symbol] + sequence
      #special = 1 # how many special symbols are in the starting sequence
      #if len(sequence) >= self.k: # trim down to k-1 length
      #  start_seq = [self.start_symbol] + sequence[-(self.k-1):]
      #elif len(sequence) < self.k-1:
      #  start_seq = [self.start_symbol for x in range(self.k - len(sequence))] + sequence
      #  special = self.k - len(sequence)
      #start = self.seq.index(start_seq)
	  
      # ASSUME: sequence is length self.k
      start_seq = sequence
      start = self.seq.index(start_seq) #self.seq.index(sequence[-(self.k):])
      
      trellis[0][start] = 1 # Initialize
	  
      j = start
      for i in range(1, n + 1):
        for char in range(self.chars_size):
          #for j in range(self.seq_size):
            temp = trellis[i-1][j] * self.transitions[j][char]
            seq = self.seq[j][1:] + [self.chars[char]] # next sequence
            s = self.seq.index(seq)
            if(temp > trellis[i][s] and self.chars[char] in string.printable):
              trellis[i][s] = temp
              bp[i][s] = j
              next_seq = seq
        j = s #next_seq
	
      #"Recursion"
      #end_seq = -1
      #for i in range(1, n + 1):
      #  for seq in range(self.seq_size):
      #    for j in range(self.seq_size):
      #      temp = trellis[i-1][j] * self.transitions[j][seq]
      #      if(temp > trellis[i][seq] and self.end_symbol not in self.seq[seq]):
      #        trellis[i][seq] = temp
      #        bp[i][seq] = j
			  
      #j = start
      #for i in range(1, n + 1):
      #  for seq in range(self.seq_size):
      #    #for j in range(self.seq_size):
      #      temp = trellis[i-1][j] * self.transitions[j][seq]
      #      if(temp > trellis[i][seq] and self.end_symbol not in self.seq[seq]):
      #        trellis[i][seq] = temp
      #        bp[i][seq] = j
      #        next_seq = seq
      #  j = next_seq
          #char = self.seq[end_seq][len(self.seq[end_seq])-1] # emission is last character in next sequence
          #char = self.seq[seq][len(self.seq[seq])-1] # emission is last character in next sequence
          #char_i = self.chars.index(char)
          #trellis[i][seq] *= self.emissions[char_i][seq]

      seq_max = -1
      vit_max = 0
      #recursion_end = self.seq[end_seq]
      #end_seq = recursion_end + [self.end_symbol]
      #end_seq.pop(0) # move window over
      #end = self.seq.index(end_seq)
	  
      
      # Find best final seq in order to go backwards
      #for seq in range(self.seq_size):
      #  for end in range(self.seq_size):
      #    if(trellis[n][seq]*self.transitions[seq][end] > vit_max and self.end_symbol in self.seq[end]):
      #      seq_max = seq
      #      vit_max = trellis[n][seq]*self.transitions[seq][end]
			
      seq_max = j # CHECK
		  
      result = [" " for x in range(n)]
      i = n
      s = seq_max
      if s == -1: # 0 probability of end_seq -> end_symbol
        # Just go back from the last character generated
        s = np.argmax(trellis[n]) #self.seq.index(recursion_end)
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
    #for i in range (5, 10+1):
    #  result = markov.generate_tweet(["i", " "], i)
    #  if all(c in string.printable for c in result):
    #    print(result)
    result = markov.generate_tweet(["i", " "], 10)
    print(result)
    #for char in markov.chars:
    #  result = markov.generate_tweet([char], 5)
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
