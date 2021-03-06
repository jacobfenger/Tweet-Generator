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
    self.k = k             # length of sequences for transition states
	
  def initialize(self, tweet_lines):
	# Build char vocab
    for line in tweet_lines:
      for char in line:
        if char not in self.chars: #and char != "":
          self.chars += char
    
    # Build sequence vocab from char vocab
    product = list(itertools.product(self.chars, repeat=self.k))
    for tuple in product:
      self.seq.append(list(tuple))
	
    # Transitions: ABC -> D (BCD | ABC)
    self.transitions = np.zeros((len(self.seq), len(self.chars)))
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
        prev_seq = line[0:self.k]
        #print(prev_seq)
        prev = self.seq.index(prev_seq)
        #self.seq_counts[prev] += 1
        
        for i in range(self.k + 1, len(line)):
          # Move window over to next sequence
          char = line[i]
          curr_seq = prev_seq + [char]
          prev_char = curr_seq.pop(0)
          #print(curr_seq)
          #index = self.seq.index(curr_seq)
          curr = self.chars.index(char)#self.chars_states[char]
          
          self.transitions[prev][curr] += 1
          #self.seq_counts[index] += 1
		  
          prev_seq = curr_seq
          prev = curr
		
    pass
  
  def topThree(self, transitions):
    max = []#[transitions[0], transitions[1], transitions[2]]
    max_i = []#[0, 1, 2]
    if transitions[0] >= transitions[1] and transitions[0] >= transitions[2]:
      if transitions[1] >= transitions[2]:
        max = [transitions[0], transitions[1], transitions[2]]
        max_i = [0,1,2]
      else:
        max = [transitions[0], transitions[2], transitions[1]]
        max_i = [0,2,1]
    elif transitions[1] >= transitions[0] and transitions[2] >= transitions[1]:
      if transitions[0] >= transitions[2]:
        max = [transitions[1], transitions[0], transitions[2]]
        max_i = [1,0,2]
      else:
        max = [transitions[1], transitions[2], transitions[0]]
        max_i = [1,2,0]
    else:
      if transitions[0] >= transitions[1]:
        max = [transitions[2], transitions[0], transitions[1]]
        max_i = [2,0,1]
      else:
        max = [transitions[2], transitions[1], transitions[0]]
        max_i = [2,1,0]
    
	
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
	
  def generate_tweet(self, sequence, length):
    tweet = sequence
    i = 0
    seq = sequence
    if len(sequence) > self.k:
      seq = sequence[(len(sequence) - self.k):]
    elif len(sequence) <= self.k-1:
      seq = [" " for x in range(self.k - len(sequence))] + sequence
    #print(seq)
    next_state = np.arange(len(self.seq))
    while i != length:
      curr = self.seq.index(seq)
      #print(self.transitions[curr])
      #next = np.random.choice(next_state,replace=True,p=self.transitions[curr])
      #next = np.argmax(self.transitions[curr])
      options = self.topThree(self.transitions[curr])
      next = self.seq[options[np.random.randint(3)]]
      #next_char = self.chars[options[np.random.randint(3)]]
      #next_seq = seq + [next_char]
      #next_seq.pop(0)
      #print(next_seq)
      tweet.append(next[self.k-1])
      i += 1
      seq = next
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
    
    num_output = 10
    #result = markov.generate_tweet(["i", " ", "w", "a", "n", "t"], 5)
    result = markov.generate_tweet(["i", " "], 10)
    #random_choice = np.random.randint(len(markov.seq))
    #input_seq = markov.seq[random_choice]
    #result = markov.generate_tweet(input_seq, 10)
    print(result)
	
    #ifs = open(tags_file, "r", encoding="ISO-8859-1")# encoding='utf-8')
    #tag_sequences = [None for x in range(num_output)]
    #for i in range(num_output):
    #  tag_sequences[i] = ifs.readline().strip().split(" ")
    #ifs.close()
	  
    #pool = mp.Pool(processes = 8)
    #with open(output_file, "w", encoding='utf-8') as ofs:
    #  for result in pool.imap(markov.generate_tweet, tag_sequences):
    #    ofs.write(str(result) + "\n")
    	  	
  time_end = time.time()
  print("\nTotal Time: ", (time_end-time_start), " s")
  
  
if __name__ == "__main__":
    main()
