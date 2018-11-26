from textgenrnn import textgenrnn
textgen = textgenrnn()
textgen.train_from_file('realDonaldTrump_tweets.csv', num_epochs=10)
textgen.generate(5)
