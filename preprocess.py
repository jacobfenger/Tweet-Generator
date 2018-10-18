import csv 

def remove_bad_tweets(tweets):
	print ''

def main():
	filename = 'realDonaldTrump_tweets.csv'
	csvr = csv.reader(open(filename))

	tweets = []
	for row in csvr:
		tweets.append(row[0].split(' '))

	print tweets[1]
        print tweets[2]
	remove_bad_tweets(tweets)

if __name__ == '__main__':
	main()
