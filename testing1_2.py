import tweepy
import matplotlib.pyplot as plt
from textblob import TextBlob
import re
#import pickle   #you can use this library to store your model in your system
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import Word
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix


consumer_key = '*****************'
consumer_key_secret = '***********************************'
access_token = '************************************'
access_token_secret = '*********************************'

auth = tweepy.OAuthHandler(consumer_key, consumer_key_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)
count_positive1 = 0
count_positive2 = 0


f=""
f1 = ""

def allOverResultPie(g3,f):
	global count_positive1
	global f
	global count_positive2
	public_tweets = api.search(g3, count=5000)
	count_positive = 0
	count_negative = 0
	count_neutral = 0
	for tweets in public_tweets:
		print(tweets.text)
		analysis = TextBlob(tweets.text)
		#print(f'Analysis: {analysis}')
		#print(analysis.sentiment)
		description_string1 = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', tweets.text,flags=re.MULTILINE)
		description_string1 = description_string1.replace("https", "")
		description_string1 = description_string1.replace("RT", "")
		description_string = re.sub(r'[^a-zA-Z]+', " ", description_string1)
		if analysis.sentiment[0] > 0:
			#print('Positive')
			f.write(description_string + "\n")
			count_positive = count_positive + 1
		elif analysis.sentiment[0] == 0:
			#print('Neutral')
			f.write(description_string + "\n")
			count_neutral = count_neutral + 1
		else:
			#print('Negative')
			f.write(description_string + "\n")
			count_negative = count_negative + 1
		print("")
	if count_positive1 == 0:
		count_positive1 = count_positive
	else:
		count_positive2 = count_positive

	# below code is to make pi chart
	labels = [g3 + '_positive', g3 + '_negative', g3 + '_neutral']

	colors = ['blue', 'yellow', 'green']

	sizes = [count_positive, count_negative, count_neutral]

	plt.pie(sizes, labels=labels, colors=colors, startangle=90, autopct='%1.1f%%')

	plt.axis('equal')

	plt.show()

	#below code is for bar chart
	labels = [g3 + '_positive', g3 + '_negative', g3 + '_neutral']
	pos = np.arange(len(labels))
	sizes = [count_positive, count_negative, count_neutral]
	plt.bar(pos, sizes, color='blue', edgecolor='black')
	plt.xticks(pos, labels)
	plt.xlabel('Tweets', fontsize=16)
	plt.ylabel('Tweets_Index', fontsize=16)
	plt.title('Barchart - Tweets index', fontsize=20)
	plt.show()









g = input("Enter first value : ")
f = open(g+".txt","w")
allOverResultPie(g,f)



g2 = input("Enter second value : ")
f1 = open(g2+".txt","w")
allOverResultPie(g2,f1)
f1.close()







def compareBoth(g,g2):
	global count_positive1,count_positive2
	# below code is to make pi chart
	labels = [g + '_positive', g2 + '_positive']

	colors = ['blue', 'yellow']

	sizes = [count_positive1, count_positive2]

	plt.pie(sizes, labels=labels, colors=colors, startangle=90, autopct='%1.1f%%')

	plt.axis('equal')

	plt.show()

	#below code is for bar chart
	labels = [g + '_positive', g2 + '_positive']
	pos = np.arange(len(labels))
	sizes = [count_positive1, count_positive2]
	plt.bar(pos, sizes, color='blue', edgecolor='black')
	plt.xticks(pos, labels)
	plt.xlabel('Tweets', fontsize=16)
	plt.ylabel('Tweets_Index', fontsize=16)
	plt.title('Barchart - Tweets index', fontsize=20)
	plt.show()

compareBoth(g,g2)


def clean_str(string):

    string = re.sub(r"\'s", "", string)
    string = re.sub(r"\'ve", "", string)
    string = re.sub(r"n\'t", "", string)
    string = re.sub(r"\'re", "", string)
    string = re.sub(r"\'d", "", string)
    string = re.sub(r"\'ll", "", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"'", "", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"[0-9]\w+|[0-9]","", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


print(f'\n\nMachine learning in progress.......\n\n')
data = pd.read_csv('dataset2.csv')
x = data['news'].tolist()
y = data['type'].tolist()

for index,value in enumerate(x):
#    print("processing data:",index)
    x[index] = ' '.join([Word(word).lemmatize() for word in clean_str(value).split()])

vect = TfidfVectorizer(stop_words='english',min_df=2)
X = vect.fit_transform(x)
Y = np.array(y)
#print(f'shape of X: {X.shape}')
#print(f'shape of X: {X}')
print("no of features extracted:",X.shape[1])
#print(f'tolist : {x}')


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

print("train size:", X_train.shape)
print("test size:", X_test.shape)

model = RandomForestClassifier(n_estimators=300, max_depth=1700,n_jobs=-1)
history = model.fit(X_train, y_train)
#print("y_test",y_test)
y_pred = model.predict(X_test)
c_mat = confusion_matrix(y_test,y_pred)
kappa_score = cohen_kappa_score(y_test,y_pred)
acc_score = accuracy_score(y_test,y_pred)
print("Confusion Matrix:\n", c_mat)
print("\nCohen's Kappa score: ",kappa_score)
print("\nAccuracy: ",acc_score)
def check_news_type(news_article):  # function to accept raw string and to provide news type(class)
    news_article = [' '.join([Word(word).lemmatize() for word in clean_str(news_article).split()])]
    features = vect.transform(news_article)
    return str(model.predict(features)[0])

read_file1 = open(g+".txt","r")
article1 = read_file1.read()
read_file1.close()

read_file2 = open(g2+".txt","r")
article2 = read_file2.read()
read_file2.close()

#check_news_type(article)
print("For "+g+" its, "+check_news_type(article1))
print("For "+g2+" its, "+check_news_type(article2))







