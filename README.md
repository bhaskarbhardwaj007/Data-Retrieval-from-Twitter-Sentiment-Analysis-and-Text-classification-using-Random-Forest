# Data-Retrieval-from-Twitter-Sentiment-Analysis-and-Text-classification-using-Random-Forest<br>
Download the dataset from given link http://mlg.ucd.ie/datasets/bbc.html<br>

# DESCRIPTION<br>
This project is my master's project that I did in my college. And it is divided into 2 parts<br>
See activity diagram for more details:-<br>
![Bhaskar](https://user-images.githubusercontent.com/52116851/90570760-3c6dd980-e165-11ea-83b0-8269e7de71c2.png)



# In the first part<br>
This project fetches the 2 given inputs from Twitter. Then it saves all tweets in 2 files. <br>
Perform sentiment analysis on Twitter. <br>
Generate the Pie-Chart and Bar-Chart according to the sentiment.
Compare the result of both inputs.<br>

# In the second part<br>
It uses the Random Forest model for Text classification. This model is trained on the BBC NEWS dataset that I mentioned to download before all the process.<br>

# Prerequisite:-<br>
pip install matplotlib
pip install textblob

# How to run the code ?<br>
Make developer account on Twitter. This process can take several days, because Twitter team will do background checking on you.<br>
Build application on your developer account.<br>
Create keys that you need for this project.<br>
See the below image for keys:-<br>
<
![api keys](https://user-images.githubusercontent.com/52116851/90570963-ab4b3280-e165-11ea-9065-31d86daca881.PNG)
These keys are :- API key = Consumer key, API secret key = consumer secret key, access token and access token secret.<br>
These are highly secretive keys, so do not share it with anyone<br>
Replace below part in code with your keys :-<br>
consumer_key = 'your key'<br>
consumer_key_secret = 'your key'<br>
access_token = 'your key'<br>
access_token_secret = 'your key'<br>
<br>
Check the directories properly.



# Feel free to use the code

# NOTE:-
For my college work I also made my own dataset. And merged it with BBC NEWS dataset to cover the more probability distribution in a class. I did it because BBC NEWS dataset contains whole article and Twitter can contain maximum of 280 characters.<br>
I have not included that part in the above code.
