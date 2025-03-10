import json

def process_tweet_file(filename):
    processed_tweets = []
    
    with open(filename, 'r') as f:
        for i, line in enumerate(f, 1):
            tweet = json.loads(line)
            # Concatenate all tokens in the text array with spaces
            full_text = f"{i}. " + ' '.join(tweet['text'])
            processed_tweets.append(full_text)
    
    return processed_tweets

# Template text
template = """<Task>
Analyze the provided tweets to determine their likely impact on the future stock price of a given company. Your answer must be one of the following: "[Positive]", "[Neutral]", or "[Negative]".

<Solving Process>
1. Identify the Target Stock: Extract the stock symbol from the tweets.
2. Filter Meaningful Tweets: Discard tweets that lack clear information about stock price movements. Focus only on those with market-moving details.
3. Tweet Analysis:
Separate the Factors: For each relevant tweet, identify Positive and Negative factors.
Assess Sentiment: Evaluate how each relevant tweet might influence investor sentiment.
Overall Sentiment Summary: Combine your analyses to conclude the overall sentiment.

<Tweets>
{}

<Output Requirement>
You must do this: Conclude with a single line that states the overall sentiment. Use one of the following tags exactly:
[Positive]
[Neutral]
[Negative]
"""

# Process the file
filename = 'tweet/preprocessed/AAPL/2014-01-02'
tweets = process_tweet_file(filename)

# Format tweets and print with template
formatted_tweets = '\n'.join(tweets)
print(template.format(formatted_tweets)) 