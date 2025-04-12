from textblob import TextBlob
import numpy as np

def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def analyze_news(news):
    sentiments = []
    for article in news:
        if 'summary' in article:
            sentiment = analyze_sentiment(article['summary'])
            sentiments.append(sentiment)
    avg_sentiment = np.mean(sentiments) if sentiments else 0
    if avg_sentiment > 0.1:
        return "Positive"
    elif avg_sentiment < -0.1:
        return "Negative"
    else:
        return "Neutral"