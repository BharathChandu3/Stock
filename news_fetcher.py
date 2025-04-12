import yfinance as yf

def fetch_news(quote):
    stock = yf.Ticker(quote)
    news = stock.news
    return news