import pandas as pd

# URLs for major indices
urls = {
    "S&P 500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "NASDAQ-100": "https://en.wikipedia.org/wiki/NASDAQ-100",
    "Dow Jones": "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
}

all_companies = pd.DataFrame()

for name, url in urls.items():
    try:
        tables = pd.read_html(url)
        if name == "S&P 500":
            df = tables[0][["Symbol", "Security"]]
        elif name == "NASDAQ-100":
            df = tables[3][["Ticker", "Company"]].rename(columns={"Ticker": "Symbol", "Company": "Security"})
        elif name == "Dow Jones":
            df = tables[1][["Symbol", "Company"]].rename(columns={"Company": "Security"})
        
        df["Index"] = name
        all_companies = pd.concat([all_companies, df], ignore_index=True)
    except Exception as e:
        print(f"Error processing {name}: {e}")

# Save to Excel
all_companies.to_excel("stock_tickers.xlsx", index=False)
print("Saved to stock_tickers.xlsx")
