# imports
import os
import time

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv


import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from pytz import timezone  # type: ignore
import matplotlib
matplotlib.use('Agg')
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient


load_dotenv()

def plot_pe_history(ticker):
    stock = yf.Ticker(ticker)

    # 1. Get historical stock prices (year-end close)
    hist = stock.history(period="5y")
    yearly_prices = hist['Close'].resample('Y').last()

    # 2. Get annual earnings per share (EPS)
    eps_data = stock.earnings  # DataFrame with 'Revenue' and 'Earnings'

    if eps_data.empty:
        print("No earnings data available.")
        return

    # 3. Compute P/E = Price / EPS
    # We'll assume EPS = Net Income / Shares Outstanding (simplified)
    # For now, use "Earnings" from `stock.earnings`, divide by some fixed shares (since shares data is limited)
    pe_ratios = []
    years = []
    
    approx_shares = 1e9  # adjust based on the company; ideally fetched dynamically

    for year in eps_data.index:
        if str(year) in yearly_prices.index.strftime('%Y'):
            price = yearly_prices[yearly_prices.index.strftime('%Y') == str(year)].values[0]
            earnings = eps_data.loc[year, "Earnings"]
            eps = earnings / approx_shares  # crude estimate
            pe = price / eps if eps != 0 else None
            pe_ratios.append(pe)
            years.append(year)

    # 4. Plot
    plt.figure(figsize=(8, 4))
    plt.plot(years, pe_ratios, marker='o')
    plt.title(f"{ticker.upper()} - Approx. P/E Ratio (Year-End)")
    plt.xlabel("Year")
    plt.ylabel("P/E Ratio")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def google_search(query: str, num_results: int = 2, max_chars: int = 500) -> list:  # type: ignore[type-arg]


    api_key = os.getenv("GOOGLE_API_KEY")
    search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

    if not api_key or not search_engine_id:
        raise ValueError("API key or Search Engine ID not found in environment variables")

    url = "https://customsearch.googleapis.com/customsearch/v1"
    params = {"key": str(api_key), "cx": str(search_engine_id), "q": str(query), "num": str(num_results)}

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(response.json())
        raise Exception(f"Error in API request: {response.status_code}")

    results = response.json().get("items", [])

    def get_page_content(url: str) -> str:
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            words = text.split()
            content = ""
            for word in words:
                if len(content) + len(word) + 1 > max_chars:
                    break
                content += " " + word
            return content.strip()
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            return ""

    enriched_results = []
    for item in results:
        body = get_page_content(item["link"])
        enriched_results.append(
            {"title": item["title"], "link": item["link"], "snippet": item["snippet"], "body": body}
        )
        time.sleep(1)  # Be respectful to the servers

    return enriched_results


def analyze_stock(ticker: str) -> dict:  # type: ignore[type-arg]

    stock = yf.Ticker(ticker)

    # Get historical data (1 year of data to ensure we have enough for 200-day MA)
    end_date = datetime.now(timezone("UTC"))
    start_date = end_date - timedelta(days=365)
    hist = stock.history(start=start_date, end=end_date)

    # Ensure we have data
    if hist.empty:
        return {"error": "No historical data available for the specified ticker."}
    info = stock.info
    # Compute basic statistics and additional metrics
    current_price = info.get("currentPrice", hist["Close"].iloc[-1])
    year_high = info.get("fiftyTwoWeekHigh", hist["High"].max())
    year_low = info.get("fiftyTwoWeekLow", hist["Low"].min())

    # Calculate 50-day and 200-day moving averages
    ma_50 = hist["Close"].rolling(window=50).mean().iloc[-1]
    ma_200 = hist["Close"].rolling(window=200).mean().iloc[-1]

    # Calculate YTD price change and percent change
    ytd_start = datetime(end_date.year, 1, 1, tzinfo=timezone("UTC"))
    ytd_data = hist.loc[ytd_start:]  # type: ignore[misc]
    if not ytd_data.empty:
        price_change = ytd_data["Close"].iloc[-1] - ytd_data["Close"].iloc[0]
        percent_change = (price_change / ytd_data["Close"].iloc[0]) * 100
    else:
        price_change = percent_change = np.nan

    # Determine trend
    if pd.notna(ma_50) and pd.notna(ma_200):
        if ma_50 > ma_200:
            trend = "Upward"
        elif ma_50 < ma_200:
            trend = "Downward"
        else:
            trend = "Neutral"
    else:
        trend = "Insufficient data for trend analysis"

    # Calculate volatility (standard deviation of daily returns)
    daily_returns = hist["Close"].pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility

    # --- Additional Financial Metrics ---
    pe_ratio = info.get("trailingPE", None)
    pb_ratio = info.get("priceToBook", None)
    book_value = info.get("bookValue", None)
    operating_margin = info.get("operatingMargins", None)
    debt_to_equity = info.get("debtToEquity", None)
    free_cash_flow = info.get("freeCashflow", None)
    return_on_capital = info.get("returnOnCapital", None)
    return_on_equity = info.get("returnOnEquity", None)


    # Result dictionary
    result = {
        "ticker": ticker,
        "current_price": current_price,
        "52_week_high": year_high,
        "52_week_low": year_low,
        "50_day_ma": ma_50,
        "200_day_ma": ma_200,
        "ytd_price_change": price_change,
        "ytd_percent_change": percent_change,
        "trend": trend,
        "volatility": volatility,
        "pe_ratio": pe_ratio,
        "pb_ratio": pb_ratio,
        "book_value": book_value,
        "operating_margin": operating_margin,
        "total_debt": debt_to_equity,
        "free_cash_flow": free_cash_flow,
        "return_on_capital": return_on_capital,
        "return_on_equity": return_on_equity,
    }

    # Convert numpy types to Python native types for better JSON serialization
    for key, value in result.items():
        if isinstance(value, np.generic):
            result[key] = value.item()

    # Generate plot
    plt.figure(figsize=(12, 6))
    plt.plot(hist.index, hist["Close"], label="Close Price")
    plt.plot(hist.index, hist["Close"].rolling(window=50).mean(), label="50-day MA")
    plt.plot(hist.index, hist["Close"].rolling(window=200).mean(), label="200-day MA")
    plt.title(f"{ticker} Stock Price (Past Year)")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True)

    # --- Add table for Additional Financial Metrics ---
    metrics = {
        "PE Ratio": pe_ratio,
        "PB Ratio": pb_ratio,
        "Operating Margin": operating_margin,
        "Debt to Equity": debt_to_equity,
        "Free Cash Flow": free_cash_flow,
        "Return on Capital": return_on_capital,
        "Return on Equity": return_on_equity,
        "Book Value": book_value,
    }
    metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
    table = plt.table(cellText=metrics_df.values,
                      colLabels=metrics_df.columns,
                      cellLoc='center',
                      loc='bottom',
                      bbox=[0.0, -0.45, 1, 0.35])
    plt.subplots_adjust(bottom=0.35)

    # Save plot to file
    os.makedirs("coding", exist_ok=True)
    plot_file_path = f"coding/{ticker}_stockprice.png"
    plt.savefig(plot_file_path)
    print(f"Plot saved as {plot_file_path}")
    result["plot_file_path"] = plot_file_path
    # plot_pe_history(ticker)

    return result