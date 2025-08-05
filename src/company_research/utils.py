# imports
import os
import time
import logging

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Set matplotlib backend to non-interactive to avoid GUI issues
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from pytz import timezone  # type: ignore
import matplotlib.pyplot as plt

from logger.log_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

load_dotenv()


def google_search(query: str, num_results: int = 2, max_chars: int = 500) -> list:
    """
    Perform a Google search and retrieve enriched results with page content.
    
    This function uses the Google Custom Search API to search for information
    and then fetches the actual content from the returned URLs to provide
    more comprehensive search results.
    
    Args:
        query (str): The search query to execute.
        num_results (int, optional): Number of search results to retrieve. 
                                   Defaults to 2.
        max_chars (int, optional): Maximum number of characters to extract 
                                  from each page's content. Defaults to 500.
    
    Returns:
        list: A list of dictionaries containing search results. Each dictionary
              contains:
              - title (str): The page title
              - link (str): The page URL
              - snippet (str): The search result snippet
              - body (str): The extracted page content (limited to max_chars)
    
    Raises:
        ValueError: If Google API key or Search Engine ID are not found in
                   environment variables.
        Exception: If the Google API request fails (non-200 status code).
    
    Example:
        >>> results = google_search("Apple Inc financial news", num_results=3)
        >>> print(f"Found {len(results)} results")
        Found 3 results
    """
    logger.info(f"Starting Google search for query: {query}")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

    if not api_key or not search_engine_id:
        logger.error("API key or Search Engine ID not found in environment variables")
        raise ValueError("API key or Search Engine ID not found in environment variables")

    url = "https://customsearch.googleapis.com/customsearch/v1"
    params = {"key": str(api_key), "cx": str(search_engine_id), "q": str(query), "num": str(num_results)}

    response = requests.get(url, params=params)

    if response.status_code != 200:
        logger.error(f"Error in API request: {response.status_code}")
        logger.error(response.json())
        raise Exception(f"Error in API request: {response.status_code}")

    results = response.json().get("items", [])
    logger.debug(f"Found {len(results)} search results")

    def get_page_content(url: str) -> str:
        """
        Extract text content from a web page.
        
        Args:
            url (str): The URL to fetch content from.
        
        Returns:
            str: The extracted text content, limited to max_chars characters.
                 Returns empty string if fetching fails.
        """
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
            logger.warning(f"Error fetching {url}: {str(e)}")
            return ""

    enriched_results = []
    for item in results:
        body = get_page_content(item["link"])
        enriched_results.append(
            {"title": item["title"], "link": item["link"], "snippet": item["snippet"], "body": body}
        )
        time.sleep(1)  # Be respectful to the servers

    logger.info(f"Successfully processed {len(enriched_results)} search results")
    return enriched_results


def analyze_stock(ticker: str) -> dict:
    """
    Perform comprehensive stock analysis and generate visualization.
    
    This function retrieves historical stock data, calculates various financial
    metrics, and generates a comprehensive analysis including price trends,
    moving averages, volatility, and key financial ratios. It also creates
    a visualization plot saved to disk.
    
    Args:
        ticker (str): The stock ticker symbol to analyze (e.g., "AAPL", "MSFT").
    
    Returns:
        dict: A dictionary containing comprehensive stock analysis data including:
              - ticker (str): The analyzed ticker symbol
              - current_price (float): Current stock price
              - 52_week_high (float): 52-week high price
              - 52_week_low (float): 52-week low price
              - 50_day_ma (float): 50-day moving average
              - 200_day_ma (float): 200-day moving average
              - ytd_price_change (float): Year-to-date price change
              - ytd_percent_change (float): Year-to-date percentage change
              - trend (str): Price trend ("Upward", "Downward", "Neutral")
              - volatility (float): Annualized volatility
              - pe_ratio (float): Price-to-Earnings ratio
              - pb_ratio (float): Price-to-Book ratio
              - book_value (float): Book value per share
              - operating_margin (float): Operating margin
              - total_debt (float): Debt-to-equity ratio
              - free_cash_flow (float): Free cash flow
              - return_on_capital (float): Return on capital
              - return_on_equity (float): Return on equity
              - plot_file_path (str): Path to the generated plot image
    
    Raises:
        Exception: If there's an error retrieving stock data or generating the plot.
    
    Example:
        >>> analysis = analyze_stock("AAPL")
        >>> print(f"Current price: ${analysis['current_price']:.2f}")
        >>> print(f"P/E ratio: {analysis['pe_ratio']:.2f}")
        Current price: $150.25
        P/E ratio: 25.30
    
    Note:
        The function generates a plot saved as "{ticker}_stockprice.png" in the
        "coding" directory, showing price history with 50-day and 200-day
        moving averages, along with a table of financial metrics.
    """
    logger.info(f"Starting stock analysis for ticker: {ticker}")

    stock = yf.Ticker(ticker)

    # Get historical data (1 year of data to ensure we have enough for 200-day MA)
    end_date = datetime.now(timezone("UTC"))
    start_date = end_date - timedelta(days=365)
    hist = stock.history(start=start_date, end=end_date)

    # Ensure we have data
    if hist.empty:
        logger.error(f"No historical data available for ticker: {ticker}")
        return {"error": "No historical data available for the specified ticker."}
    
    logger.debug(f"Retrieved {len(hist)} days of historical data for {ticker}")
    
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

    logger.debug(f"Calculated financial metrics for {ticker}")

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
    logger.debug(f"Generating stock price plot for {ticker}")
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
    logger.info(f"Stock analysis plot saved as {plot_file_path}")
    result["plot_file_path"] = plot_file_path

    logger.info(f"Stock analysis completed successfully for {ticker}")
    return result
