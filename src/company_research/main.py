# Standard library imports
import os
import time
from datetime import datetime, timedelta

# Third-party imports
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from pytz import timezone  # type: ignore
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Project imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient

from utils import google_search, analyze_stock

load_dotenv()

class CompanyResearch:
    def __init__(self, model_name="gpt-4o"):
        self.model_client = OpenAIChatCompletionClient(model=model_name)

    def get_tools(self):
        google_search_tool = FunctionTool(
            google_search, description="Search Google for information, returns results with a snippet and body content"
        )
        stock_analysis_tool = FunctionTool(
            analyze_stock, description="Analyze stock data and generate a plot"
        )
        return google_search_tool, stock_analysis_tool

    def get_agents(self, google_search_tool, stock_analysis_tool):
        search_agent = AssistantAgent(
            name="Google_Search_Agent",
            model_client=self.model_client,
            tools=[google_search_tool],
            description="Search Google for information, returns top 2 results with a snippet and body content",
            system_message="You are a helpful AI assistant. Solve tasks using your tools.",
        )

        stock_analysis_agent = AssistantAgent(
            name="Stock_Analysis_Agent",
            model_client=self.model_client,
            tools=[stock_analysis_tool],
            description="Analyze stock data and generate a plot",
            system_message="Perform data analysis using your tools and provide the latest news about the company.",
        )

        report_agent = AssistantAgent(
            name="Report_Agent",
            model_client=self.model_client,
            description="Generate a report based the search and results of stock analysis",
            system_message=(
                "You are a value investing analyst trained in the style of Benjamin Graham and Warren Buffett. "
                "Use The Intelligent Investor principles to evaluate companies based on: "
                "- Intrinsic value (DCF or earnings power) "
                "- Margin of safety (target 20–40%) "
                "- Financial ratios (P/E, ROE, Debt-to-Equity, EPS growth) "
                "- Economic moat and competitive advantages "
                "- Long-term stability and conservative assumptions. "
                "- Respond with a recommendation: BUY, HOLD, or AVOID."
                "Show the following information of the mentioned stock: "
                "- stock_price (float): Mention the current stock price "
                "- p_e_ratio (float): Mention the P/E ratio of the stock "
                "- p_b_ratio (float): Mention the P/B ratio of the stock "
                "- debt_to_equity_ratio (float): Mention the debt to equity ratio of the stock "
                "- return_on_equity (float): Mention the return on equity of the stock "
                "- free_cash_flow (float): The free cash flow of the stock "
                "- analyst_recommendation (string): The analyst recommendation for the stock "
                "- company_info (string): The company information "
                "- company_news (string): The company's latest news "
                "Always include sources. Use tables to display data. "
                "When you are done with generating the report, reply with TERMINATE."
            )
            # system_message="You are a helpful assistant that can generate a comprehensive report on a given topic based on search and stock analysis and the financial_analyst agent. When you done with generating the report, reply with TERMINATE.",
        )
        return search_agent, stock_analysis_agent, report_agent

    async def execute(self, stock_name: str) -> None:
        google_search_tool, stock_analysis_tool = self.get_tools()
        search_agent, stock_analysis_agent, report_agent = self.get_agents(google_search_tool, stock_analysis_tool)
        team = RoundRobinGroupChat(
            [stock_analysis_agent, search_agent, report_agent],
            max_turns=3
        )

        stream = team.run_stream(task=f"Write a financial report on {stock_name} using value investing principles and summarize the latest news from other agents output. ")

        final_report = ""
        await Console(stream)  # Display output to console

        await self.model_client.close()

if __name__ == "__main__":
    import asyncio
    stock_name = "SUNPHARMA.NS"
    research = CompanyResearch()
    asyncio.run(research.execute(stock_name))
