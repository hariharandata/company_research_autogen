# Standard library imports
from datetime import datetime
import logging

# Third-party imports
from dotenv import load_dotenv
from datetime import datetime

# Project imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient

from utils import google_search, analyze_stock
from logger.log_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

load_dotenv()
today_str = datetime.today().strftime("%B %d, %Y")  #


class CompanyResearch:
    """
    A class for conducting comprehensive company research using AI agents.
    
    This class orchestrates multiple AI agents to perform financial analysis,
    news research, and generate investment reports for given stock tickers.
    
    Attributes:
        model_client: The OpenAI chat completion client for agent communication.
    """
    
    def __init__(self, model_name: str = "gpt-4o"):
        """
        Initialize the CompanyResearch class.
        
        Args:
            model_name (str): The OpenAI model to use for agent communication.
                             Defaults to "gpt-4o".
        """
        self.model_client = OpenAIChatCompletionClient(model=model_name)
        logger.info(f"Initialized CompanyResearch with model: {model_name}")

    def get_tools(self):
        """
        Create and configure the tools used by the AI agents.
        
        Returns:
            tuple: A tuple containing (google_search_tool, stock_analysis_tool)
                   where each tool is a FunctionTool instance.
        """
        logger.debug("Setting up tools for agents")
        google_search_tool = FunctionTool(
            google_search, description="Search Google for information, returns results with a snippet and body content"
        )
        stock_analysis_tool = FunctionTool(
            analyze_stock, description="Analyze stock data and generate a plot"
        )
        return google_search_tool, stock_analysis_tool

    def get_agents(self, google_search_tool, stock_analysis_tool):
        """
        Create and configure the AI agents for company research.
        
        Args:
            google_search_tool: FunctionTool for performing Google searches.
            stock_analysis_tool: FunctionTool for analyzing stock data.
        
        Returns:
            tuple: A tuple containing (search_agent, stock_analysis_agent, report_agent)
                   where each agent is an AssistantAgent instance.
        
        The agents created are:
        - Google_Search_Agent: Searches for latest company news and information
        - Stock_Analysis_Agent: Analyzes stock data and generates financial metrics
        - Report_Agent: Generates comprehensive investment reports using value investing principles
        """
        logger.debug("Creating agents for company research")
        search_agent = AssistantAgent(
            name="Google_Search_Agent",
            model_client=self.model_client,
            tools=[google_search_tool],
            description="Search Google for information, returns top 2 results with a snippet and body content",
            system_message=(
                f"You search Google for the company's latest news **as of {today_str}** "
                "using the attached tool and return the top 5 results with short summaries, including the date and source."
            )
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
        logger.info("Successfully created all agents")
        return search_agent, stock_analysis_agent, report_agent

    async def execute(self, stock_name: str) -> None:
        """
        Execute the complete company research workflow for a given stock.
        
        This method orchestrates the entire research process:
        1. Creates tools and agents
        2. Initializes a team chat with the agents
        3. Runs the research task
        4. Displays results to console
        5. Cleans up resources
        
        Args:
            stock_name (str): The stock ticker symbol to research (e.g., "AAPL", "ITC.NS").
        
        Returns:
            None
        
        Raises:
            Exception: If there's an error during the research process.
        """
        logger.info(f"Starting company research for stock: {stock_name}")
        google_search_tool, stock_analysis_tool = self.get_tools()
        search_agent, stock_analysis_agent, report_agent = self.get_agents(google_search_tool, stock_analysis_tool)
        team = RoundRobinGroupChat(
            [stock_analysis_agent, search_agent, report_agent],
            max_turns=3
        )

        logger.debug("Initializing team chat with agents")
        stream = team.run_stream(task=f"Write a financial report on {stock_name} using value investing principles and summarize the latest news from other agents output. ")

        await Console(stream)  # Display output to console

        logger.info("Company research completed successfully")
        await self.model_client.close()


if __name__ == "__main__":
    import asyncio
    stock_name = "Eli Lilly"
    logger.info("Starting company research application")
    research = CompanyResearch()
    asyncio.run(research.execute(stock_name))
