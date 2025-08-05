# Company Research AI Agent System

A sophisticated multi-agent AI system for comprehensive financial analysis and investment research using value investing principles.

## 🧠 Theory Behind the AI Agents

### Multi-Agent Architecture
This system implements a **collaborative multi-agent architecture** where specialized AI agents work together to perform complex financial analysis tasks. Each agent has a specific role and expertise:

#### 1. **Stock Analysis Agent**
- **Purpose**: Performs technical and fundamental analysis
- **Expertise**: Financial metrics, price trends, volatility analysis
- **Tools**: Yahoo Finance API, statistical analysis
- **Output**: Comprehensive stock data and metrics

#### 2. **Google Search Agent**
- **Purpose**: Gathers latest market news and company information
- **Expertise**: Web scraping, content extraction, news aggregation
- **Tools**: Google Custom Search API, BeautifulSoup
- **Output**: Recent news, market sentiment, company updates

#### 3. **Report Generation Agent**
- **Purpose**: Synthesizes information and provides investment recommendations
- **Expertise**: Value investing principles, financial analysis
- **Tools**: OpenAI GPT-4, structured analysis
- **Output**: Investment recommendations and comprehensive reports

### Value Investing Framework
The system is built on **Benjamin Graham and Warren Buffett's value investing principles**:

- **Intrinsic Value Analysis**: DCF and earnings power valuation
- **Margin of Safety**: Target 20-40% safety buffer
- **Financial Ratios**: P/E, ROE, Debt-to-Equity, EPS growth
- **Economic Moat**: Competitive advantages and long-term stability
- **Conservative Assumptions**: Risk-averse analysis approach

## 🏗️ How It Works

### 1. **Agent Initialization**
```python
# Create specialized agents with specific roles
search_agent = AssistantAgent(name="Google_Search_Agent", ...)
stock_analysis_agent = AssistantAgent(name="Stock_Analysis_Agent", ...)
report_agent = AssistantAgent(name="Report_Agent", ...)
```

### 2. **Tool Integration**
Each agent has access to specialized tools:
- **Stock Analysis Tool**: Fetches financial data and generates charts
- **Google Search Tool**: Retrieves latest news and market information

### 3. **Collaborative Workflow**
```python
# Agents work in a round-robin fashion
team = RoundRobinGroupChat([stock_analysis_agent, search_agent, report_agent])
```

### 4. **Analysis Output**
- Captures all agent interactions
- Displays real-time analysis results
- Provides comprehensive financial insights

## 🛠️ Frameworks & Technologies Used

### Core AI Framework
- **AutoGen**: Microsoft's multi-agent conversation framework
- **OpenAI GPT-4**: Advanced language model for analysis
- **Function Calling**: Tool integration for external APIs

### Data & Analysis
- **yfinance**: Yahoo Finance data extraction
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib**: Financial chart generation

### Web Scraping & APIs
- **Google Custom Search API**: News and information retrieval
- **BeautifulSoup**: Web content parsing
- **requests**: HTTP client for API calls

### Development & Deployment
- **Python 3.8+**: Core programming language
- **asyncio**: Asynchronous programming for concurrent operations
- **logging**: Structured logging with YAML configuration
- **Make**: Build automation and task management


## 🚀 Quick Start Guide

### Prerequisites
```bash
# Python 3.8 or higher
python3 --version

# Virtual environment (recommended)
python3 -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd company_research
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
# Create .env file in src/company_research/
cp .env.example .env

# Add your API keys
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id
```

### Running the Application

#### Method 1: Using Make (Recommended)
```bash
make run
```

#### Method 2: Direct Python execution
```bash
python3 src/company_research/main.py
```

#### Method 3: Custom stock analysis
```python
from src.company_research.main import CompanyResearch
import asyncio

async def analyze_stock(ticker):
    research = CompanyResearch()
    await research.execute(ticker)

# Run analysis
asyncio.run(analyze_stock("AAPL"))
```

## 📊 Output & Analysis

### Real-Time Analysis
The system provides real-time analysis output including:

- **Stock Data**: Current price, P/E ratio, market cap
- **Technical Analysis**: Moving averages, volatility, trends
- **Fundamental Metrics**: ROE, debt ratios, cash flow
- **Market News**: Latest company and industry news
- **Financial Charts**: Price history and technical indicators

### Analysis Structure
- **Stock Analysis**: Technical and fundamental metrics
- **Market Research**: Latest news and market sentiment
- **Financial Insights**: Comprehensive financial analysis

## 🔧 Configuration

### Logging Configuration
```yaml
# logger/logging_config.yaml
version: 1
formatters:
  simple:
    format: '%(asctime)s [%(threadName)-15s] %(levelname)-8s - %(message)s'
handlers:
  console:
    class: logging.StreamHandler
    formatter: simple
  file:
    class: logging.handlers.RotatingFileHandler
    filename: messaging.log
    maxBytes: 10485760
    backupCount: 20
```

### Environment Variables
```bash
# Required
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
GOOGLE_SEARCH_ENGINE_ID=...

# Optional
LOG_LEVEL=INFO
MODEL_NAME=gpt-4o
```

## 🧪 Testing

### Run Tests
```bash
# Unit tests
make unit

# Integration tests
make test-integration

# All tests
make test-all
```

### Code Quality
```bash
# Linting
make lint

# Formatting
make format

# Coverage
make coverage
```

## 📈 Example Usage

### Basic Stock Analysis
```bash
# Analyze ITC.NS (Indian Tobacco Company)
make run
```

### Custom Analysis
```python
from src.company_research.main import CompanyResearch
import asyncio

async def main():
    research = CompanyResearch(model_name="gpt-4o")
    
    # Analyze multiple stocks
    stocks = ["AAPL", "MSFT", "GOOGL"]
    for stock in stocks:
        await research.execute(stock)

asyncio.run(main())
```

## 🔍 Understanding the Output

### Stock Analysis Metrics
- **Current Price**: Real-time stock price
- **P/E Ratio**: Price-to-Earnings ratio
- **P/B Ratio**: Price-to-Book ratio
- **Debt-to-Equity**: Financial leverage
- **ROE**: Return on Equity
- **Free Cash Flow**: Operating cash flow
- **Volatility**: Price volatility analysis
- **Moving Averages**: 50-day and 200-day trends

### Investment Recommendations
The system provides three types of recommendations:
- **BUY**: Strong value proposition
- **HOLD**: Neutral position
- **AVOID**: Poor value or high risk

## 🛡️ Error Handling & Logging

### Logging Levels
- **INFO**: General application flow
- **DEBUG**: Detailed debugging information
- **WARNING**: Non-critical issues
- **ERROR**: Critical errors

### Common Issues & Solutions

#### 1. API Key Errors
```bash
# Check environment variables
echo $OPENAI_API_KEY
echo $GOOGLE_API_KEY
```

#### 2. Matplotlib Backend Issues
```python
# Fixed in utils.py
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
```

#### 3. Network Connectivity
```bash
# Check internet connection
ping api.openai.com
ping finance.yahoo.com
```

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
make dev-install

# Set up pre-commit hooks
pre-commit install
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Add comprehensive docstrings
- Write unit tests for new features

## 📚 Advanced Features

### Custom Agent Development
```python
# Create custom agent
custom_agent = AssistantAgent(
    name="Custom_Analysis_Agent",
    model_client=model_client,
    tools=[custom_tool],
    system_message="Your custom system message"
)
```

### Tool Integration
```python
# Add custom tools
custom_tool = FunctionTool(
    custom_function,
    description="Custom tool description"
)
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **AutoGen Team**: Multi-agent conversation framework
- **OpenAI**: Advanced language models
- **Yahoo Finance**: Financial data API
- **Google**: Search API for market research

## 📞 Support

For issues and questions:
1. Check the [Issues](https://github.com/your-repo/issues) page
2. Review the logging output for error details
3. Ensure all environment variables are set correctly

---

*This system is designed for educational and research purposes. Always consult with qualified financial advisors before making investment decisions.*