import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from cvxopt import matrix, solvers
import yfinance as yf
import os
import json
import matplotlib.pyplot as plt
import re

# OpenAI API Key
client = OpenAI(api_key="")  # insert your open api key here

# Function to fetch stock price data from Yahoo Finance
def fetch_stock_data(stocks, years):
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=years)
    
    # Fetch adjusted close prices
    stock_data = yf.download(stocks, start=start_date, end=end_date)["Adj Close"]
    return stock_data

# Function to check whether stock price data is available from Yahoo Finance
def check_tickers(tickers):
    """Check which tickers have historical price data available."""
    valid_tickers = []
    excluded_tickers = []

    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=years)
    
    for ticker in tickers:
        try:
            # Fetch data for each ticker
            data = yf.download(ticker, start=start_date, end=end_date)
            if not data.empty:
                valid_tickers.append(ticker)
            else:
                excluded_tickers.append(ticker)
        except Exception as e:
            excluded_tickers.append(ticker)

    return valid_tickers, excluded_tickers

# Function to fetch recent news from Yahoo Finance
def fetch_yahoo_news(stock):
    ticker = yf.Ticker(stock)
    return ticker.news

# Function to generate commentary using OpenAI
def generate_commentary(stock, news_articles):
    prompt = (
        f"Provide a detailed commentary on the recent news about {stock}. "
        "Consider the following headlines:\n"
    )
    for article in news_articles[:5]:
        prompt += f"- {article['title']}\n"
    prompt += (
        "Summarize whether the overall sentiment is positive, negative, or neutral, "
        "and what this means for the stock's future prospects.\n"
    )
    
    try:
        completion = client.chat.completions.create(
            #model="gpt-3.5-turbo",
            model="gpt-4o-mini",
            messages=[
                {"role": "system", 
                 #"content": "You are a financial analyst."}
                
                 "content": """You are a senior financial analyst providing concise, objective stock sentiment analysis. 
            Your response should:
            - Be data-driven and professional
            - Clearly state overall sentiment (Positive/Negative/Neutral)
            - Provide 3-5 key supporting reasons
            - Avoid speculation or emotional language
            - Use financial terminology appropriately
            - Conclude with a brief investment perspective"""}
                ,
                {"role": "user", 
                 "content": prompt},
            ],
            max_tokens=150,
            temperature=0.5,
            frequency_penalty=0.0,  # Reduces repetitive words
            presence_penalty=0.0,   # Encourages new topics
            n=1             # Number of response variations
            
        )
        commentary = completion.choices[0].message.content
    except Exception as e:
        commentary = f"Error generating commentary: {e}"
    
    return commentary


# Function to generate commentary on the portfolio
def generate_portfolio_commentary(investment_amount, risk_aversion, stocks, weights, annualized_return, sharpe_ratio,selected_model):
    """
    Use OpenAI's GPT model to generate commentary on a portfolio.

    Parameters:
    - investment_amount: Total investment amount in dollars.
    - risk_aversion: User's risk preference (e.g., "low", "medium", "high").
    - stocks: Selected stocks
    - weights: indicates what percentage of portfolio that a specific stock makes up.
    - annualized_return: Expected annual return of the portfolio.
    - portfolio_std_dev: Standard deviation (risk) of the portfolio in dollars.
    - sharpe_ratio: Sharpe ratio of the portfolio.

    Returns:
    - A string containing the portfolio commentary.
    """
    # Construct the prompt
    prompt = f"""
    You are a financial advisor. A portfolio has been generated based on the following metrics:
    - Investment Amount: ${investment_amount:,.2f}
    - Risk Appetite: {risk_aversion}
    - Stocks : {stocks}
    - weights : {weights}
    - Expected Annual Return:{expected_return:,.2f}
    - Sharpe Ratio: {sharpe_ratio:.2f}

    Provide a detailed and professional commentary on the portfolio. Include:
    1. An assessment of the expected return relative to the risk appetite.
    2. An analysis of the portfolio's risk and its implications.
    3. A comment on the efficiency of the portfolio based on the Sharpe ratio.
    4. Suggestions for improvements or validation that the portfolio aligns with the given risk preference.
    Ensure the commentary is easy to understand, engaging, and insightful.
    """
    try:
        completion = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", 
                 "content": """You are a senior financial analyst providing concise, objective portfolio analysis. 
            Your response should:
            - Be data-driven and professional
            - Provide a detailed breakdown of the portfolio's current composition
            - Assess the portfolio's performance relative to benchmark indices
            - Analyze risk characteristics based on the investor's risk tolerance
            - Explain investment rationale behind current allocation
            - Use clear, concise language
            - Limit to max 150 words 
            - Use financial terminology appropriately
            - Translate complex financial metrics into understandable language"""}
                ,
                {"role": "user", 
                 "content": prompt},
            ],
            max_tokens=300,
            temperature=0.5,
            frequency_penalty=0.0,  # Reduces repetitive words
            presence_penalty=0.0,   # Encourages new topics
            n=1 # Number of response variations
            
        )
        commentary = completion.choices[0].message.content
    except Exception as e:
        commentary = f"Error generating commentary: {e}"
    return commentary


# Function to calculate daily returns
def calculate_daily_returns(price_data):
    returns =  price_data.pct_change().dropna()
    #log_returns = np.log(1 + returns)
    #return log_returns
    return returns

# Function to calculate expected returns and covariance matrix
def calculate_metrics(returns):
    expected_returns = returns.mean()
    covariance_matrix = returns.cov()
    return expected_returns, covariance_matrix

# Function to optimize portfolio
def optimize_portfolio(expected_returns, covariance_matrix, risk_aversion):
    n = len(expected_returns)
    P = matrix(covariance_matrix.values)
    q = matrix(np.zeros(n))
    G = matrix(-np.eye(n))
    h = matrix(np.zeros(n))
    A = matrix(1.0, (1, n))
    b = matrix(1.0)
    
    sol = solvers.qp(risk_aversion * P, q, G, h, A, b)
    weights = np.array(sol['x']).flatten()
    portfolio_return = np.dot(weights, expected_returns.values)
    return weights, portfolio_return

# Function to allocate investment
def allocate_investment(weights, investment_amount):
    allocations = weights * investment_amount
    return allocations


def calculate_portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate = 0.02):
    """Calculate expected return, risk (standard deviation), and Sharpe ratio."""
    expected_return = np.dot(weights, mean_returns)
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_std_dev = np.sqrt(portfolio_variance) 

    # Annualize return using compounding
    annualized_expected_return = (1 + expected_return) ** 252 - 1
    
    # Annualize standard deviation
    trading_days = 252
    annualized_std_dev = portfolio_std_dev * np.sqrt(trading_days)
    
    # Calculate Sharpe Ratio (annualized)
    sharpe_ratio = (annualized_expected_return - risk_free_rate) / annualized_std_dev
    
    # calculate sharpe ratio    
    #sharpe_ratio = (expected_return - risk_free_rate) / portfolio_std_dev
    
    #return expected_return, portfolio_std_dev, sharpe_ratio
    return annualized_expected_return, annualized_std_dev, sharpe_ratio


def parse_user_input(user_input,selected_model):
    """
    Parse user input to extract investment details, risk appetite, and stock preferences.
    """
    prompt_messages = [
        {
             "role": "system",
             "content": """You are a financial input parser for portfolio optimization.

             Tasks:
             1.Extract investment amount, stock symbols or company names, risk appetite, time horizon
             2.Convert any company names to stock symbols if required.
             3.If user does not specify any stocks or companies, pick 3 to 5 bluechip stocks randomly from S&P 500.
             4.If the user specifies any specific sectors, pick stocks only from those sectors. 
             5.If user specifies the number of stocks to pick, limit the number of stocks to that. 
             6.Assume medium risk if not specified.
             7.Use the number of years specified for historical stock performance, if not use 5 years as the default range. 
             8.Indicate as True or False if the user provided incomplete information and any assumptions had to be made.

             Output:
             Return all extracted data as a JSON object with the following structure:
             {
                 "investment_amount": number,
                 "risk_appetite": string,
                 "stocks": string,
                 "num_stocks": number,
                 "historical_years": number,
                 "assumptions_flag":boolean
             }"""
         },
        {
            "role": "user",
            "content": f"""Parse this investment request and provide the results as JSON with these fields:
            - investment_amount: numerical value in dollars
            - risk_appetite: one of "Low", "Medium", or "High"
            - stocks: comma-separated list of stock symbols or names of companies
            - num_stocks: numerical value
            - historical_years: number of years of historical data to analyze (1-10)
            
            Input: {user_input}"""
        }
     ]

    try:
        response = client.chat.completions.create(
            model=selected_model,
            messages=prompt_messages,
            temperature=0.3,  # Lower temperature for more consistent parsing
            max_tokens=150,
            response_format={"type": "json_object"}  # Force JSON response
        )

        # Extract the content from the response
        parsed_data = json.loads(response.choices[0].message.content)
        
        # Format the response
        return {
            "investment_amount": float(parsed_data.get("investment_amount",0)),
            "risk_appetite": parsed_data.get("risk_appetite", "Medium"),
            "stocks": [s.strip() for s in parsed_data.get("stocks", "").split(",") if s.strip()],
            "num_stocks": int(parsed_data.get("num_stocks", 0)),
            "historical_years": int(parsed_data.get("historical_years", 0)),
            "assumptions_flag": int(parsed_data.get("assumptions_flag", False))
        }
    except Exception as e:
        print(f"Error parsing input: {str(e)}")
        return {
            "investment_amount": 10000,  # Default values
            "risk_appetite": "Medium",
            "stocks": ["AAPL", "MSFT", "GOOGL"],
            "num_stocks": 3,
            "historical_years": 5,
            "assumptions_flag":True
        }

def fill_missing_inputs(parsed_input):
    """
    Fill any missing inputs in the parsed data with default values.
    
    Args:
        parsed_data (dict): The parsed JSON data that may have missing values
        
    Returns:
        dict: Complete data with default values filled in for any missing fields
    """
    # Define default values
    defaults = {
        "investment_amount": 10000.0,  # Default $10,000 investment
        "risk_appetite": "Medium",     # Default medium risk
        "stocks": "AAPL,MSFT,GOOGL",  # Default blue-chip tech stocks
        "num_stocks": 3,              # Default number of stocks
        "historical_years": 5 
    }
    
    # Check each field and fill with defaults if missing or invalid
    filled_data = {}
    
    # Handle investment_amount
    try:
        amount = float(parsed_input.get("investment_amount", defaults["investment_amount"]))
        filled_data["investment_amount"] = amount if amount > 0 else defaults["investment_amount"]
    except (ValueError, TypeError):
        filled_data["investment_amount"] = defaults["investment_amount"]
    
    # Handle risk_appetite
    risk = parsed_input.get("risk_appetite", defaults["risk_appetite"])
    valid_risk_levels = ["Low", "Medium", "High"]
    filled_data["risk_appetite"] = risk if risk in valid_risk_levels else defaults["risk_appetite"]
    
    # Handle stocks
    stocks = parsed_input.get("stocks", defaults["stocks"])
    if isinstance(stocks, list):
        stocks = ",".join(stocks)
    if not stocks or not any(stocks.strip()):
        stocks = defaults["stocks"]
    filled_data["stocks"] = stocks
    
    # Handle num_stocks
    try:
        num = int(parsed_input.get("num_stocks", defaults["num_stocks"]))
        filled_data["num_stocks"] = num if num > 0 else defaults["num_stocks"]
    except (ValueError, TypeError):
        filled_data["num_stocks"] = defaults["num_stocks"]

    # Handle historical_years
    try:
        years = int(parsed_input.get("historical_years", defaults["historical_years"]))
        # Limit historical years to reasonable range (1-10 years)
        if years < 1:
            years = defaults["historical_years"]
        elif years > 10:
            years = 10
        filled_data["historical_years"] = years
    except (ValueError, TypeError):
        filled_data["historical_years"] = defaults["historical_years"]
    
    # Ensure num_stocks matches the actual number of stocks if stocks were provided
    stock_list = [s.strip() for s in filled_data["stocks"].split(",") if s.strip()]
    if filled_data["num_stocks"] > len(stock_list):
        # If requested more stocks than provided, add default stocks
        default_stock_list = defaults["stocks"].split(",")
        additional_stocks_needed = filled_data["num_stocks"] - len(stock_list)
        stock_list.extend(default_stock_list[:additional_stocks_needed])
        filled_data["stocks"] = ",".join(stock_list)
    elif filled_data["num_stocks"] < len(stock_list):
        # If requested fewer stocks than provided, truncate the list
        stock_list = stock_list[:filled_data["num_stocks"]]
        filled_data["stocks"] = ",".join(stock_list)
    
    # Convert stocks string back to list format if needed
    filled_data["stocks"] = [s.strip() for s in filled_data["stocks"].split(",") if s.strip()]
    
    return filled_data

def plot_portfolio(weights, tickers):
    """Plot pie chart of portfolio weights and bar chart of expected returns."""
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Pie chart for portfolio allocation
    ax[0].pie(weights, labels=tickers, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 14})
    ax[0].set_title('Portfolio Allocation')
    
    # Bar chart for weights
    ax[1].bar(tickers, weights * 100)  # Convert to percentage for better visualization
    ax[1].set_title('Portfolio Weights (%)')
    ax[1].set_ylabel('Weight (%)')
    
    plt.tight_layout()
    
    return fig



def validate_investment_description(user_input):
    """
    Validate the user's investment description input.
    
    Checks:
    1. Minimum meaningful length
    2. Presence of key financial terms
    3. Reasonable numeric values
    4. Meaningful content structure
    
    Returns:
    - Boolean: Whether input is valid
    - String: Error message or empty string
    """
    # Remove leading/trailing whitespace
    cleaned_input = user_input.strip()
    
    # Check for minimum input length
    if len(cleaned_input) < 20:
        return False, "Please provide a more detailed description of your investment goals."
    
    # Check for key financial markers
    financial_markers = [
       r'\$\d+',  # Dollar amount
       r'invest(ment)?', 
       r'(buy|purchase)',
       r'portfolio',
       r'(low|medium|high)-risk',
       r'(stock|stocks|etf|bonds?)',
       r'\d+(\.\d+)?%\s*(annual|historical)\s*return', # Historical performance range
       r'(average|historical)\s*return\s*of\s*\d+(\.\d+)?%', # Performance percentages
       r'(last|past)\s*\d+\s*(year|years)\s*of\s*(data|performance|analysis)', # Historical time range
       r'\d+\s*(year|years)\s*of\s*historical\s*(performance|data)' # Alternative time range pattern

       # Flexible stock symbols and company names
       r'\b[A-Z]{1,5}\b', # Stock tickers (uppercase, 1-5 characters)
       r'\b[A-Z][a-z]+(\s+[A-Z][a-z]+)?\b', # Company names (allow multi-word names)
            
       # Sector names
    r'\b(tech|technology|pharmaceutical|pharma|healthcare|biotech|finance|financial|banking|energy|oil|gas|renewable|automotive|tech|software|industrial|manufacturing|retail|consumer|agriculture|telecom|telecommunications)\b'
                    
    ]
    
    # Verify input contains at least two financial markers
    markers_found = sum(1 for marker in financial_markers if re.search(marker, cleaned_input, re.IGNORECASE))
    
    if markers_found < 2:
        return False, "Your description seems incomplete. Please include investment amount, risk appetite, stock tickers or company names etc."
    
    # Check for unreasonable numeric values
    try:
        # Extract potential investment amount
        amounts = re.findall(r'\$(\d+(?:,\d+)?)', cleaned_input)
        if amounts:
            amount = float(amounts[0].replace(',', ''))
            if amount < 100 or amount > 1000000 :
                return False, "Please enter a realistic investment amount between $100 and $1,000,000."
    except ValueError:
        return False, "There seems to be an issue with the numeric values in your description."
    
    return True, ""



    

 # Custom CSS for compact and right-aligned selector
    st.markdown("""
    <style>
    /* Reduce width of selectbox */
    .stSelectbox > div > div {
        width: 200px !important;
        float: right;
    }
    
    /* Adjust page layout to accommodate right-aligned selector */
    .block-container {
        padding-top: 1rem;
    }
    
    /* Add some subtle styling to model selector */
    .stSelectbox > label {
        font-size: 0.85rem;
        color: #666;
    }
    </style>
    """, unsafe_allow_html=True)


    




# Streamlit App
st.title("Portfolio Builder App")


# Add CSS for styling
st.markdown(
    """
    <style>
    .model-select {
        display: flex;
        justify-content: flex-end; /* Align to the right */
    }
    .stSelectbox {
        width: 200px !important; /* Adjust width to make it smaller */
    }
    </style>
    """,
    unsafe_allow_html=True
)
available_models = ["gpt-4o-mini","gpt-4o","gpt-3.5-turbo"]

# Model selection dropdown
st.markdown('<div class="model-select">', unsafe_allow_html=True)
selected_model = st.selectbox(
    "Select AI Model for Analysis",
    options=available_models,
    index=0,
    help="Choose LLM"
)
st.markdown('</div>', unsafe_allow_html=True)


# Inputs

# User input text area
user_input = st.text_area(
        "Describe your investment goals",
        "I want to invest $10,000 in a medium-risk portfolio of tech stocks like AAPL, MSFT, and GOOGL",
        help="Example: I want to invest $5000 in a low-risk portfolio of 5 blue-chip stocks"
)


# Process and display results when "Optimize Portfolio" is clicked
if st.button("Submit"):
    
    
    if not user_input:
        st.error("Please provide your portfolio requirements.")
    else:

        is_valid, error_message = validate_investment_description(user_input)
        
        if not is_valid:
            st.error(error_message)
        else:

            # Parse user input
            st.write("Parsing your input...")
            parsed_input = parse_user_input(user_input,selected_model)
    
            if parsed_input.get("assumptions_flag") == 1:
                st.warning("""
                The system has made some assumptions based on your input. If any of these assumptions do not align with your intent, please refine the question further.
                """)
    
            # Fill missing inputs
            filled_data = fill_missing_inputs(parsed_input)
            
            # Get values from the filled inputs
            investment_amount = filled_data["investment_amount"]
            risk_appetite = filled_data["risk_appetite"]
            stocks = filled_data["stocks"]
            num_stocks = filled_data["num_stocks"]
            years =  filled_data["historical_years"]
            
            # check whether a ticker has data
            valid_tickers, excluded_tickers = check_tickers(stocks)
        
            if excluded_tickers:
                st.warning(f"The following stocks were excluded due to lack of historical data: {', '.join(excluded_tickers)}")
            
            if not valid_tickers:
                st.error("No valid stocks available for optimization.")
                st.stop()
            
            # Risk aversion mapping
            risk_aversion_map = {"Low": 0.5, "Medium": 1.0, "High": 2.0}
            risk_aversion = risk_aversion_map[risk_appetite]
    
            formatted_stocks = ', '.join(valid_tickers)
    
            st.subheader("Investment Overview")
            
            #  display the inputs
            st.write(f"Investment Amount: ${investment_amount}")
            st.write(f"Risk Appetite: {risk_appetite}")
            st.write(f"Selected Stocks: {formatted_stocks}")
            #st.write(f"Number of Stocks: {num_stocks}")
            st.write(f"Look back period (years): {years}")
            
    
            # Fetch stock data
            st.info("Fetching stock data...")
            price_data = fetch_stock_data(valid_tickers, years)
    
            # Calculate daily returns
            returns = calculate_daily_returns(price_data)
    
            # Compute metrics
            expected_returns, covariance_matrix = calculate_metrics(returns)
    
            # Optimize portfolio
            weights, portfolio_expected_return = optimize_portfolio(expected_returns, covariance_matrix, risk_aversion)
    
            # Allocate investment
            allocations = allocate_investment(weights, investment_amount)
    
           
            st.subheader("Optimal Portfolio Weights and Investment Allocation")
            # Plotting the Portfolio Allocation and Weights
            fig = plot_portfolio(weights, valid_tickers)
            st.pyplot(fig)
    
             # Display results
           
            weights_df = pd.DataFrame({
                "Stock": valid_tickers,
                "Weight (%)": (weights * 100).round(2),
                "Allocation ($)": allocations.round(2)
            })
            #st.table(weights_df)
    
            st.dataframe(weights_df.style.set_properties(**{'text-align': 'left'}).format({
                "Weight (%)": "{:.2f}",
                "Allocation ($)": "${:,.2f}"
            }))
    
            st.subheader("Expected Portfolio Return")
            #st.write(f"**{portfolio_expected_return:.2%} per day**")
    
            
            # Annualized ROI
            annualized_return = (1 + portfolio_expected_return) ** 252 - 1
    
            # Monthly ROI
            monthly_return = (1 + portfolio_expected_return) ** 21 - 1
    
            st.write(f"**{annualized_return:.2%} annualized_return**")
    
            # calculate_portfolio_performance   
            expected_return, portfolio_std_dev, sharpe_ratio = calculate_portfolio_performance(weights,
                                                                                                   expected_returns.values,
                                                                                                   covariance_matrix)
    
    
            # Convert annualized return and risk to dollar terms
            expected_return_in_dollars = expected_return * investment_amount
            std_dev_in_dollars = portfolio_std_dev * investment_amount
    
            st.write(f"\n**Expected Annual Return:** ${expected_return_in_dollars:.2f}")
            #st.write(f"**Portfolio Risk (Standard Deviation):** ${std_dev_in_dollars:.2f}")
            st.write(f"**Sharpe Ratio:** {sharpe_ratio:.2f}")
    
    
            # Portfolio Commentary
            st.subheader("Portfolio Commentary")
            portfolio_commentary = generate_portfolio_commentary(investment_amount, risk_appetite, formatted_stocks, weights, annualized_return, sharpe_ratio,selected_model)
            st.write(portfolio_commentary)   
