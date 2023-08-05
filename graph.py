import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

def stock_price_trend_graph(prices_df, start_date, end_date):
    fig, ax = plt.subplots(figsize=(14, 7))
    for c in prices_df.columns:
        plt.plot(prices_df.index, prices_df[c], lw=3, alpha=0.8,label=c)
    plt.legend(loc='upper left', fontsize=12)
    plt.ylabel('price in VND')
    plt.title('Stock price from {} to {}'.format(start_date, end_date))
    return fig

def daily_returns_stock(returns_df):
    fig, ax = plt.subplots(figsize=(14, 7))
    for c in returns_df.columns:
        plt.plot(returns_df.index, returns_df[c], lw=3, alpha=0.8, label=c)
    plt.legend(loc='upper right', fontsize=12)
    plt.ylabel('daily returns')
    plt.title('Daily returns of Stocks')
    return fig 

def simulated_portfolio_graph(expected_Return, expected_Volatility, max_sharpe_index, min_volality_index, sharpe_Ratio):
    fig, ax = plt.subplots(figsize=(14, 7))
    plt.scatter(expected_Volatility, expected_Return, c= sharpe_Ratio, cmap='YlGnBu', marker='o', s=10, alpha=0.3)

    plt.colorbar()
    plt.scatter(expected_Volatility[max_sharpe_index], expected_Return[max_sharpe_index], marker='*', color='r', s=500, label='Maximum Sharpe ratio')
    plt.scatter(expected_Volatility[min_volality_index], expected_Return[min_volality_index], marker='*', color='g', s=500, label='Minimum volatility')

    plt.legend(labelspacing=0.8)
    plt.title('Simulated Portfolio')
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Annualised Log Return')
    return fig 

def portfolio_optimization_graph(returns_df, mean_returns, expected_Volatility, expected_Return, sharpe_Ratio, max_sharpe_opt_Volatility, max_sharpe_opt_Return, min_vol_opt_Volatility, min_vol_opt_Return, volatility_opt, simulate_returns):
    fig, ax = plt.subplots(figsize=(10, 7))
    vol = np.std(returns_df) * np.sqrt(250)
    rt = mean_returns * 250

    ax.scatter(vol, rt, marker='o', s=200)
    for i, txt in enumerate(returns_df.columns):
        ax.annotate(txt, (vol[i],rt[i]), xytext=(10,0), textcoords='offset points')

    plt.scatter(expected_Volatility, expected_Return, c= sharpe_Ratio, cmap='YlGnBu', marker='o', s=10, alpha=0.3)

    plt.colorbar()

    plt.scatter(max_sharpe_opt_Volatility, max_sharpe_opt_Return, marker='*', color='r', s=500, label='Optimal maximum Sharpe ratio')
    plt.scatter(min_vol_opt_Volatility, min_vol_opt_Return, marker='*', color='g', s=500, label='Optimal minimum volatility')

    plt.plot(volatility_opt, simulate_returns, linestyle='-.', color='black', label='efficient frontier')
    plt.legend(labelspacing=0.8)
    plt.title('Simulated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Annualised Log Return')

    return fig

def capital_allocation_line_graph(a, risk_free_rate, expected_Return, expected_Volatility, sharpe_Ratio, 
                                max_sharpe_opt_Return, max_sharpe_opt_Volatility, max_sharpe_opt_sharpe_Ratio,
                                min_vol_opt_Volatility, min_vol_opt_Return, 
                                volatility_opt, simulate_returns,):
    
    CAL_line = []
    CAL_line.append([0, risk_free_rate])
    CAL_line.append([max_sharpe_opt_Volatility, max_sharpe_opt_Volatility * max_sharpe_opt_sharpe_Ratio + risk_free_rate])
    CAL_line.append([max_sharpe_opt_Volatility * 1.2, max_sharpe_opt_Volatility * max_sharpe_opt_sharpe_Ratio * 1.2 + risk_free_rate])
    CAL_line = np.array(CAL_line)
    
    utility = []
    CAL_x = []
    CAL_y = []

    for er in np.linspace(risk_free_rate, max(expected_Return), 100):
        sd = (er - risk_free_rate)/(max_sharpe_opt_sharpe_Ratio)
        u = er - 0.5 * a * (sd ** 2)
        CAL_x.append(sd)
        CAL_y.append(er)
        utility.append(u)

    utility_index = np.argmax(utility)

    fig, ax = plt.subplots(figsize=(10, 7))
    plt.scatter(expected_Volatility, expected_Return, c= sharpe_Ratio, cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()

    plt.scatter(max_sharpe_opt_Volatility, max_sharpe_opt_Return, marker='*', color='r', s=500, label='Optimal maximum Sharpe ratio')
    ax.annotate('Optimal Maximum Sharpe ratio', (max_sharpe_opt_Volatility - 0.2, max_sharpe_opt_Return + 0.01), xytext=(10,0), textcoords='offset points')

    plt.scatter(min_vol_opt_Volatility, min_vol_opt_Return, marker='*', color='g', s=500, label='Optimal minimum volatility')
    ax.annotate('Optimal Minimum volatility', (min_vol_opt_Volatility, min_vol_opt_Return), xytext=(10,0), textcoords='offset points')

    plt.scatter(CAL_x[utility_index],CAL_y[utility_index], marker='*', color='gray', s=500, label="Investor's Optimal Portfolio")
    ax.annotate("Optimal Investor's Portfolio", (CAL_x[utility_index],CAL_y[utility_index]), xytext=(10,0), textcoords='offset points')

    plt.plot(CAL_line[:,0], CAL_line[:,1],"-", label='Capital Allocation Line')
    plt.plot(volatility_opt, simulate_returns, linestyle='-.', color='black', label='efficient frontier')
    plt.legend(loc= 0, labelspacing=0.8)
    plt.title('Optimal Portfolio Recommendation')
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Annualised Log Return')
    return fig