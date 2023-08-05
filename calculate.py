import numpy as np
import pandas as pd
import scipy.optimize as sco
import datetime

import warnings
warnings.filterwarnings("ignore")

from vnstock import *

def calculate_prices_df(stock_list, start_date, end_date):
    data = []
    for stock in stock_list:
        stock_df = stock_historical_data(symbol= stock, start_date = start_date, end_date = end_date)
        data.append(stock_df['Close'])
    
    prices_df = pd.DataFrame(data).T
    prices_df.columns = stock_list
    return prices_df

def calculate_returns_df(prices_df):
    returns_df = prices_df / prices_df.shift(1)
    returns_df = np.log(returns_df[1:])
    return returns_df

# calculate 
def calculate(w, mean_returns, cov_matrix, risk_free_rate):
        # Expected log return
        expected_Return = np.sum(mean_returns * w) * 250
        # Expected volatility
        expected_Volatility = np.sqrt(w.T @ cov_matrix @ w) * np.sqrt(250)
        # Sharpe Ratio
        sharpe_Ratio = (expected_Return - risk_free_rate)/ expected_Volatility
        return expected_Return, expected_Volatility, sharpe_Ratio

def simulated_portfolios(n_portfolios, n_stocks, mean_returns, cov_matrix, risk_free_rate):
    weight = np.zeros((n_portfolios, n_stocks))
    expected_Return = np.zeros(n_portfolios)
    expected_Volatility = np.zeros(n_portfolios)
    sharpe_Ratio = np.zeros(n_portfolios)

    for i in range(n_portfolios):
        # generate random weight vector
        w = np.array(np.random.random(n_stocks))
        w /= np.sum(w)
        weight[i] = w
        expected_Return[i], expected_Volatility[i], sharpe_Ratio[i] = calculate(w, mean_returns, cov_matrix, risk_free_rate)

    result_table = pd.concat([pd.Series(expected_Return), pd.Series(expected_Volatility), pd.Series(sharpe_Ratio)], axis=1)
    result_table.columns= ['Return', 'Volatility', 'Sharpe_Ratio']

    return result_table, weight, expected_Return, expected_Volatility, sharpe_Ratio

def calculate_max_sharpe_opt_allocation(n_stocks, mean_returns, cov_matrix, risk_free_rate):
    def negativeSR(w):
        w = np.array(w)
        R = np.sum(mean_returns * w) * 250
        V = np.sqrt(w @ cov_matrix @ w) * np.sqrt(250)
        SR = (R  - risk_free_rate)/ V
        return -SR

    w0 = [0.5] * n_stocks                                              # initial weight
    bounds = tuple([(0,1) for i in range(n_stocks)])
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    w_opt_sharpe = sco.minimize(fun= negativeSR,
                            x0= w0,
                            bounds= bounds,
                            constraints= constraints,
                            method= 'SLSQP')
    
    return w_opt_sharpe

def calculate_min_vol_opt_allocation(n_stocks,  cov_matrix):
    def minimize_Volatility(w):
        W = np.array(w)
        V = np.sqrt(w @ cov_matrix @ W) * np.sqrt(250)
        return V

    w0 = [0.5] * n_stocks                                              # initial weight
    bounds = tuple([(0,1) for i in range(n_stocks)])
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    w_opt_vol = sco.minimize(fun= minimize_Volatility,
                        x0= w0,
                        bounds= bounds,
                        constraints= constraints,
                        method= 'SLSQP')
    return w_opt_vol

def calculate_opt_allocation(n_stocks, mean_returns, cov_matrix, expected_Return, max_sharpe_index, min_volality_index):
    def minimize_Volatility(w):
        W = np.array(w)
        V = np.sqrt(W @ cov_matrix @ W) * np.sqrt(250)
        return V

    def get_Return(w):
        W = np.array(w)
        R = np.sum(mean_returns * W) * 250
        return R

    w0 = [0.5] * n_stocks                                              # initial weight
    bounds = tuple([(0,1) for i in range(n_stocks)])
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: get_Return(x) - R})

    simulate_returns = np.linspace(expected_Return[min_volality_index], expected_Return[max_sharpe_index], 50)
    volatility_opt = []

    for R in simulate_returns:
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                    {'type': 'eq', 'fun': lambda x: get_Return(x) - R})

        opt = sco.minimize(minimize_Volatility, 
                        w0,
                        method= 'SLSQP',
                        bounds= bounds,
                        constraints= constraints)

        volatility_opt.append(opt['fun'])

    return volatility_opt, simulate_returns