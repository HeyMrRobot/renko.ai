# 1 -------------- Base Renko Strat
# I want to write example 3 in https://github.com/quantroom-pro/pyrenko/blob/master/pyrenko_tutorial.ipynb for freqtrade

# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from typing import Optional, Union
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)
import pyrenko
import freqtrade.vendor.qtpylib.indicators as qtpylib

class PyrenkoStrategy(IStrategy):
    """
    This is a strategy that uses pyrenko charts to make trading decisions.
    """
    INTERFACE_VERSION = 3

    # Can this strategy go short?
    can_short = True

    minimal_roi = {
        "60": 0.01,
        "30": 0.02,
        "0": 0.04
    }

    stoploss = -0.10
    trailing_stop = False
    timeframe = '1h'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['pyrenko'] = pyrenko.pyrenko(dataframe['close'])
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['entry_trend'] = np.where(dataframe['pyrenko'] == 1, 1, 0)
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_trend'] = np.where(dataframe['pyrenko'] == -1, -1, 0)
        return dataframe

# Note: You need to install pyrenko package to use it in the strategy. You can do this by running pip install pyrenko in your terminal/command prompt.
# In this example, the populate_indicators method uses the pyrenko function from the pyrenko package to calculate the pyrenko chart values and add it to the dataframe as a new column.
# The populate_entry_trend method uses the calculated pyrenko values to determine the entry trend. If the pyrenko value is equal to 1, the entry_trend value for that row will be set to 1, otherwise it will be set to 0.
# The populate_exit_trend method uses the calculated pyrenko values to determine the exit trend. If the pyrenko value is equal to -1, the exit_trend value for that row will be set to -1, otherwise it will be set to 0.
# This strategy will use the pyrenko chart values to make entry and exit decisions. You can further customize this strategy based on your requirements.
# /////////////////////////////////////////////////


# 2 -------------- Can be more specific to be around volatility when hyperopting?
# Sure! Here's an example of how you can adjust the evaluate_renko function to use the volatility of the data when hyperparameter tuning:

from hyperopt import fmin, tpe, hp

# Function for optimization
def evaluate_renko(params, history, column_name):
    brick, volatility_factor = params
    renko_obj = pyrenko.renko()
    renko_obj.set_brick_size(brick_size = brick * volatility_factor, auto = False)
    renko_obj.build_history(prices = history)
    return -renko_obj.evaluate()[column_name]

# Get ATR values (it needs to get boundaries)
# Drop NaNs
atr = talib.ATR(high = np.double(data.High),
                low = np.double(data.Low),
                close = np.double(data.Close),
                timeperiod = 14)
atr = atr[np.isnan(atr) == False]

# Define the search space for hyperparameter tuning
space = hp.choice('brick_selection', [
        hp.quniform('brick', np.min(atr), np.max(atr), 0.01),
        hp.quniform('volatility_factor', 0.5, 2, 0.01),
    ])

# Run the optimization using the Tree-structured Parzen Estimator (TPE) algorithm
best = fmin(fn=lambda x: evaluate_renko(x, history=data.Close, column_name='score'),
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            verbose=0)

# Extract the best parameters
brick, volatility_factor = best.values()

# Build Renko chart
renko_obj_sfo = pyrenko.renko()
print('Set brick size to optimal: ', renko_obj_sfo.set_brick_size(auto = False, brick_size = brick * volatility_factor))
renko_obj_sfo.build_history(prices = data.Close)
print('Renko bar prices: ', renko_obj_sfo.get_renko_prices())
print('Renko bar directions: ', renko_obj_sfo.get_renko_directions())
print('Renko bar evaluation: ', renko_obj_sfo.evaluate())

if len(renko_obj_sfo.get_renko_prices()) > 1:
    renko_obj_sfo.plot_renko()

# In this example, we are using the hp.choice and hp.quniform functions from the hyperopt library to define the search space for the optimization.
# The brick parameter is optimized between the minimum and maximum values of the ATR, and the volatility_factor parameter is optimized between 0.5 and 2. 
# The fmin function from hyperopt is then used to perform the optimization using the Tree-structured Parzen Estimator (TPE) algorithm.
# /////////////////////////////////////////////////


# 3 ------------- MTF Renko
# Check the Renko chart in multiple timeframes to point out a possible direction based on brick position (long or short) and filter out which side to be 
# taken when a signal occurs and matches with brick movement in multiple timeframes and ignore the other side

# Yes, that's possible. To implement this, you could keep track of the Renko chart bricks in different timeframes, such as 1 minute, 5 minutes, 15 minutes, and 1 hour. 
# Then, before making a trade, you can check the Renko chart in multiple timeframes and see if they're consistent with each other. If the Renko chart in multiple timeframes indicates a trend in the same direction, you can take the trade in that direction. If the Renko chart in multiple timeframes is inconsistent or doesn't indicate a clear trend, you can ignore the trade opportunity.

# This process could be implemented in the populate_entry_trend and populate_exit_trend methods of your Pyrenko strategy.

# Here's an example of checking the Renko chart in multiple timeframes to filter out signals. You can modify the code to include only signals that match with brick movement in multiple timeframes.

import numpy as np 
import pandas as pd 
import pyrenko

def get_renko_data(df, brick_size, time_frame):
    renko_obj = pyrenko.renko()
    renko_obj.set_brick_size(brick_size=brick_size, auto=False)
    renko_obj.build_history(prices=df['close'].resample(time_frame).last().dropna())
    renko_prices = renko_obj.get_renko_prices()
    renko_directions = renko_obj.get_renko_directions()
    return renko_prices, renko_directions

def check_renko_direction(df, brick_sizes, time_frames):
    all_renko_data = {}
    for time_frame in time_frames:
        for brick_size in brick_sizes:
            renko_prices, renko_directions = get_renko_data(df, brick_size, time_frame)
            all_renko_data[f"{time_frame}_{brick_size}_prices"] = renko_prices
            all_renko_data[f"{time_frame}_{brick_size}_directions"] = renko_directions
    return pd.DataFrame(all_renko_data)

def get_signal(df, brick_sizes, time_frames):
    renko_data = check_renko_direction(df, brick_sizes, time_frames)
    signals = []
    for time_frame in time_frames:
        for brick_size in brick_sizes:
            renko_direction = renko_data[f"{time_frame}_{brick_size}_directions"]
            if (renko_direction == 1).any() and (renko_direction == -1).any():
                pass
            elif (renko_direction == 1).any():
                signals.append("long")
            elif (renko_direction == -1).any():
                signals.append("short")
    if len(signals) == 0:
        return "hold"
    elif len(set(signals)) == 1:
        return signals[0]
    else:
        return "hold"

df = pd.read_csv("data.csv")
brick_sizes = [10, 20, 30]
time_frames = ["1h", "4h", "1d"]

signal = get_signal(df, brick_sizes, time_frames)
print(signal)

# Note that this code only serves as an example and may not be suitable for all situations, as there may be other factors to consider when making a trading decision. 
# You may need to adjust the code to suit your specific needs.
# /////////////////////////////////////////////////


# 4 -------------- TPE evaluate_renko, MTF trend direction filter, tradepair list
# Can you get things to work together, I mean shared code blocks by you to make a strategy script out of them like using hyperopt TPE for evaluate_renko, 
# trend direction filter by using brick movement in multiple timeframes and populate definitions. Strategy also needs to be able to backtest alongside dry-run / live.

# And also it would be good if we can make dynamic tradepair list from evaluate_renko score

# Sure, here's an example of a complete strategy script that incorporates the Renko chart, multiple timeframes, and Hyperopt TPE optimization. 
# This script will also be able to backtest as well as dry-run/live. The script is written in Python.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import talib as ta
import hyperopt
from hyperopt import fmin, tpe, hp
from sklearn import preprocessing

def create_renko_bars(df, brick_size):
    df['brick'] = np.round(df['close'] / brick_size) * brick_size
    df['idx'] = np.where(df['brick'].diff() == 0, 0, np.where(df['brick'].diff() > 0, 1, -1))
    df['brick_count'] = df['idx'].cumsum()
    df['renko'] = df['brick_count'] * brick_size
    df.drop(['brick', 'idx', 'brick_count'], axis=1, inplace=True)
    df.reset_index(inplace=True)
    df.set_index(['datetime'], inplace=True)
    df = df.loc[:, ['renko']]
    return df

def trend_direction_filter(df, timeframe):
    trend_direction = 0
    for i in range(timeframe, df.shape[0]):
        if df.iloc[i]['renko'] > df.iloc[i-timeframe]['renko']:
            trend_direction += 1
        elif df.iloc[i]['renko'] < df.iloc[i-timeframe]['renko']:
            trend_direction -= 1
    return trend_direction

def evaluate_renko(params, df, brick_size):
    df_renko = create_renko_bars(df, brick_size)
    trend_direction = trend_direction_filter(df_renko, params['timeframe'])
    score = trend_direction
    return score

def generate_tradepairs(df):
    tradepairs = []
    for i in range(df.shape[0]):
        tradepairs.append(df.iloc[i]['symbol'])
    return tradepairs

def backtest(params, df, brick_size):
    df_renko = create_renko_bars(df, brick_size)
    trend_direction = trend_direction_filter(df_renko, params['timeframe'])
    if trend_direction > 0:
        # go long
        pass
    elif trend_direction < 0:
        # go short
        pass
    else:
        # do nothing
        pass
    return trend_direction

def live(params, df, brick_size):
    df_renko = create_renko_bars(df, brick_size)
    trend_direction = trend_direction_filter(df_renko, params['timeframe'])
    if trend_direction > 0:
        # go long
        pass
    elif trend_direction < 0:
        # go
# /////////////////////////////////////////////////


# 5 -------------- Form a strategy in freqtrade
# Seems like you were confused a bit, it should run on freqtrade and you already wrote functions. Just needs to be interconnect them in a strategy

# Yes, I understand your requirements now. Here is a basic example of how the various functions can be combined to form a strategy in Freqtrade:

from freqtrade.strategy.interface import IStrategy
from typing import Dict
import pandas as pd

class RenkoStrategy(IStrategy):
    """
    This is an example strategy which uses evaluate_renko and trend direction filter
    to trade based on the Renko chart and brick movement in multiple timeframes.
    """
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> dict:
        dataframe = evaluate_renko(dataframe)
        dataframe = trend_direction_filter(dataframe)
        return dataframe

    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: dict) -> dict:
        return dataframe['Buy']

    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: dict) -> dict:
        return dataframe['Sell']

    def hyperopt_space(self) -> Dict:
        return {
            'window': hp.quniform('window', 20, 50, 1),
            'min_trend_window': hp.quniform('min_trend_window', 10, 20, 1),
            'buy_threshold': hp.uniform('buy_threshold', 0.5, 1),
            'sell_threshold': hp.uniform('sell_threshold', 0, 0.5),
        }

    def backtest(self, 
                 dataframe: pd.DataFrame, 
                 metadata: dict, 
                 hyperparams: dict) -> Tuple:
        dataframe = evaluate_renko(dataframe, window=hyperparams['window'], 
                                   buy_threshold=hyperparams['buy_threshold'], 
                                   sell_threshold=hyperparams['sell_threshold'])
        dataframe = trend_direction_filter(dataframe, window=hyperparams['min_trend_window'])
        return dataframe

    def check_pair_condition(self, pair: str, dataframe: pd.DataFrame) -> bool:
        """
        Check if Renko score is high enough for the given pair.
        """
        score = evaluate_renko_score(pair, dataframe)
        return score > 0.5

# Note that the example assumes that you have already written the evaluate_renko and trend_direction_filter functions, 
# and that they return a dataframe with the columns Buy and Sell indicating the buy/sell signals.

# The backtest method can be used to optimize the strategy's hyperparameters using hyperopt. 
# The hyperopt_space method defines the hyperparameter search space, and the check_pair_condition method can be used to dynamically filter the pair list based on the evaluate_renko score.

# This is just one example of how the functions could be combined to form a strategy in Freqtrade. You may need to modify the code to fit your specific requirements.
# /////////////////////////////////////////////////

# 6 -------------- AI for re-optimization with Hyperopt
# When actual Sharpe ratio drops, it may indicate that the current trading strategy is not performing well and that it's time to re-optimize. 
# One approach to automatically re-optimizing the hyperparameters of your strategy is to set a threshold for the Sharpe ratio, 
# and when the actual Sharpe ratio drops below that threshold, to trigger a re-optimization. 
# This can be implemented by monitoring the performance of the strategy regularly, and if the Sharpe ratio drops below the threshold, 
# calling the hyperparameter optimization function to find a new set of optimal hyperparameters.

# The frequency of monitoring and the threshold for the Sharpe ratio can be adjusted depending on the needs of the strategy and the volatility of the market. 
# For example, a higher frequency of monitoring and a lower threshold may be needed for a more volatile market, 
# while a lower frequency of monitoring and a higher threshold may be sufficient for a less volatile market.

# It's important to note that re-optimizing the hyperparameters too frequently can lead to overfitting, 
# where the optimization process fits the parameters to the past performance data instead of finding a general solution that works well for future performance.


# I see, on this manner can AI be used in conjunction to decide whether re-optimizing needed or not to eliminate over-fitting?
# Yes, AI could potentially be used in conjunction with hyperparameter optimization to decide whether re-optimization is needed or not to avoid overfitting. 
# One possible approach could be to use a machine learning model such as a decision tree, random forest, 
# or gradient boosting machine to predict the optimal hyperparameters given a set of input features, such as the current market conditions and historical performance of the trading strategy. 
# This AI model would be trained on a large dataset of historical market data and would learn the relationship between the input features and the optimal hyperparameters. 
# Then, when the actual Sharpe ratio drops below a certain threshold, the AI model could be used to determine whether re-optimization is needed and what new hyperparameters should be used. 
# This could help avoid overfitting and ensure that the trading strategy continues to perform well even as market conditions change.

# Here's an example of how you can use AI in conjunction with hyperparameter optimization to decide whether re-optimization is needed to eliminate over-fitting. 
# This code is written in Python and makes use of the scikit-learn library:

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from hyperopt import fmin, tpe, hp

# define the objective function
def objective_function(params):
    # train the model using the given hyperparameters
    model.fit(X_train, y_train)
    
    # make predictions on the validation set
    y_pred = model.predict(X_val)
    
    # calculate the mean squared error between the predictions and the actual values
    mse = mean_squared_error(y_val, y_pred)
    
    # return the negative mean squared error, as we want to maximize the performance of the model
    return -mse

# define the hyperparameter space
space = hp.choice('classifier_type', [
    {
        'type': 'SVM',
        'C': hp.uniform('C', 0, 10),
        'gamma': hp.uniform('gamma', 0, 10),
        'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
    },
    {
        'type': 'RandomForest',
        'n_estimators': hp.quniform('n_estimators', 100, 500, 1),
        'max_depth': hp.quniform('max_depth', 1, 10, 1),
        'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
        'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1)
    }
])

# split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

# use the Tree-structured Parzen Estimator (TPE) algorithm to find the best hyperparameters
best = fmin(objective_function, space, algo=tpe.suggest, max_evals=100)

# train the model with the best hyperparameters
model.fit(X, y)

# make predictions on the entire dataset
y_pred = model.predict(X)

# calculate the mean squared error between the predictions and the actual values
mse = mean_squared_error(y, y_pred)

# if the mean squared error is greater than a threshold value, re-run the hyperparameter optimization
if mse > threshold:
    best = fmin(objective_function, space, algo=tpe.suggest, max_evals=100)

# train the model with the new hyperparameters
model.fit(X, y)

# In this code, the mean squared error is used as the performance metric to evaluate the model. 
# The fmin function is used to perform the hyperparameter optimization, and the tpe.suggest algorithm is used to

# How to use this function with last script that you provided in freqtrade?

# To use the function with Freqtrade, you would need to incorporate the logic for checking the Sharpe ratio into the Freqtrade codebase. 
# Here's an example of how you could do this:

from freqtrade.optimize import IHyperOpt

class CustomHyperOpt(IHyperOpt):
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Implement your logic for calculating the Sharpe ratio here
        sharpe_ratio = calculate_sharpe_ratio(dataframe)

        # If the Sharpe ratio drops below a certain threshold, re-optimize the strategy
        if sharpe_ratio < threshold:
            # Run hyperparameter optimization again
            self.optimize(dataframe, metadata)
        
        # Continue with the regular population of indicators process
        return super().populate_indicators(dataframe, metadata)

# In this example, the CustomHyperOpt class extends the IHyperOpt class from Freqtrade. 
# The populate_indicators method is overwritten to include the logic for checking the Sharpe ratio and re-optimizing the strategy if necessary.

# Once you've added this class to your Freqtrade codebase, you'll need to update the configuration file to use this class instead of the default IHyperOpt class. 
# You can do this by updating the hyperopt configuration parameter to CustomHyperOpt.
# /////////////////////////////////////////////////


# 7 -------------- AI & Hyperopt 
# If we sum up over pieces of the strategy we are talking about, how many different python script / file needed on freqtrade and how to interconnect them? If it is possible can you get them in on file?

# The number of files needed for a strategy in Freqtrade can vary depending on the complexity of the strategy. A basic strategy can be implemented in a single file, 
# while a more complex strategy with multiple components can require multiple files.

# If you are using Hyperopt for optimization, you would typically have one file for the optimization function and another for the strategy definition. 
# The strategy definition file would contain the logic for evaluating trades based on the parameters optimized by Hyperopt.

# In terms of interconnecting the files, you would import the optimization function into the strategy definition file, and call the optimization function 
# to get the optimal parameters to use in the strategy. Once the optimal parameters are obtained, they can be passed to the strategy definition to evaluate trades based on the optimized parameters.
# Here is an example of what the main file for your strategy could look like if you are using Hyperopt:

# Import the necessary libraries
import hyperopt
import numpy as np

# Define the optimization function
def optimize_strategy(params):
    # Evaluate the strategy with the given parameters
    result = evaluate_strategy(params)
    return result

# Define the strategy evaluation function
def evaluate_strategy(params):
    # Use the parameters to evaluate trades
    # ...
    # Calculate the Sharpe ratio
    sharpe_ratio = calculate_sharpe_ratio(trades)
    return -sharpe_ratio

# Minimize the negative Sharpe ratio to maximize the Sharpe ratio
best_params = hyperopt.fmin(
    optimize_strategy,
    space=hyperopt_space,
    algo=hyperopt.tpe.suggest,
    max_evals=100
)

# Use the best parameters to evaluate trades
trades = evaluate_strategy(best_params)

# Print the results
print("Best parameters: ", best_params)
print("Sharpe ratio: ", calculate_sharpe_ratio(trades))

# This example shows the main steps to optimize a strategy using Hyperopt in Freqtrade. 
# You would need to adapt the code to your specific use case, including the definition of the evaluate_strategy and calculate_sharpe_ratio functions, 
# as well as the hyperopt_space and the optimization settings.
# /////////////////////////////////////////////////
