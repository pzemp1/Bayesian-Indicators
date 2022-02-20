#Various Strategies will be formulated here

#First Strategy which was done, was the moving average cross over.
#Next I want to focus on Bollinger Bands

#New Bayesian Indicator
'''
Bayesian Indicator
1. Select Large Window [                ]
2. Select Small Window             [    ]

Large Window generates P(S(i)). Where S - {Buy, Sell}

Small Window generates P(S(i-1) | S(i)).
Transition probabilities.
'''
import numpy as np
import pandas as pd
from scipy.stats import norm
import plotly.graph_objects as go

def getData():
    CryptoData = []
    Variables = ["BNBUSDT", "ADAUSDT", "BTCUSDT", "ETHUSDT",
                 "NEOUSDT", "QTUMUSDT", "XRPUSDT", "LTCUSDT"]

    for x in Variables:
        path = 'CryptoMinuteData/Binance_' + x + "_1h.csv"
        data = pd.read_csv(path)
        CryptoData.append(data)

    return CryptoData, Variables

def SimpleBayesianIndicator(data, BW, SW):

    BigWindow = data[-BW:]
    SmallWindow = data[-SW:]
    transition_matrix = np.full((2,2),1/(SW+3))

    for i in range(1, SW):
        prev = SmallWindow[i-1]
        curr = SmallWindow[i]
        transition_matrix[prev,curr] += 1/(SW+3)
    #Note SmallWindow[-1] is the "current" value
    return BayesianProbability(transition_matrix, BigWindow, SmallWindow[-1])


# Will give you the probabilty for all cases.
# Evidence is e, so the previous most value.
def BayesianProbability(transition_matrix, BigWindow, e):
    Ones = np.count_nonzero(BigWindow == 1)/len(BigWindow)
    Zeroes = 1 - Ones
    H1 = Ones
    H2 = Zeroes
    E = transition_matrix[e,0]*H2 + transition_matrix[e,1]*H1
    Res1 = (transition_matrix[e,0]*H2)/E
    Res2 = (transition_matrix[e,1]*H1)/E
    lists = []

    if e == 1:
        Up = tuple([1,1])
        Down = tuple([1,0])
    else:
        Up = tuple([0,1])
        Down = tuple([0,0])

    lists.append(Signal)
    Results = {Down: Res1, Up: Res2}
    #Results ([Before, After] : Prob, [Before, After] : Prob)
    return Results

def BayesianSequence(Signals, Res, FrequencyTable):
    #print(Signals)
    Signals = np.flip(Signals) # Reverses list I think
    target = Signals[0]
    count = 1

    for i in range (1, len(Signals)):
        if Signals[i] == target:
            count += 1
        else:
            break
    frequency = count

    #Next we need to
    #We already now our "Res" values
    code = str()
    if target == 1:
        code = "Up"
    else:
        code = "Down"

    #Now lets compute the frequencies;
    total_frequencies = sum(FrequencyTable[code])
    #hypothesis 1 says the sequence will extend
    h1Likelihood = (FrequencyTable[code][frequency+1])/total_frequencies
    #hypothesis 2 says the sequence will not extend
    h2Likelihood = (FrequencyTable[code][frequency])/total_frequencies

    keys = list(Res.keys())
    Down = keys[0]
    Up = keys[1]
    #So our before is Down[0] and Up[0]
    #We need to sort it our on what the target is doing.
    #If target is 0, we need to look at sequences of 0

    if target == 0:
        #We know that Down key would be the one holding the prior
        h1prior = Res[Down]
        h2prior = Res[Up]
        Denominator = h1prior * h1Likelihood + h2prior*h2Likelihood
        Results = dict()
        h1posterior = h1prior * h1Likelihood / Denominator
        h2posterior = h2prior * h2Likelihood / Denominator
        Results[Down] = h1posterior
        Results[Up] = h2posterior

    elif target == 1: # if target is 1, we need to look at sequences of 1
        h1prior = Res[Up]
        h2prior = Res[Down]
        Denominator = h1prior * h1Likelihood + h2prior * h2Likelihood
        Results = dict()
        h1posterior = h1prior * h1Likelihood / Denominator
        h2posterior = h2prior * h2Likelihood / Denominator
        Results[Up] = h1posterior
        Results[Down] = h2posterior

    return Results

def Streaks(df):
    df['pct_change'] = df['close'].pct_change()
    decrease = df['pct_change'] < 0
    index = decrease.index[decrease != decrease.shift(1)]
    length = index[1:] - index[:-1]
    streak = pd.Series([decrease[0]] * len(length))
    streak.loc[streak.index%2 == 1] = ~decrease[0]
    indicator = (streak == True)
    streak.loc[indicator] = "Down"
    streak.loc[~indicator] = "Up"
    decrease = pd.DataFrame({"Streak" : streak, "Length" : length, "Count" : 1})
    decrease = decrease.groupby(["Streak", "Length"]).sum()

    return decrease

def reset_my_index(df):
    res = df[::-1].reset_index(drop=True)
    return (res)

if __name__ == "__main__":
    CryptoData, Variables = getData()
    #data = np.random.binomial(n=1, p=0.5, size=[30000])
    TotalCounter = 0
    CorrectCounter1 = 0
    CorrectCounter2 = 0

    #Simulate Data from Brownian motion starting from 10k and going to 50k
    #Approximately 40 iterations. Increment 1k each iteration
    Increment = 1000
    n = 10000
    datapoints = []
    UniqueUpStreaks = []
    UpStreaksCount = []
    UniqueDownStreaks = []
    DownStreaksCount = []


    for data in CryptoData:


        for i in range (0, 2000):

            Signal = np.where((data['pct_change'] > 0), 1, 0).copy()
            z = Streaks(data)
            #print(z)
            #print(sum(z['Count']['Down']))
            #print(sum(z['Count']['Up']))

            prediction = Signal[-1+i]
            Signal = Signal[:len(Signal)-1+i]

            res = SimpleBayesianIndicator(Signal, 20, 10)
            max_key = max(res, key=res.get)

            if max_key[1] == prediction:
                CorrectCounter1 += 1

            #x = list(res.keys())
            #bayesian_factor = res[x[0]]/res[x[1]]
            Signal = Signal[-20:]
            Res = BayesianSequence(Signal, res, z['Count'])
            max_key = max(Res, key=Res.get)

            if max_key[1] == prediction:
                CorrectCounter2 += 1

        #x = list(Res.keys())
        #bayesian_factor = Res[x[0]]/Res[x[1]]


    print(CorrectCounter1/7)
    print(CorrectCounter2/7)