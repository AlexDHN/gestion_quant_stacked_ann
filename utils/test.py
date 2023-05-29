import numpy as np
from scipy.stats import skew, kurtosis
from scipy.stats import norm
from scipy.stats import chi2
import pandas as pd



def statistics(array):
    return np.mean(array), np.std(array), skew(array), kurtosis(array)


def kupiec_test(stock: pd.Series, volatility=None, confidence=0.025, chi_squared_pr=0.025, days=10, ret = False):
    """
    Check if the number of VaR excesses are align with the level of confidence chosen.
    H0: VaR model is accurate
    H1: VaR model is not accurate
    if pvalue > critical proba ==> accept H0
    https://www.youtube.com/watch?v=lc8q18FyZuU&t=1225s is a great explaination
    :param days: days to compute VaR
    :param confidence: VaR confidence
    :param stock: time series
    :param confidence_level: confidence level test statistic
    :return: kupiec test
    """
    n = len(stock) - 1
    # compute returns of the power sample size
    ret = stock.pct_change() if not ret else stock
    mu = np.mean(ret)
    vol = np.std(ret) if volatility is None else volatility
    # VCV VaR
    VCV = VAR(confidence, vol, mu, days)
    violation = ret < VCV.reshape(-1, 1)

    # number of violation
    pi_exp = confidence
    n0, n1 = n - sum(violation), sum(violation)
    pi_obs = n1 / n

    # kupiec statistic
    LR = pi_exp ** n1 * (1 - pi_exp) ** n0 / (pi_obs ** n1 * (1 - pi_obs) ** n0)
    kupiec_stat = -2 * np.log(LR)

    # if pvalue < 5% then unreliable confidence level
    pvalue = (1-chi2.cdf(kupiec_stat[0], 1))*100
    # if pvalue > chi squared pr it means that LR < critical value of the chi squared of the test so H0 validated
    return {'kupiec_stat':kupiec_stat, 'VaR_accpeted': pvalue > chi_squared_pr}


def interval_forecast_test(stock: pd.Series, confidence=0.01, confidence_level=0.01, days=10, vol=None):
    """
    Validate that VaR excesses are iid and in line with chosen confidence level: Christoffersen (1998)
    H0: VaR model is accurate
    H1: VaR model is not accurate
    if pvalue > critical proba ==> accept H0
    https://www.youtube.com/watch?v=lc8q18FyZuU&t=1225s is a great explatation
    :param days: days to compute VaR
    :param confidence: VaR confidence
    :param stock: time series
    :param confidence_level: confidence level test statistic
    :return: Christoffersen's test
    """
    n = len(stock) - 1
    # compute returns of the power sample size
    ret = stock.pct_change()
    mu = np.mean(ret)
    if vol is None:
        vol = np.std(ret)
    # VCV VaR
    VCV = VAR(confidence, vol, mu, days)
    violation = ret < VCV

    # number of violation
    n0, n1 = n - sum(violation), sum(violation)
    pi_obs = n1 / n

    n01, n10, n11 = find_ns(violation)
    n00 = n0 - n01
    pi_01 = n01 / (n00 + n01)
    pi_11 = n11 / (n11 + n10)
    # stat de test
    LR = (pi_obs ** n1 * (1 - pi_obs) ** n0) / (pi_01 ** n01 * (1 - pi_01) ** n00 * pi_11 ** n11 * (1 - pi_11) ** n10)
    christoffersen_stat = -2 * np.log(LR)

    # if pvalue < 5% then unreliable confidence level
    pvalue = 1 - chi2.cdf(christoffersen_stat, 1)
    # if pvalue > 0.05 it means that LR < critical value of the test so H0 validated
    return christoffersen_stat, pvalue > confidence_level, violation


def find_ns(response_violation: np.array):
    """
    allow to compute the number of 0 followed by 1, nb of 1 followed by 0, and the nb of 1 followed by 1
    :param response_violation: array of 1 and 0
    :return: n01, n10, n11
    """
    n01, n10, n11 = 0, 0, 0
    for result in range(len(response_violation) - 1):
        if response_violation[result] == 0 and response_violation[result + 1] == 1:
            n01 += 1
        elif response_violation[result] == 1 and response_violation[result + 1] == 0:
            n10 += 1
        elif response_violation[result] == 1 and response_violation[result + 1] == 1:
            n11 += 1
        else:
            pass  # n00 is calculated later
    return n01, n10, n11


def unconditional_expected_shortfall_test(stock, confidence=0.05, days=10, ret=False, nb_scenario=5000, vol=None, build_pval=False):
    """
    AC2 test: Implementation based on mathworks ressources which followed Acerbi, C., and B. Szekely. Backtesting Expected Shortfall
    H0: reject VaR and CVAR (Fredriksson & Johansson 2020, A comparative empirical evaluation of different backtests)
    :param ret: does the input stock is return or prices series ? False = Price series
    :param nb_scenario: nb scenario for the simualtion of pvalues
    :param stock: series, list
    :param confidence:
    :param days:
    :return: AC1 test
    """
    if ret:
        n = len(stock) - 1
        # compute returns of the power sample size
        ret = stock.pct_change()
    else:
        n = len(stock) # assume no na
        # compute returns of the power sample size
        ret = stock
    mu = np.mean(ret)
    if vol is None:
        vol = np.std(ret)
    # VCV VaR
    VCV = VAR(confidence, vol, mu, days).reshape(-1, 1)
    ES = CVAR(confidence, vol, mu, days).reshape(-1, 1)
    violation = ret < VCV

    # nominator: Rt * indicatrice(1 if Rt < -Var)
    nominator = ret * violation
    res = sum(nominator / ES)
    stat = res / (n * confidence) + 1
    if build_pval:
        return stat[0]
    simul_stat_test = simulation_pvalue(ret,vol, confidence, nb_scenario=nb_scenario) # should not work yet

    # mean breaches
    pval = sum(simul_stat_test <= stat) / nb_scenario
    critical_val = np.quantile(stat, 1-confidence)
    # reject test if Ptest (confidence) > Pval
    # if pval <= confidence we do not reject VaR & CVaR because we reject H0
    return {'pvalue':pval, 'critical_value':critical_val, "(C)VaR rejected":pval <= confidence}


def simulation_pvalue(retu, vol, conf, nb_scenario=500):
    # We have to generate a return serie: we estimate it through the
    # parameters of the model we used to compute the volatility
    # work on vol simulation by simulation process
    statistics = []
    for sim in range(nb_scenario):
        create_new_vol_frame = np.random.choice(range(0, len(vol)), size=(len(vol), 1))
        X = vol[create_new_vol_frame] # Later work
        statistic = unconditional_expected_shortfall_test(retu, vol=X, confidence=conf, ret=False, build_pval=True)
        statistics.append(statistic)
    return statistics


def CVAR(alpha, vol, mu, days, series=None):
    """
    Model has to be adapted to receive series of volatility and thus to return a series of CVAR
    Conditional Value-at-Risk (CVaR) in normal linear VaR model: loss expected in case VaR is triggered
    :param alpha: confidence
    :param vol: volatility
    :param mu: mean
    :param days:
    :param series: return series used in rolling CVAR calculation
    :return: CVAR
    """
    if series is not None:
        mu = np.mean(series)
        vol = np.std(series) if type(vol) != pd.Series else vol
    return -(1 / alpha) * norm.pdf(norm.ppf(alpha)) * vol * np.sqrt(days / 252) - mu


def VAR(alpha, vol, mu, days):
    """
    Value-at-Risk
        Model has to be adapted to receive series of volatility and thus to return a series of CVAR
    :param alpha: confidence
    :param vol: volatility
    :param mu: mean
    :param days:
    :return: VAR
    """
    return (norm.ppf(alpha) * vol) * np.sqrt(days / 252) - mu


