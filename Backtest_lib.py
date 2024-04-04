import pandas as pd
import numpy as np
import matplotlib as plt
import ffn
import scipy


import os.path
#print(os.getcwd())
#dirRawData = "../RawData/"

#xls = pd.ExcelFile(dirRawData + 'weight_return.xlsx')
#weight_eq_wkly  = pd.read_excel(xls, 'weight_eq_wkly', index_col=0)
#simple_return  = pd.read_excel(xls, 'simple_return', index_col=0)

    
    
# input assignment
nav_lst = []
intial_nav = 1
n = 0
#return_ = simple_return
#weights = weight_eq_wkly
#fee = 0
#transaction_lst = {}
#riskfree_rate = 0.03
    

# nav simulation

def Back_test_sim(intial_nav ,return_ , weights, fee ):
    
    # dummy variables 
    nav = intial_nav
    nav_lst = []
    transaction_feelst = {}
    transaction_ratiolst = {}
    n = 0
    weight_before = 0
    sumtransaction_ratio = 0
    transaction_lst = {}
    
    # loop for sim
    for date_ in return_.index :
        r = return_[return_.index == date_].T[date_]
        
        # rellallacing
        if date_  in  weights.index : 
            weight = weights[weights.index == date_].T[date_]
            
            #transaction_fee
            transaction = weight_before - weight
            transaction_lst.update({date_: transaction})
            transaction_ratio = abs(weight_before - weight).sum()
            transaction_fee = transaction_ratio * (1+fee)
            sumtransaction_ratio += transaction_ratio
            
            # dic_
            transaction_feelst.update({date_: transaction_fee})
            transaction_ratiolst.update({date_: transaction_ratiolst})
            
            
        else:
            weight = weight *(1+r)/ ((weight *(1+r)).sum())
        r_port =( weight * r).sum()
        nav = nav * (1+r_port)
        nav_lst.append(nav)


        weight_before = weight
    df = return_
    df['Port_nav'] = nav_lst
    df['Port'] = df['Port_nav'].pct_change()
    r = df['Port'].dropna()
        
    return r, transaction_lst, nav_lst, sumtransaction_ratio, transaction_feelst,transaction_ratiolst



#_________________________________________________State___________________________________________


def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3


def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

def compound(r):
    """
    returns the result of compounding the set of returns in r
    """
    return np.expm1(np.log1p(r).sum())


def annualize_rets(r, periods_per_year=12):
    """
    Annualizes a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1

def returns_Y_M (r):
    df_monthly_returns = r.resample('M').ffill()
    df_Year_returns = r.resample('Y').ffill()
    return df_monthly_returns,  df_Year_returns


def annualize_vol(r, periods_per_year):
    """
    Annualizes the vol of a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    return r.std()*(periods_per_year**0.5)


def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of returns
    """
    # convert the annual riskfree rate to per period
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol


import scipy.stats
def is_normal(r, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a Series is normal or not
    Test is applied at the 1% level by default
    Returns True if the hypothesis of normality is accepted, False otherwise
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(is_normal)
    else:
        statistic, p_value = scipy.stats.jarque_bera(r)
        return p_value > level



def drawdown(return_series: pd.Series):
    """Takes a time series of asset returns.
       returns a DataFrame with columns for
       the wealth index, 
       the previous peaks, and 
       the percentage drawdown
    """
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({"Wealth": wealth_index, 
                         "Previous Peak": previous_peaks, 
                         "Drawdown": drawdowns})

def drawdown_group(df,index_list):
    group_max,dd_date = index_list
    ddGroup = df[df['Previous Peak'] == group_max]
    group_length = len(ddGroup)
    group_dd = ddGroup['Drawdown'].min()
    group_dd_length = len(ddGroup[ddGroup.index <= dd_date])
    group_start = ddGroup[0:1].index[0]
    group_end = ddGroup.tail(1).index[0]
    group_rec = group_length - group_dd_length
    #print (group_start,group_end,group_dd,dd_date,group_dd_length,group_rec,group_length)
    return group_start,group_end,group_max,group_dd,dd_date,group_dd_length,group_rec,group_length


def group_drawdown(r):
    
    df = drawdown(r)
    
    groups = df.groupby(df['Previous Peak'])
    topdrowdown = groups['Previous Peak','Drawdown'].apply(
        lambda g: g[g['Drawdown'] == g['Drawdown'].min()]).sort_values(
        'Drawdown', ascending=True)
    
    
    
    dd_col = ('start_date',
              'end_date',
              'previous_peeks',
              'drowdown',
              'drowdown_date',
              'drowdown_length',
              'recovery_length',
              'total_length')
    df_dd = pd.DataFrame(columns = dd_col)
    for i in range(1,len(topdrowdown)):
        index_list = topdrowdown[i-1:i].index.tolist()[0]
        #start,end,peak,dd,dd_date,dd_length,dd_rec,total_length = drawdown_group(df,index_list)
        start,end,peak,dd,dd_date,dd_length,dd_rec,total_length = drawdown_group(df,index_list)
        df_dd.loc[i-1] = drawdown_group(df,index_list)

    
    return topdrowdown , df_dd
    

def semideviation(r):
    """
    Returns the semideviation aka negative semideviation of r
    r must be a Series or a DataFrame, else raises a TypeError
    """
    if isinstance(r, pd.Series):
        is_negative = r < 0
        return r[is_negative].std(ddof=0)
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(semideviation)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


def var_historic(r, level=5):
    """
    Returns the historic Value at Risk at a specified level
    i.e. returns the number such that "level" percent of the returns
    fall below that number, and the (100-level) percent are above
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


from scipy.stats import norm
def var_gaussian(r, level=5, modified=False):
    """
    Returns the Parametric Gauusian VaR of a Series or DataFrame
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    """
    # compute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 -3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() + z*r.std(ddof=0))


def portfolio_return(weights, returns):
    """
    Computes the return on a portfolio from constituent returns and weights
    weights are a numpy array or Nx1 matrix and returns are a numpy array or Nx1 matrix
    """
    return weights.T @ returns


def portfolio_vol(weights, covmat):
    """
    Computes the vol of a portfolio from a covariance matrix and constituent weights
    weights are a numpy array or N x 1 maxtrix and covmat is an N x N matrix
    """
    return (weights.T @ covmat @ weights)**0.5


def plot_ef2(n_points, er, cov):
    """
    Plots the 2-asset efficient frontier
    """
    if er.shape[0] != 2 or er.shape[0] != 2:
        raise ValueError("plot_ef2 can only plot 2-asset frontiers")
    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
    })
    return ef.line(x="Volatility", y="Returns", style=".-")





from scipy.optimize import minimize

def minimize_vol(target_return, er, cov):
    """
    Returns the optimal weights that achieve the target return
    given a set of expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    return_is_target = {'type': 'eq',
                        'args': (er,),
                        'fun': lambda weights, er: target_return - portfolio_return(weights,er)
    }
    weights = minimize(portfolio_vol, init_guess,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,return_is_target),
                       bounds=bounds)
    return weights.x


def tracking_error(r_a, r_b):
    """
    Returns the Tracking Error between the two return series
    """
    return np.sqrt(((r_a - r_b)**2).sum())


                        
def msr(riskfree_rate, er, cov):
    """
    Returns the weights of the portfolio that gives you the maximum sharpe ratio
    given the riskfree rate and expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio
        of the given portfolio
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol
    
    weights = minimize(neg_sharpe, init_guess,
                       args=(riskfree_rate, er, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x



def Risk_contributions_cal (weights ,returns ):
    
    weights_ = weights.iloc[-1:].values[0]
    # Calculate covariance matrix
    cov_matrix = returns.cov()

    # Portfolio variance
    port_variance = np.dot(weights_.T, np.dot(cov_matrix, weights_))

    # Marginal Risk Contribution (MRC)
    mrc = np.dot(cov_matrix, weights_) / port_variance

    # Total Risk Contribution (TRC)
    trc = mrc * weights_

    # Normalize risk contributions to sum up to 1
    risk_contributions = trc / np.sum(trc)
    
    lst_risk_contributions  = {}

    # Display risk contributions
    for i, rc in enumerate(risk_contributions):
        print(f"Risk contribution of Asset {i+1}: {rc:.2f}")
        lst_risk_contributions[weights .columns[i]] =  rc
    return lst_risk_contributions
        







def summary_stats(r, riskfree_rate=0.03):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    ann_r = r.aggregate(annualize_rets, periods_per_year=12)
    ann_vol = r.aggregate(annualize_vol, periods_per_year=12)
    ann_sr = r.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=12)
    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified=True)
    hist_cvar5 = r.aggregate(cvar_historic)
    return pd.DataFrame({
        "Annualized Return": [ann_r],
        "Annualized Vol": [ann_vol],
        "Skewness": [skew],
        "Kurtosis": [kurt],
        "Cornish-Fisher VaR (5%)": [cf_var5],
        "Historic CVaR (5%)": [hist_cvar5],
        "Sharpe Ratio": [ann_sr],
        "Max Drawdown": [dd]
    })









#_________________________________________________Blacktest_class_______________________________________

# if  swap cost pl. minus return (new return = return - swap cost)

class Basic_backtest(object):
    import pandas as pd
    import numpy as np
    import matplotlib as plt
    import scipy
    
    
    def __init__(self, weight, return_univ, intial_nav=1, riskfree_rate=0, fee = 0, periods_per_year =200 ):
        self.weight = weight
        self.return_univ = return_univ
        self.riskfree_rate = riskfree_rate
        self.fee = fee

        self.intial_nav = intial_nav
        self.transaction_lst = {}
        self.transaction_ratiolst = {}
        self.df =self.return_univ
        self.periods_per_year = periods_per_year        
        
        self.r,self.transaction_lst, self.nav_lst, self.sumtransaction_ratio, self.transaction_feelst,self.transaction_ratiolst = Back_test_sim(intial_nav ,return_univ, weight, fee )
        self.skewness = skewness(self.r)
        self.kurtosis = kurtosis(self.r)
        self.compound = compound(self.r)
        self.annualize_rets = annualize_rets(self.r, self.periods_per_year )
        self.df_monthly_returns,  self.df_Year_returns = returns_Y_M (self.r)
        self.annualize_vol = annualize_vol(self.r, self.periods_per_year)
        self.sharpe_ratio = sharpe_ratio(self.r, self.riskfree_rate, self.periods_per_year)
        self.is_normal =  is_normal(self.r, level=0.01)
        
        self.group_drawdown  = group_drawdown(self.r)
        self.semideviation = semideviation(self.r)
        self.drawdown = drawdown(self.r)
        
        
        self.var_historic = var_historic(self.r, level=5)
        self.cvar_historic = cvar_historic(self.r, level=5)
        self.var_gaussian = var_gaussian(self.r, level=5, modified=False)
        self.summary_stats = summary_stats(r = self.r, riskfree_rate=self.riskfree_rate)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
################################################################################
print("#"*10)
print("")




#if __name__ == "__main__": 
    #p1 = Basic_backtest(weight= weight_eq_wkly,  return_univ =  simple_return )
#    p1 = Basic_backtest(weight_eq_wkly, simple_return, intial_nav=1, riskfree_rate=0, fee = 0)
#    p1summary_stats = p1.summary_stats
    

print("Magicians are blowing themselves up all over the place.")
print("")
print("#"*10)



## you can use ffn module for stat
#import ffn
#stats = p1.r.calc_stats()
#stats.display()
#p1.r.display_lookback_returns()




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
     