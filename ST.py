import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from joblib import Parallel, delayed
import pickle
import os 
import inspect
import scipy as sp
import itertools as it
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import copy
import sys
import  Backtest_lib
import time
import GlidePath as GP
import pandas as pd
import plotly.express as px


return_frame =pd.read_excel("Asset_class_return.xlsx",index_col = "Date")
#return_frame_a['Risk_Assets'] =( return_frame_a['SET_Risk_Assets_r']  +  return_frame_a['Unhedged__Risk_Assets_r '])/2
#return_frame_a['Risk_free_Assets'] =( return_frame_a['Corp1_3_r ']  +  return_frame_a['Gov1_3_r'])/2
#return_frame = return_frame_a[['Risk_Assets','Risk_free_Assets' ] ]


# Function to create and display a Plotly line chart in a Streamlit app
def display_wealth_over_time(Wealthplot):
    # Generate the Plotly line chart
    fig = px.line(Wealthplot, title='Wealth Over Time')
    
    # Update the layout to set the y-axis range, ensuring it starts at 0 and goes up to 120% of the max value
    fig.update_layout(yaxis_range=[0, Wealthplot.max().max() * 1.2])
    
    return fig

def annualize_rets(_05, _25, _50,_75, _95) : 
    dic_annualize_rets = { 'value' : 'Return',
                            'Percentile 5 ' : Backtest_lib.annualize_rets(_05,12 ),
                          'Percentile 25 ' : Backtest_lib.annualize_rets(_25,12 ),
                          'Percentile 50 ' : Backtest_lib.annualize_rets(_50,12 ),
                          'Percentile 75 ' : Backtest_lib.annualize_rets(_75,12 ),
                          'Percentile 95 ' : Backtest_lib.annualize_rets(_95,12 ),
    }
    results_df = pd.DataFrame(dic_annualize_rets, index=[0] )
    return results_df

def annualize_vol(_05, _25, _50,_75, _95) : 
    dic_annualize_vol= { 'value' : 'Vol',
                          'Percentile 5 ' :Backtest_lib.annualize_vol(_05,12 ),
                          'Percentile 25 ' : Backtest_lib.annualize_vol(_25,12 ),
                          'Percentile 50 ' : Backtest_lib.annualize_vol(_50,12 ),
                          'Percentile 75 ' : Backtest_lib.annualize_vol(_75,12 ),
                          'Percentile 95 ' : Backtest_lib.annualize_vol(_95,12 ),
    }
    results_df = pd.DataFrame(dic_annualize_vol, index=[0] )
    return results_df



# Streamlit app interface
st.title('Financial Projection Calculator')

# Inputs
start_age = st.sidebar.number_input('Start Age', min_value=0, max_value=120, value=40)
retire_age = st.sidebar.number_input('Retirement Age', min_value=0, max_value=120, value=60)
death_age = st.sidebar.number_input('Death Age', min_value=0, max_value=120, value=80)
#n_block_bt = st.sidebar.number_input('Number of Blocks Between Periods', min_value=1, value=22)
n_block_bt = 22 # period_sim
wealth_indi  = st.sidebar.number_input('wealth_Initial', value=15000 )
initial_salary = st.sidebar.number_input('Initial Salary', value=15000)
inflation_rate = st.sidebar.number_input('Inflation Rate(for after retrie)', value=0.0)
replacement_cost = st.sidebar.number_input('Replacement Cost', value=0.3)
interest_rate = st.sidebar.number_input('Interest Rate(for cal sharpe ratio)', value=0.004)
salary_growth = st.sidebar.number_input('Salary Growth', value=0.055)
own_contri_rate = st.sidebar.number_input('Own Contribution Rate', value=0.05)



investment_hori_mth = (retire_age-start_age)*12 #random_return_size
ret = return_frame.values
n_day_in_mth = 22
n_sim = 50000
n_mth_retire_to_die = (death_age - retire_age)*12 


employer_contri=np.zeros(investment_hori_mth)
employer_contri[(retire_age  - start_age)*12:] =  st.sidebar.number_input('employer Contribution Rate', value=0.05)


st.sidebar.write("Port")
# Define the column names and values
columns = ['Thai Equity', 'US Equity', 'Corp1_3_r' , 'Gov1_3_r' ]
Thai_Equity =  st.number_input('Thai_Equity ', value=0.4)
US_Equity =  st.number_input('US_Equity', value=0.4)
Corp1_3 =  st.number_input('Corp Bons 1-3 ', value=0.1)
Gov1_3   =  st.number_input('Gov Bons 1-3 ', value=0.1)




# Define the column names and values

values = [0.05, 0.05, 0.4, 0.4]
# Create the dataframe with 50 rows of the given values
w_after_re  = pd.DataFrame([values] * (death_age  - retire_age)*12, columns=columns)


# Button to perform calculation
if st.button('Calculate Financials'):
    # Create a progress bar with an initial value of 0
    progress_bar = st.progress(0)


    # Loop to update the progress bar
    for percentage in range(100):
        # Wait for 0.1 seconds
        time.sleep(0.05)
        # Update the progress bar
        progress_bar.progress(percentage + 1)

    w_before_re_values = [Thai_Equity , US_Equity, Corp1_3, Gov1_3  ]
    w_before_re = pd.DataFrame([w_before_re_values ] * (retire_age - start_age )*12 , columns=columns)    
    w_all_til_die  = pd.concat([w_before_re, w_after_re], ignore_index=True)

    
    ogp = GP.OptimalGlidePath()
    dict_sim_all = ogp.block_bt_lifepath(ret,investment_hori_mth+n_mth_retire_to_die,n_day_in_mth,n_sim)
    dict_sim = {ii:ret_sim[:investment_hori_mth] for ii,ret_sim in dict_sim_all.items()}
    dict_sim_retire = {ii:ret_sim[investment_hori_mth:] for ii,ret_sim in dict_sim_all.items()}
    salary_cf,invest_amount = ogp.get_investment_cf(initial_salary,salary_growth,
                                                    investment_hori_mth,own_contri_rate,
                                                    employer_contri)
    invest_amount[0]+= wealth_indi
    
    
    
    w_all = w_all_til_die.iloc[:investment_hori_mth]
    w_retire = w_all_til_die.iloc[investment_hori_mth:].reset_index(drop=True)
    # if (w_all<0).any().any(): return -1 #0
    w_all_ = w_all.values
    final_cf_sim,dict_cashflow_sim = ogp.get_cf_final(dict_sim,w_all_,invest_amount,rebal_all_date=True,rebal_freq=6)
    ret_after_retire = GP.get_return_from_dict_sim(dict_sim_retire, w_retire.values)
    last_salary = salary_cf[-1]
    cf_retirement = ogp.get_retirement_cf(final_cf_sim,last_salary,
                                          replacement_cost,n_mth_retire_to_die,
                                          inflation_rate,ret_after_retire)

    prob_ = 1-(cf_retirement<0).any(axis=1).sum()/cf_retirement.shape[0]
    #st.write(pd.DataFrame(cf_retirement))
    #st.write(pd.DataFrame(salary_cf))
    #st.write(last_salary)
    st.write(replacement_cost)
    st.write(inflation_rate)
    st.write(w_retire)
    st.write(pd.DataFrame(dict_sim_retire [0]))



    
    

    df_cashflow_sim =  pd.DataFrame(dict_cashflow_sim ).T.quantile(q=[0.05,0.25, 0.5, 0.75,0.95]).T.astype(int)
    df_cashflow_sim['invest_amount'] =  pd.DataFrame(invest_amount).astype(int)
    df_retirement_sim  = pd.DataFrame(cf_retirement ).quantile(q=[0.05,0.25, 0.5, 0.75,0.95]).T.astype(int)
    df_retirement_sim['invest_amount']   = pd.DataFrame(invest_amount).astype(int)
    Wealthplot = pd.concat([df_cashflow_sim 
                             ,df_retirement_sim], ignore_index=True)
    
    _05 = ((df_cashflow_sim[0.25] - df_cashflow_sim['invest_amount']) /df_cashflow_sim[0.25].shift(1)) -1
    _25 = ((df_cashflow_sim[0.25] - df_cashflow_sim['invest_amount']) /df_cashflow_sim[0.25].shift(1)) -1
    _50 = ((df_cashflow_sim[0.50] - df_cashflow_sim['invest_amount']) /df_cashflow_sim[0.50].shift(1)) -1
    _75 =((df_cashflow_sim[0.75] - df_cashflow_sim['invest_amount']) /df_cashflow_sim[0.75].shift(1)) -1
    _95 =((df_cashflow_sim[0.95] - df_cashflow_sim['invest_amount']) /df_cashflow_sim[0.95].shift(1)) -1
    
    
    dd = (pd.DataFrame(dict_cashflow_sim ).T.sort_values(by=((retire_age - start_age ) * 12 )-1 , ascending=False).astype(int).iloc[2500:2501,:].T ) 
    dd['invest_amount'] = pd.DataFrame(invest_amount).astype(int)
    dd_2= ((dd[dd.columns[0]] - df_cashflow_sim['invest_amount']) /dd[dd.columns[0]].shift(1)) -1
    Drawdown = Backtest_lib.drawdown(dd_2).min()
    

    # Display the results in a table
    st.write("Simulation")
    success_rate =  prob_  *100
    st.header('success_rate ' +str(int(success_rate *100 ))+ ' %' )
    
    summarise_sta = pd.concat ([ Backtest_lib.summary_stats(_05, riskfree_rate=0.015) ,
                Backtest_lib.summary_stats(_25, riskfree_rate=0.015),
                Backtest_lib.summary_stats(_50, riskfree_rate=0.015),
                Backtest_lib.summary_stats(_75, riskfree_rate=0.015),
                Backtest_lib.summary_stats(_95, riskfree_rate=0.015), ]
    )
    # List to be added as a new column
    new_column_data = ['Percentile 5 ', 'Percentile 25 ', 'Percentile 50 ', 'Percentile 75 ', 'Percentile 95 ']
    # Adding the list as a new column to the DataFrame
    summarise_sta['Percentile'] = new_column_data
    summarise_sta.set_index('Percentile', inplace=True)

    st.write(summarise_sta[["Annualized Return", "Annualized Vol", "Cornish-Fisher VaR (5%)", "Sharpe Ratio"]])


    display_wealth_over_time_fig =  display_wealth_over_time(Wealthplot)
    st.plotly_chart(display_wealth_over_time_fig)
    st.write("Drawdown : ", Backtest_lib.drawdown(dd_2).min()['Drawdown'])
    

