import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from datetime import datetime, timedelta,date
import numpy as np
from yahoo_fin.stock_info import get_data
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import statsmodels.api as sm


st.title("Autocorrelation Periods")
st.write("Choose an asset and a time window to find the most statistically similar period in history")

option=st.selectbox('Choose an Asset', ('S&P500', 'S&P/TSX Composite'))

def set_data_and_ticker(option):
    if option=='S&P500':
        f_name='sp_500_all_history.csv'
        ticker='^GSPC'
    elif option=='S&P/TSX Composite':
        f_name='TSX_comp_all_history.csv'
        ticker='^GSPTSE'
    else:
        print("What?")
    #print(pd.read_csv(f_name).columns)
    try:
        df=pd.read_csv(f_name, parse_dates=['Date'])
    except:
        df=pd.read_csv(f_name, parse_dates=['Unnamed: 0'])
        df.rename({'Unnamed: 0':'Date'},axis=1,inplace=True)
    
    yesterday=pd.Timestamp(date.today()-timedelta(days=1))
    if df['Date'].max()<yesterday:
        try:  
            query_start=(df['Date'].max()+timedelta(days=1)).date()
            query_end=date.today()       
            print("------", query_start, query_end,"-----")

            new_info=get_data(ticker, start_date=query_start, end_date=query_end, interval="1d")
            df.set_index('Date',inplace=True)
            df=pd.concat([df,new_info],axis=0)
            df.to_csv(f_name)
        except:
            df.set_index('Date',inplace=True)
            print("Oh oh")
    else:
        df.set_index('Date',inplace=True)
    return ticker, df.dropna()


def find_peak(df):
    max_close_date=df[df['close']==df['close'].max()].index[0]
    return df.loc[max_close_date:].shape[0]


def test_autocorrelation(df,amp):
    most_recent=df.iloc[-amp:,:-1]
    final=pd.DataFrame(columns=['open', 'high', 'low', 'close', 'adjclose', 'volume', 'start_range'])
    for i in range(1,df.shape[0]-amp):
        r_ini=-amp-i
        r_end=-i
        last_per=df.iloc[r_ini:r_end,:-1]
        new_start=last_per.index.min()
        new_end=last_per.index.max()
        coefs={}
        for c in most_recent.columns:
            coef=np.corrcoef(most_recent[c],last_per[c], rowvar=False)[0][1]
            coefs[c]=coef
        coefs['start_range']=new_start
        coefs['end_range']=new_end
        final=pd.concat([final,pd.DataFrame(coefs,index=[0])],axis=0)
    return most_recent, final 


def period_overlay(valid_max_row, valid_max_year, df, most_recent):
    l_date=valid_max_row['start_range'].item()
    u_date=valid_max_row['end_range'].item()

    plot_amp=round(most_recent.shape[0]/252,0)
    period_lim_b=valid_max_year-plot_amp
    period_lim_u=valid_max_year+plot_amp+1

    df_p=df.reset_index().rename({'index':'Date'},axis=1)
    overlay_df=df_p[df_p['Date'].between(l_date, u_date)]
    
    overlay_df['Most_recent_ov']=most_recent['close'].array
    plot_period=df_p[df_p['Date'].dt.year.between(period_lim_b,period_lim_u)]
    name_1=str(period_lim_b)+" to "+str(period_lim_u)
    title=str(option)+"Close Price - "+str(l_date.month)+"/"+str(l_date.year)+" to "+str(u_date.month)+"/"+str(u_date.year)
    return l_date, u_date, df_p, overlay_df, plot_period, name_1, title

def make_stats(df_p, l_date,u_date, most_recent):
    stat_df=df_p[df_p['Date'].between(l_date, u_date)]
    most_recent_f=df_p[df_p['Date'].between(most_recent.index.min(), most_recent.index.max())]
    stat_df['key_date']=most_recent_f['Date'].array
    stat_df=stat_df.merge(most_recent_f, how='left', left_on='key_date', right_on='Date').fillna(0)

    stats={}
    for i in ['open', 'high', 'low', 'close', 'adjclose','volume']:
        stats_res=[]
        X=stat_df[i+'_x']
        X = sm.add_constant(X)
        Y=stat_df[i+'_y']
        mod = sm.OLS(Y,X)
        fii = mod.fit()
        stats_res.append(fii.rsquared)
        stats_res.append(fii.params[1])
        stats_res.append(fii.f_pvalue)
        stats[i]=stats_res
    return pd.DataFrame(stats, index=['R_squared','Beta','Sig_F'])


ticker, df=set_data_and_ticker(option)

peak_period=find_peak(df)

st.write(str(option)+' peaked '+str(peak_period)+' periods ago')

amp=st.slider('Select the number of periods to compare'
              ,min_value=30
              ,max_value=1000
              , value=peak_period,step=1)

if st.button("Execute"):

    amp=int(amp)
    st.text('Loading, pls hold.')
    progress_bar = st.progress(0) 
 
    progress_bar.progress(25)  

    most_recent, final=test_autocorrelation(df,amp)

    progress_bar.progress(50)
    progress_bar.progress(100)
    
    recent_date_start=most_recent.index.min().strftime("%d/%m/%Y")
    recent_date_end=most_recent.index.max().strftime("%d/%m/%Y")

    st.write('Your period selection is '+ str(recent_date_start)+' - '+str(recent_date_end))


    thresh=0.7
    top=final[(final['close']>=thresh)|(final['close']<=-thresh)]
    top['month_period']=top.start_range.apply(lambda x: x-timedelta(x.day-1))
    top.reset_index(inplace=True)

    st.header('Historic Correlation Distribution')
    st.write('Distribution of correlation between the closing prices in your period selection and all same-sized periods in the asset history')

    fig=px.histogram(final, x='close')

    st.plotly_chart(fig)

    st.write('Periods in History with more than 70% correlation with the most recent period')


    positive=top[(top.end_range.dt.year<2023)&(top.close>0)]['month_period'] #.value_counts().reset_index()
    negative=top[(top.end_range.dt.year<2023)&(top.close<0)]['month_period'] #.value_counts().reset_index()

    fig2=make_subplots(rows=1, cols=2)

    fig2.add_trace(
    go.Histogram(x=positive, nbinsx=40,name='Positively Correlated'),
    row=1,col=2
    )

    fig2.add_trace(
    go.Histogram(x=negative, nbinsx=40, name='Negatively Correlated'),
    row=1,col=1
    )

    st.plotly_chart(fig2)

    ##########

    st.header('Most similar period to your period selection',3)


    valid_max_1=top[top.end_range.dt.year<2023]['close'].max()
    valid_max_1_year=top[top['close']==valid_max_1]['start_range'].dt.year.item()
    valid_max_2=top[top.end_range.dt.year<valid_max_1_year]['close'].max()
    valid_max_2_year=top[top['close']==valid_max_2]['start_range'].dt.year.item()
    valid_max_1_row=top[top['close']==valid_max_1]
    valid_max_2_row=top[top['close']==valid_max_2]

    l_date, u_date, df_p, overlay_df, plot_period, name_1, title=period_overlay(valid_max_1_row, valid_max_1_year, df, most_recent)

    similar_per_start_1=l_date.strftime("%d/%m/%Y")
    similar_per_end_1=u_date.strftime("%d/%m/%Y")

    st.write(str(similar_per_start_1)+' - '+str(similar_per_end_1))


    fig3=make_subplots(specs=[[{'secondary_y':True}]])

    fig3.add_trace(
    go.Line(x=plot_period['Date'], y=plot_period['close'], name=name_1),
    secondary_y=False
    )

    fig3.add_trace(
    go.Line(x=overlay_df['Date'], y=overlay_df['Most_recent_ov'], name='Most Recent Period'),
    secondary_y=True
    )

    fig3.update_traces(line=dict(width=3))    
    fig3.update_layout(
        height=600, 
        width=800,  
        title=dict(text=title),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        yaxis2=dict(showgrid=False),
        legend=dict(
            orientation='h',  
            yanchor='bottom',  
            y=1.02,  
            xanchor='right',  
            x=1  
        )
        )
    fig3.add_vline(x=l_date, line_dash="dash")
    fig3.add_vline(x=u_date, line_dash="dash")

    st.plotly_chart(fig3)

    st.write('Stats and comparisson with other price metrics and volume')
    stat_df=make_stats(df_p, l_date,u_date, most_recent).fillna(0)
    stat_df=stat_df.applymap(lambda x:x.replace(np.nan,0) if x==np.nan else x)

    html_string = stat_df.to_html(index=True, float_format="{:.2f}".format)
    centered_html_string = f"<div style='text-align: center; padding-left: 100px'>{html_string}</div>"
    st.write(centered_html_string, unsafe_allow_html=True)



    st.header('Second Most similar period to your period selection',2)

    l_date, u_date, df_p, overlay_df, plot_period, name_1, title=period_overlay(valid_max_2_row, valid_max_2_year, df, most_recent)

    similar_per_start_2=l_date.strftime("%d/%m/%Y")
    similar_per_end_2=u_date.strftime("%d/%m/%Y")

    st.write(str(similar_per_start_2)+' - '+str(similar_per_end_2))

    fig4=make_subplots(specs=[[{'secondary_y':True}]])

    fig4.add_trace(
    go.Line(x=plot_period['Date'], y=plot_period['close'], name=name_1),
    secondary_y=False
    )

    fig4.add_trace(
    go.Line(x=overlay_df['Date'], y=overlay_df['Most_recent_ov'], name='Most Recent Period'),
    secondary_y=True
    )

    fig4.update_traces(line=dict(width=3))
    
    fig4.update_layout(
        height=600,
        width=800, 
        title=dict(text=title),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        yaxis2=dict(showgrid=False),        
        legend=dict(
            orientation='h',  
            yanchor='bottom',  
            y=1.02,  
            xanchor='right',  
            x=1  
        )
    )
    fig4.add_vline(x=l_date, line_dash="dash")
    fig4.add_vline(x=u_date, line_dash="dash")

    st.plotly_chart(fig4)

    st.write('Stats and comparisson with other price metrics and volume')
    stat_df=make_stats(df_p, l_date,u_date, most_recent).fillna(0)
    stat_df=stat_df.applymap(lambda x:x.replace(np.nan,0) if x==np.nan else x)

    html_string = stat_df.to_html(index=True, float_format="{:.2f}".format)
    centered_html_string = f"<div style='text-align: center; padding-left: 100px'>{html_string}</div>"
    st.write(centered_html_string, unsafe_allow_html=True)

    st.write('\n')
    st.write('\n')

    st.write('*If you would like to:*')
    list_items = ["Implement advance data analytics in your company"
                  ,"Learn how to build this tools"
                  ,"access the underlying data"]
    list_html = "<ul>" + "".join(["<li>{}</li>".format(item) for item in list_items]) + "</ul>"

    st.write(list_html, unsafe_allow_html=True)

    st.write("Contct me on Linkedin [link] (https://www.linkedin.com/juanraful/)")
    st.write("\n")


    st.write('*Other applications*')
    list_items = ["Improve forecasting: Find the best period in history to train your forecast model"
                  ,"Compare companies on different life stages: Compare a company recent financials to the historic financials of a large corporation"
                  ,"Undestand traffic patterns: find if there are significant patterns in a website traffic data"]
    list_html = "<ul>" + "".join(["<li>{}</li>".format(item) for item in list_items]) + "</ul>"

    st.write(list_html, unsafe_allow_html=True)




