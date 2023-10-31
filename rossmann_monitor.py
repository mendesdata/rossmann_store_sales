# Import Libraries

import pandas               as pd
import numpy                as np
import streamlit            as st
import plotly.express       as px
import plotly.graph_objects as go

import requests
import json
import math
import datetime
import os

from  api.rossmann.Rossmann import Rossmann
from  sklearn.metrics       import mean_absolute_error, mean_absolute_percentage_error

def absolute_error(y, yhat):
    error = np.sum( yhat - y )
    return round( error, 2 )

def sum_absolute_error( y, yhat ):
    error = np.sum( np.abs( y - yhat ) )
    return round( error, 2 )


def load_data( file ):
    # colect original test dataset
    df = pd.read_csv( file )
    
    # remove close day
    df = df[ ( df['open'] != 0 ) ]

    return df

def grouped_stores( df ):
    # store´s performance
    df_store_sum = df[['store', 'sales', 'sales_predictions']].groupby( 'store' ).sum().reset_index()
    df_store_avg = df[['store', 'sales', 'absolute_error']].groupby( 'store' ).mean().reset_index().rename( columns = { 'sales' : 'sales_avg',
                                                                                                                        'absolute_error' : 'MAE' } )
    
    df_store = pd.merge( df_store_sum, df_store_avg, how='inner', on='store' )
    df_store['MAPE'] = round( ( df_store['MAE'] / df_store['sales_avg'] ) * 100, 2)
    df_result = df_store.sort_values( 'MAPE').reset_index()

    return df_result

def score_of_stores( df, score_radio, score_slider):
    df_store = grouped_stores( df )
    df_store = df_store.sort_values( 'MAPE', ascending= ( score_radio == 'Best' ) ).reset_index()
    df_store = df_store.head(score_slider)

    return df_store['store'].values 

def error_range( df ):
    df['range'] = df['MAPE'].apply( lambda x : 'Range 1: Until 5%'       if ( x <= 5.0  ) else 
                                               'Range 2: Between 5-10%'  if ( x <= 10.0 ) else 
                                               'Range 3: Between 10-15%' if ( x <= 15.0 ) else 
                                               'Range 4: Between 15-20%' if ( x <= 20.0 ) else 
                                               'Range 5: Over 20%' )

    return df['range'].values

def apply_model( x_test ):
    # drop sales columns
    y_test = x_test['sales'].values
    x_test = x_test.drop( ['sales'], axis=1 )

    # convert dataframe to json
    data_json = json.dumps( x_test.to_dict( orient='records' ) )

    # local request
    local_url = 'http://0.0.0.0:5000/rossmann/predict'
    url = os.environ.get( 'url', local_url)

    # Render´s Server request
    #url = 'https://rossmann-app-9l04.onrender.com/rossmann/predict'


    # API CALL  
    header = { 'Content-type' : 'application/json' }
    response = requests.post( url, data=data_json, headers=header ) 

    # return dataframe with predictions
    df = pd.DataFrame( response.json(), columns=response.json()[0].keys() )
    df = df[['store', 'date', 'sales_predictions']]

    df['sales']          = y_test
    df['absolute_error'] = np.abs( y_test - df['sales_predictions'].values )

    return df

def model_metrics( df ):
    stores            = df['store'].nunique()
    sales             = df['sales'].sum() 
    sales_predictions = df['sales_predictions'].sum() 
    sales_average     = df['sales'].mean()
    MAE               = df['absolute_error'].mean()
    MAPE              = round( ( MAE / sales_average ) * 100, 2 )
    SAR               = sum_absolute_error( df['sales'].values , df['sales_predictions'].values )

    with st.container():
        col1, col2, col3, col4, col5, col6 = st.columns ( 6, gap='small' )
    
        with col1:
            st.metric( label='Stores',  value='{:,.0f}'.format( stores ), help='Number of stores' )      

        with col2:
            st.metric( label='Sales Average',  value='{:,.2f}'.format( sales_average ), delta='{:,.2f}'.format( sales ) , help='Sales average')      

        with col3:
            st.metric( label='MAE',  value='{:,.2f}'.format( MAE ), delta='{:,.2f}'.format( MAPE )+'%',  help='Mean Absolute Error')                   

        with col4:
            st.metric( label='Sales Predictions',  value='{:,.2f}'.format( sales_predictions), help='Sales predictions')  

        with col5:
            st.metric( label='Best Scenario',  value='{:,.2f}'.format( sales_predictions + SAR ),  help='Error')      

        with col6:
            st.metric( label='Worst Scenario',  value='{:,.2f}'.format( sales_predictions - SAR ),  help='Error')      

        return None
    
def error_average_chart( df ):
    df_grouped = df[['date', 'absolute_error', 'sales']].groupby( 'date' ).mean().reset_index()

    with st.container():
        fig = go.Figure()
        fig.add_trace(go.Scatter( x=df_grouped['date'].values,
                                  y=df_grouped['absolute_error'].values,
                                  mode='lines',
                                  name='Absolute Error Average',
                                  line=dict( color='royalblue' )
                                 ) )
        
        fig.add_trace(go.Scatter( x=df_grouped['date'].values,
                                  y=df_grouped['sales'].values,
                                  mode='lines',
                                  name='Sales Average',
                                  line=dict( color='firebrick' )
                                  ) )
        
        fig.update_layout( title='Sales and Absolute Errors', xaxis_title='Days', yaxis_title='Average' )
        st.plotly_chart( fig, use_container_width=True )

    return None

def error_range_chart( df ):
    df_error_range         = df[['store', 'absolute_error', 'sales']].groupby( 'store' ).mean().reset_index()
    df_error_range['MAPE'] = df_error_range['absolute_error'] / df_error_range['sales'] * 100
    df_error_range['range'] = error_range( df_error_range )
    
    df_error_range = ( df_error_range[['range', 'store']]
                                        .groupby('range')
                                        .count()
                                        .sort_values('range')
                                        .reset_index()
                                        .rename( columns = {'store' : 'count_range' } ) )
    
    acum_range = []
    sum_range  = 0
    for v in df_error_range['count_range'].values:
        sum_range = sum_range + v
        acum_range.append( sum_range )

    df_error_range['acum_count_range'] = acum_range
    df_error_range['diff_count_range'] = df_error_range['acum_count_range'] - df_error_range['count_range']

    list_range            = df_error_range['range'].values
    list_count_range      = df_error_range['count_range'].values
    list_acum_count_range = df_error_range['acum_count_range'].values
    list_diff_count_range = df_error_range['diff_count_range'].values

    total_count = df_error_range['count_range'].sum() 

    text_count_range = []
    for v in list_count_range:
        text_count_range.append( str( v ) + ' (' + round( v / total_count * 100, 2 ).astype( str ) + '%)' )

    text_acum_count_range = []
    for v in list_acum_count_range:
        text_acum_count_range.append( str( v ) + ' (' + round( v / total_count * 100, 2 ).astype( str ) + '%)' )

    fig = go.Figure(data=[
            go.Bar(name='Total range', x=list_range, y=list_count_range     , textposition='auto', text=text_count_range),
            go.Bar(name='Accumulated', x=list_range, y=list_diff_count_range, textposition='auto', text=text_acum_count_range ) ] )
    fig.update_layout(barmode='stack',  title='Quantity of stores x Errors range', xaxis_title='Errors range', yaxis_title='Quantity of stores' )
    st.plotly_chart( fig, use_container_width=True )

    return None

def averaging_models_comparative( df_predictions, df_original ):
    # create rossmann model dataframe
    df_rossmann = df_predictions.copy()
    df_rossmann['model'] = 'Rossmann'
    MAE_rossmann = df_rossmann['absolute_error'].mean()

    # create simple average model dataframe
    df_avg_full = df_predictions.copy()
    df_avg_full['sales_predictions'] = df_original['sales'].mean()
    df_avg_full['absolute_error']    = np.abs( df_avg_full['sales'] - df_avg_full['sales_predictions'] )
    df_avg_full['model']             = 'Simple Average'
    
    # create average store dataframe
    df_avg_store = df_predictions.copy()
    df_avg_store_grouped = df_avg_store[['store', 'sales']].groupby( 'store' ).mean().reset_index().rename( columns = {'sales' :  'avg_store' } )
    df_avg_store = pd.merge( df_avg_store, df_avg_store_grouped, how='left', on='store' )
    df_avg_store['sales_predictions'] = df_avg_store['avg_store']
    df_avg_store['absolute_error']    = np.abs( df_avg_store['sales'] - df_avg_store['sales_predictions'] )
    df_avg_store['model']             = 'Store Average'
    df_avg_store = df_avg_store.drop( columns=['avg_store'], axis=1 )

    # create average store_date dataframe
    df_avg_store_date                      = df_predictions.copy()
    df_avg_store_date_grouped              = df_avg_store_date[['store', 'date', 'sales']].groupby( ['store','date'] ).mean().reset_index().rename( columns = {'sales' :  'avg_store_date' } )
    df_avg_store_date                      = pd.merge( df_avg_store_date, df_avg_store_date_grouped, how='left', on='store' )
    df_avg_store_date['sales_predictions'] = df_avg_store_date['avg_store_date']
    df_avg_store_date['absolute_error']    = np.abs( df_avg_store_date['sales'] - df_avg_store_date['sales_predictions'] )
    df_avg_store_date['model']             = 'Store Date Average'
    df_avg_store_date                      = df_avg_store_date.drop( columns=['avg_store_date'], axis=1 )

    # concatenate all models dataframe
    df_pred = pd.concat( [df_rossmann, df_avg_full, df_avg_store, df_avg_store_date] )

    df_pred_model = df_pred[['model', 'absolute_error']].groupby(['model']).mean().sort_values('absolute_error').reset_index()
    df_pred_model['absolute_error_perc']   = ( df_pred_model['absolute_error'] - MAE_rossmann ) / MAE_rossmann * 100
    df_pred_model['absolute_error_text']   = df_pred_model.apply( lambda x: ( str ( round( x['absolute_error']     , 2 ) ) + ' ( +' +  
                                                                              str ( round( x['absolute_error_perc'], 2 ) ) + '% )' ), axis=1 )
    
    fig = px.bar( df_pred_model, x='model', y='absolute_error', text='absolute_error_text') 
    fig.update_layout(title='MAE x Models Chart', yaxis_title='MAE - Mean Absolute Error', xaxis_title='Models' )
    st.plotly_chart( fig, use_container_width=True )

    df_pred_store_model = df_pred[['store', 'model', 'absolute_error']].groupby(['store', 'model']).mean().sort_values('absolute_error').reset_index()
    df_pred_store       = df_pred_store_model[['store', 'absolute_error']].groupby(['store']).min().sort_values('absolute_error').reset_index()
    df_pred_store       = pd.merge( df_pred_store, df_pred_store_model, how='inner', on=['store', 'absolute_error'])
    df_pred_store       = df_pred_store[['model', 'store']].groupby('model').count().reset_index().rename( columns= {'store': 'count'})

    count_sum = df_pred_store['count'].sum()
    df_pred_store['count_text'] = df_pred_store['count'].apply( lambda x: ( str( x ) + ' ( ' +  
                                                                            str( round( x / count_sum * 100, 2 ) ) + '% )' ) )

    fig = px.bar( df_pred_store, x='model', y='count', text='count_text') 
    fig.update_layout(title='Quantity of stores x Models Chart', yaxis_title='Quantity of stores', xaxis_title='Models' )
    st.plotly_chart( fig, use_container_width=True )
    
    return None

def table_of_stores( df ):
    df_table = grouped_stores( df )
    df_table['range'] = error_range( df_table )

    st.dataframe(df_table, hide_index=True )
    return None


# >>> Main Function
def main():
    # configure page title
    st.set_page_config( page_title='Rossmann Monitor - Main Page', layout='wide' )  

    # create test dataset
    x_test = load_data( 'data/test.csv' )

    start_date_period =  x_test['date'].min()
    end_date_period   =  x_test['date'].max()
    days = str( ( pd.to_datetime( end_date_period ) - pd.to_datetime( start_date_period ) ).days+1 )
    
    subtitle = '*Start date:*' + '    **' + start_date_period + '**    '  + '*End date:*' + '    **' + end_date_period + '**    '  + '*Days:*' + '    **' + days + '**'

    st.title( 'Welcome to Rossmann Model Monitor' )
    st.markdown( subtitle )

    # sidebar area
    st.sidebar.markdown('# Filters')

    option = st.sidebar.radio( "How do you want to apply the model ?",
                                ["All stores", "Score of stores", "Choose stores"],
                                index=0 )
    
    if option == 'Score of stores':
        score_radio = st.sidebar.radio( "What kind of score ?",
                                        ["Best", "Worst"],
                                        index=0 )
        
        score_slider = st.sidebar.slider('Choose the range', min_value=1, max_value=x_test['store'].nunique())
    
    
    elif option == 'Choose stores':
        list_stores = x_test['store'].unique() 

        choose_select = st.sidebar.multiselect ( 'Select stores:', list_stores )
    
    button = st.sidebar.button('Apply Model', type='primary')

    if button:
        df_pred = apply_model( x_test )

        if option == 'Score of stores':
            df_pred = df_pred[ df_pred['store'].isin( score_of_stores( df_pred, score_radio, score_slider ) ) ]

        elif option == 'Choose stores':
            df_pred = df_pred[ df_pred['store'].isin( choose_select ) ]

        model_metrics( df_pred )

        tab1, tab2, tab3, tab4 = st.tabs( ['Error Average', 
                                           'Error Range', 
                                           'Averaging Models Comparative', 
                                           'Table of Stores'] )

        with tab1: 
            error_average_chart( df_pred )   

        with tab2:
            error_range_chart( df_pred )            

        with tab3: 
            averaging_models_comparative( df_pred, x_test )

        with tab4: 
            table_of_stores( df_pred )

    return None
   
# >>> Call Main Function
if __name__ == "__main__":
    main()
    