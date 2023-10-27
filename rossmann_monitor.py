# Import Libraries

import pandas               as pd
import numpy                as np
import streamlit            as st
import plotly.express       as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

import requests
import json
import math
import datetime

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

def apply_model( x_test ):
    # drop sales columns
    y_test = x_test['sales'].values
    x_test = x_test.drop( ['sales'], axis=1 )

    # convert dataframe to json
    data_json = json.dumps( x_test.to_dict( orient='records' ) )

    # local request
    url = 'http://0.0.0.0:5000/rossmann/predict'

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
            st.metric( label='Sales Predictions',  value='{:,.2f}'.format( sales_predictions), help='Sales predictions')  

        with col3:
            st.metric( label='Sales Average',  value='{:,.2f}'.format( sales_average ), delta='{:,.2f}'.format( sales ) , help='Sales average')      

        with col4:
            st.metric( label='MAE',  value='{:,.2f}'.format( MAE ), delta='{:,.2f}'.format( MAPE )+'%',  help='Mean Absolute Error')                   

        with col5:
            st.metric( label='Best Scenario',  value='{:,.2f}'.format( sales_predictions + SAR ),  help='Error')      

        with col6:
            st.metric( label='Worst Scenario',  value='{:,.2f}'.format( sales_predictions - SAR ),  help='Error')      

        return None
    
def sales_predictions_byday_chart( df ):
    df_grouped = df[['date', 'absolute_error', 'sales']].groupby( 'date' ).mean().reset_index()

    with st.container():
        fig = go.Figure()
        fig.add_trace(go.Scatter( x=df_grouped['date'].values,
                                  y=df_grouped['absolute_error'].values,
                                  mode='lines',
                                  name='Absolute error average',
                                  line=dict( color='royalblue' )
                                 ) )
        
        fig.add_trace(go.Scatter( x=df_grouped['date'].values,
                                  y=df_grouped['sales'].values,
                                  mode='lines',
                                  name='sales average',
                                  line=dict( color='firebrick' )
                                  ) )
        
        fig.update_layout(title='Sales and Sales Predictions in the next 6 weeks',
                   xaxis_title='Dates',
                   yaxis_title='Amount ( in millions )')

        st.plotly_chart( fig, use_container_width=True )

    return None

def error_distribuition_chart( df, MAPE_model ):
    half_MAPE_model = ( MAPE_model/2 )    

    st.markdown( MAPE_model)
    st.markdown( half_MAPE_model)

    df_error_level         = df[['store', 'absolute_error', 'sales']].groupby( 'store' ).mean().reset_index()
    df_error_level['MAPE'] = df_error_level['absolute_error'] / df_error_level['sales'] * 100

    df_error_level['error_level'] = df_error_level['MAPE'].apply( lambda x : 'L1' if ( x <= half_MAPE_model )                                         else 
                                                                             'L2' if ( ( x > half_MAPE_model ) & ( x <= MAPE_model ) )                else 
                                                                             'L3' if ( ( x > MAPE_model ) & ( x <= ( MAPE_model+half_MAPE_model ) ) ) else 
                                                                             'L4' )
    
    st.dataframe(df_error_level)
    
    df_error_level = df_error_level[['error_level', 'store']].groupby('error_level').count().reset_index().rename( columns = {'store' : 'count' } )

    total_count = df_error_level['count'].sum()

    df_error_level['error_level_percentage'] = df_error_level['count'] / total_count * 100
    df_error_level['error_level_text']       = df_error_level['count'].astype(str) + ' (' + round( df_error_level['error_level_percentage'], 2 ).astype(str) + '%)'
    
    fig = px.bar( df_error_level, x='error_level', y='count', text='error_level_text', 
                  labels={ 'error_level' : 'Error Level: <L1> Low   <L2> Medium   <L3> Medium High   <L4> High',
                           'count' : 'Store count' } )
    st.plotly_chart( fig, use_container_width=True )

    return None

def grouped_stores( df ):
    with st.container():
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

    #df = df[ df['store'].isin( df_store['store'].values ) ]

    return df_store['store'].values 

# >>> Main Function
def main():
    # configure page title
    st.set_page_config( page_title='Rossmann Monitor - Main Page', layout='wide' )  

    # create test dataset
    x_test = load_data( 'data/test.csv' )

    start_date_period = x_test['date'].min()
    end_date_period   = x_test['date'].max()

    st.markdown( '# Welcome to Rossmann Monitor' )
    st.markdown(  start_date_period )
    st.markdown( end_date_period)

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

        MAPE_model = round( ( df_pred['absolute_error'].mean() / df_pred['sales'].mean() ) * 100, 2 )

        if option == 'Score of stores':
            df_pred = df_pred[ df_pred['store'].isin( score_of_stores( df_pred, score_radio, score_slider ) ) ]

        elif option == 'Choose stores':
            df_pred = df_pred[ df_pred['store'].isin( choose_select ) ]

        model_metrics( df_pred )

        tab1, tab2, tab3 = st.tabs(['Sales and Sales Predictions by Day', 'Error Distribuition Chart', 'Stores Score Table'])

        with tab1: 
            sales_predictions_byday_chart( df_pred )   

        with tab2:
            error_distribuition_chart( df_pred, MAPE_model )            

        with tab3: 
            df_store = grouped_stores( df_pred )
            st.dataframe(df_store.head(600), hide_index=True )


    return None
   
# >>> Call Main Function
if __name__ == "__main__":
    main()
    