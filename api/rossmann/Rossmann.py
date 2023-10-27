import pickle
import inflection
import pandas as pd
import numpy as np
import math
import datetime

class Rossmann( object ):
    def __init__(self):
        self.home_path= '/home/datamendes/comunidadeds/projetos/rossmann_store_sales'
        self.competition_distance_scaler   = pickle.load( open( self.home_path + '/parameters/competition_distance_scaler.pkl', 'rb' ) )
        self.promo2_time_week_scaler       = pickle.load( open( self.home_path + '/parameters/promo2_time_week_scaler.pkl', 'rb' ) )
        self.competition_time_month_scaler = pickle.load( open( self.home_path + '/parameters/competition_time_month_scaler.pkl', 'rb' ) )
        self.store_type_scaler             = pickle.load( open( self.home_path + '/parameters/store_type_scaler.pkl', 'rb' ) )
        self.year_scaler                   = pickle.load( open( self.home_path + '/parameters/year_scaler.pkl', 'rb' ) )


    def data_cleaning( self, df ):
        ## Rename Columns
        cols_old = ['Store', 'DayOfWeek', 'Date', 'Customers', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment', 'CompetitionDistance',
                    'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']


        snakecase = lambda x: inflection.underscore( x )
        cols_new = list( map( snakecase, cols_old ) )

        # rename
        df.columns = cols_new

        # change data type "date"
        df['date'] = pd.to_datetime( df['date'] )

        # competition_distance - competition_distance with NA values means "no competitor around". Set max value = 200000
        max_value = df['competition_distance'].max()
        df['competition_distance'] = df['competition_distance'].apply( lambda x: 200000 if math.isnan( x ) else x )

        # competition_open_since_month and competition_open_since_year - Set month and year of sale as default value
        df['competition_open_since_month'] = df.apply( lambda x: x['date'].month if math.isnan( x['competition_open_since_month'] ) else x['competition_open_since_month'], axis=1 )
        df['competition_open_since_year']  = df.apply( lambda x: x['date'].year if math.isnan( x['competition_open_since_year'] )   else x['competition_open_since_year'], axis=1 )

        # promo2_since_week  and promo2_since_year - Set month and year of sale as default value
        df['promo2_since_week'] = df.apply( lambda x: x['date'].week if math.isnan( x['promo2_since_week'] ) else x['promo2_since_week'], axis=1 )
        df['promo2_since_year'] = df.apply( lambda x: x['date'].year if math.isnan( x['promo2_since_year'] ) else x['promo2_since_year'], axis=1 )


        # promo_interval - first, create a new column(month_map) with the month of sale  
        month_map = { 1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec' }
        df['promo_interval'].fillna( 0, inplace=True )
        df['month_map'] = df['date'].dt.month.map( month_map )

        # second, create a new column(is_promo) to check two conditions:  if promo_interval is active (1) and if month_map is inside promo_interval. (0) No, (1) Yes
        df['is_promo'] = df[['promo_interval', 'month_map']].apply( lambda x: 0 if x['promo_interval'] == 0 else 1 if x['month_map'] in x['promo_interval'].split( ',' ) else 0, axis=1 ) 


        # change data types from float to int
        df['competition_open_since_month'] = df['competition_open_since_month'].astype( int )
        df['competition_open_since_year'] = df['competition_open_since_year'].astype( int )
        df['promo2_since_week'] = df['promo2_since_week'].astype( int )
        df['promo2_since_year'] = df['promo2_since_year'].astype( int )

        return df

    def feature_engineering( self, df ):
        # year
        df['year'] = df['date'].dt.year

        # month
        df['month'] = df['date'].dt.month

        # day
        df['day'] = df['date'].dt.day

        # week of year
        df['week_of_year'] = df.apply( lambda x: datetime.date( x['year'], x['month'], x['day'] ).isocalendar().week, axis=1 )

        # year week
        df['year_week'] = df.apply( lambda x: str( x['year'] ) + '-' + str( x['week_of_year'] ), axis=1 )

        # from competition_open_since columns
        df['competition_since']      = df.apply( lambda x: datetime.datetime( year=x['competition_open_since_year'], month=x['competition_open_since_month'], day=1 ), axis=1 )
        df['competition_time_month'] = df.apply( lambda x: ( ( x['date'] - x['competition_since'] ) / 30 ).days, axis=1 ).astype( int )

        # from promo2_since columns
        df['promo2_since']     = df['promo2_since_year'].astype( str ) + '-' + df['promo2_since_week'].astype( str )
        df['promo2_since']     = df['promo2_since'].apply( lambda x: datetime.datetime.strptime( x +'-1', '%Y-%W-%w' ) - datetime.timedelta( days=7 ) )
        df['promo2_time_week'] = df.apply( lambda x: ( ( x['date'] - x['promo2_since'] ) / 7 ).days, axis=1 ).astype( int )

        # assortment level: a = basic, b = extra, c = extended
        df['assortment'] = df['assortment'].apply( lambda x: 'basic' if x == 'a' else 'extra' if x =='b' else 'extended' )

        #StateHoliday - indicates a state holiday - a = public holiday, b = easter holiday, c = christmas, 0 = regular day
        df['state_holiday'] = df['state_holiday'].apply( lambda x: 'public holiday' if x == 'a' else 'easter holiday' if x == 'b' else 'christmas' if x == 'c' else 'regular day' )

        # remove lines where there are no sales 
        df = df[ ( df['open'] != 0 ) ]

        # customers -> Quantidade de clientes nas lojas, não é possível saber quantos clientes estarão nas lojas na predição. 
        # open -> Se a loja está aberta ou não. Quando está fechada não há vendas, ou seja, devem ser consideradas apenas as linhas com loja aberta ( open != 0 )
        # sales -> Valor total em vendas. Quando não há vendas desconsiderar linhas ( sales > 0 )
        # colunas que foram criadas apenas para auxiliar a geração de outras também devem ser excluídas. Ex: promo_interval, month_map
        cols_drop = ['customers', 'open', 'promo_interval', 'month_map']
        df = df.drop( cols_drop, axis=1 )

        return df

    def data_preparation( self, df ):
        df['promo2_time_week']       = self.promo2_time_week_scaler.fit_transform( df[['promo2_time_week']].values )
        df['competition_distance']   = self.competition_distance_scaler.fit_transform( df[['competition_distance']].values )
        df['competition_time_month'] = self.competition_time_month_scaler.fit_transform( df[['competition_time_month']].values )
        df['year']                   = self.year_scaler.fit_transform( df[['year']].values )
        df['store_type']             = self.store_type_scaler.fit_transform( df['store_type'] )

        # Apply One-Hot Encoding
        df = pd.get_dummies( df, prefix=['state_holiday'], columns=['state_holiday'] )

        # Apply Ordinal Encoding
        assortment_dict = { 'basic' : 1, 'extra' : 2, 'extended' : 3 }
        df['assortment'] = df['assortment'].map( assortment_dict )

        # Calculate sin and cos 
        df['month_sin'] = df['month'].apply( lambda x: np.sin( x * ( 2 * np.pi/12 ) ) )
        df['month_cos'] = df['month'].apply( lambda x: np.cos( x * ( 2 * np.pi/12 ) ) )

        # Calculate sin and cos 
        df['day_of_week_sin'] = df['day_of_week'].apply( lambda x: np.sin( x * ( 2 * np.pi/7 ) ) )
        df['day_of_week_cos'] = df['day_of_week'].apply( lambda x: np.cos( x * ( 2 * np.pi/7 ) ) )

        # Calculate sin and cos 
        df['day_sin'] = df['day'].apply( lambda x: np.sin( x * ( 2 * np.pi/30 ) ) )
        df['day_cos'] = df['day'].apply( lambda x: np.cos( x * ( 2 * np.pi/30 ) ) )

        # Calculate sin and cos 
        df['week_of_year_sin'] = df['week_of_year'].apply( lambda x: np.sin( x * ( 2 * np.pi/52 ) ) )
        df['week_of_year_cos'] = df['week_of_year'].apply( lambda x: np.cos( x * ( 2 * np.pi/52 ) ) )

        # features selected 
        #cols_selected = ['store', 'promo', 'store_type', 'assortment', 'competition_distance', 'competition_open_since_month',
        #                 'competition_open_since_year', 'promo2', 'promo2_since_week', 'promo2_since_year', 'competition_time_month', 'promo2_time_week', 
        #                 'day_of_week_sin', 'day_of_week_cos', 'day_sin', 'day_cos', 'week_of_year_sin', 'week_of_year_cos', 'month_sin', 'month_cos']
        
        cols_selected = ['store', 'promo', 'store_type', 'assortment', 'competition_distance', 'competition_open_since_month',
                         'competition_open_since_year', 'promo2', 'promo2_since_week', 'promo2_since_year', 'competition_time_month', 'promo2_time_week', 
                         'day_of_week_sin', 'day_of_week_cos', 'day_sin', 'day_cos', 'week_of_year_cos', 'month_cos']

        

        return df[cols_selected]
    
    def get_prediction( self, model, original_data, test_data ):
        # prediction
        pred = model.predict( test_data )

        # join pred into original data
        original_data['sales_predictions'] = np.expm1( pred )

        return original_data.to_json( orient='records', date_format='iso' )
