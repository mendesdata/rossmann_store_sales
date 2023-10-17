from   flask             import Flask, request, Response
from   rossmann.Rossmann import Rossmann
import pandas as pd
import pickle


# loading model 
model = pickle.load( open( '/home/datamendes/comunidadeds/projetos/rossmann_store_sales/model/model_rossmann.pkl', 'rb' ) )

# Initialize API
app = Flask( __name__ )
@app.route( '/rossmann/predict', methods=['POST'] )
def rossmann_predict():
    test_json = request.get_json()

    # there is data
    if test_json:

        # Unique Example
        if isinstance( test_json, dict ):
            test_raw = pd.DataFrame( test_json, index=[0] )
        
        # Multiple Examples
        else:
            test_raw = pd.DataFrame( test_json, columns=test_json[0].keys() )

        # Instanciate Rossmann Class
        pipeline = Rossmann()

        # Data Cleaning
        df1 = pipeline.data_cleaning( test_raw )

        # Feature Engineering
        df2 = pipeline.feature_engineering( df1 )

        # Data Preparation
        df3 = pipeline.data_preparation( df2 )

        # Prediction
        df_response = pipeline.get_prediction( model, test_raw, df3 )

        return df_response

    # there is no data
    else:
        return Response( '{}', status=200, mimetype='application/json' )
    

    

if __name__ == '__main__':
    app.run( '0.0.0.0' )