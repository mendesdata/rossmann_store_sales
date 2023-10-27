from   flask             import Flask, request, Response
from   api.rossmann.Rossmann import Rossmann
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
            df_test_raw = pd.DataFrame( test_json, index=[0] )
        
        # Multiple Examples
        else:
            df_test_raw = pd.DataFrame( test_json, columns=test_json[0].keys() )

        # Instanciate Rossmann Class
        pipeline = Rossmann()

        # Data Cleaning
        df_test = pipeline.data_cleaning( df_test_raw )

        # Feature Engineering
        df_test = pipeline.feature_engineering( df_test )

        # Data Preparation
        df_test = pipeline.data_preparation( df_test )

        # Prediction
        df_response = pipeline.get_prediction( model, df_test_raw, df_test )

        return df_response

    # there is no data
    else:
        return Response( '{}', status=200, mimetype='application/json' )
    

    

if __name__ == '__main__':
    app.run( '0.0.0.0' )