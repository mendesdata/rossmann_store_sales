import pandas as pd
import json
import requests

from Flask import flask, request, Response

# BOT token constant
TOKEN = '6406278029:AAHkTUkj9tS2YJvzdo1l0yz6hQrSKQywwgY'

# Info about BOT
#https://api.telegram.org/bot6406278029:AAHkTUkj9tS2YJvzdo1l0yz6hQrSKQywwgY/getMe

# get_update
#https://api.telegram.org/bot6406278029:AAHkTUkj9tS2YJvzdo1l0yz6hQrSKQywwgY/getUpdates

# get_update
#https://api.telegram.org/bot6406278029:AAHkTUkj9tS2YJvzdo1l0yz6hQrSKQywwgY/sendMessage?chat_id

def send_message( chat_id, text ):
    url = https://api.telegram.org/bot{}.format( TOKEN )
    url = url + 'sendMessage?chat_id={}'.format( chat_id )

    response = requests.post( url, json={'text' : text})
    print(f'Status Code { response.status_code }' )

    return None

def load_data( store_id ):
    # load datasets: prod + store
    df_prod_raw  = pd.read_csv( '../data/prod.csv' )
    df_store_raw = pd.read_csv( '../data/store.csv' )

    # merging datasets: prod + store
    df_prod = pd.merge( df_prod_raw, df_store_raw, how='left', on='Store' )
    df_prod = df_prod.drop( "Id", axis=1 )
    df_prod.head()

    # choose store for prediction
    df_test = df_prod[  df_prod['Store'].isin( [store_id] ) ]

    # remove close day
    df_test = df_test[ df_test['Open'] != 0 ]
    df_test = df_test[ ~df_test['Open'].isnull() ]


    # convert dataframe to json
    data = json.dumps( df_test.to_dict( orient='records' ) )

    return data

def predict( data ):
    # API CALL - Render´s Server request
    url = 'https://rossmann-app-9l04.onrender.com/rossmann/predict'
    header = { 'Content-type' : 'application/json' }
    response = requests.post( url, data=data, headers=header )
    print(f'Status Code: { response.status_code } ' )

    # return predictions - converts json to dataframe
    df = pd.DataFrame( response.json(), columns=response.json()[0].keys() )

    return df

# d2 = d1[['store', 'predictions']].groupby( 'store' ).sum().reset_index()

# for i in range( len( d2 ) ):
#    print(f'Store Number { d2.loc[i, "store"] } will sell R${ round( d2.loc[i, "predictions"], 2 )} in the next 6 weeks' )

# API Initialize
app = Flask( __name__ )
@app.route( '/', methods=['GET', 'POST'] )

def parse_message( message ):
    chat_id = message['message']['chat']
    return chat_id, store_id

def index():
    if request.method == 'POST':
        message = request.get_json()

        chat_id, store_id = parse_message ( message )
    else:
        return '<h1> Rossmann Telegram BOT </h1>'

if __name__ == '__main__':
    app.run( host='0.0.0.0', port=5000 )

6406278029:AAHkTUkj9tS2YJvzdo1l0yz6hQrSKQywwgY