from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json


def main():    
    print("Init Data")   
    
    url = 'https://sandbox-api.coinmarketcap.com/v1/cryptocurrency/quotes/historical'
        
    parameters = {
      'interval':'5m',
      'count':'20',
      'symbol':'BTC',
      'convert':'USD'
    }
    
    headers = {
      'Accepts': 'application/json',
      'X-CMC_PRO_API_KEY': '7b070305-6d08-488d-935a-7645431b5a11',
    }
    
    
    session = Session()
    session.headers.update(headers)
    
    try:
      response = session.get(url, params=parameters)
      print(response)
      data = json.loads(response.text)
      print(data)
    except (ConnectionError, Timeout, TooManyRedirects) as e:
      print(e)
  

if __name__ == "__main__":
    main()
