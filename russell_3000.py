import requests

url = 'https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund&asOfDate=20221230'
response = requests.get(url)

with open('russell-3000.csv', 'wb') as f:
    f.write(response.content)