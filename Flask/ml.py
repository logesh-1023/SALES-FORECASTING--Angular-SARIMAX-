from flask import Flask,request,jsonify
app=Flask(__name__)
from flask_cors import CORS
CORS(app)
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
import matplotlib.pyplot as  plot
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

@app.route('/flask',methods=['POST'])
def hello():
    global qemail
    f=request.files['file']
    f.save(f.filename+".csv")
    qemail=request.form['email']
    return jsonify("success")

@app.route('/flask',methods=['GET'])
def result():
    resultz=""
    li=[]
    # CSV LOADING 
    power_bi=pd.read_csv('onFileSelected.csv')

    df=power_bi[['Month','Stocks_Sold']].copy()

    df.columns=["Month","Sales"]
    ## Drop last 2 rows
    df.drop(106,axis=0,inplace=True)
    df.drop(105,axis=0,inplace=True)

    # Convert Month into Datetime
    df['Month']=pd.to_datetime(df['Month'])

    df.set_index('Month',inplace=True)

    ### Testing For Stationarity

    from statsmodels.tsa.stattools import adfuller
    test_result=adfuller(df['Sales'])

    def adfuller_test(sales):
        result=adfuller(sales)
        labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
        for value,label in zip(result,labels):
            li.append(label+' : '+str(value))
        if result[1] <= 0.05:
            li.append("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
        else:
            li.append("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")

    adfuller_test(df['Sales'])

    #Differencing
    df['Sales First Difference'] = df['Sales'] - df['Sales'].shift(1)
    df['Sales'].shift(1)

    #Seasonal 

    df['Seasonal First Difference']=df['Sales']-df['Sales'].shift(12)

    adfuller_test(df['Seasonal First Difference'].dropna())

    #Sarimax
    import statsmodels.api as sm
    model=sm.tsa.statespace.SARIMAX(df['Sales'],order=(1, 1, 1),seasonal_order=(1,1,1,12))
    results=model.fit()

    #Prediction
    df['forecast']=results.predict(start=90,end=103,dynamic=True)
    df[['Sales','forecast']].plot(figsize=(12,8))

    #future Prediction
    from pandas.tseries.offsets import DateOffset

    future_dates=[df.index[-1]+ DateOffset(months=x)for x in range(0,24)]
    future_datest_df=pd.DataFrame(index=future_dates[1:],columns=df.columns)

    future_df=pd.concat([df,future_datest_df])
    future_df['forecast'] = results.predict(start = 104, end = 120, dynamic= True)  
    future_df[['Sales','forecast']].plot(figsize=(12, 8))
    plot.show()

    #PowerBI 
    power1=power_bi[['Month','CustomerID','State','Region']].copy()
    power1=power1.join(df['Sales'])
    v=[]
    for value in df['Sales']:
            v.append(value)     
    mySeries=pd.Series(v)
     
    power1['Sales']=mySeries

    power1.to_csv('processed.csv')

    #mailer

    fromaddr = "logeshfire6@gmail.com"
    toaddr =qemail
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = "PROCESSED FILE AFTER MACHINE LEARINING"
    body = "HI,This is the processed file attached for future purpose!"
    msg.attach(MIMEText(body, 'plain'))
    filename = "processed.csv"
    attachment = open(r"C:\Users\91701\OneDrive\Desktop\Kaar_technologies\processed.csv", "rb")
    p = MIMEBase('application', 'octet-stream')
    p.set_payload((attachment).read())
    encoders.encode_base64(p)
    p.add_header('Content-Disposition', "attachment; filename= %s" % filename)
    msg.attach(p)
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login(fromaddr, "jatxquxctpxcqklg")
    text = msg.as_string()
    s.sendmail(fromaddr, toaddr, text)
    s.quit()
    
    resultz=",".join(li)

    return jsonify(resultz)


if __name__=="__main__": 
    app.run(debug=True,port=5000)

