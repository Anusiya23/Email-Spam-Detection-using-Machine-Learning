import numpy as np
import pandas as pd

import pymysql
pymysql.install_as_MySQLdb()
import MySQLdb

import nltk
import re

gmail_list=[]
password_list=[]
gmail_list1=[]
password_list1=[]

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from nltk.corpus import stopwords

stop_word = stopwords.words('english')

from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()


from flask import Flask,request,jsonify, render_template



import joblib
import os

script_directory = os.path.dirname(os.path.realpath(__file__))

file_path = os.path.join(script_directory, 'final_pickle_model.pkl')

if os.path.exists(file_path):
    model = joblib.load(file_path)
else:
    print(f"Error: File '{file_path}'not found.")

from flask import Flask,request,jsonify, render_template
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('register.html')


  

@app.route('/register',methods=['POST'])
def register():

    int_features2=[str(x) for x in request.form.values()]

    r1=int_features2[0]
    print(r1)
    
    r2=int_features2[1]
    print(r2)
    logu1=int_features2[0]
    passw1=int_features2[1]



    import MySQLdb

    db = MySQLdb.connect(host="localhost", user="root", passwd="", db="ddbb")


    cursor = db.cursor()
    cursor.execute("SELECT user FROM user_register")
    result1 = cursor.fetchall()
    

    for row1 in result1:
        print(row1)
        print(row1[0])
        gmail_list1.append(str(row1[0]))

    print(gmail_list1)
    if logu1 in gmail_list1:
        return render_template('register.html',text="This Username is already in use")
    
    else:
        sql = "INSERT INTO user_register(user,password) VALUES(%s,%s)"
        val = (r1,r2)

        try:
            cursor.execute(sql,val)
            db.commit()

        except:
            db.rollback()

        db.close()
        return render_template('register.html',text="Successfully  Registered")


@app.route('/login')
def login():
    # Your login route logic here
    return render_template('login.html')

@app.route('/logedin', methods=['POST'])

def logedin():

    int_features3 = [str(x) for x in request.form.values()]
    print(int_features3)
    logu=int_features3[0]
    passw=int_features3[1]

    import MySQLdb

    db = MySQLdb.connect(host="localhost", user="root", passwd="", db="ddbb")


    cursor = db.cursor()
    cursor.execute("SELECT user FROM user_register")
    result1 = cursor.fetchall()
    

    for row1 in result1:
        print(row1)
        print(row1[0])
        gmail_list.append(str(row1[0]))

    print(gmail_list)
    cursor1=db.cursor()
    cursor1.execute("SELECT password FROM user_register")
    result2=cursor1.fetchall()


    for row2 in result2:
        print(row2)
        print(row2[0])
        password_list.append(str(row2[0]))

    print(password_list)
    print(gmail_list.index(logu))
    print(password_list.index(passw))


    try:
       password_index = password_list.index(passw)
    except ValueError:
       password_index = -1



    if gmail_list.index(logu) == password_index:
        return render_template('index1.html')
    else:
        return render_template('login.html',text='Use Proper Username and Password')
    
    


@app.route('/production')
def production():
    return render_template('index1.html')


@app.route('/production/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [str(x) for x in request.form.values()]
    a=int_features

    msg=str(a)

    filter_sentence = ''

    sentence = re.sub(r'[^\w\s]','',msg)

    words = nltk.word_tokenize(sentence)

    words = [w for w in words if not w in stopwords.words('english')]


    for word in words:
        filter_sentence = filter_sentence + ' ' + str(lemmatizer.lemmatize(word)).lower()

        data= [filter_sentence]

    print(data)



    my_prediction = model.predict(data)
    my_prediction = int(my_prediction)
    print(my_prediction)

    if my_prediction==1:
        print("This Mail is Spam")

        return render_template('index1.html',prediction_text="This Mail is Spam")
    else:
        print("This Mail is Real")
        return render_template('index1.html',prediction_text="This Mail is Real")



if  __name__ == "__main__":
    app.run(debug=False)
    
