import numpy as np
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')



nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')




data = pd.read_csv('spam.csv', encoding='latin1')
print(data.head())
print(data.shape)

data=data.drop_duplicates()
print(data.shape)

print(data.isnull().sum())

data=data.dropna()

print(data.isnull().sum())


data['Label']=data['Label'].map({'ham':0,'spam':1})

print(data.head())

data["Label"].value_counts().plot(kind="bar",color=["salmon","lightblue"])
plt.xlabel("0 =Real , 1 = Spam ")
plt.title("Email Spam Prediction")
plt.show()



s = "!</> hello please$$ </>^!!!%%&&%$@@@attend^^^&&!% </>*@% the&& @@@class##%^^&!@# %%$"
s = re.sub(r'[^\w\s]','',s)
print(s)


k=nltk.word_tokenize("Hello how are you")
print(k)

from nltk.corpus import stopwords

stop_words = stopwords.words('english')
print(stop_words)

sentence = "Covid-19 pandemic has impacted many countries and what it did to economy is very stressful"

words = nltk.word_tokenize(sentence)
words = [w for w in words if w not in stop_words]

print(words)


from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()

input_str="been had done language cities mice"

input_str=nltk.word_tokenize(input_str)

for word in input_str:
    print(lemmatizer.lemmatize(word))




for index,row in data.iterrows():
    filter_sentence = ''


    sentence = row['EmailText']
    sentence = re.sub(r'[^\w\s]','',sentence)

    words = nltk.word_tokenize(sentence)

    words = [w for w in words if  not w in stop_words]

    for word in words:
        filter_sentence = filter_sentence + ' ' + str(lemmatizer.lemmatize(word)).lower()
        

    data.loc[index,'text'] = filter_sentence



print(data.head())





X = data['EmailText']
y= data['Label']
print(X)
print(y)




from sklearn.model_selection import train_test_split
# Remove rows with NaN values
data = data.dropna()

# Reset indices after dropping NaN values
data.reset_index(drop=True, inplace=True)

# Split data into features and target
X = data['EmailText']
y = data['Label']

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Reset indices after splitting
X_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)

# Check lengths of X_train and y_train
print('Length of X_train:', len(X_train))
print('Length of y_train:', len(y_train))

# Rest of your code for model training and evaluation


print('X_train :', len(X_train))
print('X_test :', len(X_test))
print('y_train :', len(y_train))
print('y_test :', len(y_test))


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


from sklearn.naive_bayes import MultinomialNB
clf1= MultinomialNB()

from sklearn.linear_model import LogisticRegression
clf2= LogisticRegression()

from xgboost import XGBClassifier
clf3= XGBClassifier()

from sklearn.ensemble import RandomForestClassifier
clf4= RandomForestClassifier(n_estimators=400)

from sklearn.neighbors import KNeighborsClassifier
clf5= KNeighborsClassifier(n_neighbors=5)

from sklearn.tree import DecisionTreeClassifier
clf6 = DecisionTreeClassifier() 

from sklearn.svm import SVC
clf7 = SVC(kernel='linear')



from sklearn.pipeline import Pipeline

model1 = Pipeline([
    ('vect',CountVectorizer()),
    ('tfidf', TfidfTransformer(norm='l2')),
    ('clf',clf1),
])

model2 = Pipeline([
    ('vect',CountVectorizer()),
    ('tfidf', TfidfTransformer(norm='l2')),
    ('clf',clf2),
])

model3 = Pipeline([
    ('vect',CountVectorizer()),
    ('tfidf', TfidfTransformer(norm='l2')),
    ('clf',clf3),
])

model4 = Pipeline([
    ('vect',CountVectorizer()),
    ('tfidf', TfidfTransformer(norm='l2')),
    ('clf',clf4),
])

model5 = Pipeline([
    ('vect',CountVectorizer()),
    ('tfidf', TfidfTransformer(norm='l2')),
    ('clf',clf5),
])

model6 = Pipeline([
    ('vect',CountVectorizer()),
    ('tfidf', TfidfTransformer(norm='l2')),
    ('clf',clf6),
])

model7 = Pipeline([
    ('vect',CountVectorizer()),
    ('tfidf', TfidfTransformer(norm='l2')),
    ('clf',clf7),
])



model1.fit(X_train, y_train)

model2.fit(X_train, y_train)

model3.fit(X_train, y_train)

model4.fit(X_train, y_train)

model5.fit(X_train, y_train)

model6.fit(X_train, y_train)

model7.fit(X_train, y_train)



predictions1 = model1.predict(X_test)

predictions2 = model2.predict(X_test)

predictions3 = model3.predict(X_test)

predictions4 = model4.predict(X_test)

predictions5 = model5.predict(X_test)

predictions6 = model6.predict(X_test)

predictions7 = model7.predict(X_test)


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions1)
print("Accuracy of Naive Bayes is {:.2f}%".format(accuracy * 100))

accuracy = accuracy_score(y_test, predictions2)
print("Accuracy of Logistic Regression is {:.2f}%".format(accuracy * 100))

accuracy = accuracy_score(y_test, predictions3)
print("Accuracy of XGB Classifier is {:.2f}%".format(accuracy * 100))

accuracy = accuracy_score(y_test, predictions4)
print("Accuracy of RandomForestClassifier is {:.2f}%".format(accuracy * 100))

accuracy = accuracy_score(y_test, predictions5)
print("Accuracy of KNN is {:.2f}%".format(accuracy * 100))

accuracy = accuracy_score(y_test, predictions6)
print("Accuracy of DecisionTree Classifier is {:.2f}%".format(accuracy * 100))

accuracy = accuracy_score(y_test, predictions7)
print("Accuracy of SVC is {:.2f}%".format(accuracy * 100))





import joblib

joblib.dump(model7, 'final_pickle_model.pkl')

final_model = joblib.load('final_pickle_model.pkl')

try:
    joblib.dump(model7, 'final_pickle_model.pkl')
    print("Model saved successfully.")
except Exception as e:
    print("Error occurred while saving the model:", e)

pred = final_model.predict(X_test)

accuracy=accuracy_score(y_test,pred)

print("Accuracy of Final Model is  {:.2f}%".format(accuracy*100))











