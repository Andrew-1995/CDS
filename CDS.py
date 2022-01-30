from _ast import In

import mapping as mapping
from flask import Flask,request,jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
from sklearn.neighbors import KNeighborsClassifier




df = pd.read_csv('students_placement.csv')

df['chest_pain']=df['chest_pain'].astype('str')
df['runny_nose']=df['runny_nose'].astype('str')
df['headache']=df['headache'].astype('str')
df['stuffy_nose']=df['stuffy_nose'].astype('str')
df['cough']=df['cough'].astype('str')
df['shortness_breath']=df['shortness_breath'].astype('str')
df['hoarseness']=df['hoarseness'].astype('str')
df['sore_throat']=df['sore_throat'].astype('str')
#df['chest_pain'] = df['chest_pain'].map({1:'yes', 0:'no'})
df.sample(5)

X = df.drop(columns=['placed'])
y = df['placed']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

#train the model
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
print(accuracy_score(y_test,y_pred))


knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X, y)


# Its important to use binary mode
knnPickle = open('model.pkl', 'wb')

# source, destination
pickle.dump(knn, knnPickle)

model = pickle.load(open('model.pkl','rb'))



app = Flask(__name__)
@app.route('/')
def index():
    return "Hello world"
@app.route('/predict',methods=['POST'])
def predict():
    age = request.form.get('age')
    bps = request.form.get('bps')
    temp = request.form.get('temp')
    chest_pain = request.form.get('chest_pain')
    runny_nose = request.form.get('runny_nose')
    headache = request.form.get('headache')
    stuffy_nose = request.form.get('stuffy_nose')
    cough = request.form.get('cough')
    shortness_breath = request.form.get('shortness_breath')
    hoarseness = request.form.get('hoarseness')
    sore_throat = request.form.get('sore_throat')

    input_query = np.array([[age,bps,temp,chest_pain,runny_nose,headache,stuffy_nose,cough,shortness_breath,hoarseness,sore_throat]])

    result = model.predict(input_query)[0]
    return jsonify({'The Result of Prediction is:':str(result)})
if __name__ == '__main__':
    app.run(debug=True)