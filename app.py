from flask import Flask,request,jsonify
import numpy as np
import pickle


import numpy as np
import pandas as pd
from basic_knn import knn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
from sklearn.neighbors import KNeighborsClassifier

from sklearn.neighbors import NearestNeighbors

df = pd.read_csv('students_placement.csv')
df.sample(5)
X = df.drop(columns=['placed'])
y = df['placed']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

#train the model
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
print(accuracy_score(y_test,y_pred))
#save the model in pickle format
#pickle.dump(knn,open('model.pkl','wb'))

#knn = NearestNeighbors(10)

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X, y)

#knn.fit(y_pred)

# Its important to use binary mode
knnPickle = open('model.pkl', 'wb')

# source, destination
pickle.dump(knn, knnPickle)


# load the model from disk
#loaded_model = pickle.load(open('model.pkl', 'rb'))
#result = loaded_model.predict(X_test)








model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)
@app.route('/')
def index():
    return "Hello world"
@app.route('/predict',methods=['POST'])
def predict():
    cgpa = request.form.get('cgpa')
    iq = request.form.get('iq')
    profile_score = request.form.get('profile_score')

    input_query = np.array([[cgpa,iq,profile_score]])

    result = model.predict(input_query)[0]
    return jsonify({'placement':str(result)})
if __name__ == '__main__':
    app.run(debug=True)
