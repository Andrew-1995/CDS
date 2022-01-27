from flask import Flask,request,jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
from sklearn.neighbors import KNeighborsClassifier


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

    input_query = np.array([[age,bps,temp,chest_pain]])

    result = model.predict(input_query)[0]
    return jsonify({'The Result of Prediction is:':str(result)})
if __name__ == '__main__':
    app.run(debug=True)
