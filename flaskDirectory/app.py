from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

@app.route('/')
def student():
   return render_template('form.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   global clf
   if request.method == 'POST':
      result = request.form
      print(result)
      input = []
      for i in result:
         input.append(float(result[i]))
      print(input)
      res = {'Diabetes Prediction':int(clf.predict([input]))}

      return render_template("result.html",result = res)

if __name__ == '__main__':

   """
   Setting up data
   """
   df = pd.read_csv('/Users/aakashsbhatia/Projects/SBUHacks/flaskDirectory/diabetes-dataset.csv')

   X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin',
           'BMI', 'DiabetesPedigreeFunction', 'Age']]
   Y = df['Outcome']

   X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
   clf = KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree')
   clf.fit(X_train, y_train)

   app.run(debug = True)