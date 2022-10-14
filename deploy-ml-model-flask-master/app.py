from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model_pickle_GradientBoostingClassifier.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    #   HighBloodPressure,HighCholesterol,CheckCholesterolwithinpastfiveyears,BMI,Smoker,HaveaStroke,Diabetes,physicalactivity
    #         ,generalhealth,difficultywalking,Sex,agecategory,grade,income,mental,physical
    data1 = request.form['HighBloodPressure']
    data2 = request.form['HighCholesterol']
    data3 = request.form['CheckCholesterolwithinpastfiveyears']
    data4 = request.form['BMI']
    
    data5 = request.form['Smoker']
    data6 = request.form['HaveaStroke']
    data7 = request.form['Diabetes']
    data8 = request.form['physicalactivity']
    
    data9 = request.form['generalhealth']
    data10 = request.form['difficultywalking']
    data11 = request.form['Sex']
    data12 = request.form['agecategory']

    data13 = request.form['grade']
    data14 = request.form['income']
    data15 = request.form['mental']
    data16 = request.form['physical']

    

    arr = np.array([[data1, data2, data3, data4, data5, data6,
     data7, data8, data9, data10, data11, data12
    ,data13, data14, data15, data16]])

    pred = model.predict(arr)
    return render_template('after.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)















