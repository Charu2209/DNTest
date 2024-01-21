from flask import Flask,render_template,request
#import pickle
import main as m
#import json

app = Flask(__name__)

@app.route('/',methods=['POST','GET'])
def new():
    return render_template('new.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    data1 = (request.form['text1'])
    data2 = (request.form['text2'])

    #data = request.json
    #data1=data.get('text1')
    #data2 = data.get('text2')
    
   # model=pickle.load(open('model.pickle','rb'))
    data = m.tagged_data(data1,data2)
    m.model_training(data)
    pred = m.similarity_score(0)[0][1]
    
    prediction_statement = f"Similarity Score is:{pred}"

    #output = {"similarity score":pred}
    return render_template('new.html',statement=prediction_statement)
    #return json.dumps(output)


if __name__=='__main__':
    app.run()