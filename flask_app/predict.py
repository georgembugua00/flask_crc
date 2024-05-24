from flask import Flask,render_template,request
import pickle
import joblib

app = Flask(__name__)

with open('best_rf_model.joblib', 'rb') as f:
    loaded_model = joblib.load(f)

@app.route('/')
def home():
    name_id = input(prompt="Enter your name: ")
    name_id = "Hello ()"
    return render_template("home.html",name_id=name_id)

@app.route("/predict",methods=["POST"])
def predict():
    person_age = int(request.form['person_age'])
    person_income = int(request.form['person_income'])
    person_emp_length = int(request.form['person_emp_length'])
    loan_amnt = int(request.form['loan_amnt'])
    loan_int_rate = float(request.form['loan_int_rate'])
    cb_person_cred_hist_length = int(request.form['cb_person_cred_hist_length'])
    result = loaded_model.predict([[person_age,person_income,person_emp_length,loan_amnt,loan_int_rate,cb_person_cred_hist_length]])[0]
    if result == 1:
        result = "high risk"
    else:
        result = "low risk"   
    
    return render_template("predict.html",result=result)




if __name__ == "__main__":
    app.run(debug=True)
