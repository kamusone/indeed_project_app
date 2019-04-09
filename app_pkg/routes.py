
from app_pkg import app
from flask import render_template
from flask import request
import pymongo
import pandas as pd
import indeed_script_01 as ind1
import indeed_script_02 as ind2
import main as Vs



@app.route('/indeed', methods=['GET', 'POST'])
def indeed():
    if request.method == 'POST':
        job=request.form['metier']
        city=request.form['ville']
        ind1.global_fuction(job, city)

    return render_template('indeed_home.html')

@app.route('/model', methods=['GET', 'POST'])
def get_model():
    if request.method == 'POST':
    	#ind2.save_indeed()
    	#ind2.save_accuracy_csv()
    	#ind2.save_confusion_matrix_csv()
    	#ind2.save_confusion_matrix_svm_csv()
    	#ind2.save_classification_report_rfc_csv()
    	#ind2.save_classification_report_svm_csv()
    	ind2.send_mail()
    	Vs.show_shar()
        

    return render_template('model.html')