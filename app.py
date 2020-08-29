from flask import Flask,render_template, redirect,request,url_for,request
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import heapq
import sys
import subprocess
import os
import re
import json
import requests
import ast

from inference import inference, see_lablador_prob, build_model

UPLOAD_PATH = 'static/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
model = build_model()

app = Flask(__name__)
app.config['UPLOAD_PATH'] = UPLOAD_PATH

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload',methods=['POST'])
def upload():
    if request.method == "POST":
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_PATH'], filename)
            file.save(file_path)
            result_top_1_label, result_top_1_prob, result_top_1_verbose = inference(model, file_path)
            result_lablador_prob_verbose = see_lablador_prob(model, file_path)
            return render_template('result.html', enumerate=enumerate, img_path=os.path.basename(file_path), result_top_1_verbose=result_top_1_verbose, result_top_1_label=result_top_1_label, result_lablador_prob_verbose=result_lablador_prob_verbose)
        else:
            return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')