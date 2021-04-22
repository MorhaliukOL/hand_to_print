import os
import tensorflow as tf

from flask import (Flask, url_for, render_template, request, flash, redirect)
from werkzeug.utils import secure_filename

from config import DevelopmentConfig
from infer import get_text_from_img

app = Flask(__name__)
app.config.from_object(DevelopmentConfig())

CWD = os.getcwd()
UPLOAD_FOLDER = os.path.join(CWD, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

MODEL = 'm1'
MODEL_PATH = os.path.join(CWD, 'saved_models', MODEL)
CHARLIST_PATH = os.path.join(CWD, 'saved_models', MODEL, 'charList.txt') 


def allowed_file(filename):
    """Return True if file extension in ALLOWED_EXTENSIONS, otherwise - False"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url) 
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            img_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(img_path)

            text, _ = get_text_from_img(img_path, CHARLIST_PATH, MODEL_PATH)
            os.remove(img_path)
            return render_template('home.html', data=text)
        else:
            flash('Not supported file format')
            return redirect(request.url)
        return redirect(request.url)

    else:
        return render_template('home.html')
