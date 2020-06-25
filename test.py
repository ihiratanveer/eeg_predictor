import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from commons import get_model

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'edf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'GET':
        return render_template('main.html')
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
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            eeg_result = get_model(filename)
            if eeg_result == 0:
                result_call = "The system detected a NORMAL EEG. You may consult your designated Neurologist for further assistance."
            elif eeg_result == 1:
                result_call = "The system detected an ABNORMAL EEG. Please consult your designated Neurologist for further assistance."
            return render_template('result.html', eeg=result_call)
        else:
            return render_template('main.html', value="Please enter .edf file only!")
@app.route('/templates/about.html')
def about():
    return render_template('about.html')

@app.route('/templates/main.html')
def main():
    return render_template('main.html')
     
      
        
        
        
    


if __name__ == '__main__':
    app.debug = True
    app.run()