from flask import Flask, request, render_template, redirect, url_for
from model.detect import detect_features
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        result = detect_features(filepath)
        return render_template('result.html', image_path=filepath, result=result)

if __name__ == '__main__':
    app.run(debug=True)
