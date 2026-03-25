from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
# Make sure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/renovate', methods=['POST'])
def renovate():
    file = request.files['room_image']
    style = request.form['style']

    if file:
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)

        # For now, just copy same image as output (No AI yet)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        file.save(output_path)

        return render_template("result.html",
                               original=upload_path,
                               renovated=output_path,
                               selected_style=style)


if __name__ == '__main__':
    app.run(debug=True)
