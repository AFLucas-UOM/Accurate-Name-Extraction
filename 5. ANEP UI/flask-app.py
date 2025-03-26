from flask import Flask, render_template, request, jsonify
import os
import time
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='assets')

# Configuration
UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
ALLOWED_EXTENSIONS = {'mp4', 'webm', 'ogg', 'mov', 'mkv'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Simulate analysis of an uploaded video.
    Expects a file under the key "video" in a POST form-data request.
    """
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Simulate analysis delay
        time.sleep(5)

        results = {
            'message': 'Analysis complete',
            'video_filename': filename,
            'extracted_names': ['Donald Trump', 'Keir Starmer', 'Boris Johnson']
        }

        return jsonify(results)
    else:
        return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    app.run(debug=True)
