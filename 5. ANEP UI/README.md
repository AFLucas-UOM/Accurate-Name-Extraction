
# ANEP - Accurate Name Extraction from News Video Graphics

This application allows users to extract names from news video graphics using advanced computer vision models.

## Project Structure

This project is split into two parts:

- **Frontend:** A React application using Tailwind CSS and shadcn/ui for the UI
- **Backend:** A Flask API (to be implemented separately)

## Frontend Setup

### Prerequisites

- Node.js (v14 or later)
- npm or yarn

### Installation

1. Clone this repository
2. Navigate to the project directory
3. Install dependencies:

```bash
npm install
# or
yarn install
```

4. Start the development server:

```bash
npm run dev
# or
yarn dev
```

The frontend will be available at http://localhost:8080

## Backend Setup (Flask)

### Prerequisites

- Python 3.8 or later
- pip

### Installation

1. Create a new directory for the backend:

```bash
mkdir -p backend
cd backend
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Flask and required packages:

```bash
pip install flask flask-cors python-dotenv
```

4. Create a basic Flask application (app.py):

```python
from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import os
import random

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Sample dummy names for demonstration
DUMMY_NAMES = [
    "Donald Trump", 
    "Joe Biden", 
    "Boris Johnson", 
    "Emmanuel Macron", 
    "Angela Merkel",
    "Vladimir Putin",
    "Justin Trudeau",
    "Narendra Modi",
    "Xi Jinping",
    "Scott Morrison"
]

@app.route('/api/analyze', methods=['POST'])
def analyze_video():
    """Simulate video analysis with a delay and return dummy results"""
    
    # Check if a file was uploaded
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    video_file = request.files['video']
    model = request.form.get('model', 'anep')
    
    # Validate the file
    if video_file.filename == '':
        return jsonify({"error": "No video file selected"}), 400
    
    # Simulate processing time
    time.sleep(5)
    
    # Generate random number of names (between 2 and 6)
    num_names = random.randint(2, 6)
    selected_names = random.sample(DUMMY_NAMES, num_names)
    
    # Create dummy results
    results = {
        "names": [
            {
                "name": name,
                "confidence": round(random.uniform(0.75, 0.98), 2),
                "timestamp": f"00:{str(random.randint(1, 59)).zfill(2)}:{str(random.randint(10, 59)).zfill(2)}"
            }
            for name in selected_names
        ],
        "model": model,
        "processingTime": "00:02:35",
        "videoMetadata": {
            "filename": video_file.filename,
            "size": 0,  # In a real app, you would get the actual file size
            "type": video_file.content_type
        }
    }
    
    return jsonify(results)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
```

5. Run the Flask application:

```bash
python app.py
```

The backend API will be available at http://localhost:5000

## Connecting Frontend to Backend

To connect the React frontend to the Flask backend, modify the `src/services/api.ts` file to use the actual Flask API endpoint instead of the simulated one.

Replace:

```typescript
// Simulate video analysis
export const analyzeVideo = async (
  video: File,
  model: string
): Promise<any> => {
  // Simulate API delay
  await new Promise((resolve) => setTimeout(resolve, 5000));
  
  // Simulate successful response
  return {
    // ... dummy data
  };
};
```

With:

```typescript
// Real API call to Flask backend
export const analyzeVideo = async (
  video: File,
  model: string
): Promise<any> => {
  const formData = new FormData();
  formData.append('video', video);
  formData.append('model', model);
  
  const response = await fetch('http://localhost:5000/api/analyze', {
    method: 'POST',
    body: formData,
  });
  
  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error || 'Analysis failed');
  }
  
  return response.json();
};
```

## Future Improvements

- Add user authentication
- Implement session management to save analysis history
- Add batch processing for multiple videos
- Integrate with cloud storage for larger video files
- Add more detailed visualization of results
- Improve error handling and user feedback
