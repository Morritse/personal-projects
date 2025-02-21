from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import io
import json
import logging
import sys
import pandas as pd
import anthropic
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_url_path='/static')
CORS(app)

# Initialize Anthropic client
api_key = os.getenv('ANTHROPIC_API_KEY')
if not api_key:
    logger.error("ANTHROPIC_API_KEY not found in environment variables")
    raise ValueError("ANTHROPIC_API_KEY environment variable is required")

client = anthropic.Client(api_key=api_key)

def analyze_csv_with_ai(data):
    """Use Claude to analyze CSV data"""
    try:
        messages = [{
            "role": "user",
            "content": f"""Analyze this robot joint speed/current data and provide insights:

Data Overview:
- {len(data['joint_data'])} robot arms
- {data['data_overview']['record_count']} data points per arm
- Speed/current measurements for each joint

Current/Speed Ratios:
{data['technical_insights']['behavior'][0]}

Please analyze this data and provide:
1. Detailed friction analysis for each joint
2. Anomaly detection and potential issues
3. Performance optimization recommendations
4. Maintenance scheduling suggestions

Format your response as JSON with this structure:
{{
    "analysis": {{
        "friction_analysis": {{
            "joint_details": {{}},
            "overall_assessment": ""
        }},
        "anomalies": [],
        "performance_insights": [],
        "maintenance_recommendations": []
    }}
}}"""
        }]

        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=messages,
            temperature=0
        )
        
        # Combine AI analysis with our basic analysis
        ai_analysis = json.loads(response.content[0].text)
        data.update(ai_analysis)
        return data
        
    except Exception as e:
        logger.error(f"Error getting AI analysis: {str(e)}")
        return data  # Return basic analysis if AI fails

def analyze_csv(file_content):
    """Analyze CSV file with robot joint data"""
    try:
        # Read the CSV content
        df = pd.read_csv(io.BytesIO(file_content))
        
        # Get arm names from column headers
        arm_names = sorted(list(set([col.split()[0].lower() for col in df.columns if 'speed' in col.lower()])))
        
        # Create joint data structure
        joint_data = {}
        for arm in arm_names:
            speed_col = next(col for col in df.columns if arm in col.lower() and 'speed' in col.lower())
            current_col = next(col for col in df.columns if arm in col.lower() and 'current' in col.lower())
            
            # Create array of speed/current pairs
            joint_data[arm] = [
                {"speed": round(float(speed), 2), "current": round(float(current), 3)} 
                for speed, current in zip(df[speed_col], df[current_col])
            ]
        
        # Calculate basic statistics for insights
        behavior = []
        optimization = []
        
        # Calculate average current/speed ratio for each arm
        ratios = {}
        for arm, data in joint_data.items():
            ratios[arm] = sum(d['current']/d['speed'] if d['speed'] != 0 else 0 for d in data) / len(data)
            ratios[arm] = round(ratios[arm], 4)
        
        # Add ratio insights
        ratio_text = "Average current/speed ratios: " + ", ".join(f"{arm}: {ratio:.4f}" for arm, ratio in ratios.items())
        behavior.append(ratio_text)
        
        # Find arm with highest ratio (most friction)
        max_ratio_arm = max(ratios.items(), key=lambda x: x[1])
        avg_ratio = sum(ratios.values())/len(ratios)
        
        if max_ratio_arm[1] > 1.5 * avg_ratio:
            behavior.append(f"{max_ratio_arm[0]} shows {round((max_ratio_arm[1]/avg_ratio - 1) * 100, 1)}% higher friction than average")
            optimization.append(f"Investigate {max_ratio_arm[0]} for potential maintenance needs")
        
        # Add general insights
        behavior.append("Current/speed relationships are consistent across most arms")
        optimization.append("Regular monitoring recommended for arms: " + 
                          ", ".join(arm for arm, ratio in ratios.items() if ratio <= 1.5 * avg_ratio))
        
        # Create initial analysis
        analysis = {
            "data_overview": {
                "analyzed_arms": arm_names,
                "record_count": len(df)
            },
            "joint_data": joint_data,
            "technical_insights": {
                "behavior": behavior,
                "optimization": optimization
            }
        }
        
        # Get additional AI insights
        enhanced_analysis = analyze_csv_with_ai(analysis)
        return enhanced_analysis
        
    except Exception as e:
        logger.error(f"Error analyzing CSV: {str(e)}")
        raise ValueError(f"Failed to analyze CSV file: {str(e)}")

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def handle_analyze():
    if request.method == 'OPTIONS':
        return '', 204
        
    logger.info("Received analyze request")
    
    try:
        if 'file' not in request.files:
            logger.error("No file provided")
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.error("No file selected")
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.csv'):
            logger.error("Invalid file type")
            return jsonify({'error': 'Please upload a CSV file'}), 400
        
        # Read and analyze file
        file_content = file.read()
        try:
            analysis = analyze_csv(file_content)
            return jsonify({'analysis': analysis})
        except Exception as e:
            logger.error(f"Error analyzing file: {str(e)}")
            return jsonify({'error': str(e)}), 400
            
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    try:
        return send_file('index.html')
    except Exception as e:
        logger.error(f"Error serving index.html: {str(e)}")
        return jsonify({'error': 'Error serving page'}), 500

@app.route('/static/<path:path>')
def serve_static(path):
    try:
        return send_from_directory('static', path)
    except Exception as e:
        logger.error(f"Error serving static file: {str(e)}")
        return jsonify({'error': 'Error serving static file'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
