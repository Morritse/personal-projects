from flask import Flask, request, jsonify, render_template
import os

# Set matplotlib config directory before importing matplotlib
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set backend to Agg for serverless environment
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from openai import OpenAI
from dotenv import load_dotenv
import traceback

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Define valid arm numbers
VALID_ARMS = ['900', '901', '902', '903', '906', '907', '909']

# Use /tmp directory for Vercel serverless environment
UPLOAD_FOLDER = '/tmp/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def analyze_data(df):
    """
    Generate analysis comments about the data using OpenAI.
    """
    # Calculate some basic statistics for raw current values
    stats = {}
    for arm in VALID_ARMS:
        current_col = f'arm{arm} j6 current'
        if current_col in df.columns:
            stats[arm] = {
                'avg_current': df[current_col].mean(),
                'max_current': df[current_col].max(),
                'min_current': df[current_col].min()
            }
    
    analysis_prompt = f"""Analyze the following raw current data from robot arms:

Statistics for each arm:
{stats}

Please provide a brief analysis focusing on:
1. Notable patterns in the raw current values
2. Any anomalies or outliers in current readings
3. Comparison of current usage between different arms
4. Key insights about arm performance based on current draw

Keep the analysis technical but concise."""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a robotics data analyst. Provide clear, technical insights about the data."},
            {"role": "user", "content": analysis_prompt}
        ]
    )
    
    return response.choices[0].message.content

def analyze_csv_with_prompt(df, plot_type, prompt):
    """
    Analyze CSV data based on a textual prompt using OpenAI.
    """
    # Get the first column name (Column A)
    index_col = df.columns[0]
    
    # Create list of valid current columns
    current_cols = [f"arm{arm} j6 current" for arm in VALID_ARMS]
    valid_cols = [col for col in current_cols if col in df.columns]
    
    ai_prompt = f"""Generate Python code to plot raw current values from the robot arms.
    
    Available columns:
    - X-axis: '{index_col}' (first column)
    - Y-axis current columns: {valid_cols}
    
    User's Request: Create a {plot_type} plot to {prompt}
    
    IMPORTANT REQUIREMENTS:
    1. Use '{index_col}' as X-axis
    2. Plot RAW current values on Y-axis (do not calculate ratios or derivatives)
    3. Only use these specific arms: {VALID_ARMS}
    4. Label axes appropriately:
       - X-axis: "{index_col}"
       - Y-axis: "Current (raw)"
    
    Generate ONLY valid Python code that will run without any syntax errors. The code should:
    1. Start with plt.clf() to clear any existing plots
    2. Create a {plot_type} plot using matplotlib
    3. Plot raw current values for each valid arm
    4. Add grid lines with grid(True)
    5. Include proper legend
    6. End with plt.tight_layout()
    7. Do NOT include plt.show()
    
    Example structure:
    plt.clf()
    fig, ax = plt.subplots(figsize=(12, 6))
    for arm in {VALID_ARMS}:
        current_col = f'arm{{arm}} j6 current'
        if current_col in df.columns:
            ax.plot(df['{index_col}'], df[current_col], label=f'Arm {{arm}}')
    ax.set_xlabel('{index_col}')
    ax.set_ylabel('Current (raw)')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    """
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a Python code generator. Provide only valid, executable Python code without any explanations or markdown formatting."},
            {"role": "user", "content": ai_prompt}
        ]
    )
    
    # Get the code and clean it up
    code = response.choices[0].message.content
    # Remove any markdown code block formatting if present
    code = code.replace('```python', '').replace('```', '').strip()
    print("Generated code:", code)  # Print for debugging
    return code

def generate_plot(df, code):
    """
    Execute Python code to generate a plot and return as base64 string.
    """
    try:
        # Create a namespace for code execution
        namespace = {
            'df': df,
            'plt': plt,
            'pd': pd,
            'np': np
        }
        
        # Execute the code
        exec(code, namespace)
        
        # Save the plot to a bytes buffer
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=300)
        img.seek(0)
        plt.close('all')  # Close all figures to free memory
        
        return base64.b64encode(img.getvalue()).decode('utf-8')
    except Exception as e:
        print("Error executing code:", str(e))  # Print for debugging
        print("Traceback:", traceback.format_exc())  # Print full traceback
        raise Exception(f"Error generating plot: {str(e)}\nCode:\n{code}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return 'About'

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    plot_type = request.form.get('plotType', 'line')  # Get plot type from dropdown
    prompt = request.form.get('prompt', 'analyze the data')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files are allowed'}), 400
    
    try:
        # Read the CSV file directly from the request stream
        df = pd.read_csv(io.StringIO(file.stream.read().decode('utf-8')))
        
        # Print DataFrame info for debugging
        print("\nDataFrame Info:")
        print(df.info())
        
        # Get analysis code from OpenAI
        analysis_code = analyze_csv_with_prompt(df, plot_type, prompt)
        
        # Generate the plot
        plot_data = generate_plot(df, analysis_code)
        
        # Generate analysis comments
        analysis = analyze_data(df)
        
        return jsonify({
            'message': f'{file.filename} uploaded and analyzed successfully',
            'plot': plot_data,
            'code': analysis_code,
            'analysis': analysis
        })
    except Exception as e:
        print("Error in upload_file:", str(e))  # Print for debugging
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
