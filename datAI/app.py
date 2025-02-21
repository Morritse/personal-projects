from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    df = pd.read_csv(file)  # Use pandas to read CSV
    # Example: Generate a summary
    summary = df.describe().to_dict()
    return jsonify({'summary': summary})


if __name__ == "__main__":
    app.run(debug=True)
