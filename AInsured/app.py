from flask import Flask, render_template, jsonify, make_response
from datetime import datetime
import queue
import threading
import time

app = Flask(__name__)

# Queue to store log messages
log_queue = queue.Queue()

# Mock data for demonstration
mock_performance = {
    'current_value': 100000.00,
    'portfolio_pl': 0.80,
    'buying_power': 200000.00,
    'nasdaq': {'change': -0.17, 'price': 80.83},
    'spy': {'change': -0.06, 'price': 506.89}
}

mock_holdings = []

def generate_logs():
    """Generate mock log messages for demonstration"""
    while True:
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_queue.put({
            'timestamp': timestamp,
            'message': f'System running normally at {timestamp}',
            'type': 'info'
        })
        time.sleep(5)

# Start log generation in background
log_thread = threading.Thread(target=generate_logs, daemon=True)
log_thread.start()

@app.route('/')
def dashboard():
    """Render the main dashboard"""
    return render_template('dashboard.html', 
                         performance=mock_performance,
                         holdings=mock_holdings)

@app.route('/api/logs')
def get_logs():
    """Get new log messages"""
    logs = []
    while not log_queue.empty():
        logs.append(log_queue.get())
    return jsonify(logs)

@app.route('/api/performance')
def get_performance():
    """Get current performance metrics"""
    return jsonify(mock_performance)

@app.route('/api/holdings')
def get_holdings():
    """Get current holdings"""
    return jsonify(mock_holdings)

if __name__ == '__main__':
    print("\nStarting AInsued Dashboard")
    print("=========================")
    print("Server running at: http://localhost:7000")
    print("Press Ctrl+C to stop the server\n")
    
    app.run(host='localhost', port=7000, debug=True)
