
from ultralytics import YOLO
import cv2
from flask import Flask, render_template_string, jsonify, Response, send_file
from datetime import datetime
import threading
import time
import pandas as pd
import os
import mysql.connector

MODEL_PATH = '/Users/adityasharma/Desktop/embedded/airplanedetection.pt'  
IP_CAM_URL = 'http://192.168.26.16:8080/video'
MYSQL_CONFIG = {
    'host': '192.168.169.16',  # add ip address of your host
    'user': 'root',       # add username of your pi created and connected to mysql
    'password': 'adit',
    'database': 'airplane_detection'
}

model = YOLO(MODEL_PATH)

log_data = []
start_time = time.time()
log_file = '/Users/adityasharma/Desktop/embedded/log.xlsx'
latest_frame = None
lock = threading.Lock()
plane_position_history = []
MAX_HISTORY = 10

app = Flask(__name__)

html = """
<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <title>Airplane Detection</title>
    <style>
        body { font-family: Arial, sans-serif; background-color: #f4f7f8; margin: 20px; padding: 20px; color: #333; }
        h2 { color: #1f4e79; }
        table { border-collapse: collapse; width: 100%; background-color: #fff; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1); }
        th, td { text-align: center; padding: 10px; border: 1px solid #ccc; }
        th { background-color: #1f4e79; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        button { background-color: #1f4e79; color: white; padding: 10px 20px; border: none; border-radius: 6px; font-size: 16px; cursor: pointer; }
        button:hover { background-color: #163b5d; }
        img { border-radius: 10px; margin: 20px 0; box-shadow: 0 0 10px rgba(0,0,0,0.2); }
        a { text-decoration: none; }
    </style>
</head>
<body>
    <h2>🛩️ Real-Time Airplane Detection Log</h2>
    <h2>Live Detection Feed</h2>
    <img src="/video_feed" width="640" height="480">
    <h2>Download Detection Log</h2>
    <a href="/download_log">
        <button>📥 Download Excel Log</button>
    </a>
    <h2>Detection Log</h2>
    <table id="log-table">
        <tr><th>Timestamp</th><th>Zone</th><th>Status</th><th>Run Time (s)</th></tr>
    </table>
    <script>
    setInterval(() => {
        fetch('/data').then(r => r.json()).then(data => {
            let table = document.getElementById('log-table');
            table.innerHTML = "<tr><th>Timestamp</th><th>Zone</th><th>Status</th><th>Run Time (s)</th></tr>";
            data.forEach(entry => {
                let row = `<tr><td>${entry['Timestamp']}</td><td>${entry['Zone']}</td><td>${entry['Status']}</td><td>${entry['Run Time (s)']}</td></tr>`;
                table.innerHTML += row;
            });
        });
    }, 1000);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(html)

@app.route('/data')
def data():
    return jsonify(log_data)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/download_log')
def download_log():
    if log_data:
        df = pd.DataFrame(log_data)
        df.to_excel(log_file, index=False, engine='openpyxl')
        return send_file(log_file, as_attachment=True)
    return "No data to download.", 404

def gen_frames():
    global latest_frame
    while True:
        with lock:
            if latest_frame is not None:
                _, buffer = cv2.imencode('.jpg', latest_frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.05)

def get_vertical_zone(box, frame_height):
    y_center = (box[1] + box[3]) / 2
    if y_center < frame_height / 3:
        return 'top'
    elif y_center < 2 * frame_height / 3:
        return 'middle'
    return 'bottom'

def is_plane_moving(box):
    global plane_position_history
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    plane_position_history.append((cx, cy))
    if len(plane_position_history) > MAX_HISTORY:
        plane_position_history.pop(0)
    if len(plane_position_history) >= 2:
        dx = plane_position_history[-1][0] - plane_position_history[0][0]
        dy = plane_position_history[-1][1] - plane_position_history[0][1]
        return (dx**2 + dy**2)**0.5 > 20
    return False

def log_airplane_event(timestamp, zone, status):
    entry = {
        'Timestamp': timestamp,
        'Zone': zone,
        'Status': status,
        'Run Time (s)': round(time.time() - start_time, 2)
    }
    log_data.append(entry)

    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO detection_logs (timestamp, zone, status, runtime)
            VALUES (%s, %s, %s, %s)
        """, (timestamp, zone, status, entry['Run Time (s)']))
        conn.commit()
        cursor.close()
        conn.close()
    except mysql.connector.Error as err:
        print(f"[MySQL Error] {err}")

    df = pd.DataFrame(log_data)
    tmp_file = '/Users/adityasharma/Desktop/embedded/log_tmp.xlsx'
    df.to_excel(tmp_file, index=False, engine='openpyxl')
    os.replace(tmp_file, log_file)

def detect_and_stream():
    global latest_frame
    for result in model(source=2, stream=True, conf=0.6):
        frame = result.plot()
        with lock:
            latest_frame = frame.copy()

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        names = result.names

        for box, cls in zip(boxes, classes):
            label = names[cls]
            if label != 'airplane':
                continue
            zone = get_vertical_zone(box, frame.shape[0])
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            status = ('In Air' if zone == 'top' else
                      'Landing' if zone == 'middle' else
                      'Landed & Moving' if is_plane_moving(box) else 'Landed & At Rest')
            print(f"✈️ Airplane detected in {zone.upper()} zone → {status}")
            log_airplane_event(timestamp, zone.capitalize(), status)

threading.Thread(target=detect_and_stream, daemon=True).start()

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)

