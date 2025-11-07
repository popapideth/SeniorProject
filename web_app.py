from flask import Flask, render_template, Response, jsonify
import threading
import time
import cv2
import json
import os
from process import ProcessFrame
from threshold import get_mediapipe_pose, get_thresholds
import os
import statistics

app = Flask(__name__)


state = {
    'last_similarity': None,
    'lock': threading.Lock()
}

# session state
session = {
    'target_reps': 0,
    'done_reps': 0,
    'running': False,
    'trainer_enabled': False,
    'keyframes': []
}

pose = get_mediapipe_pose()
cap = cv2.VideoCapture(0)
user_camera_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
user_camera_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
thresholds = get_thresholds(user_camera_width, user_camera_height)


user_data = {
    "reps": []
}

def _similarity_cb(val):
    try:
        if val is None:
            return

        # รองรับทั้ง dict
        if isinstance(val, dict):
            similarity = float(val.get("similarity", 0))
            depth = val.get("depth")
            user_vec = val.get("user_vec")
            rep_number = val.get("rep_number")
            timestamp = val.get("timestamp", time.time())
        else:
            similarity = float(val)
            depth = None
            user_vec = None
            rep_number = session.get("done_reps", 0) + 1
            timestamp = time.time()

        # อัปเดต similarity ล่าสุด
        with state['lock']:
            state['last_similarity'] = similarity

        if not session.get('running') or session.get('done_reps', 0) >= session.get('target_reps', 0):
            return
        session['done_reps'] = rep_number

        # ดึง keyframe ล่าสุดจาก status.json
        status_path = os.path.join('static', 'status.json')
        latest_kf = None
        if os.path.exists(status_path):
            try:
                with open(status_path, 'r', encoding='utf-8') as fh:
                    status = json.load(fh)
                if status.get('keyframes'):
                    latest_kf = status['keyframes'][-1]
            except Exception as e:
                print("Error reading status.json in callback:", e)

        # สร้าง record สำหรับเก็บค่าทั้งหมด
        record = {
            'trainer': latest_kf.get('trainer') if latest_kf else None,
            'user_image': latest_kf.get('user_image') if latest_kf else None,
            'similarity': round(similarity, 2),
            'depth': depth,
            'user_vec': user_vec,
            'timestamp': int(timestamp * 1000),
            'rep_number': rep_number
        }

        # เก็บลง session
        session.setdefault('keyframes', []).append(record)

        # เก็บลง database
        user_data["reps"].append(record)
        print(f"user_data['rep']: {user_data['reps']}")

        # ถ้าครบ reps แล้วหยุด
        if session['done_reps'] >= session.get('target_reps', 0):
            session['running'] = False

    except Exception as e:
        print("Error in similarity callback:", e)

processor = ProcessFrame(thresholds=thresholds, flip_frame=True, similarity_callback=_similarity_cb)

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        if session.get('running'):
            frame = processor.process(frame, pose)
        else:
            # put text
            ignore = True
        sim = None
        with state['lock']:
            sim = state.get('last_similarity')
            # put text
        if sim is not None:
            # print(f"Current similarity: {sim}%")
            None
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/video')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/similarity')
def get_similarity():
    with state['lock']:
        val = state.get('last_similarity')
    
    try:
        # Get status.json data
        status_path = os.path.join('static', 'status.json')
        status_data = {'keyframes': [], 'rounds_count': 0}
        if os.path.exists(status_path):
            with open(status_path, 'r') as f:
                status_data = json.load(f)
        
        # Get latest keyframe data
        latest_keyframe = None
        if status_data['keyframes']:
            latest_keyframe = status_data['keyframes'][-1]
        
        current_info = {
            'similarity': val,
            'depth': None,
            'depth_text': 'N/A',
            'keyframe_number': len(status_data['keyframes']),
            'user_image': latest_keyframe.get('user_image') if latest_keyframe else None
        }
        
        # Get current depth from processor
        if hasattr(processor, 'state_tracker'):
            depth_index = next((i for i, d in enumerate(processor.state_tracker['DISPLAY_DEPTH']) if d), None)
            if depth_index is not None:
                depth_map = {
                    0: "Quarter Squat (45°)",
                    1: "Half Squat (60°)",
                    2: "Parallel Squat (90°)",
                    3: "Full Squat (120°)",
                    4: "Improper Squat"
                }
                current_info['depth'] = depth_index
                current_info['depth_text'] = depth_map.get(depth_index, 'N/A')
                
                if latest_keyframe and 'depth' not in latest_keyframe:
                    latest_keyframe['depth'] = depth_index
                    status_data['keyframes'][-1] = latest_keyframe
                    with open(status_path, 'w') as f:
                        json.dump(status_data, f)
        
    except Exception as e:
        print(f"Error in get_similarity: {e}")
        
    return jsonify(current_info)


@app.route('/start', methods=['POST'])
def start_session():
    from flask import request
    data = request.get_json() or {}
    reps = int(data.get('reps', 0))
    if reps <= 0:
        return jsonify({'error': 'reps must be > 0'}), 400
    
    # Reset state
    with state['lock']:
        state['last_similarity'] = None
    
    # reset session
    session['target_reps'] = reps
    session['done_reps'] = 0
    session['running'] = True
    session['keyframes'] = []
    session['trainer_enabled'] = False
    
    # Clear previous status and keyframes
    status = {
        "keyframes": [],
        "rounds_count": 0
    }
    
    try:
        # Ensure keyframes directory exists
        keyframes_dir = os.path.join('static', 'keyframes')
        os.makedirs(keyframes_dir, exist_ok=True)
        
        # Reset status.json
        with open(os.path.join('static', 'status.json'), 'w') as f:
            json.dump(status, f)
            
        # Clear previous keyframes
        for file in os.listdir(keyframes_dir):
            if file.endswith('.jpg') or file.endswith('.png'):
                try:
                    os.remove(os.path.join(keyframes_dir, file))
                except:
                    pass
    except Exception as e:
        print(f"Error clearing files: {e}")
    
    # Reset processor state
    if hasattr(processor, 'state_tracker'):
        processor.state_tracker['COUNT_DEPTH'] = processor.state_tracker['COUNT_DEPTH'] * 0
        processor.state_tracker['DISPLAY_DEPTH'] = processor.state_tracker['DISPLAY_DEPTH'] * False
        processor.state_tracker['rounds_count'] = 0
        processor.state_tracker['state_seq'] = []
        processor.state_tracker['selected_frame'] = []
        processor.state_tracker['selected_frame_count'] = 0
        processor.state_tracker['COMPLETE_STATE'] = 0
        processor.state_tracker['IMPROPER_STATE'] = 0
        processor.state_tracker['prev_knee_angle'] = 0
        processor.state_tracker['stable_pose_time_count'] = 0
        processor.state_tracker['POINT_OF_MISTAKE'] = processor.state_tracker['POINT_OF_MISTAKE'] * False
        
    return jsonify({
        'ok': True, 
        'target_reps': reps,
        'done_reps': 0,
        'running': True,
        'keyframes': []
    })


@app.route('/toggle_trainer', methods=['POST'])
def toggle_trainer():
    session['trainer_enabled'] = not session.get('trainer_enabled', False)
    return jsonify({'trainer_enabled': session['trainer_enabled']})


@app.route('/status')
def status():
    st1 = time.time()
    try:
        status_path = os.path.join('static', 'status.json')
        status_data = {'keyframes': [], 'rounds_count': 0}
        if os.path.exists(status_path):
            with open(status_path, 'r', encoding='utf-8') as f:
                status_data = json.load(f)
                # None

        current_depth = None
        depth_text = "Preparing..."
        if hasattr(processor, 'state_tracker'):
            display_depth = processor.state_tracker['DISPLAY_DEPTH']
            for i, is_active in enumerate(display_depth):
                if is_active:
                    current_depth = i
                    depth_map = {
                        0: "Quarter Squat (45°)",
                        1: "Half Squat (60°)",
                        2: "Parallel Squat (90°)",
                        3: "Full Squat (120°)",
                        4: "Improper Squat"
                    }
                    depth_text = depth_map.get(i, "Unknown")
                    break

        with state['lock']:
            similarity = state.get('last_similarity')
            if similarity is not None:
                similarity = round(float(similarity), 2)

        latest_keyframe = None
        if status_data['keyframes']:
            latest_keyframe = status_data['keyframes'][-1]

        # รวมข้อมูลจาก user_data["reps"]
        last_record = None
        total_reps = 0
        if 'user_data' in globals() and 'reps' in user_data and len(user_data["reps"]) > 0:
            total_reps = len(user_data["reps"])
            last_record = user_data["reps"][-1]

        # รวมทั้งหมดใน response
        response_data = {
            'target_reps': session.get('target_reps', 0),
            'done_reps': status_data.get('rounds_count', 0),
            'running': session.get('running', False),
            'trainer_enabled': session.get('trainer_enabled', False),
            'keyframes': status_data.get('keyframes', []),
            'similarity': similarity if similarity is not None else "Waiting...",
            'depth': depth_text,
            'depth_value': current_depth,
            'latest_keyframe': latest_keyframe,
            'total_records': total_reps,
            'last_record': last_record  #  ข้อมูลล่าสุดจาก user_data["reps"]
        }

        # Debug log
        elasped = time.time() - st1
        if elasped >=1.0:
            print(f"Status response: {response_data}")
            st1 = time.time()

        return jsonify(response_data)

    except Exception as e:
        print(f"Error in status endpoint: {e}")
        return jsonify({
            'error': str(e),
            'target_reps': 0,
            'done_reps': 0,
            'running': False,
            'trainer_enabled': False,
            'keyframes': [],
            'current_state': {},
            'last_similarity': None
        })



@app.route('/trainer_exists')
def trainer_exists():
    trainer_path = os.path.join(os.path.dirname(__file__), 'static', 'trainer.mp4')
    return jsonify({'exists': os.path.exists(trainer_path)})



@app.route('/stop', methods=['POST'])
def stop_session_route():
    # Just stop the session without resetting
    session['running'] = False
    return jsonify({'ok': True})

## ตอนนี้ใช้ sims ในการคำนวน มี rule ที่ข้าวเขียนไว้แบบเทียบองศา ##
##เก็บลง database ## for rep in user_data["reps"]
@app.route('/summary')
def summary():
    # Compute simple stats from recorded similarities
    kfs = session.get('keyframes', []) or []
    total = len(kfs)
    sims = [float(k.get('similarity') or 0.0) for k in kfs]
    avg = round(statistics.mean(sims), 2) if sims else None
    # define correctness threshold
    CORRECT_THRESH = 80.0
    correct = sum(1 for s in sims if s >= CORRECT_THRESH)
    incorrect = total - correct

    return jsonify({
        'total': total,
        'correct': correct,
        'incorrect': incorrect,
        'average_similarity': avg,
        'threshold': CORRECT_THRESH
    })


#ดึงข้อมูล keyframes ทั้งหมด 


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
