from flask import Flask, render_template, Response, jsonify
import threading
import time
import cv2
import json
import os
from process2 import ProcessFrame
from threshold import get_mediapipe_pose, get_thresholds
import statistics

app = Flask(__name__)


state = {
    'last_similarity': None,
    'lock': threading.Lock(),
    'play_sound': False  # เพิ่มสถานะสำหรับการเล่นเสียง
}

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

################## เก็บข้อมูลลง database ####################
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
            user_vec = processor.state_tracker.get('latest_user_vec') if hasattr(processor, 'state_tracker') else None
            rep_number = session.get("done_reps", 0) + 1
            timestamp = time.time()

        # อัปเดต similarity ล่าสุด
        with state['lock']:
            state['last_similarity'] = similarity

        if not session.get('running') or session.get('done_reps', 0) >= session.get('target_reps', 0):
            return
        session['done_reps'] = rep_number

        # path to status.json (process2 writes entries)
        status_path = os.path.join('static', 'status.json')

        # สร้าง record สำหรับเก็บค่าทั้งหมด
        filename = f"frame_{int(time.time() * 1000)}.jpg"
        
        # แปลงค่า depth เป็นข้อความ
        depth_map = {
            0: "Quarter Squat (45°)",
            1: "Half Squat (60°)",
            2: "Parallel Squat (90°)",
            3: "Full Squat (120°)",
            4: "Improper Squat"
        }
        depth_text = depth_map.get(depth, "Unknown") if isinstance(depth, int) else "Unknown"
        
        record = {
            'trainer': None,  # ไม่ต้องดึงจาก latest_kf แล้ว
            'user_image': f"/static/keyframes/{filename}",  # ใส่ path แบบเต็ม
            'similarity': round(similarity, 2),
            'depth': depth,  # เก็บค่าตัวเลข depth
            'depth_text': depth_text,  # เก็บข้อความที่แปลงแล้ว
            'user_vec': user_vec,
            'timestamp': int(timestamp * 1000),
            'rep_number': rep_number
        }


        if processor.state_tracker.get('COMPLETE_STATE', 0) == 1:
            state['play_sound'] = True
            try:
                latest_img = None
                for _ in range(6):
                    if os.path.exists(status_path):
                        try:
                            with open(status_path, 'r', encoding='utf-8') as f:
                                sd = json.load(f)
                            if sd.get('keyframes'):
                                latest = sd['keyframes'][-1]
                                latest_img = latest.get('user_image')
                                break
                        except Exception:
                            pass
                    time.sleep(0.1)
                if latest_img:
                    record['user_image'] = latest_img
            except Exception as e:
                print('Error polling status.json for latest keyframe:', e)

        # เก็บลง session
        session.setdefault('keyframes', []).append(record)

        # เก็บลง database มุมของแต่ละจุด 4 จุด
        user_data["reps"].append(record)

        # ถ้าครบ reps แล้วหยุด
        if session['done_reps'] >= session.get('target_reps', 0):
            session['running'] = False
            processor.state_tracker['running'] = False  # เพิ่มการหยุดที่ processor ด้วย

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
            print(f"Current similarity: {sim}%")
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index3.html')

@app.route('/test-sounds')
def test_sounds():
    return render_template('test_sound.html')


@app.route('/video')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start', methods=['POST'])
def start_session():
    from flask import request
    data = request.get_json() or {}
    reps = int(data.get('reps', 0))
    if reps <= 0:
        return jsonify({'error': 'reps must be > 0'}), 400
    
    with state['lock']:
        state['last_similarity'] = None
    
    session['target_reps'] = reps
    session['done_reps'] = 0
    session['running'] = True
    session['keyframes'] = []
    session['trainer_enabled'] = False
    
    status = {
        "keyframes": [],
        "rounds_count": 0
    }
    
    try:
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
    try:
        status_path = os.path.join('static', 'status.json')
        status_data = {'keyframes': [], 'rounds_count': 0}
        if os.path.exists(status_path):
            with open(status_path, 'r', encoding='utf-8') as f:
                status_data = json.load(f)

        current_depth = None
        depth_text = "Preparing..."
        user_vec = None
        
        if status_data.get('keyframes'):
            latest_kf = status_data['keyframes'][-1]
            current_depth = latest_kf.get('depth')
            user_vec = latest_kf.get('user_vec')
            
        if user_vec is None and len(user_data.get('reps', [])) > 0:
            latest_record = user_data['reps'][-1]
            user_vec = latest_record.get('user_vec')
            
        if user_vec is None and hasattr(processor, 'state_tracker'):
            user_vec = processor.state_tracker.get('latest_user_vec')
        
        if current_depth is None and hasattr(processor, 'state_tracker'):
            display_depth = processor.state_tracker['DISPLAY_DEPTH']
            for i, is_active in enumerate(display_depth):
                if is_active:
                    current_depth = i
                    break
                    
        depth_map = {
            0: "Quarter Squat (45°)",
            1: "Half Squat (60°)",
            2: "Parallel Squat (90°)",
            3: "Full Squat (120°)",
            4: "Improper Squat"
        }
        if isinstance(current_depth, int):
            depth_text = depth_map.get(current_depth, "Unknown")

        with state['lock']:
            similarity = state.get('last_similarity')
            if similarity is not None:
                similarity = round(float(similarity), 2)

        if status_data['keyframes']:
            depth_map = {
                0: "Quarter Squat (45°)",
                1: "Half Squat (60°)",
                2: "Parallel Squat (90°)",
                3: "Full Squat (120°)",
                4: "Improper Squat"
            }

            for idx, kf in enumerate(status_data['keyframes']):
                if 'rep_number' not in kf:
                    kf['rep_number'] = idx + 1
                
                if 'depth' not in kf:
                    if idx < len(user_data.get('reps', [])):
                        depth_value = user_data['reps'][idx].get('depth')
                        if depth_value is not None and isinstance(depth_value, int):
                            kf['depth'] = depth_map.get(depth_value, "Unknown")
                    if 'depth' not in kf and hasattr(processor, 'state_tracker'):
                        display_depth = processor.state_tracker['DISPLAY_DEPTH']
                        for i, is_active in enumerate(display_depth):
                            if is_active:
                                kf['depth'] = depth_map.get(i, "Unknown")
                                break


        # รวมข้อมูลจาก user_data["reps"]
        last_record = None
        total_reps = 0
        if 'user_data' in globals() and 'reps' in user_data and len(user_data["reps"]) > 0:
            total_reps = len(user_data["reps"])
            last_record = user_data["reps"][-1]

        # ตรวจสอบว่าถ้าครบ reps ให้หยุดการทำงาน
        if status_data.get('rounds_count', 0) >= session.get('target_reps', 0):
            session['running'] = False
            
        # รวมทั้งหมดใน response (ไม่รวม keyframes)
        if user_vec is not None:
            print(f"Found user_vec to send: {user_vec[:50]}..." if isinstance(user_vec, (list, tuple)) else user_vec)
            
        response_data = {
            'target_reps': session.get('target_reps', 0),
            'done_reps': status_data.get('rounds_count', 0),
            'running': session.get('running', False),
            'trainer_enabled': session.get('trainer_enabled', False),
            'similarity': similarity if similarity is not None else "Waiting...",
            'depth': depth_text,
            'depth_value': current_depth,
            'user_vec': user_vec,  # เพิ่ม user_vec เข้าไปใน response
            'total_records': total_reps,
            'play_sound': state.get('play_sound', False)
        }
        if state.get('play_sound'):
            state['play_sound'] = False

        print(f"Status response: {response_data}")
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
###################เก็บลง database ## for rep in user_data["reps"]###########
@app.route('/summary')
def summary():
    kfs = session.get('keyframes', []) or []
    total = len(kfs)
    sims = [float(k.get('similarity') or 0.0) for k in kfs]
    avg = round(statistics.mean(sims), 2) if sims else None
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


# ดึงข้อมูล keyframes แยกออกมาต่างหาก
@app.route('/get_keyframes')
def get_keyframes():
    try:
        status_path = os.path.join('static', 'status.json')
        status_data = {'keyframes': [], 'rounds_count': 0}
        if os.path.exists(status_path):
            with open(status_path, 'r', encoding='utf-8') as f:
                status_data = json.load(f)

        def get_depth_text(depth_value):
            depth_map = {
                0: "Quarter Squat (45)",
                1: "Half Squat (60)",
                2: "Parallel Squat (90)",
                3: "Full Squat (120)",
                4: "Improper Squat"
            }
            if depth_value is not None and isinstance(depth_value, int):
                return depth_map.get(depth_value, "Unknown")
            return "Unknown"

        # Process keyframes data and remove consecutive duplicate images
        cleaned = []
        last_img = None
        if status_data.get('keyframes'):
            for idx, kf in enumerate(status_data['keyframes']):
                img = kf.get('user_image')
                # skip if same image as previous entry (duplicate)
                if img and img == last_img:
                    continue
                if 'rep_number' not in kf:
                    kf['rep_number'] = len(cleaned) + 1

                # จัดการค่า depth
                depth_value = None
                depth_text = "Unknown"

                # 1. ลองดึงจาก keyframe ก่อน
                if 'depth' in kf:
                    if isinstance(kf['depth'], int):
                        depth_value = kf['depth']
                    elif isinstance(kf['depth'], str):
                        depth_text = kf['depth']  # ถ้าเป็น string ใช้ค่านั้นเลย

                # 2. ถ้าไม่มีใน keyframe ลองดึงจาก user_data
                if depth_value is None and idx < len(user_data.get('reps', [])):
                    depth_value = user_data['reps'][idx].get('depth')
                
                # 3. ถ้าเป็นค่าตัวเลข แปลงเป็นข้อความ
                if isinstance(depth_value, int):
                    depth_map = {
                        0: "Quarter Squat (45°)",
                        1: "Half Squat (60°)",
                        2: "Parallel Squat (90°)",
                        3: "Full Squat (120°)",
                        4: "Improper Squat"
                    }
                    depth_text = depth_map.get(depth_value, "Unknown")

                # อัพเดทข้อมูลใน keyframe
                kf['depth'] = depth_value
                kf['depth_text'] = depth_text
                
                cleaned.append(kf)
                last_img = img

        return jsonify({
            'keyframes': cleaned,
            'total_reps': len(cleaned),
            'rounds_count': len(cleaned)
        })
    except Exception as e:
        print(f"Error in get_keyframes endpoint: {e}")
        return jsonify({
            'keyframes': [],
            'total_reps': 0,
            'rounds_count': 0
        })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
