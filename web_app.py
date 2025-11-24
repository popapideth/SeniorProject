import cv2
import json
import numpy as np
import os
import statistics
import threading
import time

from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request
from threshold import get_mediapipe_pose, get_thresholds
from process2 import ProcessFrame

# backend
from backend.models.db_connection import get_db_connection

current_session_id = None
app = Flask(__name__)

state = {
    'last_similarity': None,
    'lock': threading.Lock(),
    'play_sound': False,
    'last_depth_text': None,
    'last_depth_value': None
}
session = {
    'target_reps': 0,
    'done_reps': 0,
    'running': False,
    'trainer_enabled': False,
    'target_depth': None,
    'keyframes': []
}

pose = get_mediapipe_pose()
cap = cv2.VideoCapture(0)
user_camera_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
user_camera_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

thresholds = get_thresholds(user_camera_width, user_camera_height)

user_data = {
    "reps": [],
}

CORRECT_THRESH =85.0

USER_DATA_PATH = os.path.join('static', 'user_data.json')

# เก็บลงDatabase ในFuncนี้ ซึ่งจะถูกเรียกทุกครั้งที่ rep สำเร็จ
def _similarity_cb(val):
    try:
        if val is None:
            return

        if isinstance(val, dict):
            similarity = float(val.get("similarity", 0))
            #confused
            rep_number = val.get("rep_number")
            timestamp = val.get("timestamp", time.time())
        else:
            similarity = float(val)
            rep_number = session.get("done_reps", 0) + 1
            timestamp = time.time()

        print(f"⭐ rep_number: {rep_number}")
            
        if isinstance(val, dict):
            depth_text = val.get('depth')
            depth_idx = val.get('depth_value')
        else:
            depth_value, depth_text = processor.get_depth(as_text=True)
            depth_idx = depth_value

        if isinstance(depth_text, (list, tuple)):
            try:
                depth_text = depth_text[1]
            except Exception:
                depth_text = str(depth_text)
        if isinstance(depth_idx, (list, tuple)):
            try:
                depth_idx = depth_idx[0]
            except Exception:
                depth_idx = None

        user_vec = (
            processor.state_tracker.get("latest_user_vec")
            if hasattr(processor, "state_tracker")
            else None
        )

        user_criteria = None
        if isinstance(val, dict):
            user_criteria = val.get('user_criteria')
            
        criteria_thresholds = {
            'head_variance': thresholds.get('EAR_DEGREE_VARIANCE', 30),
            'knee_variance': thresholds.get('KNEE_EXTEND_BEYOND_TOE', 15 * (user_camera_width / 640)),
            'heel_variance': thresholds.get('HEEL_FLOAT_VARIANCE', 15),
            'trunk_variance': thresholds.get('NEUTRAL_BIAS_TRUNK_TIBIA_ANGLE', 10),
        }

        criteria_pass = True
        criteria_results = {}
        if user_criteria:
            for k, v in user_criteria.items():
                th = criteria_thresholds.get(k)
                if th is not None:
                    criteria_results[k] = abs(v) <= th
                    if not criteria_results[k]:
                        criteria_pass = False
                else:
                    criteria_results[k] = None

        if user_criteria:
            print("User criteria:")
            for k, v in user_criteria.items():
                passed = criteria_results.get(k)
                print(f"  {k}: {v} {'✓' if passed else '✗'} (threshold: {criteria_thresholds.get(k)})")

        with state['lock']:
            state['last_similarity'] = similarity

        if not session.get('running') or session.get('done_reps', 0) >= session.get('target_reps', 0):
            return

        session['done_reps'] = rep_number
        sim_val = round(float(similarity), 2)

        try:
            depth_idx_normalized = depth_idx[0] if isinstance(depth_idx, (list, tuple)) and len(depth_idx) > 0 else depth_idx
        except Exception:
            depth_idx_normalized = depth_idx

        target_depth = session.get('target_depth')
        target_txt = processor.DEPTH_MAP.get(target_depth)
        
        print(f"!!!!!!!!!!!!!!!! << Target depth: {target_depth} ({target_txt})")
        
        depth_matches = (depth_idx_normalized == target_depth) if target_depth is not None else True

        is_correct = (sim_val >= CORRECT_THRESH) and depth_matches and criteria_pass

        record = {
            "user_image": f"/static/keyframes/frame_{int(timestamp * 1000)}.jpg",
            "similarity": sim_val,
            "depth": depth_text,
            "depth_value": depth_idx_normalized,
            "target_txt": target_txt,
            "target_depth": target_depth,
            "user_vec": user_vec,
            "timestamp": int(timestamp * 1000),
            "rep_number": rep_number + 1,
            "isCorrect": bool(is_correct),
            "depth_match": bool(depth_matches),
            "user_criteria": user_criteria,
            "criteria_results": criteria_results,
        }

        session.setdefault("keyframes", []).append(record)
        user_data.setdefault("reps", []).append(record)
        save_user_data()
        print(f"user_data['rep']: {user_data['reps']}")

        with app.app_context():
            saveToDatabase(record)

        try:
            with state['lock']:
                state['last_depth_text'] = depth_text
                state['last_depth_value'] = depth_idx
                state['play_sound'] = True
        except Exception:
            pass

        if session["done_reps"] >= session.get("target_reps", 0):
            session["running"] = False
            processor.state_tracker["running"] = False

    except Exception as e:
        print("Error in similarity callback:", e)


processor = ProcessFrame(thresholds=thresholds, similarity_callback=_similarity_cb)

def gen_frames():
    try:
        s_gf = time.time()  
        while True:
            success, frame = cap.read()
            frame = cv2.flip(frame, 1)
            
            if not success:
                break
            if session.get('running'):
                frame = processor.process(frame, pose)
            else:
                ignore = True
            sim = None
            with state['lock']:
                sim = state.get('last_similarity')

            elasped_s_gf = time.time() - s_gf
            if sim is not None and session['running'] == True and elasped_s_gf>=1.0:
                print(f"Current similarity: {sim}%")
                s_gf = time.time()

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    except GeneratorExit:
        print("Client disconnected.")
    except Exception as e:
        print("Error in gen_frames:", e)
        
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
    data = request.get_json() or {}
    reps = int(data.get('reps', 0))
    # เลือก 4 แบบ
    depth_raw = data.get('depth', None)
    try:
        target_depth = int(depth_raw) if depth_raw is not None and str(depth_raw) != '' else None
    except Exception:
        target_depth = None

    if reps <= 0:
        return jsonify({'error': 'reps must be > 0'}), 400

    with state['lock']:
        state['last_similarity'] = None
        state['play_sound'] = False
    session.update({
        'target_reps': reps,
        'done_reps': 0,
        'running': True,
        'keyframes': [],
        'trainer_enabled': False,
        'target_depth': target_depth
    })

    # รีเซ็ตไฟล์บันทึก
    try:
        global user_data
        user_data = {"reps": []}
        save_user_data()

        status_path = os.path.join('static', 'status.json')
        keyframes_dir = os.path.join('static', 'keyframes')

        os.makedirs(keyframes_dir, exist_ok=True)

        with open(status_path, 'w', encoding='utf-8') as f:
            json.dump({'keyframes': [], 'rounds_count': 0}, f, indent=2)

    except Exception as e:
        print(f"[ERROR] Failed to initialize new session: {e}")

    try:
        if hasattr(processor, 'state_tracker'):
            tracker = processor.state_tracker

            for key in [
                "COUNT_DEPTH", "DISPLAY_DEPTH", "POINT_OF_MISTAKE",
                "state_seq", "selected_frame", "COMPLETE_STATE",
                "IMPROPER_STATE", "stable_pose_time_count"
            ]:
                if key in tracker:
                    val = tracker[key]
                    if isinstance(val, (list, tuple)):
                        tracker[key] = [0] * len(val)
                    elif isinstance(val, (int, float, bool)):
                        tracker[key] = 0
                    else:
                        tracker[key] = type(val)() 

            tracker["rounds_count"] = 0
            tracker["selected_frame_count"] = 0
            tracker["prev_knee_angle"] = 0
            tracker["running"] = True

            # รีเซ็ตค่าความลึก
            
            tracker["DISPLAY_DEPTH"] = [False] * len(
                getattr(processor, "DEPTH_MAP", {0: ""})
            )

    except Exception as e:
        print(f"[ERROR] Failed to reset processor tracker: {e}")

    global current_session_id
    with app.app_context():
        conn = get_db_connection()
        cursor = conn.cursor()

        exercise_name = "squat"
        created_time = datetime.now().isoformat()

        query  = ''' INSERT INTO public.sessions
        (exercise_name,total_count,correct_count,incorrect_count,avg_Accuracy_percent,created_time)
        VALUES (%s,%s,%s,%s,%s,%s)
        RETURNING session_id'''
        cursor.execute(query,(exercise_name,0,0,0,0,created_time))

        current_session_id = cursor.fetchone()[0]#เก็บ session_id
        conn.commit()
        cursor.close()
        conn.close()
    print("🔵 New session created:", current_session_id)

    return jsonify({
        'ok': True,
        'target_reps': reps,
        'target_depth': target_depth,
        'done_reps': 0,
        'running': True,
        'keyframes': [],
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

        current_depth_value, current_depth_text = processor.get_depth(as_text=True)
        current_depth_idx = current_depth_value

        if (current_depth_text is None or current_depth_text == "Unknown" or current_depth_idx is None):
            with state['lock']:
                last_dt = state.get('last_depth_text')
                last_dv = state.get('last_depth_value')
            if last_dt is not None:
                current_depth_text = last_dt
            if last_dv is not None:
                current_depth_idx = last_dv

        if (current_depth_text is None or current_depth_text == "Unknown"):
            if status_data.get('keyframes'):
                last_kf = status_data['keyframes'][-1]
                if last_kf.get('depth'):
                    current_depth_text = last_kf.get('depth')
                elif last_kf.get('depth_text'):
                    current_depth_text = last_kf.get('depth_text')
            elif user_data.get('reps'):
                last_rep = user_data['reps'][-1]
                if last_rep.get('depth'):
                    current_depth_text = last_rep.get('depth')
        if current_depth_idx is None:
            if status_data.get('keyframes'):
                last_kf = status_data['keyframes'][-1]
                if last_kf.get('depth_value') is not None:
                    current_depth_idx = last_kf.get('depth_value')
            elif user_data.get('reps'):
                last_rep = user_data['reps'][-1]
                if last_rep.get('depth_value') is not None:
                    current_depth_idx = last_rep.get('depth_value')

        # หา user_vec ล่าสุดด
        user_vec = None
        if status_data.get('keyframes'):
            user_vec = status_data['keyframes'][-1].get('user_vec')

        if user_vec is None and user_data.get('reps'):
            user_vec = user_data['reps'][-1].get('user_vec')

        if user_vec is None and hasattr(processor, 'state_tracker'):
            user_vec = processor.state_tracker.get('latest_user_vec')

        # similarity ล่าสุด
        with state['lock']:
            similarity = state.get('last_similarity')
            if similarity is not None:
                similarity = round(float(similarity), 2)

        # อัปเดตข้อมูล rep_number ให้ครบ
        for idx, kf in enumerate(status_data.get('keyframes', [])):
            if 'rep_number' not in kf:
                kf['rep_number'] = idx + 1
            if 'depth' not in kf:
                kf['depth'] = current_depth_idx
            if 'depth_text' not in kf:
                kf['depth_text'] = current_depth_text

        # ตรวจสอบจำนวนรอบที่เสร็จ
        done_reps = status_data.get('rounds_count', 0)
        target_reps = session.get('target_reps', 0)
        if done_reps >= target_reps and target_reps > 0:
            session['running'] = False

        response_data = {
            'target_reps': target_reps,
            'done_reps': done_reps,
            'running': session.get('running', False),
            'trainer_enabled': session.get('trainer_enabled', False),
            'similarity': similarity if similarity is not None else "Waiting...",
            'depth': current_depth_text,
            'depth_value': current_depth_idx,
            'user_vec': user_vec,
            'total_records': len(user_data.get('reps', [])),
            'play_sound': state.get('play_sound', False)
        }

        if state.get('play_sound'):
            state['play_sound'] = False

        print(f"[STATUS] {response_data}")
        return jsonify(response_data)

    except Exception as e:
        print(f"[ERROR] in /status: {e}")
        return jsonify({
            'error': str(e),
            'target_reps': 0,
            'done_reps': 0,
            'running': False,
            'trainer_enabled': False,
            'similarity': None,
            'depth': "Unknown",
            'depth_value': None,
            'user_vec': None,
            'play_sound': False
        })

@app.route('/trainer_exists')
def trainer_exists():
    trainer_path = os.path.join(os.path.dirname(__file__), 'static', 'trainer.mp4')
    return jsonify({'exists': os.path.exists(trainer_path)})

@app.route('/get_reps')
def get_reps():
    try:
        summary = calculate_summary()
        reps = user_data.get('reps', [])

        return jsonify({
            'reps': reps,
            'average': summary()['average'],
            'total': summary()['total'],
            'dept_correct': summary()['dept_correct'],
            'correct': summary()['correct'],
            'incorrect': summary()['incorrect'],
        })

    except Exception as e:
        print('Error in get_reps endpoint:', e)
        return jsonify({
            'reps': [],
            'average': None,
            'total': 0,
            'correct': 0,
            'incorrect': 0
        })

@app.route('/stop', methods=['POST'])
def stop_session_route():
    session['running'] = False
    return jsonify({'ok': True})

## ตอนนี้ใช้ sims ในการคำนวน มี rule ที่ข้าวเขียนไว้แบบเทียบองศา ##
@app.route('/summary')
def summary():
    return jsonify({
        'total': calculate_summary()['total'],
        'dept_correct': calculate_summary()['dept_correct'],
        'correct': calculate_summary()['correct'],
        'incorrect': calculate_summary()['incorrect'],
        'average_similarity': calculate_summary()['average'],
    })

@app.route('/get_keyframes')
def get_keyframes():
    try:
        status_path = os.path.join('static', 'status.json')
        status_data = {'keyframes': [], 'rounds_count': 0}
        if os.path.exists(status_path):
            with open(status_path, 'r', encoding='utf-8') as f:
                status_data = json.load(f)

        cleaned = []
        last_img = None

        depth_idx = processor.get_depth()
        dv, depth_text = processor.get_depth(as_text=True)
        if depth_idx is None:
            depth_idx = dv

        if status_data.get('keyframes'):
            for idx, kf in enumerate(status_data['keyframes']):
                img = kf.get('user_image')
                if img and img == last_img:
                    continue

                if 'rep_number' not in kf:
                    kf['rep_number'] = len(cleaned) + 1

                depth_value = depth_idx if depth_idx is not None else kf.get('depth')
                depth_str = depth_text if (depth_text is not None and depth_text != "Unknown") else kf.get('depth_text')

                if depth_value is not None:
                    kf['depth'] = depth_value
                else:
                    kf.pop('depth', None)

                if depth_str is not None and depth_str != "Unknown":
                    kf['depth_text'] = depth_str
                else:
                    kf.pop('depth_text', None)

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
        
def calculate_summary():
    reps = user_data.get('reps', [])
    target_depth = session.get('target_depth', None)
    
    # filter
    if target_depth is not None:
        filtered = [r for r in reps if r.get('depth_value') == target_depth]
    else:
        filtered = reps

    total = len(reps)
    dept_correct = len(filtered)


    sims = [float(r.get('similarity') or 0.0) for r in filtered]
    avg = round(statistics.mean(sims), 2) if sims else None

    CORRECT_THRESH = 85.0

    correct = 0
    for r in filtered:
        sim_val = float(r.get('similarity') or 0.0)
        depth_idx = r.get('depth_value')
        try:
            depth_idx_normalized = depth_idx[0] if isinstance(depth_idx, (list, tuple)) and len(depth_idx) > 0 else depth_idx
        except Exception:
            depth_idx_normalized = depth_idx
            
        
        depth_matches = (depth_idx_normalized == target_depth) if target_depth is not None else True
        
   #######ยังได้เช็ค
        user_criteria = None
        if isinstance(r, dict):
            user_criteria = r.get('user_criteria')
            
        criteria_thresholds = {
            'head_variance': thresholds.get('EAR_DEGREE_VARIANCE', 30),
            'knee_variance': thresholds.get('KNEE_EXTEND_BEYOND_TOE', 15 * (user_camera_width / 640)),
            'heel_variance': thresholds.get('HEEL_FLOAT_VARIANCE', 15),
            'trunk_variance': thresholds.get('NEUTRAL_BIAS_TRUNK_TIBIA_ANGLE', 10),
        }

        criteria_pass = True
        criteria_results = {}
        if user_criteria:
            for k, v in user_criteria.items():
                th = criteria_thresholds.get(k)
                if th is not None:
                    criteria_results[k] = abs(v) <= th
                    if not criteria_results[k]:
                        criteria_pass = False
                else:
                    criteria_results[k] = None

        if (sim_val >= CORRECT_THRESH) and (depth_matches) and (criteria_pass):
            correct += 1
            
    #################

    incorrect = total - correct

    return {
        'total': total,
        'dept_correct': dept_correct,
        'correct': correct,
        'incorrect': incorrect,
        'average': avg,
    }


def load_user_data():
    global user_data
    try:
        if os.path.exists(USER_DATA_PATH):
            with open(USER_DATA_PATH, 'r', encoding='utf-8') as f:
                user_data = json.load(f)
                if 'reps' not in user_data or not isinstance(user_data['reps'], list):
                    user_data = {'reps': []}
        else:
            user_data = {'reps': []}
    except Exception as e:
        print(f"Failed to load user_data from {USER_DATA_PATH}:", e)
        user_data = {'reps': []}

    try:
        reps = user_data.get('reps', [])
        changed = False
        for rep in reps:
            d = rep.get('depth')
            if isinstance(d, (list, tuple)):
                try:
                    rep['depth_value'] = d[0]
                except Exception:
                    rep['depth_value'] = None
                try:
                    rep['depth'] = d[1]
                except Exception:
                    rep['depth'] = 'Unknown'
                changed = True
        if changed:
            save_user_data()
    except Exception:
        pass

def save_user_data():
    def convert_np(obj):
        if isinstance(obj, dict):
            return {k: convert_np(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_np(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        else:
            return obj
    try:
        os.makedirs(os.path.dirname(USER_DATA_PATH), exist_ok=True)
        tmp = USER_DATA_PATH + '.tmp'
        user_data_clean = convert_np(user_data)
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(user_data_clean, f, ensure_ascii=False, indent=2)
        os.replace(tmp, USER_DATA_PATH)
    except Exception as e:
        print(f"Failed to save user_data to {USER_DATA_PATH}:", e)

load_user_data()

def saveToDatabase(record):
    try:
        global current_session_id
        with app.app_context():
            conn = get_db_connection()
            cursor = conn.cursor()
            
            if current_session_id is None:
                print("ERROR: No active session_id, cannot save rep")
                return
            
            # Data from reps:[] to repetitions
            user_image = record.get("user_image",None)
            accuracy_percent = record.get("similarity", 0)
            depth = record.get("depth",None)
            depth_value = record.get("depth_value", 0)
            user_vec = record.get("user_vec") or []
            if len(user_vec) == 4 :
                shoulder_angle,hip_angle,knee_angle,ankle_angle = user_vec
            else:
                shoulder_angle=hip_angle=knee_angle=ankle_angle = None
            timestamp = record.get("timestamp")
            created_time = (
                datetime.fromtimestamp(timestamp/1000.0).isoformat()
                if timestamp else None
            )
            rep_number = record.get("rep_number" , 0)
            isCorrect = record.get("isCorrect", None)
            session_id = current_session_id

            insertRepetition_query = '''INSERT INTO public.repetitions
                (session_id,reps_number,iscorrect,depth_value,shoulder_angle,hip_angle,knee_angle,ankle_angle,accuracy_percent,created_time)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'''
            cursor.execute(insertRepetition_query, (session_id,rep_number,isCorrect,depth_value,shoulder_angle,hip_angle,knee_angle,ankle_angle,accuracy_percent,created_time))       
            conn.commit()

            print("📥 Repetition saved! 📥")

            # Insert data to sessions
            summary = calculate_summary()
            total_count = summary['total']
            correct_count = summary['correct']
            incorrect_count = summary['incorrect']
            avg_accuracy_percent = summary['average']

            update_session_query = '''UPDATE public.sessions
            SET
                total_count = %s,
                correct_count = %s,
                incorrect_count = %s,
                avg_accuracy_percent = %s
            WHERE session_id = %s
            '''
            cursor.execute(update_session_query, (total_count,correct_count,incorrect_count,avg_accuracy_percent,session_id))
            conn.commit()
            cursor.close()
            conn.close()
            print("📦 DB saved successfully!")
            return {
                'message': 'Session with Repetitions created success',
                'session_id': session_id
            }
    except Exception as e:
        print(f"Error saving to DB: {e}")
        return {
            'message': 'Error saving to DB',
            'error': str(e)
        }
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)