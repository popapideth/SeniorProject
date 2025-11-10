import cv2
import numpy as np
import mediapipe as mp
import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
#import seaborn as sns

from screeninfo import get_monitors
from utils import get_chosen_joints_coord, find_angle, findDistance, findModeKneeAngle, _show_feedback, save_keyframe_csv, save_keyframe_json, scaledTo, draw_rounded_rect, draw_text, cropEasy, overlayImage, baseWidthHeight



import os, json, time as _time

KEYFRAME_DIR = os.path.join("static", "keyframes")
STATUS_JSON = os.path.join("static", "status.json")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_keyframe_image(frame_bgr, role="user"):
    ensure_dir(KEYFRAME_DIR)
    ts = int(_time.time() * 1000)
    fname = f"{role}_{ts}.jpg"
    full_path = os.path.join(KEYFRAME_DIR, fname)
    import cv2
    cv2.imwrite(full_path, frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return "/" + full_path.replace(os.path.sep, "/")

def append_status_entry(user_image_url=None, similarity=None, rounds_count=None, user_vec=None, **kwargs):
    ensure_dir(os.path.dirname(STATUS_JSON))
    status = {"keyframes": [], "rounds_count": 0}
    try:
        if os.path.exists(STATUS_JSON):
            with open(STATUS_JSON, "r", encoding="utf-8") as fh:
                status = json.load(fh)
    except Exception:
        status = {"keyframes": [], "rounds_count": 0}

    entry = {
        "timestamp": int(_time.time() * 1000),
        "user_image": user_image_url,
        "similarity": float(similarity) if similarity is not None else None,
        "user_vec": user_vec,
        "rounds_count": int(rounds_count) if rounds_count is not None else int(status.get("rounds_count", 0)),
    }
    # เพิ่ม field อื่นๆ ที่ส่งเข้ามาผ่าน kwargs
    entry.update(kwargs)

    status.setdefault("keyframes", []).append(entry)
    status["rounds_count"] = entry["rounds_count"]
    if len(status["keyframes"]) > 200:
        status["keyframes"] = status["keyframes"][-200:]
    with open(STATUS_JSON, "w", encoding="utf-8") as fh:
        json.dump(status, fh, ensure_ascii=False)
    return status

class ProcessFrame:
    
    def __init__(self, thresholds, similarity_callback=None):
        self.st = time.time()
        self.flip_frame = flip_frame
            
        # Define text properties
        self.fontFace_ptf = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale_ptf = 1
        self.thickness_ptf = 2
        self.linetype = cv2.LINE_AA

        # BGR
        self.COLORS = {
            'navy_blue': (128, 0, 0),
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'light_green': (144, 238, 144),
            'yellow': (0, 255, 255),
            'magenta': (255, 0, 255),
            'white': (255, 255, 255),
            'cyan': (255, 255, 0),
            'light_blue': (255, 200, 100),
            'light_purple': (255, 229, 204),
            'orange': (0, 128, 255),
            'pink': (255, 51, 255),
            'neo_blue': (255, 80, 80)
        }

        self.left_joints = [11, 13, 15, 23, 25, 27, 29, 31]
        self.left_connections = [(11, 13),
                            (13, 15),
                            (11, 23),
                            (23, 25),
                            (25, 27)]

        self.right_joints = [12, 14, 16, 24, 26, 28, 30, 32]
        self.right_connections = [(12, 14),
                            (14, 16),
                            (12, 24),
                            (24, 26),
                            (26, 28)]

        self.dict_features = {}
        self.left_features = {
            'shoulder': 11,
            'elbow': 13,
            'wrist': 15,
            'hip': 23,
            'knee': 25,
            'ankle': 27,
            'heel': 29,
            'foot': 31
        }

        self.right_features = {
            'shoulder': 12,
            'elbow': 14,
            'wrist': 16,
            'hip': 24,
            'knee': 26,
            'ankle': 28,
            'heel': 30,
            'foot': 32
        }

        self.dict_features['left'] = self.left_features
        self.dict_features['right'] = self.right_features
        self.dict_features['nose'] = 0
        self.direction = None

        self.state_tracker = {
            'state_seq': [],

            'COMPLETE_STATE': 0,
            'IMPROPER_STATE': 0,
            
            'latest_user_vec': None,  # เพิ่มตัวแปรเก็บ vector ล่าสุด

            'prev_knee_angle': 0,

            'rounds_count': 0,
            'stable_pose_time_count': 0,
            'stable_pose_end_time_active':time.perf_counter(),

            'selected_frame': [],
            'selected_frame_count': 0,

            'prev_keyframe': None,
            'keyframe': None,

            'Evaluation_Process': False,

            # แสดงว่าอยู่ความลึกใด  [quarter, half, parallel, full, improper]->[False, False, False, False, False]
            'DISPLAY_DEPTH': np.full((5,), False),

            # จำนวนครั้งที่ทำของแต่ละความลึก [quarter, half, parallel, full, improper]- >[0, 0, 0, 0, 0]
            'COUNT_DEPTH': np.zeros((5,), dtype=np.int64),

            'INCORRECT_POSTURE': False,

            # จุดผิดพลาด [โน้มตัวไปด้านหน้าเกินไป, โน้มไปด้านหลังเยอะเกิน, เข่าเลยปลายเท้า,
            #           สวอตไม่ตรงตามต้องการ, เท้าลอย, Bias-Trunk-tibia]
            'POINT_OF_MISTAKE': np.full((6,), False),
            'pic_at_point_of_mistake': np.full((6,), None),

            'INACTIVE_TIME': 0.0,
            'INACTIVE_TIME_FRONT': 0.0,
            'start_inactive_time': time.perf_counter(),
            'start_inactive_time_front': time.perf_counter(),
        }

        self.MISTAKE_ID_MAP = {
            0: ('TOO BEND FORWARD', 300, self.COLORS['neo_blue']),
            1: ('TOO BEND BACKWARDS', 250, self.COLORS['neo_blue']),
            2: ('KNEE EXTEND BEYOND TOE', 200, self.COLORS['neo_blue']),
            3: ('SQUAT INCORRECT DEPTH', 150, self.COLORS['neo_blue']),
            4: ('FOOT FLOATING', 100, self.COLORS['neo_blue']),
            5: ('BIAS TRUNK TIBIA', 50, self.COLORS['neo_blue']),
        }
        self.SQUAT_DEPTH_ID_MAP = {
            0: ('QUARTER SQUAT', 350, self.COLORS['orange']),
            1: ('HALF SQUAT', 350, self.COLORS['orange']),
            2: ('PARALLEL SQUAT', 350, self.COLORS['orange']),
            3: ('FULL SQUAT', 350, self.COLORS['orange']),
            4: ('NOT IN RANGE DEFINE', 350, self.COLORS['orange']),
        }

        self.thresholds = thresholds

        # optional callback function to receive similarity percentage (float)
        self.similarity_callback = similarity_callback

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

        self.monitor = get_monitors()[0]
        self.user_screen_width = self.monitor.width
        self.user_screen_height = self.monitor.height
        self.use_user_screen = False

    def update_state_sequence(self, state):
        if state == 's2':
            if (
                ('s3' not in self.state_tracker['state_seq']) and (self.state_tracker['state_seq'].count('s2')) == 0) or \
                (('s3' in self.state_tracker['state_seq']) and (self.state_tracker['state_seq'].count('s2') == 1) and (len(self.state_tracker['selected_frame']) > 0)
                ):
                self.state_tracker['state_seq'].append(state)

        elif state == 's3':
            if (state not in self.state_tracker['state_seq']) and 's2' in self.state_tracker['state_seq']:
                self.state_tracker['state_seq'].append(state)

    def get_state(self, hip_angle, knee_angle, thresholds):
        knee = None

        if thresholds['HIP_VERT']['STAND'][0] <= knee_angle <= thresholds['HIP_VERT']['STAND'][1]:
            knee = 1
        elif thresholds['HIP_VERT']['SQUATTING'][0] <= hip_angle <= thresholds['HIP_VERT']['SQUATTING'][1] and \
                thresholds['KNEE_VERT']['SQUATTING'][0] <= knee_angle <= thresholds['KNEE_VERT']['SQUATTING'][1]:
            knee = 2
        return f's{knee}' if knee else None

    def process(self, frame: np.array, pose):
        frame_height, frame_width, _ = frame.shape

        if self.flip_frame:
            frame = cv2.flip(frame, 1)
        

        # Recolor image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        image.flags.writeable = True
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            init_landmarks = results.pose_landmarks.landmark

            # Render detection
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                    self.mp_drawing.DrawingSpec(
                                        color = self.COLORS['light_green'], thickness=1, circle_radius=2),
                                    self.mp_drawing.DrawingSpec(
                                        color = self.COLORS['light_blue'], thickness=1, circle_radius=2),
                                    )

            if init_landmarks[self.mp_pose.PoseLandmark.NOSE].visibility > 0.5:

                nose_coord = get_chosen_joints_coord(
                    init_landmarks, self.dict_features, 'nose', frame_width, frame_height)
                left_shoulder_coord, left_elbow_coord, left_wrist_coord, left_hip_coord, left_knee_coord, left_ankle_coord, left_foot_coord = \
                    get_chosen_joints_coord(
                        init_landmarks, self.dict_features, 'left', frame_width, frame_height)
                right_shoulder_coord, right_elbow_coord, right_wrist_coord, right_hip_coord, right_knee_coord, right_ankle_coord, right_foot_coord = \
                    get_chosen_joints_coord(
                        init_landmarks, self.dict_features, 'right', frame_width, frame_height)

                offset_shoulder_x = findDistance(
                    left_shoulder_coord, right_shoulder_coord)
                offset_ankle_x = findDistance(
                    left_ankle_coord, right_ankle_coord)

                # --------------> Start turning sideways towards the camera.
                if offset_shoulder_x > self.thresholds['OFFSET_SHOULDERS_X'] or offset_ankle_x > self.thresholds['OFFSET_ANKLES_X']: #threshold
                    display_inactivity = False

                    # หากผิดพลาด หรือ ไม่กลับมาทำต่อเนื่องในเวลาที่กำหนด
                    if self.state_tracker['state_seq'] is not None:
                        self.state_tracker['state_seq'] = []
                        self.state_tracker['Evaluation_Process'] = False

                    end_time = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME_FRONT'] += end_time - \
                        self.state_tracker['start_inactive_time_front']
                    self.state_tracker['start_inactive_time_front'] = end_time
                    
                    if self.state_tracker['INACTIVE_TIME_FRONT'] >= self.thresholds['INACTIVE_THRESH']: #thresholds
                        display_inactivity = True

                    if display_inactivity:
                        self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                        self.state_tracker['start_inactive_time_front'] = time.perf_counter()
                        self.state_tracker['INCORRECT_POSTURE'] = False
                        self.state_tracker['selected_frame'] = []

                    self.state_tracker['start_inactive_time'] = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME'] = 0.0

                else:
                    self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                    self.state_tracker['start_inactive_time_front'] = time.perf_counter()

                    left_shoulder_z = init_landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].z
                    right_shoulder_z = init_landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].z

                    if left_shoulder_z < right_shoulder_z:
                        chosen_joints = self.left_joints
                        chosen_connections = self.left_connections

                        shoulder_coord = left_shoulder_coord
                        elbow_coord = left_elbow_coord
                        wrist_coord = left_wrist_coord
                        hip_coord = left_hip_coord
                        knee_coord = left_knee_coord
                        ankle_coord = left_ankle_coord
                        foot_coord = left_foot_coord

                        multiplier = -1

                    elif left_shoulder_z > right_shoulder_z:
                        chosen_joints = self.right_joints
                        chosen_connections = self.right_connections

                        shoulder_coord = right_shoulder_coord
                        elbow_coord = right_elbow_coord
                        wrist_coord = right_wrist_coord
                        hip_coord = right_hip_coord
                        knee_coord = right_knee_coord
                        ankle_coord = right_ankle_coord
                        foot_coord = right_foot_coord

                        multiplier = 1

                    # calculating angle
                    try:
                        shoulder_angle = find_angle(
                            hip_coord, elbow_coord, shoulder_coord)
                        hip_angle = find_angle(
                            (hip_coord[0], 0), shoulder_coord, hip_coord)
                        knee_angle = find_angle(
                            (knee_coord[0], 0), hip_coord, knee_coord)
                        ankle_angle = find_angle(
                            (ankle_coord[0], 0), knee_coord, ankle_coord)

                        current_state = self.get_state(int(hip_angle), int(knee_angle), self.thresholds)
                        self.update_state_sequence(current_state)

                        # draw text
                        cv2.putText(frame, str(shoulder_angle), shoulder_coord,
                                    self.fontFace_ptf, self.fontScale_ptf, self.COLORS['yellow'], 2)
                        cv2.putText(frame, str(hip_angle), hip_coord,
                                    self.fontFace_ptf, self.fontScale_ptf, self.COLORS['yellow'], 2)
                        cv2.putText(frame, str(knee_angle), knee_coord,
                                    self.fontFace_ptf, self.fontScale_ptf, self.COLORS['yellow'], 2)
                        cv2.putText(frame, str(ankle_angle), ankle_coord,
                                    self.fontFace_ptf, self.fontScale_ptf, self.COLORS['yellow'], 2)

                        # draw ellipse
                        cv2.ellipse(frame, (shoulder_coord[0], shoulder_coord[1]-5), (30, 30), angle=0, startAngle=90, endAngle=90 - (multiplier*shoulder_angle),
                                    color=self.COLORS['white'], thickness=1) if shoulder_angle > 0 else None
                        cv2.ellipse(frame, (hip_coord[0], hip_coord[1]-10), (30, 30), angle=0, startAngle=-90, endAngle=-90 + multiplier*hip_angle,
                                    color=self.COLORS['white'], thickness=1) if hip_angle > 0 else None
                        cv2.ellipse(frame, (knee_coord[0], knee_coord[1]-10), (30, 30), angle=0, startAngle=-90, endAngle=-90 - multiplier*knee_angle,
                                    color=self.COLORS['white'], thickness=1) if knee_angle > 0 else None
                        cv2.ellipse(frame, (ankle_coord[0], ankle_coord[1]-10), (30, 30), angle=0, startAngle=-90, endAngle=-90 + multiplier*ankle_angle,
                                    color=self.COLORS['white'], thickness=1) if ankle_angle > 0 else None

                        # draw perpendicular line
                        for idx in chosen_joints:
                            if 16 < idx < 29:
                                landmark = init_landmarks[idx]
                                cx, cy = int(
                                    landmark.x * frame_width), int(landmark.y * frame_height)
                                cv2.line(frame, (cx, cy), (cx, cy-40),
                                        self.COLORS['light_purple'], 2, lineType=self.linetype)
                    except:
                        pass

                    # draw line joint
                    for start_idx, end_idx in chosen_connections:
                        start_landmark = init_landmarks[start_idx]
                        end_landmark = init_landmarks[end_idx]

                        x1, y1 = int(
                            start_landmark.x * frame_width), int(start_landmark.y * frame_height)
                        x2, y2 = int(
                            end_landmark.x * frame_width), int(end_landmark.y * frame_height)

                        cv2.line(frame, (x1, y1), (x2, y2),
                                self.COLORS['light_green'], 2)

                    # draw dot joint
                    for idx in chosen_joints:
                        if idx < 29:
                            landmark = init_landmarks[idx]
                            cx, cy = int(
                                landmark.x * frame_width), int(landmark.y * frame_height)
                            cv2.circle(frame, (cx, cy), 5,
                                    color=self.COLORS['light_green'], thickness=-5)

                    # ------------------------------------------ After calculate angle to change state
                    if current_state == 's1':
                        if ('s3' in self.state_tracker['state_seq']) and len(self.state_tracker['state_seq']) == 3:
                            self.state_tracker['COMPLETE_STATE'] += 1
                            self.state_tracker['Evaluation_Process'] = True

                        # กรณี s1 ไป s2 แล้วกลับมา s1 ไม่ครบ 3 state
                        elif 's2' in self.state_tracker['state_seq'] and len(self.state_tracker['state_seq']) == 1:
                            self.state_tracker['IMPROPER_STATE'] += 1

                        self.state_tracker['state_seq'] = []
                        self.state_tracker['prev_knee_angle'] = None

                        self.state_tracker['stable_pose_time_count'] = 0
                        self.state_tracker['selected_frame_count'] = 0
                        self.state_tracker['start_inactive_time_front'] = time.perf_counter()

                    else:
                        if current_state == 's2':

                            if self.state_tracker['INCORRECT_POSTURE']:
                                self.state_tracker['DISPLAY_DEPTH'][:] = False
                                self.state_tracker['POINT_OF_MISTAKE'][:] = False
                                self.state_tracker['INCORRECT_POSTURE'] = not self.state_tracker['INCORRECT_POSTURE']

                            if self.state_tracker['prev_knee_angle'] is not None:
                                delta = abs(knee_angle - self.state_tracker['prev_knee_angle'])
                                # นิ่ง threshold
                                if delta < self.thresholds['DELTA_TRANS_KNEE_ANGLE']:
                                    stable_pose_end_time = time.perf_counter()
                                    self.state_tracker['stable_pose_time_count'] += stable_pose_end_time - self.state_tracker['stable_pose_end_time_active']
                                    self.state_tracker['stable_pose_end_time_active'] = stable_pose_end_time

                                else:
                                    self.state_tracker['stable_pose_time_count'] = 0

                                # ถ้านิ่งต่อเนื่องพอ -> เก็บ keyframe
                                # threshold
                                if self.state_tracker['stable_pose_time_count'] >= self.thresholds['STABLE_POSE_TIME_COUNT']:

                                    self.state_tracker['selected_frame'].append({
                                        'frame': frame.copy(),
                                        'angles': {
                                            'shoulder': shoulder_angle,
                                            'hip': hip_angle,
                                            'knee': knee_angle,
                                            'ankle': ankle_angle
                                        },
                                        'landmarks': init_landmarks,
                                    })

                                    print(
                                        f"select frame: {self.state_tracker['selected_frame'][-1]['angles']}")

                                    self.state_tracker['selected_frame_count'] += 1
                                    self.state_tracker['stable_pose_time_count'] = 0
                                    self.state_tracker['start_inactive_time_front'] = time.perf_counter()

                                if len(self.state_tracker['selected_frame']):
                                    self.update_state_sequence('s3')

                            self.state_tracker['prev_knee_angle'] = knee_angle

                        else:
                            None

                    # -----------------> Evaluation keyframe
                    # True to Do
                    if current_state == 's1' and self.state_tracker['Evaluation_Process']:

                        angle_list = [item['angles']['knee']
                                    for item in self.state_tracker['selected_frame']]

                        self.state_tracker['keyframe'] = self.state_tracker['selected_frame'][findModeKneeAngle(
                            angle_list)]

                        show_keyframe = save_keyframe_json(
                            keyframe=self.state_tracker['keyframe'], option=1)
                        
                        
                        print(f"Keyframe: {self.state_tracker['keyframe']['angles']}")
                        
                        #cosine 127,39,98,32
                        # เก็บค่า trainer vector สำหรับเปรียบเทียบ
                        w1 = 0.1
                        w2 = 0.2
                        w3 = 0.6
                        w4 = 0.1
                        trainer_vec = np.array([127, 39, 98, 32], dtype=float)
                        
                        # สร้าง user vector จากมุมต่างๆ
                        user_vec = np.array([
                            self.state_tracker['keyframe']['angles']['shoulder'],
                            self.state_tracker['keyframe']['angles']['hip'],
                            self.state_tracker['keyframe']['angles']['knee'], 
                            self.state_tracker['keyframe']['angles']['ankle']
                        ], dtype=float)
                        
                        # เก็บค่า vector ล่าสุดใน state_tracker เพื่อส่งไปยัง frontend
                        self.state_tracker['latest_user_vec'] = user_vec.tolist()

                        # Normalize to 0-1 by dividing by 180 and clipping
                        trainer_normalized = np.clip(trainer_vec / 180.0, 0, 1)
                        user_normalized = np.clip(user_vec / 180.0, 0, 1)
                        
                        ''' scaler = MinMaxScaler()
                        stacked = np.vstack([trainer_vec, user_vec])
                        stacked_scaled = scaler.fit_transform(stacked)

                        trainer_normalized = stacked_scaled[0]
                        user_normalized = stacked_scaled[1] '''
                        
                        ''' co_sim = float(cosine_similarity([trainer_norm], [user_norm ])[0][0])
                        co_percentage = round(co_sim * 100, 2) '''

                        diff = 1 - np.abs(trainer_normalized - user_normalized)
                        angle_similarity = diff * 100 #ของแต่ละมุม
                        total_similarity = angle_similarity.mean() #ค่าเฉลี่ย


                        print("Similarity per angle:", angle_similarity)
                        print(f"Average similarity: {total_similarity:.2f}%")
                        print(f"---------------------------------------")
                        print(f"---------------------------------------")
                        print("trainer_vec:", trainer_vec)
                        print("user_vec:", user_vec)
                        print("trainer_normalized:", trainer_normalized)
                        print("user_normalized:", user_normalized)
                        print(f"---------------------------------------") 
                        
                        
                        kf_hip_coord = show_keyframe['landmarks'][chosen_joints[3]]
                        kf_knee_coord = show_keyframe['landmarks'][chosen_joints[4]]
                        kf_ankle_coord = show_keyframe['landmarks'][chosen_joints[5]]
                        kf_foot_coord = show_keyframe['landmarks'][chosen_joints[7]]

                        kf_hip_angle = show_keyframe['angles']['hip']
                        kf_knee_angle = show_keyframe['angles']['knee']
                        kf_ankle_angle = show_keyframe['angles']['ankle']

                        if self.thresholds['SQUAT_DEPTHS']['QUARTER'][0] <= kf_knee_angle <= self.thresholds['SQUAT_DEPTHS']['QUARTER'][1]:
                            self.state_tracker['DISPLAY_DEPTH'][0] = True
                            self.state_tracker['COUNT_DEPTH'][0] += 1

                        elif self.thresholds['SQUAT_DEPTHS']['HALF'][0] <= kf_knee_angle <= self.thresholds['SQUAT_DEPTHS']['HALF'][1]:
                            self.state_tracker['DISPLAY_DEPTH'][1] = True
                            self.state_tracker['COUNT_DEPTH'][1] += 1

                        elif self.thresholds['SQUAT_DEPTHS']['PARALLEL'][0] <= kf_knee_angle <= self.thresholds['SQUAT_DEPTHS']['PARALLEL'][1]:
                            self.state_tracker['DISPLAY_DEPTH'][2] = True
                            self.state_tracker['COUNT_DEPTH'][2] += 1

                        elif self.thresholds['SQUAT_DEPTHS']['FULL'][0] <= kf_knee_angle <= self.thresholds['SQUAT_DEPTHS']['FULL'][1]:
                            self.state_tracker['DISPLAY_DEPTH'][3] = True
                            self.state_tracker['COUNT_DEPTH'][3] += 1

                        else:
                            self.state_tracker['DISPLAY_DEPTH'][4] = True
                            self.state_tracker['COUNT_DEPTH'][4] += 1
                            self.state_tracker['POINT_OF_MISTAKE'][3] = True
                            self.state_tracker['INCORRECT_POSTURE'] = True
                            
                        current_depth = None
                        depth_text = "Unknown"
                        display_depth = self.state_tracker.get('DISPLAY_DEPTH', [])
                        for i, active in enumerate(display_depth):
                            if active:
                                current_depth = i
                                depth_text = self.DEPTH_MAP.get(i, "Unknown")
                                break

                        try:
                            if self.similarity_callback is not None:
                                data = {
                                    "similarity": round(float(total_similarity), 2),
                                    "depth": depth_text,
                                    "depth_value": current_depth,
                                    "user_vec": user_vec.tolist(),
                                    "rep_number": self.state_tracker['rounds_count'],
                                    "timestamp": time.time()
                                }
                                self.similarity_callback(data)
                                print(f"DEBUG - Sending depth data: {depth_text} (value: {current_depth})")
                        except Exception as e:
                            print(f"Error in similarity callback: {str(e)}")
                            pass

                        # threshold
                        if abs(kf_knee_coord['x'] - kf_foot_coord['x'])*frame_width > self.thresholds['KNEE_EXTEND_BEYOND_TOE']:
                            self.state_tracker['POINT_OF_MISTAKE'][2] = True
                            self.state_tracker['pic_at_point_of_mistake'][2] = cropEasy(show_keyframe['frame'],
                                                                                (kf_knee_coord['x'], kf_knee_coord['y']),
                                                                                (kf_foot_coord['x'], kf_foot_coord['y'])
                                                                                )
                            self.state_tracker['INCORRECT_POSTURE'] = True

                        # threshold
                        if abs(kf_hip_angle - kf_ankle_angle) > self.thresholds['NEUTRAL_BIAS_TRUNK_TIBIA_ANGLE']:
                            self.state_tracker['POINT_OF_MISTAKE'][5] = True
                            self.state_tracker['pic_at_point_of_mistake'][5] = cropEasy(show_keyframe['frame'],
                                                                                (kf_hip_coord['x'], kf_hip_coord['y']),
                                                                                (kf_ankle_coord['x'], kf_ankle_coord['y'])
                                                                                )
                            self.state_tracker['INCORRECT_POSTURE'] = True

                        print(
                            f"รอบที่ {self.state_tracker['rounds_count']+1} เสร็จแล้ว")
                        self.state_tracker['rounds_count'] += 1
                        try:
                            if 'show_keyframe' in locals() and show_keyframe is not None and 'frame' in show_keyframe:
                                user_img_url = save_keyframe_image(show_keyframe['frame'], role="user")
                                try:
                                    rounds = int(self.state_tracker.get('rounds_count', 0))
                                except Exception:
                                    rounds = None
                                try:
                                    sim_val = float(total_similarity)
                                except Exception:
                                    sim_val = None
                                user_vec_data = self.state_tracker.get('latest_user_vec')
                                append_status_entry(
                                    user_image_url=user_img_url,
                                    rounds_count=rounds,
                                    similarity=sim_val,
                                    user_vec=user_vec_data,
                                    depth=depth_text,
                                    depth_text=depth_text,
                                    depth_value=current_depth,
                                    rounds_count=rounds
                                )
                        except Exception as _e:
                            pass
                    

                        print('---------------------------------------')

                        self.state_tracker['selected_frame'] = []
                        self.state_tracker['Evaluation_Process'] = False

                    else:
                        self.state_tracker['Evaluation_Process'] = False

                # ------------------------------------------
            else:
                end_time = time.perf_counter()
                self.state_tracker['INACTIVE_TIME'] += end_time - \
                    self.state_tracker['start_inactive_time']

                display_inactivity = False

                if self.state_tracker['INACTIVE_TIME'] >= self.thresholds['INACTIVE_THRESH']: #thresholds
                    display_inactivity = True

                self.state_tracker['start_inactive_time'] = end_time

                if display_inactivity:
                    self.state_tracker['start_inactive_time'] = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME'] = 0.0
                    self.state_tracker['selected_frame'] = []
                    self.state_tracker['INCORRECT_POSTURE'] = False

                self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                self.state_tracker['start_inactive_time_front'] = time.perf_counter()

        except Exception as e:  # ตรวจจับไม่ได้สักจุด
            el = time.time() - self.st
            if el >= 1.0:
                print(f'ตรงนี้ {e}')
                self.st = time.time()
            pass
        return frame

    DEPTH_MAP = {
        0: "Quarter Squat (45)",
        1: "Half Squat (60)",
        2: "Parallel Squat (90)",
        3: "Full Squat (120)",
        4: "Improper Squat"
    }

    def get_depth(self, as_text=False):
        if not hasattr(self, "state_tracker"):
            return (None, "Unknown") if as_text else None

        display_depth = self.state_tracker.get("DISPLAY_DEPTH", [])
        if not isinstance(display_depth, (list, np.ndarray)) or len(display_depth) == 0:
            return (None, "Unknown") if as_text else None

        for i, active in enumerate(display_depth):
            if active:
                depth_text = self.DEPTH_MAP.get(i, "Unknown")
                self.state_tracker["DISPLAY_DEPTH"][:] = False
                return (i, depth_text) if as_text else i

        return (None, "Unknown") if as_text else None