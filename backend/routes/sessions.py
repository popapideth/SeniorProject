from flask import Flask, Blueprint,render_template, request, redirect, url_for,jsonify,make_response
from backend.models.db_connection import get_db_connection
from collections import OrderedDict
import json
import datetime

sessions_bp = Blueprint('sessions',__name__)

@sessions_bp.route('/sessions', methods=['GET'])
def readAllSessions():
    conn = get_db_connection()
    cursor = conn.cursor()
    query ="SELECT * FROM public.sessions"
    cursor.execute(query)
    AlldataSessions = cursor.fetchall()
    cursor.close()
    conn.close()

    result = []
    for row in AlldataSessions:
        # OrderedDict เเพื่อคงลำดับ key ตามที่กำหนด
        result.append(OrderedDict([
            ("session_id", row[0]),
            ("exercise_name", row[1]),
            ("total_count", row[2]),
            ("correct_count", row[3]),
            ("incorrect_count", row[4]),
            ("avg_Accuracy_percent", row[5]),
            ("avg_Shoulder_angle", row[6]),
            ("avg_Hip_angle", row[7]),
            ("avg_Knee_angle", row[8]),
            ("avg_Ankle_angle", row[9]),
            ("created_time", row[10]),
        ]))
    #   XX ห้ามใช้ jsonify() เพราะเปลี่ยนลำดับ key มั่ว XX
    # ใช้ make_response + json.dumps เพื่อคงลำดับ key
    response = make_response(json.dumps({"data": result}, ensure_ascii=False, indent=2))
    response.headers["Content-Type"] = "application/json"
    return response
@sessions_bp.route('/sessions/<id>', methods=['GET'])
def readSessionsById(id):
    conn = get_db_connection()
    cursor = conn.cursor()
    query ="SELECT * FROM public.sessions WHERE session_id = %s "
    cursor.execute(query, (id,))
    AlldataSessions = cursor.fetchall()
    cursor.close()
    conn.close()

    result = []
    for row in AlldataSessions:
        # OrderedDict เเพื่อคงลำดับ key ตามที่กำหนด
        result.append(OrderedDict([
            ("session_id", row[0]),
            ("exercise_name", row[1]),
            ("total_count", row[2]),
            ("correct_count", row[3]),
            ("incorrect_count", row[4]),
            ("avg_Accuracy_percent", row[5]),
            ("avg_Shoulder_angle", row[6]),
            ("avg_Hip_angle", row[7]),
            ("avg_Knee_angle", row[8]),
            ("avg_Ankle_angle", row[9]),
            ("created_time", row[10]),
        ]))
    #   XX ห้ามใช้ jsonify() เพราะเปลี่ยนลำดับ key มั่ว XX
    # ใช้ make_response + json.dumps เพื่อคงลำดับ key
    response = make_response(json.dumps({"data": result}, ensure_ascii=False, indent=2))
    response.headers["Content-Type"] = "application/json"
    return response
@sessions_bp.route('/sessionsdetail/<id>', methods = ['GET'])
def readSessionDetailById(id):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = '''SELECT s.session_id, s.exercise_name, s.total_count, s.correct_count, s.incorrect_count, s."avg_Accuracy_percent", r.reps_number, r.iscorrect, r.accuracy_percent
    FROM public.sessions s  
    JOIN public.repetitions r ON s.session_id = r.session_id
    WHERE s.session_id = %s'''
    cursor.execute(query, (id,))
    AlldataSessionDetail = cursor.fetchall()
    cursor.close()
    conn.close()

    result = []
    for row in AlldataSessionDetail:
        result.append(OrderedDict([
            ("session_id", row[0]),
            ("exercise_name", row[1]),
            ("total_count", row[2]),
            ("correct_count", row[3]),
            ("incorrect_count" , row[4]),
            ("avg_Accuracy_percent" , row[5]),
            ("reps_number", row[6]),
            ("isCorrect" , row[7]),
            ("accuracy_percent", row[8]),
        ]))
     #   XX ห้ามใช้ jsonify() เพราะเปลี่ยนลำดับ key มั่ว XX
    # ใช้ make_response + json.dumps เพื่อคงลำดับ key
    response = make_response(json.dumps({"data": result}, ensure_ascii=False, indent=2))
    response.headers["Content-Type"] = "application/json"
    return response 
@sessions_bp.route('/sessionswithlandmarks/<id>', methods=['GET'])
def readSessionsWithLandmarks(id):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = '''SELECT s.session_id, s.exercise_name, s.total_count, s.correct_count, s.incorrect_count, s."avg_Accuracy_percent"
    , r.reps_number, r.iscorrect, r.accuracy_percent
    ,l.landmark_index, l.x, l.y, l.z, l.visibility
    FROM public.sessions s  
    JOIN public.repetitions r ON s.session_id = r.session_id
    JOIN public.landmarks l ON r.repetition_id = l.repetition_id
    WHERE s.session_id = %s'''
    cursor.execute(query, (id,))
    AlldataSessionsWithLandmarks = cursor.fetchall()
    cursor.close()
    conn.close()

    result = []
    for row in AlldataSessionsWithLandmarks:
        result.append(OrderedDict([
            ("session_id", row[0]),
            ("exercise_name", row[1]),
            ("total_count", row[2]),
            ("correct_count", row[3]),
            ("incorrect_count" , row[4]),
            ("avg_Accuracy_percent" , row[5]),
            ("reps_number", row[6]),
            ("isCorrect" , row[7]),
            ("accuracy_percent", row[8]),
            ("landmark_index" , row[9]),
            ("x" , row[10]),
            ("y" , row[11]),
            ("z" , row[12]),
            ("visibility" , row[13])
        ]))
     #   XX ห้ามใช้ jsonify() เพราะเปลี่ยนลำดับ key มั่ว XX
    # ใช้ make_response + json.dumps เพื่อคงลำดับ key
    response = make_response(json.dumps({"data": result}, ensure_ascii=False, indent=2))
    response.headers["Content-Type"] = "application/json"
    return response 
@sessions_bp.route('/createSession', methods=['POST'])
def createSession():
    data = request.get_json()
    exercise_name = data.get('exercise_name')
    total_count = data.get('total_count',0)
    correct_count = data.get('correct_count',0)
    incorrect_count = data.get('incorrect_count',0)
    avg_Accuracy_percent = data.get('avg_Accuracy_percent',0)
    avg_shoulder_angle = data.get('avg_shoulder_angle',0)
    avg_hip_angle = data.get('avg_hip_angle',0)
    avg_knee_angle = data.get('avg_knee_angle',0)
    avg_ankle_angle = data.get('avg_ankle_angle',0)
    created_time = data.get('created_time')

    conn = get_db_connection()
    cursor = conn.cursor()
    insert_query = '''INSERT INTO public.sessions 
    (exercise_name, total_count, correct_count, incorrect_count,avg_Accuracy_percent, avg_shoulder_angle, avg_hip_angle, avg_knee_angle, avg_ankle_angle, created_time ) 
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'''
    cursor.execute(insert_query, (exercise_name, total_count,correct_count,incorrect_count,avg_Accuracy_percent,avg_shoulder_angle, avg_hip_angle,avg_knee_angle,avg_ankle_angle,created_time,))
    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({'message' : 'session created success'}, 201)

sessions_bp.route('/createsessionwithrepetition' ,methods=['POST'])
def createSessionWithRepetition():
    try:
        data = request.get_json()
        # sessionData 
        exercise_name = data.get('exercise_name')
        total_count = data.get('total_count',0)
        correct_count = data.get('correct_count',0)
        incorrect_count = data.get('incorrect_count',0)
        avg_Accuracy_percent = data.get('avg_Accuracy_percent',0)
        avg_shoulder_angle = data.get('avg_shoulder_angle',0)
        avg_hip_angle = data.get('avg_hip_angle',0)
        avg_knee_angle = data.get('avg_knee_angle',0)
        avg_ankle_angle = data.get('avg_ankle_angle',0)
        created_time = data.get('created_time')
        # repetitionData
        repetitions = data.get("repetitions", [])

        conn = get_db_connection()
        cursor = conn.cursor()
        insertSession_query = '''INSERT INTO public.sessions
        (exercise_name,total_count,correct_count,incorrect_count,avg_Accuracy_percent,created_time)
        VALUES (%s,%s,%s,%s,%s,%s,)
        RETURNING session_id'''
        cursor.execute(insertSession_query, (exercise_name,total_count,correct_count,incorrect_count,avg_Accuracy_percent,created_time))

        session_id = cursor.fetchone()[0]
        for rep in repetitions:
            insertRepetition_query = '''INSERT INTO public.repetitions
            (session_id,reps_number,iscorrect,depth_value,shoulder_angle,hip_angle,knee_angle,ankle_angle,accuracy_percent,created_time)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'''
            cursor.execute(insertRepetition_query,
                session_id,
                rep.get('reps_number'),
                rep.get('iscorrect'),
                rep.get('depth_value'),
                rep.get('shoulder_angle'),
                rep.get('hip_angle'),
                rep.get('knee_angle'),
                rep.get('ankle_angle'),
                rep.get('accuracy_percent'),
                rep.get('created_time'),
            )
        conn.commit()
        cursor.close()
        conn.close()
    
        return jsonify({'message': 'Session with Repetitios created success','session_id': session_id }, 201)
    except Exception as e:
        return  jsonify({'error': str(e)}), 500
            

