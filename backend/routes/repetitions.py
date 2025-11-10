from flask import Flask, Blueprint,render_template, request, redirect, url_for,jsonify,make_response
from backend.models.db_connection import get_db_connection
from collections import OrderedDict
import json

repetitions_bp = Blueprint('repetitions',__name__)

@repetitions_bp.route('/repetitions', methods = ['GET'])
def readAllRepetitions():
    conn = get_db_connection()
    cursor = conn.cursor()
    query = "SELECT * FROM public.repetitions"
    cursor.execute(query)
    AlldataRepetitions = cursor.fetchall()
    cursor.close()
    conn.close()

    result = []
    for row in AlldataRepetitions:
        result.append(OrderedDict([
            ("repetition_id" , row[0]),
            ("session_id", row[1]),
            ("reps_number", row[2]),
            ("iscorrect", row[3]),
            ("shoulder_angle", row[4]),
            ("hip_angle", row[5]),
            ("knee_angle", row[6]),
            ("ankle_angle", row[7]),
            ("accuracy_percent", row[8]),
            ("created_time", row[9]),
        ]))
    #   XX ห้ามใช้ jsonify() เพราะเปลี่ยนลำดับ key มั่ว XX
    # ใช้ make_response + json.dumps เพื่อคงลำดับ key
    response = make_response(json.dumps({"data": result}, ensure_ascii=False, indent=2))
    response.headers["Content-Type"] = "application/json"
    return response
@repetitions_bp.route('/repetitions/session/<id>', methods = ['GET'])
def readRepetitionsById(id):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = "SELECT * FROM public.repetitions WHERE session_id = %s"
    cursor.execute(query,(id,))
    AlldataRepetitions = cursor.fetchall()
    cursor.close()
    conn.close()

    result = []
    for row in AlldataRepetitions:
        result.append(OrderedDict([
            ("repetition_id" , row[0]),
            ("session_id", row[1]),
            ("reps_number", row[2]),
            ("iscorrect", row[3]),
            ("shoulder_angle", row[4]),
            ("hip_angle", row[5]),
            ("knee_angle", row[6]),
            ("ankle_angle", row[7]),
            ("accuracy_percent", row[8]),
            ("created_time", row[9]),
        ]))
    #   XX ห้ามใช้ jsonify() เพราะเปลี่ยนลำดับ key มั่ว XX
    # ใช้ make_response + json.dumps เพื่อคงลำดับ key
    response = make_response(json.dumps({"data": result}, ensure_ascii=False, indent=2))
    response.headers["Content-Type"] = "application/json"
    return response
@repetitions_bp.route('/repetitionsdetail/<id>', methods = ['GET'])
def readRepetiotionsDetailById(id):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = '''SELECT s.session_id, s.exercise_name, r.repetition_id, r.reps_number, r.iscorrect, r.shoulder_angle,r.hip_angle, r.knee_angle, r.ankle_angle, r.accuracy_percent
    FROM public.sessions s
    JOIN public.repetitions r ON s.session_id = r.session_id
    WHERE r.repetition_id = %s '''
    cursor.execute(query, (id,))
    AlldataRepetitionsDetail = cursor.fetchall()
    cursor.close()
    conn.close()

    result = []
    for row in AlldataRepetitionsDetail:
        result.append(OrderedDict([
            ("session_id", row[0]),
            ("exercise_name", row[1]),
            ("repetition_id" , row[2]),
            ("reps_number" , row[3]),
            ("iscorrect" , row[4]),
            ("shoulder_angle", row[5]),
            ("hip_angle", row[6]),
            ("knee_angle", row[7]),
            ("ankle_angle", row[5]),
            ("accuracy_percent", row[9]),
        ]))
    #   XX ห้ามใช้ jsonify() เพราะเปลี่ยนลำดับ key มั่ว XX
    # ใช้ make_response + json.dumps เพื่อคงลำดับ key
    response = make_response(json.dumps({"data": result}, ensure_ascii=False, indent=2))
    response.headers["Content-Type"] = "application/json"
    return response
@repetitions_bp.route('/createrepetition', methods = ['POST'])
def createRepetition():
    data = request.get_json()
    session_id = data.get('session_id')
    reps_number = data.get('reps_number',0)
    iscorrect = data.get('iscorrect')
    shoulder_angle = data.get('shoulder_angle',0)
    hip_angle = data.get('hip_angle',0)
    knee_angle = data.get('knee_angle',0)
    ankle_angle = data.get('ankle_angle',0)
    accuracy_percent = data.get('accuracy_percent',0)
    created_time = data.get('created_time')

    conn = get_db_connection()
    cursor = conn.cursor()
    insert_query = '''INSERT INTO public.repetitions
    (session_id, reps_number, iscorrect, shoulder_angle, hip_angle, knee_angle, ankle_angle, accuracy_percent, created_time)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)'''
    cursor.execute(insert_query, (session_id,reps_number,iscorrect,shoulder_angle,hip_angle,knee_angle,ankle_angle,accuracy_percent,created_time, ))
    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({'message' : 'repetition created success'}, 201)
@repetitions_bp.route('/createrepetitionwithlandmarks', methods=['POST'])
def createRepetitionWithLandmarks():
    try:
        data = request.get_json()
        # repetitionData
        session_id = data.get('session_id')
        reps_number = data.get('reps_number',0)
        iscorrect = data.get('iscorrect')
        shoulder_angle = data.get('shoulder_angle',0)
        hip_angle = data.get('hip_angle',0)
        knee_angle = data.get('knee_angle',0)
        ankle_angle = data.get('ankle_angle',0)
        accuracy_percent = data.get('accuracy_percent',0)
        created_time = data.get('created_time')
        # landmarkData
        landmarks = data.get('landmarks', [])

        conn = get_db_connection()
        cursor = conn.cursor()
        insertRepetition_query = '''INSERT INTO public.repetitions
        (session_id, reps_number, iscorrect, shoulder_angle, hip_angle, knee_angle, ankle_angle, accuracy_percent, created_time)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        RETURNING repetition_id'''
        cursor.execute(insertRepetition_query, (session_id,reps_number,iscorrect,shoulder_angle,hip_angle,knee_angle,ankle_angle,accuracy_percent,created_time, ))

        repetition_id = cursor.fetchone()[0]
        for landmark in landmarks:
            insertLandmark_query = '''INSERT INTO public.landmarks
            (repetition_id, landmark_index, x,y,z, visibility)
            VALUES (%s,%s,%s,%s,%s,%s)'''
            cursor.execute(insertLandmark_query, repetition_id,landmark.get('landmark_index'),
            landmark.get('x',0),landmark.get('y',0),landmark.get('z',0), landmark.get('visibility',0))
            
        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({'message': 'Repetition with landmarks created success','repetition_id': repetition_id }, 201)
    except Exception as e:
        return  jsonify({'error': str(e)}), 500