from flask import Flask, Blueprint,render_template, request, redirect, url_for,jsonify,make_response
from backend.models.db_connection import get_db_connection
from collections import OrderedDict
import json

landmarks_bp = Blueprint('landmarks',__name__)

@landmarks_bp.route('/landmark', methods = ['GET'])
def readAllLandmarks():
    conn = get_db_connection()
    cursor = conn.cursor()
    query = "SELECT * FROM public.landmarks"
    cursor.execute(query)
    AlldataLandmarks = cursor.fetchall()
    cursor.close()
    conn.close()

    result = []
    for row in AlldataLandmarks:
        result.append(OrderedDict([
            ("landmark_id" , row[0]),
            ("repetition_id" , row[1]),
            ("landmark_index" , row[2]),
            ("X" , row[3]),
            ("Y" , row[4]),
            ("Z" , row[5]),
            ("visibility" , row[6]),
        ]))
    #   XX ห้ามใช้ jsonify() เพราะเปลี่ยนลำดับ key มั่ว XX
    # ใช้ make_response + json.dumps เพื่อคงลำดับ key
    response = make_response(json.dumps({"data": result}, ensure_ascii=False, indent=2))
    response.headers["Content-Type"] = "application/json"
    return response
@landmarks_bp.route('/landmark/<id>', methods = ['GET'])
def readLandmarksById(id):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = "SELECT * FROM public.landmarks WHERE repetition_id = %s"
    cursor.execute(query, (id,))
    AlldataLandmarks = cursor.fetchall()
    cursor.close()
    conn.close()

    result = []
    for row in AlldataLandmarks:
        result.append(OrderedDict([
            ("landmark_id" , row[0]),
            ("repetition_id" , row[1]),
            ("landmark_index" , row[2]),
            ("X" , row[3]),
            ("Y" , row[4]),
            ("Z" , row[5]),
            ("visibility" , row[6]),
        ]))
    #   XX ห้ามใช้ jsonify() เพราะเปลี่ยนลำดับ key มั่ว XX
    # ใช้ make_response + json.dumps เพื่อคงลำดับ key
    response = make_response(json.dumps({"data": result}, ensure_ascii=False, indent=2))
    response.headers["Content-Type"] = "application/json"
    return response
@landmarks_bp.route('/landmarkdetail/<id>' , methods = ['GET'])
def readLandmarkDetailById(id):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = '''SELECT r.repetition_id, r.reps_number, l.landmark_index, l.x, l.y, l.z, l.visibility
    FROM public.landmarks l
    JOIN public.repetitions r ON l.repetition_id = r.repetition_id
    WHERE l.repetition_id = %s '''
    cursor.execute(query, (id,))
    AlldataLandmarkDetail = cursor.fetchall()
    cursor.close()
    conn.close()

    result = []
    for row in AlldataLandmarkDetail:
        result.append(OrderedDict([
            ("repetition_id" , row[0]),
            ("reps_number" , row[1]),
            ("landmark_index" , row[2]),
            ("x" , row[3]),
            ("y" , row[4]),
            ("z" , row[5]),
            ("visibility" , row[6]),
        ]))
    #   XX ห้ามใช้ jsonify() เพราะเปลี่ยนลำดับ key มั่ว XX
    # ใช้ make_response + json.dumps เพื่อคงลำดับ key
    response = make_response(json.dumps({"data": result}, ensure_ascii=False, indent=2))
    response.headers["Content-Type"] = "application/json"
    return response
@landmarks_bp.route('/createLandmark' , methods = ['POST'])
def createLandmark():
    data = request.get_json()
    repetition_id = data.get('repetition_id')
    landmark_index = data.get('landmark_index' )
    x = data.get('x',0)
    y = data.get('y',0)
    z = data.get('z',0)
    visibility = data.get('visibility', 0)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    insert_query = '''INSERT INTO public.landmarks
    (repetition_id, landmark_index, x,y,z,visibility)
    VALUES (%s,%s,%s,%s,%s,%s)'''
    cursor.execute(insert_query, (repetition_id, landmark_index, x,y,z,visibility,))
    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({'message': 'landmarks created success'}, 201)