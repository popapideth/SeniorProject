from flask import Flask, render_template, request, redirect, url_for,jsonify,make_response
import psycopg2
from dotenv import load_dotenv
import os
import json

# load_dotenv(dotenv_path=".env")

# # ดึงค่าตัวแปรจาก .env
# DB_HOST = os.getenv("HOST")
# DB_PORT = os.getenv("PORT")
# DB_USER = os.getenv("DB_USER")
# DB_PASSWORD = os.getenv("DB_PASSWORD")
# DB_NAME = os.getenv("DB_NAME")

# print("user",DB_USER)
# print("password",DB_PASSWORD)
# print("database",DB_NAME)
# print("host",DB_HOST)
# print("port",DB_PORT)
# def get_db_connection():
#     connection = psycopg2.connect(
#         host = DB_HOST,
#         port = DB_PORT,
#         user = DB_USER,
#         password = DB_PASSWORD,
#         dbname = DB_NAME
#     )
#     return connection


app = Flask(__name__)

# import blueprints 
from backend.routes.sessions import sessions_bp
from backend.routes.repetitions import repetitions_bp
from backend.routes.landmarks import landmarks_bp
# register blueprint
app.register_blueprint(sessions_bp)
app.register_blueprint(repetitions_bp)
app.register_blueprint(landmarks_bp)

@app.route('/db')
def home():
    return {"message" : "Backend is running successfully!!!"}


if __name__== '__main__':
    app.run(debug=True, use_reloader=True)