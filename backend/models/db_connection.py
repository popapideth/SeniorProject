import psycopg2
from dotenv import load_dotenv
import os
import json

load_dotenv(dotenv_path=".env")

# ดึงค่าตัวแปรจาก .env
DB_HOST = os.getenv("HOST")
DB_PORT = os.getenv("PORT")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

def get_db_connection():
    connection = psycopg2.connect(
        host = DB_HOST,
        port = DB_PORT,
        user = DB_USER,
        password = DB_PASSWORD,
        dbname = DB_NAME
    )
    return connection