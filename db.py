import mysql.connector

def get_connection():
    return mysql.connector.connect(
        host="192.168.0.218",      # Hostname
        port=3306,                 # Port
        user="guardian",           # Username
        password="guardian@123",   # 비밀번호
        database="guardian"     # DB 이름 (공백 포함 시 그대로 문자열로)
    )
