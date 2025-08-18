# routes/auth.py
from flask import Blueprint, request, session, redirect, url_for
from db import get_connection
import bcrypt

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/')
def index():
    return redirect(url_for('auth.login'))

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = None
        cursor = None

        try:
            conn = get_connection()
            cursor = conn.cursor(dictionary=True)

            # 첫 번째 SELECT : 사용자 정보 조회
            cursor.execute(
                "SELECT user_id, name, password_hash FROM users WHERE name=%s",
                (username,)
            )
            user = cursor.fetchone()
            # 남아 있는 결과 버퍼 비우기
            cursor.fetchall()

            if user and bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
                session['user_id'] = user['user_id']

                # 두 번째 SELECT : role 조회
                cursor.execute(
                    "SELECT role FROM users WHERE user_id=%s",
                    (user['user_id'],)
                )
                role_row = cursor.fetchone()
                cursor.fetchall()  # 혹시 남은 결과 비우기

                role = role_row['role'] if role_row else ''

                if role == 'admin':
                    return redirect(url_for('admin.admin_page'))
                return redirect(url_for('profile.profile'))
            else:
                return '로그인 실패 - 아이디 또는 비밀번호를 확인하세요.'

        except Exception as e:
            return f"DB 에러 발생: {e}"

        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None:
                conn.close()

    # GET 요청일 때 로그인 폼 반환
    return '''
    <form method="post">
      Username: <input name="username"><br>
      Password: <input name="password" type="password"><br>
      <input type="submit" value="Login">
    </form>
    '''
