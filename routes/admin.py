# routes/admin.py
from flask import Blueprint, session, redirect, url_for
from db import get_connection

admin_bp = Blueprint('admin', __name__)

@admin_bp.route('/admin')
def admin_page():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))

    user_id = session['user_id']

    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT role FROM users WHERE user_id=%s", (user_id,))
        user = cursor.fetchone()
        if not user or user['role'] != 'admin':
            return "접근 권한이 없습니다.", 403

        cursor.execute("SELECT * FROM user_request ORDER BY created_at DESC")
        requests = cursor.fetchall()

        html = "<h2>관리자 페이지</h2><ul>"
        for r in requests:
            html += f"<li>요청ID {r['id']}: 사용자 {r['user_id']} - {r['request_type']} - {r['request_detail']} - 상태: {r.get('delivery_status', 'N/A')} - 위치: {r.get('destination', '')} ({r['created_at']})</li>"
        html += "</ul>"
        return html

    except Exception as e:
        return f"DB 에러 발생: {e}"
    finally:
        if cursor: cursor.close()
        if conn: conn.close()
