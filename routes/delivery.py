# routes/delivery.py
from flask import Blueprint, request, session, jsonify
from db import get_connection

delivery_bp = Blueprint('delivery', __name__)

@delivery_bp.route('/delivery_status', methods=['POST'])
def delivery_status():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': '로그인이 필요합니다.'})

    user_id = session['user_id']
    data = request.get_json()
    status = data.get('status')

    if status not in ['완료', '처리중']:
        return jsonify({'success': False, 'message': '잘못된 상태 값입니다.'})

    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM user_request WHERE user_id=%s ORDER BY created_at DESC LIMIT 1", (user_id,))
        row = cursor.fetchone()
        if not row:
            return jsonify({'success': False, 'message': '업데이트할 요청이 없습니다.'})

        latest_request_id = row[0]
        cursor.execute("UPDATE user_request SET delivery_status=%s WHERE id=%s", (status, latest_request_id))
        conn.commit()

        return jsonify({'success': True, 'message': '배송 상태가 업데이트되었습니다.'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'DB 에러 발생: {e}'})
    finally:
        if cursor: cursor.close()
        if conn: conn.close()
