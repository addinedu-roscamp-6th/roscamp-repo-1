from flask import Blueprint, request, session, redirect, url_for, render_template
from db import get_connection
import random

profile_bp = Blueprint('profile', __name__)

@profile_bp.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))

    user_id = session['user_id']
    message = ''
    show_delivery_popup = False

    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)

        if request.method == 'POST':
            items_str = request.form.get('items', '')
            location = request.form.get('location', '')
            request_detail = items_str if items_str else '내용없음'
            request_type = request.form.get('type', 'coffee')

            cursor.execute("""
                INSERT INTO user_request (user_id, request_type, request_detail, delivery_status, destination, created_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
            """, (user_id, request_type, request_detail, '처리중', location))
            conn.commit()
            show_delivery_popup = True

        cursor.execute("SELECT user_id, name, role, seat FROM users WHERE user_id=%s", (user_id,))
        user = cursor.fetchone()

        # 랜덤 로봇 위치 (예시)
        robot_location = f"로봇 위치: ({random.randint(0,100)}, {random.randint(0,100)})"

        # -------- 팝업 스크립트 생성 부분 --------
        delivery_popup_script = ""
        if show_delivery_popup:
            delivery_popup_script = f"""
            <script>
              async function updateDeliveryStatus(status) {{
                try {{
                  await fetch('/delivery_status', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{status: status}})
                  }});
                }} catch(e) {{
                  console.error("배송 상태 업데이트 실패:", e);
                }}
              }}

              async function askDelivery() {{
                if (confirm('물건을 받으셨나요?')) {{
                  document.getElementById('deliveryStatus').innerText = '배송이 완료되었습니다.';
                  await updateDeliveryStatus('완료');
                }} else {{
                  document.getElementById('deliveryStatus').innerText =
                    '{robot_location} 로봇 위치를 전송했습니다. 잠시만 기다려 주세요.';
                  await updateDeliveryStatus('처리중');
                  setTimeout(askDelivery, 5000);
                }}
              }}

              window.addEventListener('load', function() {{
                setTimeout(askDelivery, 5000);
              }});
            </script>
            """
        # --------------------------------------

        return render_template(
            'profile.html',
            user=user,
            message=message,
            robot_location=robot_location,
            delivery_popup_script=delivery_popup_script  # 템플릿에서 {{ delivery_popup_script|safe }}로 출력
        )

    except Exception as e:
        return f"DB 에러 발생: {e}"
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
