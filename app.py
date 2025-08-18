# app.py
from flask import Flask
from config import SECRET_KEY
from routes.auth import auth_bp
from routes.profile import profile_bp
from routes.delivery import delivery_bp
from routes.admin import admin_bp
import markupsafe
import re

app = Flask(__name__)
app.secret_key = SECRET_KEY

# 블루프린트 등록
app.register_blueprint(auth_bp)
app.register_blueprint(profile_bp)
app.register_blueprint(delivery_bp)
app.register_blueprint(admin_bp)

def escapejs_filter(value):
    if value is None:
        return ''
    # 문자열 변환
    value = str(value)
    replacements = {
        '\\': '\\\\',
        '\'': '\\\'',
        '"': '\\"',
        '\r': '\\r',
        '\n': '\\n',
        '</': '<\\/',
    }
    pattern = re.compile('|'.join(re.escape(key) for key in replacements.keys()))
    escaped = pattern.sub(lambda match: replacements[match.group(0)], value)
    return markupsafe.Markup(escaped)

app.jinja_env.filters['escapejs'] = escapejs_filter

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
