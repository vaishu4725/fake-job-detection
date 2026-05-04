"""
app.py — TrueHire Backend (Flask)
REST API for job portal with ML fraud detection
"""

import os
import sys
import jwt
import joblib
import datetime
from functools import wraps
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

# Add ML path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ml'))

from db import get_db
from config import Config

app = Flask(__name__, static_folder='../frontend', static_url_path='')
app.config.from_object(Config)
CORS(app)

# ── Load ML model ──
MODEL_PATH      = os.path.join(os.path.dirname(__file__), '..', 'ml', 'models', 'pac_model.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), '..', 'ml', 'models', 'tfidf_vectorizer.pkl')
META_COLS_PATH  = os.path.join(os.path.dirname(__file__), '..', 'ml', 'data', 'meta_features.npy')

ml_model     = None
ml_vectorizer = None
ml_meta_cols = None

def load_ml_model():
    global ml_model, ml_vectorizer, ml_meta_cols
    try:
        if os.path.exists(MODEL_PATH):
            ml_model      = joblib.load(MODEL_PATH)
            ml_vectorizer = joblib.load(VECTORIZER_PATH)
            ml_meta_cols  = joblib.load(META_COLS_PATH)
            print("[ML] Model loaded successfully.")
        else:
            print("[ML] Model not found. Run ml/train_model.py first.")
    except Exception as e:
        print(f"[ML] Load error: {e}")


# ── JWT helpers ──
def create_token(user_id, role):
    payload = {
        'user_id': user_id,
        'role': role,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=7)
    }
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            return jsonify({'message': 'Token missing'}), 401
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            request.user_id = data['user_id']
            request.user_role = data['role']
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token expired'}), 401
        except:
            return jsonify({'message': 'Invalid token'}), 401
        return f(*args, **kwargs)
    return decorated


def require_role(role):
    def decorator(f):
        @wraps(f)
        @token_required
        def wrapped(*args, **kwargs):
            if request.user_role != role:
                return jsonify({'message': f'Access denied. {role} role required.'}), 403
            return f(*args, **kwargs)
        return wrapped
    return decorator


def run_fraud_detection(job_dict):
    """Run ML model on job posting. Returns label and confidence."""
    if ml_model is None:
        return 'pending', 0.0
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ml'))
        from tfidf_features import transform_single
        from train_model import predict_job
        result = predict_job(job_dict, ml_model, ml_vectorizer, ml_meta_cols)
        return result['label'], result['confidence']
    except Exception as e:
        print(f"[ML] Prediction error: {e}")
        return 'pending', 0.0


# ── STATIC ROUTES ──
@app.route('/')
def serve_index():
    return send_from_directory('../frontend', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('../frontend', path)


# ════════════════════════════════════════
# AUTH ROUTES
# ════════════════════════════════════════

@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data:
        return jsonify({'message': 'No data provided'}), 400

    name     = data.get('name', '').strip()
    email    = data.get('email', '').strip().lower()
    password = data.get('password', '')
    role     = data.get('role', 'seeker')

    if not name or not email or not password:
        return jsonify({'message': 'Name, email and password are required'}), 400
    if len(password) < 6:
        return jsonify({'message': 'Password must be at least 6 characters'}), 400

    db = get_db()
    cursor = db.cursor()
    try:
        cursor.execute("SELECT id FROM users WHERE email=%s", (email,))
        if cursor.fetchone():
            return jsonify({'message': 'Email already registered'}), 409

        hashed = generate_password_hash(password)
        cursor.execute("""
            INSERT INTO users (name, email, password_hash, role, phone, created_at)
            VALUES (%s, %s, %s, %s, %s, NOW())
        """, (name, email, hashed, role, data.get('phone', '')))
        user_id = cursor.lastrowid

        if role == 'seeker':
            cursor.execute("""
                INSERT INTO seeker_profiles (user_id, skills, experience, bio, created_at)
                VALUES (%s, %s, %s, '', NOW())
            """, (user_id, data.get('skills', ''), data.get('experience', 0) or 0))
        elif role == 'company':
            cursor.execute("""
                INSERT INTO company_profiles
                (user_id, industry, year_founded, description, website, created_at)
                VALUES (%s, %s, %s, %s, '', NOW())
            """, (user_id, data.get('industry', ''),
                  data.get('year_founded') or None,
                  data.get('description', '')))

        db.commit()
        token = create_token(user_id, role)
        return jsonify({
            'token': token,
            'user': {'id': user_id, 'name': name, 'email': email, 'role': role}
        }), 201
    except Exception as e:
        db.rollback()
        return jsonify({'message': str(e)}), 500
    finally:
        cursor.close()


@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    email    = data.get('email', '').strip().lower()
    password = data.get('password', '')
    role     = data.get('role', 'seeker')

    db = get_db()
    cursor = db.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM users WHERE email=%s AND role=%s", (email, role))
        user = cursor.fetchone()
        if not user or not check_password_hash(user['password_hash'], password):
            return jsonify({'message': 'Invalid email or password'}), 401

        token = create_token(user['id'], user['role'])
        return jsonify({
            'token': token,
            'user': {'id': user['id'], 'name': user['name'], 'email': user['email'], 'role': user['role']}
        })
    finally:
        cursor.close()


# ════════════════════════════════════════
# JOBS ROUTES
# ════════════════════════════════════════

@app.route('/api/jobs', methods=['GET'])
def get_jobs():
    q         = request.args.get('q', '')
    location  = request.args.get('location', '')
    status    = request.args.get('status', '')
    job_type  = request.args.get('type', '')
    min_sal   = request.args.get('min_salary', '')
    limit     = int(request.args.get('limit', 20))
    verified  = request.args.get('verified', '')

    db = get_db()
    cursor = db.cursor(dictionary=True)
    try:
        sql = """
            SELECT j.id, j.title, j.location, j.salary_range, j.job_type,
                   j.experience_required, j.ml_label, j.ml_confidence,
                   j.created_at, u.name AS company_name
            FROM jobs j
            JOIN users u ON j.company_id = u.id
            WHERE j.is_active = 1
        """
        params = []
        if q:
            sql += " AND (j.title LIKE %s OR j.description LIKE %s)"
            params += [f'%{q}%', f'%{q}%']
        if location:
            sql += " AND j.location LIKE %s"
            params.append(f'%{location}%')
        if status:
            sql += " AND j.ml_label = %s"
            params.append(status)
        if job_type:
            sql += " AND j.job_type = %s"
            params.append(job_type)
        if verified == 'true':
            sql += " AND j.ml_label = 'genuine'"
        sql += " ORDER BY j.created_at DESC LIMIT %s"
        params.append(limit)

        cursor.execute(sql, params)
        jobs = cursor.fetchall()
        return jsonify({'jobs': jobs, 'total': len(jobs)})
    finally:
        cursor.close()


@app.route('/api/jobs/<int:job_id>', methods=['GET'])
def get_job(job_id):
    db = get_db()
    cursor = db.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT j.*, u.name AS company_name, cp.industry, cp.website,
                   cp.description AS company_description
            FROM jobs j
            JOIN users u ON j.company_id = u.id
            LEFT JOIN company_profiles cp ON cp.user_id = u.id
            WHERE j.id = %s
        """, (job_id,))
        job = cursor.fetchone()
        if not job:
            return jsonify({'message': 'Job not found'}), 404
        return jsonify(job)
    finally:
        cursor.close()


@app.route('/api/jobs', methods=['POST'])
@token_required
def post_job():
    if request.user_role != 'company':
        return jsonify({'message': 'Only companies can post jobs'}), 403

    data = request.get_json()
    required = ['title', 'description']
    for field in required:
        if not data.get(field):
            return jsonify({'message': f'{field} is required'}), 400

    # Run ML fraud detection
    ml_label, ml_conf = run_fraud_detection(data)

    db = get_db()
    cursor = db.cursor()
    try:
        # Get company name for context
        cursor.execute("SELECT name FROM users WHERE id=%s", (request.user_id,))
        company = cursor.fetchone()

        cursor.execute("""
            INSERT INTO jobs (company_id, title, description, requirements, location,
                              salary_range, job_type, experience_required, contact_mobile,
                              deadline, ml_label, ml_confidence, is_active, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 1, NOW())
        """, (request.user_id, data['title'], data['description'],
              data.get('requirements', ''), data.get('location', ''),
              data.get('salary_range', ''), data.get('job_type', 'Full-time'),
              data.get('experience_required', 0) or 0,
              data.get('contact_mobile', ''), data.get('deadline'),
              ml_label, ml_conf))
        job_id = cursor.lastrowid
        db.commit()
        return jsonify({'id': job_id, 'ml_label': ml_label, 'ml_confidence': ml_conf,
                        'message': 'Job posted successfully'}), 201
    except Exception as e:
        db.rollback()
        return jsonify({'message': str(e)}), 500
    finally:
        cursor.close()


@app.route('/api/jobs/<int:job_id>', methods=['DELETE'])
@token_required
def delete_job(job_id):
    db = get_db()
    cursor = db.cursor()
    try:
        cursor.execute("UPDATE jobs SET is_active=0 WHERE id=%s AND company_id=%s",
                       (job_id, request.user_id))
        db.commit()
        return jsonify({'message': 'Job deleted'})
    finally:
        cursor.close()


@app.route('/api/jobs/<int:job_id>/apply', methods=['POST'])
@token_required
def apply_job(job_id):
    if request.user_role != 'seeker':
        return jsonify({'message': 'Only seekers can apply'}), 403

    db = get_db()
    cursor = db.cursor(dictionary=True)
    try:
        # Check duplicate
        cursor.execute("SELECT id FROM applications WHERE job_id=%s AND seeker_id=%s",
                       (job_id, request.user_id))
        if cursor.fetchone():
            return jsonify({'message': 'Already applied to this job'}), 409

        # Fetch job and seeker profile for match check
        cursor.execute("SELECT * FROM jobs WHERE id=%s", (job_id,))
        job = cursor.fetchone()
        cursor.execute("SELECT * FROM seeker_profiles WHERE user_id=%s", (request.user_id,))
        seeker = cursor.fetchone()

        # Basic profile match check
        match_status = 'matched'
        if seeker and job:
            seeker_skills = set((seeker.get('skills') or '').lower().split(','))
            job_req = (job.get('requirements') or '').lower()
            matched = any(s.strip() in job_req for s in seeker_skills if s.strip())
            if not matched and seeker_skills != {''}:
                match_status = 'mismatch'

        cursor.execute("""
            INSERT INTO applications (job_id, seeker_id, status, match_status, applied_at)
            VALUES (%s, %s, 'pending', %s, NOW())
        """, (job_id, request.user_id, match_status))
        db.commit()
        return jsonify({'message': 'Application submitted', 'match_status': match_status}), 201
    except Exception as e:
        db.rollback()
        return jsonify({'message': str(e)}), 500
    finally:
        cursor.close()


@app.route('/api/jobs/<int:job_id>/applicants', methods=['GET'])
@token_required
def job_applicants(job_id):
    if request.user_role != 'company':
        return jsonify({'message': 'Access denied'}), 403
    db = get_db()
    cursor = db.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT u.name, u.email, u.phone, sp.skills, sp.experience,
                   a.status, a.match_status, a.applied_at
            FROM applications a
            JOIN users u ON a.seeker_id = u.id
            LEFT JOIN seeker_profiles sp ON sp.user_id = u.id
            JOIN jobs j ON a.job_id = j.id
            WHERE a.job_id=%s AND j.company_id=%s
        """, (job_id, request.user_id))
        return jsonify({'applicants': cursor.fetchall()})
    finally:
        cursor.close()


# ════════════════════════════════════════
# SEEKER ROUTES
# ════════════════════════════════════════

@app.route('/api/seeker/dashboard', methods=['GET'])
@token_required
def seeker_dashboard():
    db = get_db()
    cursor = db.cursor(dictionary=True)
    try:
        cursor.execute("SELECT COUNT(*) AS cnt FROM applications WHERE seeker_id=%s", (request.user_id,))
        total = cursor.fetchone()['cnt']
        cursor.execute("SELECT COUNT(*) AS cnt FROM applications WHERE seeker_id=%s AND status='pending'", (request.user_id,))
        pending = cursor.fetchone()['cnt']
        cursor.execute("""
            SELECT a.id, j.title AS job_title, u.name AS company_name,
                   a.status, a.applied_at, j.ml_label
            FROM applications a
            JOIN jobs j ON a.job_id = j.id
            JOIN users u ON j.company_id = u.id
            WHERE a.seeker_id=%s ORDER BY a.applied_at DESC LIMIT 5
        """, (request.user_id,))
        recent = cursor.fetchall()

        cursor.execute("SELECT * FROM seeker_profiles WHERE user_id=%s", (request.user_id,))
        profile = cursor.fetchone() or {}
        fields = ['name_set','skills','experience','bio','preferred_location','expected_salary']
        filled = sum(1 for f in ['skills','experience','bio'] if profile.get(f))
        profile_score = int((filled / 3) * 100)

        return jsonify({
            'total_applications': total, 'pending': pending,
            'saved': 0, 'profile_score': profile_score,
            'recent_applications': recent
        })
    finally:
        cursor.close()


@app.route('/api/seeker/applications', methods=['GET'])
@token_required
def seeker_applications():
    db = get_db()
    cursor = db.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT j.title AS job_title, u.name AS company_name, j.location,
                   a.status, a.applied_at, j.ml_label
            FROM applications a
            JOIN jobs j ON a.job_id = j.id
            JOIN users u ON j.company_id = u.id
            WHERE a.seeker_id=%s ORDER BY a.applied_at DESC
        """, (request.user_id,))
        return jsonify({'applications': cursor.fetchall()})
    finally:
        cursor.close()


@app.route('/api/seeker/profile', methods=['GET', 'PUT'])
@token_required
def seeker_profile():
    db = get_db()
    cursor = db.cursor(dictionary=True)
    try:
        if request.method == 'GET':
            cursor.execute("""
                SELECT u.name, u.phone, sp.skills, sp.experience,
                       sp.bio, sp.preferred_location, sp.expected_salary
                FROM users u LEFT JOIN seeker_profiles sp ON sp.user_id = u.id
                WHERE u.id=%s
            """, (request.user_id,))
            return jsonify(cursor.fetchone() or {})
        else:
            data = request.get_json()
            cursor.execute("UPDATE users SET name=%s, phone=%s WHERE id=%s",
                           (data.get('name'), data.get('phone'), request.user_id))
            cursor.execute("""
                INSERT INTO seeker_profiles (user_id, skills, experience, bio,
                                             preferred_location, expected_salary, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
                ON DUPLICATE KEY UPDATE
                  skills=VALUES(skills), experience=VALUES(experience),
                  bio=VALUES(bio), preferred_location=VALUES(preferred_location),
                  expected_salary=VALUES(expected_salary)
            """, (request.user_id, data.get('skills'), data.get('experience'),
                  data.get('bio'), data.get('preferred_location'), data.get('expected_salary')))
            db.commit()
            return jsonify({'message': 'Profile updated'})
    except Exception as e:
        db.rollback()
        return jsonify({'message': str(e)}), 500
    finally:
        cursor.close()


# ════════════════════════════════════════
# COMPANY ROUTES
# ════════════════════════════════════════

@app.route('/api/company/dashboard', methods=['GET'])
@token_required
def company_dashboard():
    db = get_db()
    cursor = db.cursor(dictionary=True)
    try:
        cursor.execute("SELECT COUNT(*) AS cnt FROM jobs WHERE company_id=%s AND is_active=1", (request.user_id,))
        total_jobs = cursor.fetchone()['cnt']
        cursor.execute("SELECT COUNT(*) AS cnt FROM applications a JOIN jobs j ON a.job_id=j.id WHERE j.company_id=%s", (request.user_id,))
        total_apps = cursor.fetchone()['cnt']
        cursor.execute("""
            SELECT j.id, j.title, j.created_at, j.ml_label,
                   COUNT(a.id) AS applicant_count
            FROM jobs j LEFT JOIN applications a ON a.job_id=j.id
            WHERE j.company_id=%s AND j.is_active=1
            GROUP BY j.id ORDER BY j.created_at DESC LIMIT 5
        """, (request.user_id,))
        recent_jobs = cursor.fetchall()
        return jsonify({
            'total_jobs': total_jobs, 'total_applicants': total_apps,
            'new_this_week': 0, 'active_jobs': total_jobs,
            'recent_jobs': recent_jobs
        })
    finally:
        cursor.close()


@app.route('/api/company/jobs', methods=['GET'])
@token_required
def company_jobs():
    db = get_db()
    cursor = db.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT j.id, j.title, j.location, j.created_at, j.ml_label, j.ml_confidence,
                   COUNT(a.id) AS applicant_count
            FROM jobs j LEFT JOIN applications a ON a.job_id=j.id
            WHERE j.company_id=%s AND j.is_active=1
            GROUP BY j.id ORDER BY j.created_at DESC
        """, (request.user_id,))
        return jsonify({'jobs': cursor.fetchall()})
    finally:
        cursor.close()


@app.route('/api/company/applicants', methods=['GET'])
@token_required
def company_applicants():
    db = get_db()
    cursor = db.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT u.name, u.email, u.phone, sp.skills, sp.experience,
                   a.status, a.match_status, a.applied_at, j.title AS job_title
            FROM applications a
            JOIN users u ON a.seeker_id = u.id
            LEFT JOIN seeker_profiles sp ON sp.user_id = u.id
            JOIN jobs j ON a.job_id = j.id
            WHERE j.company_id=%s ORDER BY a.applied_at DESC
        """, (request.user_id,))
        return jsonify({'applicants': cursor.fetchall()})
    finally:
        cursor.close()


@app.route('/api/company/profile', methods=['GET', 'PUT'])
@token_required
def company_profile():
    db = get_db()
    cursor = db.cursor(dictionary=True)
    try:
        if request.method == 'GET':
            cursor.execute("""
                SELECT u.name, u.phone, cp.industry, cp.year_founded,
                       cp.description, cp.website
                FROM users u LEFT JOIN company_profiles cp ON cp.user_id=u.id
                WHERE u.id=%s
            """, (request.user_id,))
            return jsonify(cursor.fetchone() or {})
        else:
            data = request.get_json()
            cursor.execute("UPDATE users SET name=%s, phone=%s WHERE id=%s",
                           (data.get('name'), data.get('phone'), request.user_id))
            cursor.execute("""
                INSERT INTO company_profiles (user_id, industry, year_founded, description, website, created_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
                ON DUPLICATE KEY UPDATE
                  industry=VALUES(industry), year_founded=VALUES(year_founded),
                  description=VALUES(description), website=VALUES(website)
            """, (request.user_id, data.get('industry'), data.get('year_founded') or None,
                  data.get('description'), data.get('website')))
            db.commit()
            return jsonify({'message': 'Profile updated'})
    except Exception as e:
        db.rollback()
        return jsonify({'message': str(e)}), 500
    finally:
        cursor.close()


if __name__ == '__main__':
    load_ml_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
