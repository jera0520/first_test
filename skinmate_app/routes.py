"""
SKINMATE 라우트 블루프린트 모듈

모든 라우트를 기능별로 분리하여 Blueprint로 관리합니다.
"""

import os
import json
import shutil
from datetime import datetime, timedelta
from flask import (
    Blueprint, render_template, request, redirect, url_for, 
    flash, session, jsonify, current_app
)

from .utils import login_required, DatabaseManager, get_current_season
from .models import AnalysisResult, RecommendationData
from .services import AnalysisService, RecommendationService, ProductRecommendationService, UserService


# Blueprint 생성
main_bp = Blueprint('main', __name__)
auth_bp = Blueprint('auth', __name__)
analysis_bp = Blueprint('analysis', __name__)
history_bp = Blueprint('history', __name__)
recommendations_bp = Blueprint('recommendations', __name__)
api_bp = Blueprint('api', __name__)


# =====================
# 메인 페이지 라우트
# =====================

@main_bp.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')


@main_bp.route('/introduction')
def introduction():
    """소개 페이지"""
    return render_template('introduction.html')


@main_bp.route('/skin_diary')
def skin_diary():
    """피부 일지 페이지"""
    if 'user_id' not in session:
        flash('피부 일지를 보려면 먼저 로그인해주세요.')
        return redirect(url_for('auth.login'))
    return render_template('skin_diary.html')


# =====================
# 인증 관련 라우트
# =====================

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """회원가입"""
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # 입력 검증
        error = None
        if not username:
            error = 'Username is required.'
        elif not password:
            error = 'Password is required.'
        elif not email:
            error = 'Email is required.'
        
        if error is None:
            try:
                user_service = UserService()
                db = DatabaseManager.get_connection()
                
                success = user_service.create_user(username, email, password, db)
                if success:
                    flash('회원가입 성공! 로그인해주세요.', 'success')
                    return redirect(url_for('auth.login'))
                else:
                    error = f"Email {email} is already registered."
            except Exception as e:
                error = f"회원가입 중 오류가 발생했습니다: {str(e)}"
        
        flash(error, 'danger')
    
    return render_template('login.html')


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """로그인"""
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        user_service = UserService()
        db = DatabaseManager.get_connection()
        
        user = user_service.authenticate_user(email, password, db)
        
        if user:
            session.clear()
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('로그인 성공!', 'success')
            return redirect(url_for('main.index'))
        else:
            flash('잘못된 이메일 또는 비밀번호입니다.', 'danger')
    
    return render_template('login.html')


@auth_bp.route('/logout')
def logout():
    """로그아웃"""
    session.clear()
    flash('로그아웃되었습니다.', 'info')
    return redirect(url_for('main.index'))


# =====================
# 피부 분석 라우트
# =====================

@analysis_bp.route('/')
def analysis():
    """분석 페이지"""
    return render_template('analysis.html')


@analysis_bp.route('/analyze', methods=['POST'])
@login_required
def analyze():
    """피부 이미지 분석"""
    try:
        # 파일 검증
        if 'image' not in request.files or request.files['image'].filename == '':
            flash('파일이 선택되지 않았습니다.')
            return redirect(request.url)
        
        file = request.files['image']
        
        # 이미지 검증 서비스 사용
        image_validator = current_app.image_validator
        file_uploader = current_app.file_uploader
        
        # 이미지 유효성 검증
        try:
            validation_result = image_validator.validate_skin_image(file, session['user_id'])
        except Exception as e:
            flash(f'이미지 검증 실패: {str(e)}')
            return redirect(url_for('analysis.analysis'))
        
        # 안전한 파일 업로드
        try:
            filename = file_uploader.secure_file_upload(file, session['user_id'])
            filepath = file_uploader.save_file(file, filename)
        except Exception as e:
            flash(f'파일 업로드 실패: {str(e)}')
            return redirect(url_for('analysis.analysis'))
        
        # 얼굴 감지
        if not image_validator.is_face_image(filepath):
            flash("얼굴이 인식되지 않습니다. 얼굴이 보이는 사진을 업로드해주세요.")
            file_uploader.delete_file(filename)
            return redirect(url_for('analysis.analysis'))
        
        # 피부 분석 서비스 사용
        analysis_service = current_app.analysis_service
        from .config import get_model_paths
        model_paths = get_model_paths(current_app)
        
        try:
            scores = analysis_service.analyze_skin_image(filepath, model_paths)
            if scores is None:
                raise Exception("모델 분석 결과가 None입니다")
        except Exception as e:
            current_app.logger.error(f"피부 분석 실패: {e}")
            flash(f'피부 점수 분석 중 오류가 발생했습니다: {str(e)}')
            file_uploader.delete_file(filename)
            return redirect(url_for('analysis.analysis'))
        
        # 추천 서비스 사용
        recommendation_service = RecommendationService()
        
        skin_type = analysis_service.determine_skin_type(scores.get('skin_type_score', 50))
        concerns = analysis_service.identify_concerns(scores)
        recommendation_text = recommendation_service.generate_recommendation_text(
            scores, session.get('username', '방문자')
        )
        
        # 제품 추천 서비스 사용
        product_service = ProductRecommendationService()
        db = DatabaseManager.get_connection()
        current_season = get_current_season()
        makeup = 'no'  # 기본값
        
        morning_routine = product_service.get_morning_routine_structure(
            db, skin_type, concerns, current_season, makeup
        )
        night_routine = product_service.get_night_routine_structure(
            db, skin_type, concerns, current_season, makeup
        )
        
        # 사용자 정보 구성
        now = datetime.now()
        user_info = {
            "username": session.get('username', '방문자'),
            "date_info": {"year": now.year, "month": now.month, "day": now.day},
            "skin_type": skin_type,
            "concerns": concerns,
            "season": current_season,
            "makeup": makeup
        }
        
        recommendations_data = {
            "user_info": user_info,
            "morning_routine": morning_routine,
            "night_routine": night_routine
        }
        
        # 세션에 추천 데이터 저장
        session['recommendations_data'] = recommendations_data
        
        # 데이터베이스에 분석 결과 저장
        scores_serializable = {k: float(v.item() if hasattr(v, 'item') else v) 
                             for k, v in scores.items()}
        
        db.execute(
            'INSERT INTO analyses (user_id, skin_type, recommendation_text, scores_json, concerns_json, image_filename) VALUES (?, ?, ?, ?, ?, ?)',
            (session['user_id'], skin_type, recommendation_text, 
             json.dumps(scores_serializable), json.dumps(concerns), filename)
        )
        db.commit()
        
        # 결과 페이지용 데이터 준비
        concern_scores = {k: v for k, v in scores.items() if k != 'skin_type_score'}
        main_score = sum(concern_scores.values()) / len(concern_scores) if concern_scores else 0
        
        result_summary = recommendation_service.generate_result_summary(
            session.get('username', '방문자'), 
            main_score, 
            skin_type, 
            [concern['name'] for concern in concerns]
        )
        
        # 파일을 static 디렉토리로 이동
        static_dir = os.path.join('static', 'uploads_temp')
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)
        shutil.move(filepath, os.path.join(static_dir, filename))
        
        # 결과 페이지 렌더링
        return render_template(
            'result.html',
            main_score=main_score,
            scores=concern_scores,
            uploaded_image=url_for('static', filename=f'uploads_temp/{filename}'),
            result_summary=result_summary,
            recommendations=recommendations_data,
            skin_type=skin_type,
            original_scores=scores_serializable
        )
        
    except Exception as e:
        current_app.logger.error(f"이미지 분석 중 오류: {e}")
        flash(f'분석 중 오류가 발생했습니다: {str(e)}')
        return redirect(url_for('analysis.analysis'))


@analysis_bp.route('/routines')
def routines():
    """루틴 페이지"""
    recommendations = session.get('recommendations_data', None)
    if not recommendations:
        flash('먼저 피부 분석을 진행해주세요.', 'info')
        return redirect(url_for('analysis.analysis'))
    return render_template('routines.html', recommendations=recommendations)


# =====================
# 기록 관리 라우트
# =====================

@history_bp.route('/')
@login_required
def history():
    """분석 기록 페이지"""
    db = DatabaseManager.get_connection()
    all_analyses = db.execute(
        'SELECT * FROM analyses WHERE user_id = ? ORDER BY analysis_timestamp DESC',
        (session['user_id'],)
    ).fetchall()
    
    return render_template('history.html', analyses=all_analyses)


@history_bp.route('/delete_analysis/<int:analysis_id>', methods=['POST'])
@login_required
def delete_analysis(analysis_id):
    """개별 분석 기록 삭제"""
    db = DatabaseManager.get_connection()
    analysis = db.execute(
        'SELECT * FROM analyses WHERE id = ? AND user_id = ?', 
        (analysis_id, session['user_id'])
    ).fetchone()
    
    if analysis is None:
        flash('존재하지 않는 분석 기록입니다.', 'danger')
        return redirect(url_for('history.history'))
    
    db.execute('DELETE FROM analyses WHERE id = ?', (analysis_id,))
    db.commit()
    flash('분석 기록이 성공적으로 삭제되었습니다.', 'success')
    return redirect(url_for('history.history'))


@history_bp.route('/delete_selected_analyses', methods=['POST'])
@login_required
def delete_selected_analyses():
    """선택된 분석 기록들 일괄 삭제"""
    analysis_ids_to_delete = request.form.getlist('analysis_ids')
    if not analysis_ids_to_delete:
        flash('삭제할 기록을 선택해주세요.', 'info')
        return redirect(url_for('history.history'))
    
    db = DatabaseManager.get_connection()
    placeholders = ','.join('?' for _ in analysis_ids_to_delete)
    query = f'DELETE FROM analyses WHERE id IN ({placeholders}) AND user_id = ?'
    
    params = analysis_ids_to_delete + [session['user_id']]
    db.execute(query, params)
    db.commit()
    
    flash('선택한 분석 기록이 성공적으로 삭제되었습니다.', 'success')
    return redirect(url_for('history.history'))


# =====================
# 추천 시스템 라우트
# =====================

@recommendations_bp.route('/')
def recommendations():
    """추천 페이지"""
    results = session.get('skin_analysis_results', None)
    if not results:
        return render_template(
            'recommendations.html',
            skin_type="분석 전",
            concerns=[],
            recommendation_text='피부 분석을 먼저 진행해주세요. <a href="/analysis">분석 페이지로 이동</a>',
            products=[],
            current_season='N/A',
            recommendations={}
        )
    
    # 피부 타입과 고민에 따른 제품 추천
    skin_type = results.get('skin_type', 'N/A')
    concerns = results.get('concerns', [])
    scores = results.get('scores', {})
    current_season = get_current_season()
    makeup = results.get('makeup', 'no')
    
    # 새로운 구조화된 추천 시스템
    product_service = ProductRecommendationService()
    db = DatabaseManager.get_connection()
    
    morning_routine = product_service.get_morning_routine_structure(
        db, skin_type, concerns, current_season, makeup
    )
    night_routine = product_service.get_night_routine_structure(
        db, skin_type, concerns, current_season, makeup
    )
    
    # 사용자 정보
    now = datetime.now()
    user_info = {
        "username": session.get('username', '방문자'),
        "date_info": {
            "year": now.year,
            "month": now.month,
            "day": now.day
        },
        "skin_type": skin_type,
        "concerns": concerns,
        "season": current_season,
        "makeup": makeup
    }
    
    # 최종 추천 구조
    recommendations_data = {
        "user_info": user_info,
        "morning_routine": morning_routine,
        "night_routine": night_routine
    }
    
    return render_template(
        'recommendations.html',
        skin_type=skin_type,
        concerns=concerns,
        recommendation_text=results.get('recommendation_text', '오류가 발생했습니다.'),
        scores=scores,
        current_season=current_season,
        makeup=makeup,
        recommendations=recommendations_data
    )


# =====================
# API 라우트
# =====================

@api_bp.route('/history')
@login_required
def api_history():
    """기록 API (차트용)"""
    start_date_str = request.args.get('start_date')
    end_date_str = request.args.get('end_date')
    
    try:
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').replace(hour=23, minute=59, second=59)
    except (ValueError, TypeError):
        end_date = datetime.now().replace(hour=23, minute=59, second=59)
    
    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').replace(hour=0, minute=0, second=0)
    except (ValueError, TypeError):
        start_date = end_date - timedelta(days=6)
        start_date = start_date.replace(hour=0, minute=0, second=0)
    
    if start_date > end_date:
        return jsonify({'error': 'Start date cannot be after end date.'}), 400
    
    db = DatabaseManager.get_connection()
    analyses = db.execute(
        'SELECT analysis_timestamp, scores_json FROM analyses WHERE user_id = ? AND analysis_timestamp BETWEEN ? AND ? ORDER BY analysis_timestamp ASC',
        (session['user_id'], start_date, end_date)
    ).fetchall()
    
    # 일별 점수 집계
    daily_scores = {}
    current_date = start_date.date()
    while current_date <= end_date.date():
        date_key = current_date.strftime('%Y-%m-%d')
        daily_scores[date_key] = {'moisture': [], 'elasticity': [], 'wrinkle': []}
        current_date += timedelta(days=1)
    
    for analysis in analyses:
        analysis_date_key = analysis['analysis_timestamp'].strftime('%Y-%m-%d')
        if analysis_date_key in daily_scores:
            try:
                scores = json.loads(analysis['scores_json'])
                daily_scores[analysis_date_key]['moisture'].append(scores.get('moisture', 0))
                daily_scores[analysis_date_key]['elasticity'].append(scores.get('elasticity', 0))
                daily_scores[analysis_date_key]['wrinkle'].append(scores.get('wrinkle', 65.0))
            except (json.JSONDecodeError, TypeError):
                continue
    
    # 그래프용 데이터 생성
    graph_dates = []
    graph_moisture = []
    graph_elasticity = []
    graph_wrinkle = []
    
    for date_key, scores_list in sorted(daily_scores.items()):
        graph_dates.append(datetime.strptime(date_key, '%Y-%m-%d').strftime('%m-%d'))
        graph_moisture.append(
            round(sum(scores_list['moisture']) / len(scores_list['moisture']), 1) 
            if scores_list['moisture'] else 0
        )
        graph_elasticity.append(
            round(sum(scores_list['elasticity']) / len(scores_list['elasticity']), 1) 
            if scores_list['elasticity'] else 0
        )
        graph_wrinkle.append(
            round(sum(scores_list['wrinkle']) / len(scores_list['wrinkle']), 1) 
            if scores_list['wrinkle'] else 0
        )
    
    return jsonify(
        graph_dates=graph_dates,
        graph_moisture=graph_moisture,
        graph_elasticity=graph_elasticity,
        graph_wrinkle=graph_wrinkle
    )


@api_bp.route('/health')
def health_check():
    """헬스 체크 API"""
    try:
        # 모델 상태 확인
        model_status = current_app.model_manager.get_model_status()
        
        # 데이터베이스 연결 확인
        db = DatabaseManager.get_connection()
        db.execute('SELECT 1').fetchone()
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'models': model_status,
            'database': 'connected'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


# =====================
# 에러 핸들러
# =====================

@main_bp.errorhandler(404)
def not_found_error(error):
    """404 에러 핸들러"""
    return render_template('errors/404.html'), 404


@main_bp.errorhandler(500)
def internal_error(error):
    """500 에러 핸들러"""
    current_app.logger.error(f'Server Error: {error}')
    return render_template('errors/500.html'), 500


@main_bp.errorhandler(413)
def file_too_large_error(error):
    """413 에러 핸들러 (파일 크기 초과)"""
    flash('업로드된 파일이 너무 큽니다. 10MB 이하의 파일을 업로드해주세요.', 'danger')
    return redirect(url_for('analysis.analysis'))
