"""
SKINMATE 애플리케이션 팩토리 모듈

Flask 애플리케이션 인스턴스를 생성하고 구성요소들을 초기화합니다.
Application Factory 패턴을 사용하여 테스트와 배포 환경에서의 유연성을 제공합니다.
"""

import os
import logging
from flask import Flask, g
from pathlib import Path

# TensorFlow 경고 메시지 숨기기
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def create_app(config_name='development'):
    """
    Flask 애플리케이션을 생성하고 설정하는 팩토리 함수
    
    Args:
        config_name (str): 설정 환경 이름 ('development', 'production', 'testing')
        
    Returns:
        Flask: 구성된 Flask 애플리케이션 인스턴스
    """
    # Flask 애플리케이션 생성 (템플릿 및 정적 파일 경로 설정)
    template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')
    static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')
    
    app = Flask(__name__, 
                template_folder=template_dir,
                static_folder=static_dir)
    
    # 설정 로드
    from .config import get_config
    app.config.from_object(get_config(config_name))
    
    # 로깅 설정
    setup_logging(app)
    
    # 인스턴스 폴더 생성
    try:
        os.makedirs(app.instance_path, exist_ok=True)
    except OSError:
        app.logger.warning(f"Could not create instance path: {app.instance_path}")
    
    # 데이터베이스 초기화 (서비스 초기화 전에 실행)
    with app.app_context():
        try:
            from .utils import DatabaseManager
            db_manager = DatabaseManager(app.config['DATABASE'])
            db_manager.init_database()
            app.logger.info("데이터베이스 초기화 완료")
        except Exception as e:
            app.logger.warning(f"데이터베이스 초기화 실패: {e}")
    
    # 서비스 초기화
    initialize_services(app)
    
    # 블루프린트 등록
    register_blueprints(app)
    
    # 템플릿 필터 등록
    register_template_filters(app)
    
    # 에러 핸들러 등록
    register_error_handlers(app)
    
    # CLI 명령어 등록
    register_cli_commands(app)
    
    app.logger.info(f"SKINMATE application created with config: {config_name}")
    
    return app


def setup_logging(app):
    """로깅 설정"""
    if not app.debug and not app.testing:
        # 프로덕션 로깅 설정
        if not os.path.exists('logs'):
            os.mkdir('logs')
        
        file_handler = logging.FileHandler('logs/skinmate.log')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.INFO)


def initialize_services(app):
    """애플리케이션 서비스 초기화"""
    from .models import ModelManager
    from .services import AnalysisService, RecommendationService
    from .utils import ImageValidator, SecureFileUploader
    
    # 모델 매니저 초기화
    model_manager = ModelManager()
    app.model_manager = model_manager
    
    # 유틸리티 초기화
    image_validator = ImageValidator()
    file_uploader = SecureFileUploader(app.config['UPLOAD_FOLDER'])
    
    # 서비스 초기화 (의존성 주입)
    analysis_service = AnalysisService(model_manager, image_validator)
    recommendation_service = RecommendationService()
    
    # 애플리케이션에 서비스 등록
    app.analysis_service = analysis_service
    app.recommendation_service = recommendation_service
    app.image_validator = image_validator
    app.file_uploader = file_uploader


def register_blueprints(app):
    """블루프린트 등록"""
    from .routes import (
        main_bp, auth_bp, analysis_bp, 
        history_bp, recommendations_bp, api_bp
    )
    
    # 메인 블루프린트 (URL 접두사 없음)
    app.register_blueprint(main_bp)
    
    # 기능별 블루프린트
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(analysis_bp, url_prefix='/analysis')
    app.register_blueprint(history_bp, url_prefix='/history')
    app.register_blueprint(recommendations_bp, url_prefix='/recommendations')
    app.register_blueprint(api_bp, url_prefix='/api')


def register_template_filters(app):
    """Jinja2 템플릿 필터 등록"""
    import json
    
    @app.template_filter('fromjson')
    def fromjson_filter(json_string):
        """JSON 문자열을 Python 객체로 변환"""
        if json_string is None:
            return []
        try:
            return json.loads(json_string)
        except (json.JSONDecodeError, TypeError):
            return []
    
    @app.template_global()
    def get_face_icon(score):
        """점수에 따른 얼굴 아이콘 반환"""
        if score is None:
            return 'default-face.png'
        
        score = float(score)
        if 0 <= score <= 19:
            return 'face5.png'
        elif 20 <= score <= 49:
            return 'face4.png'
        elif 50 <= score <= 60:
            return 'face3.png'
        elif 61 <= score <= 90:
            return 'face2.png'
        elif 91 <= score <= 100:
            return 'face1.png'
        else:
            return 'default-face.png'


def register_error_handlers(app):
    """에러 핸들러 등록"""
    from flask import jsonify, render_template, request
    
    @app.errorhandler(404)
    def not_found_error(error):
        if request.is_json:
            return jsonify({'error': 'Resource not found'}), 404
        return render_template('errors/404.html'), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error(f'Server Error: {error}')
        if request.is_json:
            return jsonify({'error': 'Internal server error'}), 500
        return render_template('errors/500.html'), 500
    
    @app.errorhandler(413)
    def file_too_large_error(error):
        app.logger.warning(f'File too large: {error}')
        if request.is_json:
            return jsonify({'error': 'File too large'}), 413
        return render_template('errors/413.html'), 413


def register_cli_commands(app):
    """CLI 명령어 등록"""
    import click
    from .utils import DatabaseManager
    
    @app.cli.command('init-db')
    @click.option('--drop', is_flag=True, help='Drop existing tables')
    def init_db_command(drop):
        """데이터베이스 초기화"""
        db_manager = DatabaseManager(app.config['DATABASE'])
        
        if drop:
            db_manager.drop_tables()
            click.echo('Dropped existing tables.')
        
        db_manager.init_database()
        click.echo('Initialized the database.')
    
    @app.cli.command('clear-models')
    def clear_models_command():
        """모델 캐시 정리"""
        if hasattr(app, 'model_manager'):
            app.model_manager.clear_model()
            click.echo('Cleared model cache.')


# 데이터베이스 연결 관리
def get_db():
    """데이터베이스 연결 가져오기"""
    if 'db' not in g:
        from .utils import DatabaseManager
        g.db = DatabaseManager.get_connection()
    return g.db


def close_db(e=None):
    """데이터베이스 연결 종료"""
    db = g.pop('db', None)
    if db is not None:
        db.close()


# 애플리케이션 컨텍스트에서 데이터베이스 정리
def init_app_context(app):
    """애플리케이션 컨텍스트 초기화"""
    app.teardown_appcontext(close_db)
