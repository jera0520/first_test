"""
SKINMATE 애플리케이션 설정 관리 모듈

환경별 설정을 안전하게 관리하고 보안 강화를 위한 설정을 제공합니다.
"""

import os
import secrets
from pathlib import Path
from datetime import timedelta


class BaseConfig:
    """기본 설정 클래스"""
    
    # 보안 설정
    SECRET_KEY = os.environ.get('SECRET_KEY') or secrets.token_hex(32)
    
    # 데이터베이스 설정
    DATABASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'instance', 'skinmate.sqlite')
    
    # 파일 업로드 설정
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    ALLOWED_MIME_TYPES = ['image/jpeg', 'image/png', 'image/gif']
    
    # 세션 설정
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    SESSION_COOKIE_SECURE = False  # 개발환경에서는 False
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # AI 모델 설정
    MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)))
    XGBOOST_MODEL_FILE = 'my_xgboost_model.pkl'
    SCALER_MODEL_FILE = 'my_scaler.pkl' 
    SELECTOR_MODEL_FILE = 'my_selector.pkl'
    
    # 이미지 처리 설정
    IMAGE_SIZE = (224, 224)
    FEATURE_DIMENSIONS = 1000
    
    # 로깅 설정
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'logs/skinmate.log'
    
    # 성능 설정
    CACHE_TIMEOUT = 300  # 5분
    RATE_LIMIT_DEFAULT = "100 per hour"
    
    @staticmethod
    def init_app(app):
        """애플리케이션별 초기화"""
        # 업로드 폴더 생성
        upload_folder = app.config.get('UPLOAD_FOLDER')
        if upload_folder:
            Path(upload_folder).mkdir(parents=True, exist_ok=True)
        
        # 로그 폴더 생성
        log_file = app.config.get('LOG_FILE')
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)


class DevelopmentConfig(BaseConfig):
    """개발 환경 설정"""
    
    DEBUG = True
    TESTING = False
    
    # 개발용 완화된 보안 설정
    SESSION_COOKIE_SECURE = False
    
    # 개발용 상세 로깅
    LOG_LEVEL = 'DEBUG'
    
    # 모델 설정 (개발용)
    MODEL_CACHE_ENABLED = True
    MODEL_TIMEOUT = 30  # 30초
    
    @staticmethod
    def init_app(app):
        BaseConfig.init_app(app)
        app.logger.info("Application started in DEVELOPMENT mode")


class ProductionConfig(BaseConfig):
    """프로덕션 환경 설정"""
    
    DEBUG = False
    TESTING = False
    
    # 강화된 보안 설정
    SECRET_KEY = os.environ.get('SECRET_KEY')
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Strict'
    PERMANENT_SESSION_LIFETIME = timedelta(hours=1)  # 더 짧은 세션
    
    # 프로덕션 데이터베이스 (환경변수에서 가져오기)
    DATABASE = os.environ.get('DATABASE_URL') or BaseConfig.DATABASE
    
    # 프로덕션 파일 업로드 (더 엄격한 제한)
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5MB
    
    # 성능 최적화
    CACHE_TIMEOUT = 600  # 10분
    MODEL_CACHE_ENABLED = True
    
    # 보안 헤더 설정
    SECURITY_HEADERS = {
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Content-Security-Policy': (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "font-src 'self';"
        )
    }
    
    @staticmethod
    def init_app(app):
        BaseConfig.init_app(app)
        
        # 프로덕션 보안 검증
        if not app.config['SECRET_KEY']:
            raise ValueError("SECRET_KEY must be set in production environment")
        
        if app.config['SECRET_KEY'] == 'supersecretkey':
            raise ValueError("Default SECRET_KEY detected in production")
        
        # 보안 헤더 적용
        @app.after_request
        def set_security_headers(response):
            for header, value in ProductionConfig.SECURITY_HEADERS.items():
                response.headers[header] = value
            return response
        
        app.logger.info("Application started in PRODUCTION mode")


class TestingConfig(BaseConfig):
    """테스트 환경 설정"""
    
    DEBUG = False
    TESTING = True
    
    # 테스트용 임시 데이터베이스
    DATABASE = ':memory:'  # 메모리 내 SQLite
    
    # 테스트용 빠른 설정
    SECRET_KEY = 'test-secret-key'
    WTF_CSRF_ENABLED = False
    
    # 테스트용 작은 파일 크기
    MAX_CONTENT_LENGTH = 1 * 1024 * 1024  # 1MB
    
    # 캐시 비활성화
    CACHE_TIMEOUT = 0
    MODEL_CACHE_ENABLED = False
    
    @staticmethod
    def init_app(app):
        BaseConfig.init_app(app)
        app.logger.info("Application started in TESTING mode")


# 설정 딕셔너리
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(config_name=None):
    """
    설정 객체를 반환합니다.
    
    Args:
        config_name (str): 설정 이름 ('development', 'production', 'testing')
        
    Returns:
        Config: 설정 클래스 객체
    """
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')
    
    return config.get(config_name, config['default'])


def validate_config(app):
    """
    애플리케이션 설정을 검증합니다.
    
    Args:
        app: Flask 애플리케이션 인스턴스
        
    Raises:
        ValueError: 설정이 유효하지 않은 경우
    """
    required_settings = ['SECRET_KEY', 'DATABASE', 'UPLOAD_FOLDER']
    
    for setting in required_settings:
        if not app.config.get(setting):
            raise ValueError(f"Required setting '{setting}' is not configured")
    
    # 파일 경로 존재 여부 확인
    upload_folder = Path(app.config['UPLOAD_FOLDER'])
    if not upload_folder.exists():
        try:
            upload_folder.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ValueError(f"Cannot create upload folder: {e}")
    
    # 모델 파일 존재 여부 확인 (개발/프로덕션 환경에서만)
    if not app.config.get('TESTING'):
        model_path = Path(app.config['MODEL_PATH'])
        model_files = [
            app.config['XGBOOST_MODEL_FILE'],
            app.config['SCALER_MODEL_FILE'],
            app.config['SELECTOR_MODEL_FILE']
        ]
        
        for model_file in model_files:
            model_file_path = model_path / model_file
            if not model_file_path.exists():
                app.logger.warning(f"Model file not found: {model_file_path}")


def get_model_paths(app):
    """
    모델 파일 경로들을 반환합니다.
    
    Args:
        app: Flask 애플리케이션 인스턴스
        
    Returns:
        dict: 모델 파일 경로 딕셔너리
    """
    model_path = Path(app.config['MODEL_PATH'])
    
    return {
        'xgboost_model': model_path / app.config['XGBOOST_MODEL_FILE'],
        'scaler_model': model_path / app.config['SCALER_MODEL_FILE'],
        'selector_model': model_path / app.config['SELECTOR_MODEL_FILE']
    }
