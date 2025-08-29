"""
SKINMATE 유틸리티 함수 모듈

공통으로 사용되는 유틸리티 함수들을 정의합니다.
파일 처리, 데이터베이스 관리, 이미지 검증, 보안 등의 기능을 제공합니다.
"""

import os
import cv2
import sqlite3
import hashlib
import time
import logging

# 선택적 임포트 (라이브러리가 없어도 실행 가능)
try:
    import magic
    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False
    magic = None
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from functools import wraps

from flask import session, request, jsonify, current_app, g, flash, redirect, url_for
from werkzeug.utils import secure_filename

from .models import ValidationError, DatabaseError


class ImageValidator:
    """
    이미지 검증 클래스
    
    업로드된 이미지의 유효성을 검사하고 얼굴 감지를 수행합니다.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
        self.allowed_mime_types = ['image/jpeg', 'image/png', 'image/gif']
    
    def is_allowed_file(self, filename: str) -> bool:
        """
        파일 확장자가 허용된 형식인지 확인
        
        Args:
            filename (str): 파일명
            
        Returns:
            bool: 허용 여부
        """
        return ('.' in filename and 
                filename.rsplit('.', 1)[1].lower() in self.allowed_extensions)
    
    def validate_mime_type(self, file_content: bytes) -> bool:
        """
        MIME 타입 검증
        
        Args:
            file_content (bytes): 파일 내용
            
        Returns:
            bool: 유효한 MIME 타입 여부
        """
        if not HAS_MAGIC:
            # magic 라이브러리가 없으면 기본적으로 허용
            self.logger.warning("python-magic 라이브러리가 없어 MIME 타입 검증을 건너뜁니다")
            return True
            
        try:
            mime_type = magic.from_buffer(file_content, mime=True)
            return mime_type in self.allowed_mime_types
        except Exception as e:
            self.logger.error(f"MIME 타입 검증 실패: {e}")
            return False
    
    def is_face_image(self, image_path: str) -> bool:
        """
        이미지에 얼굴이 포함되어 있는지 확인
        
        Args:
            image_path (str): 이미지 파일 경로
            
        Returns:
            bool: 얼굴 감지 여부
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                self.logger.error(f"이미지를 읽을 수 없습니다: {image_path}")
                return False
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
            )
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=8
            )
            
            return len(faces) > 0
            
        except Exception as e:
            self.logger.error(f"얼굴 감지 오류: {e}")
            return False
    
    def validate_skin_image(self, file, user_id: int) -> Dict[str, Any]:
        """
        피부 분석용 이미지 종합 검증
        
        Args:
            file: Flask 파일 객체
            user_id (int): 사용자 ID
            
        Returns:
            Dict[str, Any]: 검증 결과 정보
            
        Raises:
            ValidationError: 검증 실패 시
        """
        if not file or not file.filename:
            raise ValidationError("파일이 선택되지 않았습니다")
        
        if not self.is_allowed_file(file.filename):
            raise ValidationError("허용되지 않는 파일 형식입니다")
        
        # 파일 크기 확인
        file.seek(0, 2)
        file_size = file.tell()
        file.seek(0)
        
        max_size = current_app.config.get('MAX_CONTENT_LENGTH', 10 * 1024 * 1024)
        if file_size > max_size:
            raise ValidationError(f"파일 크기가 너무 큽니다: {file_size} bytes")
        
        if file_size == 0:
            raise ValidationError("빈 파일은 업로드할 수 없습니다")
        
        # MIME 타입 확인
        file_content = file.read()
        file.seek(0)
        
        if not self.validate_mime_type(file_content):
            raise ValidationError("유효하지 않은 파일 형식입니다")
        
        # magic 라이브러리가 있을 때만 MIME 타입 정보 추가
        mime_type = 'unknown'
        if HAS_MAGIC:
            try:
                mime_type = magic.from_buffer(file_content, mime=True)
            except:
                mime_type = 'unknown'
        
        return {
            'file_size': file_size,
            'mime_type': mime_type,
            'original_filename': file.filename
        }


class SecureFileUploader:
    """
    안전한 파일 업로드 클래스
    
    파일을 안전하게 업로드하고 관리합니다.
    """
    
    def __init__(self, upload_folder: str):
        self.upload_folder = Path(upload_folder)
        self.upload_folder.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def secure_file_upload(self, file, user_id: int) -> str:
        """
        안전한 파일 업로드 처리
        
        Args:
            file: Flask 파일 객체
            user_id (int): 사용자 ID
            
        Returns:
            str: 저장된 파일명
            
        Raises:
            ValueError: 업로드 실패 시
        """
        # 파일 크기 검증
        file.seek(0, 2)
        file_size = file.tell()
        file.seek(0)
        
        max_size = current_app.config.get('MAX_CONTENT_LENGTH', 10 * 1024 * 1024)
        if file_size > max_size:
            raise ValueError(f"파일 크기가 너무 큽니다: {file_size} bytes")
        
        if file_size == 0:
            raise ValueError("빈 파일은 업로드할 수 없습니다")
        
        # 안전한 파일명 생성
        file_ext = Path(file.filename).suffix.lower()
        timestamp = int(time.time())
        file_hash = hashlib.md5(
            f"{user_id}_{timestamp}_{file.filename}".encode()
        ).hexdigest()[:8]
        secure_filename_new = f"user_{user_id}_{timestamp}_{file_hash}{file_ext}"
        
        return secure_filename_new
    
    def save_file(self, file, filename: str) -> str:
        """
        파일을 디스크에 저장
        
        Args:
            file: Flask 파일 객체
            filename (str): 저장할 파일명
            
        Returns:
            str: 저장된 파일의 전체 경로
        """
        filepath = self.upload_folder / filename
        file.save(str(filepath))
        return str(filepath)
    
    def delete_file(self, filename: str) -> bool:
        """
        파일 삭제
        
        Args:
            filename (str): 삭제할 파일명
            
        Returns:
            bool: 삭제 성공 여부
        """
        try:
            filepath = self.upload_folder / filename
            if filepath.exists():
                filepath.unlink()
                return True
            return False
        except Exception as e:
            self.logger.error(f"파일 삭제 실패: {e}")
            return False


class DatabaseManager:
    """
    데이터베이스 관리 클래스
    
    데이터베이스 연결, 초기화, 쿼리 실행 등을 관리합니다.
    """
    
    def __init__(self, database_path: str):
        self.database_path = database_path
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def get_connection():
        """현재 요청의 데이터베이스 연결 가져오기"""
        if 'db' not in g:
            g.db = sqlite3.connect(
                current_app.config['DATABASE'],
                detect_types=sqlite3.PARSE_DECLTYPES
            )
            g.db.row_factory = sqlite3.Row
        return g.db
    
    def init_database(self):
        """데이터베이스 초기화"""
        try:
            schema_path = Path(__file__).parent.parent / 'schema.sql'
            if not schema_path.exists():
                raise DatabaseError(f"스키마 파일을 찾을 수 없습니다: {schema_path}")
            
            # 직접 SQLite 연결 생성 (Flask 컨텍스트가 없을 수 있으므로)
            import sqlite3
            db = sqlite3.connect(self.database_path)
            db.row_factory = sqlite3.Row
            
            with open(schema_path, 'r', encoding='utf-8') as f:
                db.executescript(f.read())
            
            db.commit()
            db.close()
            
            self.logger.info("데이터베이스 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"데이터베이스 초기화 실패: {e}")
            raise DatabaseError(f"데이터베이스 초기화 실패: {str(e)}")
    
    def drop_tables(self):
        """모든 테이블 삭제"""
        try:
            db = self.get_connection()
            
            # 모든 테이블 목록 가져오기
            tables = db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            
            # 각 테이블 삭제
            for table in tables:
                db.execute(f"DROP TABLE IF EXISTS {table['name']}")
            
            db.commit()
            self.logger.info("모든 테이블 삭제 완료")
            
        except Exception as e:
            self.logger.error(f"테이블 삭제 실패: {e}")
            raise DatabaseError(f"테이블 삭제 실패: {str(e)}")
    
    def safe_execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        """
        안전한 쿼리 실행
        
        Args:
            query (str): SQL 쿼리
            params (tuple): 쿼리 매개변수
            
        Returns:
            sqlite3.Cursor: 쿼리 결과
        """
        try:
            db = self.get_connection()
            return db.execute(query, params)
        except Exception as e:
            self.logger.error(f"쿼리 실행 실패: {e}")
            raise DatabaseError(f"쿼리 실행 실패: {str(e)}")


# 데코레이터 함수들
def login_required(f):
    """로그인 필요 데코레이터"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            if request.is_json:
                return jsonify({'error': '로그인이 필요합니다'}), 401
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function


def validate_input(schema_class):
    """입력 검증 데코레이터"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                schema = schema_class()
                if request.is_json:
                    data = schema.load(request.get_json())
                else:
                    data = schema.load(request.form)
                request.validated_data = data
                return f(*args, **kwargs)
            except Exception as e:
                if request.is_json:
                    return jsonify({'error': f'입력 검증 실패: {str(e)}'}), 400
                flash(f'입력 검증 실패: {str(e)}', 'danger')
                return redirect(request.referrer or url_for('main.index'))
        return decorated_function
    return decorator


def rate_limit(max_requests: int = 60, window_seconds: int = 60):
    """레이트 리미팅 데코레이터"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # 간단한 IP 기반 레이트 리미팅 구현
            client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
            
            # 실제 구현에서는 Redis나 메모리 캐시를 사용해야 함
            # 여기서는 간단한 예시만 제공
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator


# 유틸리티 함수들
def get_current_season() -> str:
    """현재 계절 반환"""
    month = datetime.now().month
    
    if month in [5, 6, 7, 8, 9]:
        return 'summer'
    elif month in [12, 1, 2]:
        return 'winter'
    else:
        return 'spring_fall'


def sanitize_input(text: str) -> str:
    """입력 텍스트 정리"""
    import re
    
    if not isinstance(text, str):
        return ""
    
    # HTML 태그 제거
    text = re.sub(r'<[^>]+>', '', text)
    
    # 위험한 문자 제거
    text = re.sub(r'[<>"\']', '', text)
    
    # 공백 정리
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def format_file_size(size_bytes: int) -> str:
    """파일 크기를 읽기 쉬운 형태로 변환"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"


def generate_csrf_token() -> str:
    """CSRF 토큰 생성"""
    import secrets
    return secrets.token_hex(16)


def verify_csrf_token(token: str) -> bool:
    """CSRF 토큰 검증"""
    session_token = session.get('csrf_token')
    return session_token and session_token == token


def clean_old_files(directory: Path, max_age_days: int = 7):
    """오래된 파일들 정리"""
    try:
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        
        for file_path in directory.iterdir():
            if file_path.is_file():
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_time:
                    file_path.unlink()
                    logging.info(f"오래된 파일 삭제: {file_path}")
    
    except Exception as e:
        logging.error(f"파일 정리 실패: {e}")


def create_thumbnail(image_path: str, thumbnail_path: str, size: tuple = (150, 150)):
    """이미지 썸네일 생성"""
    try:
        from PIL import Image
        
        with Image.open(image_path) as img:
            img.thumbnail(size, Image.Resampling.LANCZOS)
            img.save(thumbnail_path, optimize=True, quality=85)
        
        return True
        
    except Exception as e:
        logging.error(f"썸네일 생성 실패: {e}")
        return False


def get_client_ip() -> str:
    """클라이언트 IP 주소 가져오기"""
    return request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)


def log_user_activity(user_id: int, activity: str, details: Dict[str, Any] = None):
    """사용자 활동 로깅"""
    try:
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'activity': activity,
            'ip_address': get_client_ip(),
            'user_agent': request.headers.get('User-Agent', ''),
            'details': details or {}
        }
        
        # 실제 구현에서는 별도의 로그 파일이나 데이터베이스에 저장
        logging.info(f"USER_ACTIVITY: {log_entry}")
        
    except Exception as e:
        logging.error(f"사용자 활동 로깅 실패: {e}")


class PerformanceMonitor:
    """성능 모니터링 클래스"""
    
    def __init__(self):
        self.start_time = None
        self.logger = logging.getLogger(__name__)
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.logger.info(f"실행 시간: {elapsed:.3f}초")
    
    def measure_function(self, func_name: str):
        """함수 실행 시간 측정 데코레이터"""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                start_time = time.time()
                try:
                    result = f(*args, **kwargs)
                    return result
                finally:
                    elapsed = time.time() - start_time
                    self.logger.info(f"{func_name} 실행 시간: {elapsed:.3f}초")
            return decorated_function
        return decorator
