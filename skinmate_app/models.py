"""
SKINMATE 데이터 모델 및 AI 모델 관리 모듈

데이터베이스 모델과 AI 모델 관리 클래스를 정의합니다.
"""

import logging
import pickle
import hashlib
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime


# AI 모델 관리자
class ModelManager:
    """
    AI 모델 관리 클래스
    
    ResNet50과 XGBoost 모델을 안전하게 로드하고 관리합니다.
    """
    
    def __init__(self):
        self._resnet_model: Optional[Any] = None
        self._xgboost_model: Optional[Any] = None
        self._scaler_model: Optional[Any] = None
        self._selector_model: Optional[Any] = None
        self._models_loaded = False
        self._logger = logging.getLogger(__name__)
    
    def get_resnet_model(self):
        """
        ResNet50 모델을 안전하게 로드하고 반환
        
        Returns:
            ResNet50 모델 인스턴스 또는 None
        """
        if self._resnet_model is None:
            try:
                import tensorflow as tf
                # TensorFlow 메모리 증가 허용 설정
                physical_devices = tf.config.experimental.list_physical_devices('GPU')
                if physical_devices:
                    tf.config.experimental.set_memory_growth(physical_devices[0], True)
                
                from tensorflow.keras.applications.resnet50 import ResNet50
                self._resnet_model = ResNet50(
                    weights='imagenet', 
                    include_top=False, 
                    pooling='avg'
                )
                self._logger.info("ResNet50 모델 로드 완료")
            except Exception as e:
                self._logger.error(f"ResNet50 모델 로드 실패: {e}")
                self._resnet_model = None
        
        return self._resnet_model
    
    def get_xgboost_model(self, model_path: str):
        """
        XGBoost 모델을 안전하게 로드하고 반환
        
        Args:
            model_path (str): 모델 파일 경로
            
        Returns:
            XGBoost 모델 인스턴스 또는 None
        """
        if self._xgboost_model is None:
            try:
                model_file = Path(model_path)
                if not model_file.exists():
                    raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
                
                with open(model_file, 'rb') as f:
                    self._xgboost_model = pickle.load(f)
                
                # 모델 유효성 검증
                if not hasattr(self._xgboost_model, 'predict'):
                    raise ValueError("유효하지 않은 XGBoost 모델입니다")
                
                self._logger.info("XGBoost 모델 로드 완료")
            except Exception as e:
                self._logger.error(f"XGBoost 모델 로드 실패: {e}")
                self._xgboost_model = None
        
        return self._xgboost_model
    
    def get_scaler_model(self, scaler_path: str):
        """
        Scaler 모델을 안전하게 로드하고 반환
        
        Args:
            scaler_path (str): Scaler 파일 경로
            
        Returns:
            Scaler 모델 인스턴스 또는 None
        """
        if self._scaler_model is None:
            try:
                scaler_file = Path(scaler_path)
                if not scaler_file.exists():
                    self._logger.warning(f"Scaler 파일을 찾을 수 없습니다: {scaler_path}")
                    return None
                
                with open(scaler_file, 'rb') as f:
                    self._scaler_model = pickle.load(f)
                
                self._logger.info("Scaler 모델 로드 완료")
            except Exception as e:
                self._logger.error(f"Scaler 모델 로드 실패: {e}")
                self._scaler_model = None
        
        return self._scaler_model
    
    def get_selector_model(self, selector_path: str):
        """
        Selector 모델을 안전하게 로드하고 반환
        
        Args:
            selector_path (str): Selector 파일 경로
            
        Returns:
            Selector 모델 인스턴스 또는 None
        """
        if self._selector_model is None:
            try:
                selector_file = Path(selector_path)
                if not selector_file.exists():
                    self._logger.warning(f"Selector 파일을 찾을 수 없습니다: {selector_path}")
                    return None
                
                with open(selector_file, 'rb') as f:
                    self._selector_model = pickle.load(f)
                
                self._logger.info("Selector 모델 로드 완료")
            except Exception as e:
                self._logger.error(f"Selector 모델 로드 실패: {e}")
                self._selector_model = None
        
        return self._selector_model
    
    def is_resnet_loaded(self) -> bool:
        """ResNet50 모델 로드 상태 확인"""
        return self._resnet_model is not None
    
    def is_xgboost_loaded(self) -> bool:
        """XGBoost 모델 로드 상태 확인"""
        return self._xgboost_model is not None
    
    def clear_models(self):
        """모든 모델 메모리 해제"""
        self._resnet_model = None
        self._xgboost_model = None
        self._scaler_model = None
        self._selector_model = None
        self._models_loaded = False
        self._logger.info("모든 모델 메모리 해제 완료")
    
    def get_model_status(self) -> Dict[str, bool]:
        """모든 모델의 로드 상태 반환"""
        return {
            'resnet_loaded': self.is_resnet_loaded(),
            'xgboost_loaded': self.is_xgboost_loaded(),
            'scaler_loaded': self._scaler_model is not None,
            'selector_loaded': self._selector_model is not None
        }


# 데이터 클래스들
class AnalysisResult:
    """피부 분석 결과를 담는 데이터 클래스"""
    
    def __init__(self, scores: Dict[str, float], skin_type: str, 
                 concerns: list, recommendation_text: str):
        self.scores = scores
        self.skin_type = skin_type
        self.concerns = concerns
        self.recommendation_text = recommendation_text
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'scores': self.scores,
            'skin_type': self.skin_type,
            'concerns': self.concerns,
            'recommendation_text': self.recommendation_text,
            'timestamp': self.timestamp.isoformat()
        }
    
    def get_main_score(self) -> float:
        """주요 점수 계산 (수분, 탄력, 주름의 평균)"""
        concern_scores = {k: v for k, v in self.scores.items() 
                         if k != 'skin_type_score'}
        return sum(concern_scores.values()) / len(concern_scores) if concern_scores else 0


class RecommendationData:
    """제품 추천 데이터를 담는 데이터 클래스"""
    
    def __init__(self, user_info: Dict, morning_routine: Dict, night_routine: Dict):
        self.user_info = user_info
        self.morning_routine = morning_routine
        self.night_routine = night_routine
        self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'user_info': self.user_info,
            'morning_routine': self.morning_routine,
            'night_routine': self.night_routine,
            'created_at': self.created_at.isoformat()
        }


class UserSession:
    """사용자 세션 데이터를 관리하는 클래스"""
    
    def __init__(self, user_id: int, username: str):
        self.user_id = user_id
        self.username = username
        self.login_time = datetime.now()
        self.last_activity = datetime.now()
        self.analysis_results: Optional[AnalysisResult] = None
        self.recommendations: Optional[RecommendationData] = None
    
    def update_activity(self):
        """마지막 활동 시간 업데이트"""
        self.last_activity = datetime.now()
    
    def set_analysis_results(self, results: AnalysisResult):
        """분석 결과 설정"""
        self.analysis_results = results
        self.update_activity()
    
    def set_recommendations(self, recommendations: RecommendationData):
        """추천 데이터 설정"""
        self.recommendations = recommendations
        self.update_activity()
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (세션 저장용)"""
        return {
            'user_id': self.user_id,
            'username': self.username,
            'login_time': self.login_time.isoformat(),
            'last_activity': self.last_activity.isoformat()
        }


# 예외 클래스들
class ModelError(Exception):
    """모델 관련 예외"""
    pass


class ModelLoadError(ModelError):
    """모델 로드 실패 예외"""
    pass


class PredictionError(ModelError):
    """예측 실패 예외"""
    pass


class ValidationError(Exception):
    """데이터 검증 실패 예외"""
    pass


class DatabaseError(Exception):
    """데이터베이스 관련 예외"""
    pass


# 유틸리티 함수들
def create_file_hash(file_content: bytes) -> str:
    """파일 내용의 해시값 생성"""
    return hashlib.md5(file_content).hexdigest()


def validate_score_range(scores: Dict[str, float]) -> bool:
    """점수 범위 유효성 검증 (0-100)"""
    for key, value in scores.items():
        if not isinstance(value, (int, float)):
            return False
        if not (0 <= value <= 100):
            return False
    return True


def sanitize_filename(filename: str) -> str:
    """파일명 안전하게 정리"""
    import re
    # 위험한 문자 제거
    safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    # 길이 제한
    return safe_filename[:100]


def get_current_season() -> str:
    """현재 계절 반환"""
    month = datetime.now().month
    
    if month in [5, 6, 7, 8, 9]:
        return 'summer'
    elif month in [12, 1, 2]:
        return 'winter'
    else:
        return 'spring_fall'
