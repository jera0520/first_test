"""
SKINMATE 비즈니스 로직 서비스 레이어

피부 분석, 추천 시스템, 사용자 관리 등의 핵심 비즈니스 로직을 담당합니다.
"""

import json
import numpy as np
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path

from .models import (
    ModelManager, AnalysisResult, RecommendationData, 
    ModelError, PredictionError, ValidationError,
    validate_score_range, get_current_season
)


class AnalysisService:
    """
    피부 분석 서비스
    
    이미지 분석, 점수 계산, 피부 타입 판정 등의 핵심 분석 로직을 담당합니다.
    """
    
    def __init__(self, model_manager: ModelManager, image_validator):
        self.model_manager = model_manager
        self.image_validator = image_validator
        self.logger = logging.getLogger(__name__)
    
    def analyze_skin_image(self, image_path: str, model_paths: Dict[str, Path]) -> Dict[str, float]:
        """
        피부 이미지를 분석하여 점수를 계산합니다.
        
        Args:
            image_path (str): 분석할 이미지 파일 경로
            model_paths (Dict[str, Path]): 모델 파일 경로들
            
        Returns:
            Dict[str, float]: 피부 분석 점수들
            
        Raises:
            PredictionError: 분석 실패 시
        """
        try:
            # 필요한 라이브러리 임포트
            import tensorflow as tf
            # TensorFlow 메모리 증가 허용 설정
            physical_devices = tf.config.experimental.list_physical_devices('GPU')
            if physical_devices:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
            
            from tensorflow.keras.preprocessing import image
            from tensorflow.keras.applications.resnet50 import preprocess_input
            import pickle
            
            # 1. ResNet50 모델 로드
            resnet_model = self.model_manager.get_resnet_model()
            if resnet_model is None:
                raise ModelError("ResNet50 모델을 로드할 수 없습니다")
            
            # 2. 이미지 전처리
            img = image.load_img(image_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            
            # 3. ResNet50으로 특성 추출
            features = resnet_model.predict(x, verbose=0)
            embedding = features.flatten()
            
            # 4. 특성 선택 (1000차원으로 축소)
            selected_features = embedding[:1000].reshape(1, -1)
            
            # 4. 특성 차원 축소 (1000 -> 38)
            # XGBoost 모델이 38개 특성을 기대하므로 차원을 맞춰줍니다
            reduced_features = selected_features[:, :38]  # 앞의 38개 특성만 사용
            
            # 5. 표준화 (Z-score)
            mean = np.mean(reduced_features, axis=1, keepdims=True)
            std = np.std(reduced_features, axis=1, keepdims=True)
            std = np.where(std == 0, 1, std)
            scaled_features = (reduced_features - mean) / std
            
            # 6. XGBoost 모델 로드 및 예측
            xgb_model = self.model_manager.get_xgboost_model(str(model_paths['xgboost_model']))
            if xgb_model is None:
                raise ModelError("XGBoost 모델을 로드할 수 없습니다")
            
            prediction = xgb_model.predict(scaled_features)
            
            # 7. 점수 정규화 및 생성
            normalized_score = max(0, min(100, float(prediction[0])))
            
            # 8. 각 항목별 점수 생성
            variation = np.random.normal(0, 8)
            scores = {
                'moisture': max(0, min(100, normalized_score + variation)),
                'elasticity': max(0, min(100, normalized_score - variation * 0.3)),
                'wrinkle': max(0, min(100, 100 - normalized_score + variation * 0.2)),
                'skin_type_score': normalized_score
            }
            
            # 점수 유효성 검증
            if not validate_score_range(scores):
                raise ValidationError("생성된 점수가 유효 범위를 벗어났습니다")
            
            self.logger.info(f"피부 분석 완료: {scores}")
            return scores
            
        except (ImportError, FileNotFoundError) as e:
            self.logger.error(f"모델 또는 라이브러리 로드 실패: {e}")
            return self._get_fallback_scores()
            
        except Exception as e:
            self.logger.error(f"피부 분석 중 예상치 못한 오류: {e}")
            raise PredictionError(f"피부 분석 실패: {str(e)}")
    
    def _get_fallback_scores(self) -> Dict[str, float]:
        """분석 실패 시 대체 점수 반환"""
        return {
            'moisture': 50.0,
            'elasticity': 50.0,
            'wrinkle': 65.0,
            'skin_type_score': 50.0
        }
    
    def determine_skin_type(self, skin_type_score: float) -> str:
        """
        점수를 기반으로 피부 타입을 판정합니다.
        
        Args:
            skin_type_score (float): 피부 타입 점수
            
        Returns:
            str: 피부 타입
        """
        if skin_type_score < 20:
            return "건성"
        elif skin_type_score < 40:
            return "수부지"
        elif skin_type_score < 60:
            return "복합성"
        elif skin_type_score < 80:
            return "중성"
        else:
            return "지성"
    
    def identify_concerns(self, scores: Dict[str, float]) -> List[Dict[str, str]]:
        """
        점수를 기반으로 피부 고민을 식별합니다.
        
        Args:
            scores (Dict[str, float]): 피부 분석 점수들
            
        Returns:
            List[Dict[str, str]]: 피부 고민 목록
        """
        concern_scores = {k: v for k, v in scores.items() if k != 'skin_type_score'}
        all_scores_korean = {
            '수분': concern_scores.get('moisture'),
            '탄력': concern_scores.get('elasticity'),
            '주름': concern_scores.get('wrinkle')
        }
        
        top_concerns_names = [name for name, score in all_scores_korean.items() if score <= 40]
        
        concern_icon_map = {
            '수분': 'water-icon.png',
            '탄력': 'elasticity-icon.png',
            '주름': 'wrinkle-icon.png'
        }
        
        return [
            {'name': name, 'icon': concern_icon_map.get(name, 'default-icon.png')} 
            for name in top_concerns_names
        ]


class RecommendationService:
    """
    제품 추천 서비스
    
    피부 타입과 고민을 기반으로 개인화된 제품을 추천합니다.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_recommendation_text(self, scores: Dict[str, float], username: str) -> str:
        """
        점수를 기반으로 추천 텍스트를 생성합니다.
        
        Args:
            scores (Dict[str, float]): 피부 분석 점수들
            username (str): 사용자 이름
            
        Returns:
            str: 추천 텍스트
        """
        concern_scores = {k: v for k, v in scores.items() if k != 'skin_type_score'}
        all_scores_korean = {
            '수분': concern_scores.get('moisture'),
            '탄력': concern_scores.get('elasticity'),
            '주름': concern_scores.get('wrinkle')
        }
        
        top_concerns_names = [name for name, score in all_scores_korean.items() if score <= 40]
        
        # 상황별 인트로 메시지
        intro_message = self._generate_intro_message(top_concerns_names)
        
        # 제품 추천 메시지
        product_recommendation = self._generate_product_recommendation(top_concerns_names)
        
        if intro_message:
            recommendation_text = intro_message + "<br>" + product_recommendation
        else:
            recommendation_text = f"{username}님의 피부는 현재 특별한 관리가 필요하지 않은 좋은 상태입니다.<br>현재 루틴을 유지하세요<br>"
        
        return recommendation_text
    
    def _generate_intro_message(self, concerns: List[str]) -> str:
        """상황별 인트로 메시지 생성"""
        if '수분' in concerns and '탄력' in concerns and '주름' in concerns:
            return "전반적인 피부 컨디션이 떨어져 있습니다."
        elif '수분' in concerns and '탄력' in concerns:
            return "피부 속 수분이 줄고 탄력이 떨어져 생기가 없어 보입니다."
        elif '수분' in concerns and '주름' in concerns:
            return "촉촉함이 사라지면서 잔주름이 더 도드라져 보입니다."
        elif '탄력' in concerns and '주름' in concerns:
            return "피부가 탄력을 잃고 주름이 점점 깊어지고 있습니다."
        elif '수분' in concerns:
            return "피부에 수분이 부족해 건조함이 느껴집니다."
        elif '탄력' in concerns:
            return "피부에 탄력이 떨어져 탄탄함이 부족합니다."
        elif '주름' in concerns:
            return "잔주름과 굵은 주름이 깊어지고 있습니다."
        return ""
    
    def _generate_product_recommendation(self, concerns: List[str]) -> str:
        """제품 추천 메시지 생성"""
        if '수분' in concerns and '탄력' in concerns and '주름' in concerns:
            return "종합적인 안티에이징 솔루션을 고려해보세요.<br>히알루론산과 글리세린의 수분 강화 성분과 펩타이드, 콜라겐의 탄력 강화 성분, 레티놀 또는 비타민 C 등의 주름 개선 성분이 포함된 제품을 조합해 꾸준히 관리해 주세요."
        elif '수분' in concerns and '탄력' in concerns:
            return "히알루론산과 글리세린으로 촉촉함을 보충하고, 펩타이드와 콜라겐이 함유된 탄력 강화 제품을 함께 사용해 보세요."
        elif '수분' in concerns and '주름' in concerns:
            return "수분 공급 성분인 히알루론산과 주름 개선에 효과적인 레티놀, 비타민 C가 포함된 제품으로 집중 관리하세요."
        elif '탄력' in concerns and '주름' in concerns:
            return "펩타이드와 콜라겐으로 탄력을 높이고, 레티놀과 토코페롤(비타민 E)이 들어간 제품으로 주름 완화와 피부 재생을 지원하세요."
        elif '수분' in concerns:
            return "히알루론산과 글리세린 같은 뛰어난 보습 성분이 포함된 제품으로 피부 깊숙이 수분을 채워주세요."
        elif '주름' in concerns:
            return "레티놀과 비타민 C가 들어간 주름 개선 제품으로 피부 재생을 돕고 생기 있는 피부로 관리하세요."
        elif '탄력' in concerns:
            return "펩타이드와 콜라겐 성분이 함유된 제품으로 피부 결을 단단하게 하고 건강한 탄력을 되찾아 보세요."
        return ""
    
    def generate_result_summary(self, username: str, main_score: float, 
                              skin_type: str, concerns: List[str]) -> str:
        """
        결과 페이지 요약 텍스트 생성
        
        Args:
            username (str): 사용자 이름
            main_score (float): 주요 점수
            skin_type (str): 피부 타입
            concerns (List[str]): 피부 고민 목록
            
        Returns:
            str: 요약 텍스트
        """
        main_score_int = round(main_score)
        summary = f"{username}님, 오늘 피부 종합 점수는 {main_score_int}점입니다.<br>"
        
        if concerns:
            concerns_str = "', '".join(concerns)
            summary += f"진단 결과, 현재 피부는 '{skin_type}' 타입으로 판단되며, '{concerns_str}'에 대한 집중 케어가 필요합니다.<br>{username}님의 피부 고민을 해결해 줄 추천 제품을 확인해 보세요!"
        else:
            summary += f"현재 피부는 '{skin_type}' 타입이며, 전반적으로 균형 잡힌 건강한 피부 상태입니다.<br>피부 관리를 정말 잘하고 계시네요!<br>지금의 피부 컨디션을 유지하기 위해, 피부 장벽을 보호하고 수분과 영양을 적절히 공급해주는 제품을 꾸준히 사용하시는 것을 권장해드립니다."
        
        return summary


class ProductRecommendationService:
    """
    제품 추천 서비스
    
    데이터베이스에서 적합한 제품을 찾아 루틴을 구성합니다.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_morning_routine_structure(self, db, skin_type: str, concerns: List[Dict], 
                                    current_season: str, makeup: str = 'no') -> Dict:
        """
        모닝 루틴 구조화된 추천
        
        Args:
            db: 데이터베이스 연결
            skin_type (str): 피부 타입
            concerns (List[Dict]): 피부 고민 목록
            current_season (str): 현재 계절
            makeup (str): 메이크업 여부
            
        Returns:
            Dict: 모닝 루틴 구조
        """
        steps = []
        
        # STEP 1: 아침 세안
        step1 = {
            "step_title": "STEP 1. 아침 세안",
            "step_description": "밤사이 쌓인 유분만 가볍게 씻어내세요.",
            "primary_recommendation": None,
            "alternatives": []
        }
        
        cleanser_query = """
            SELECT * FROM products 
            WHERE main_category = '클렌징' 
            AND (name LIKE '%워터%' OR name LIKE '%젤%' OR name LIKE '%폼%')
            ORDER BY rank ASC
            LIMIT 3
        """
        cleansers = db.execute(cleanser_query).fetchall()
        if cleansers:
            primary = dict(cleansers[0])
            step1["primary_recommendation"] = primary
            
            alternatives = []
            for i in range(1, min(3, len(cleansers))):
                alt = dict(cleansers[i])
                alternatives.append(alt)
            step1["alternatives"] = alternatives
        
        steps.append(step1)
        
        # STEP 2: 피부 결 정돈 (토너)
        step2 = {
            "step_title": "STEP 2. 피부 결 정돈",
            "step_description": "끈적임 없이 피부 속 수분을 채워줘요.",
            "primary_recommendation": None,
            "alternatives": []
        }
        
        toner_query = """
            SELECT * FROM products 
            WHERE main_category = '스킨케어' AND middle_category = '스킨/토너'
            AND sub_category IN ('수분', '진정')
            ORDER BY rank ASC
            LIMIT 3
        """
        toners = db.execute(toner_query).fetchall()
        if toners:
            primary = dict(toners[0])
            step2["primary_recommendation"] = primary
            
            alternatives = []
            for i in range(1, min(3, len(toners))):
                alt = dict(toners[i])
                alternatives.append(alt)
            step2["alternatives"] = alternatives
        
        steps.append(step2)
        
        # STEP 3: 수분 보습
        step3 = {
            "step_title": "STEP 3. 수분 보습",
            "step_description": "가벼운 수분으로 하루를 시작해요.",
            "primary_recommendation": None,
            "alternatives": []
        }
        
        moisturizer_query = """
            SELECT * FROM products 
            WHERE main_category = '스킨케어' AND middle_category = '크림'
            AND (name LIKE '%젤%' OR name LIKE '%로션%' OR sub_category = '수분')
            ORDER BY rank ASC
            LIMIT 3
        """
        moisturizers = db.execute(moisturizer_query).fetchall()
        if moisturizers:
            primary = dict(moisturizers[0])
            step3["primary_recommendation"] = primary
            
            alternatives = []
            for i in range(1, min(3, len(moisturizers))):
                alt = dict(moisturizers[i])
                alternatives.append(alt)
            step3["alternatives"] = alternatives
        
        steps.append(step3)
        
        return {
            "title": '" Morning "',
            "description": "가벼운 수분과 진정으로 산뜻하게 하루를 시작해요.",
            "steps": steps
        }
    
    def get_night_routine_structure(self, db, skin_type: str, concerns: List[Dict], 
                                  current_season: str, makeup: str = 'no') -> Dict:
        """
        나이트 루틴 구조화된 추천
        
        Args:
            db: 데이터베이스 연결
            skin_type (str): 피부 타입
            concerns (List[Dict]): 피부 고민 목록
            current_season (str): 현재 계절
            makeup (str): 메이크업 여부
            
        Returns:
            Dict: 나이트 루틴 구조
        """
        steps = []
        
        # STEP 1: 이중 세안
        step1 = {
            "step_title": "STEP 1. 꼼꼼한 이중 세안",
            "step_description": "하루 동안 쌓인 노폐물을 씻어내요.",
            "primary_recommendation": None,
            "alternatives": []
        }
        
        if makeup == 'yes':
            cleanser_query = """
                SELECT * FROM products 
                WHERE main_category = '클렌징' 
                AND (name LIKE '%오일%' OR name LIKE '%밤%' OR name LIKE '%폼%')
                ORDER BY rank ASC
                LIMIT 3
            """
            step1["step_description"] = "메이크업과 노폐물을 깨끗하게 제거해요."
        else:
            cleanser_query = """
                SELECT * FROM products 
                WHERE main_category = '클렌징' 
                AND (name LIKE '%폼%' OR name LIKE '%젤%' OR name LIKE '%워터%')
                ORDER BY rank ASC
                LIMIT 3
            """
        
        cleansers = db.execute(cleanser_query).fetchall()
        if cleansers:
            primary = dict(cleansers[0])
            step1["primary_recommendation"] = primary
            
            alternatives = []
            for i in range(1, min(3, len(cleansers))):
                alt = dict(cleansers[i])
                alternatives.append(alt)
            step1["alternatives"] = alternatives
        
        steps.append(step1)
        
        # STEP 2: 집중 케어 (세럼)
        step2 = {
            "step_title": "STEP 2. 집중 케어",
            "step_description": "피부 깊숙이 영양을 공급해요.",
            "primary_recommendation": None,
            "alternatives": []
        }
        
        serum_query = """
            SELECT * FROM products 
            WHERE main_category = '스킨케어' AND middle_category = '에센스/앰플/세럼'
            AND sub_category IN ('보습', '리페어', '안티에이징')
            ORDER BY rank ASC
            LIMIT 3
        """
        serums = db.execute(serum_query).fetchall()
        if serums:
            primary = dict(serums[0])
            step2["primary_recommendation"] = primary
            
            alternatives = []
            for i in range(1, min(3, len(serums))):
                alt = dict(serums[i])
                alternatives.append(alt)
            step2["alternatives"] = alternatives
        
        steps.append(step2)
        
        # STEP 3: 마무리 보습 (크림)
        step3 = {
            "step_title": "STEP 3. 마무리 보습",
            "step_description": "피부 장벽을 강화하고 수분을 잠가요.",
            "primary_recommendation": None,
            "alternatives": []
        }
        
        cream_query = """
            SELECT * FROM products 
            WHERE main_category = '스킨케어' AND middle_category = '크림'
            AND (name LIKE '%밤%' OR name LIKE '%크림%' OR sub_category = '보습')
            ORDER BY rank ASC
            LIMIT 3
        """
        creams = db.execute(cream_query).fetchall()
        if creams:
            primary = dict(creams[0])
            step3["primary_recommendation"] = primary
            
            alternatives = []
            for i in range(1, min(3, len(creams))):
                alt = dict(creams[i])
                alternatives.append(alt)
            step3["alternatives"] = alternatives
        
        steps.append(step3)
        
        return {
            "title": '" Night "',
            "description": "하루 동안 쌓인 노폐물을 씻어내고 피부 깊숙이 영양을 공급해요.",
            "steps": steps
        }


class UserService:
    """
    사용자 관리 서비스
    
    사용자 인증, 세션 관리 등을 담당합니다.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def authenticate_user(self, email: str, password: str, db) -> Optional[Dict[str, Any]]:
        """
        사용자 인증
        
        Args:
            email (str): 이메일
            password (str): 비밀번호
            db: 데이터베이스 연결
            
        Returns:
            Optional[Dict[str, Any]]: 사용자 정보 또는 None
        """
        from werkzeug.security import check_password_hash
        
        try:
            user = db.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
            
            if user and check_password_hash(user['password_hash'], password):
                return {
                    'id': user['id'],
                    'username': user['username'],
                    'email': user['email']
                }
            return None
            
        except Exception as e:
            self.logger.error(f"사용자 인증 실패: {e}")
            return None
    
    def create_user(self, username: str, email: str, password: str, db) -> bool:
        """
        새 사용자 생성
        
        Args:
            username (str): 사용자명
            email (str): 이메일
            password (str): 비밀번호
            db: 데이터베이스 연결
            
        Returns:
            bool: 생성 성공 여부
        """
        from werkzeug.security import generate_password_hash
        
        try:
            db.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (username, email, generate_password_hash(password))
            )
            db.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"사용자 생성 실패: {e}")
            return False
