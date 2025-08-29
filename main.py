"""
화해 뷰티 제품 데이터 수집 메인 스크립트
20년차 개발자의 실용적 접근: 안정성과 모니터링 우선
"""

import logging
import sys
import time
from datetime import datetime
from typing import List, Dict

from crawler import HwahaeAPICrawler
from database import ProductDatabase

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crawler.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DataCollectionPipeline:
    """데이터 수집 파이프라인 클래스"""
    
    def __init__(self):
        self.crawler = HwahaeAPICrawler()
        self.database = ProductDatabase()
        self.start_time = None
        self.end_time = None
    
    def run(self) -> Dict:
        """
        전체 데이터 수집 파이프라인을 실행합니다.
        
        Returns:
            실행 결과 통계
        """
        self.start_time = datetime.now()
        logger.info("=== 데이터 수집 파이프라인 시작 ===")
        
        try:
            # 1. 크롤링 실행
            logger.info("1단계: 웹 크롤링 시작")
            products = self.crawl_products()
            
            if not products:
                logger.error("크롤링된 제품이 없습니다.")
                return self._create_error_result("크롤링 실패")
            
            # 2. 데이터베이스 저장
            logger.info("2단계: 데이터베이스 저장 시작")
            saved_count = self.save_products(products)
            
            # 3. 데이터 정리
            logger.info("3단계: 데이터 정리 시작")
            self.cleanup_database()
            
            # 4. 결과 통계
            self.end_time = datetime.now()
            result = self._create_success_result(products, saved_count)
            
            logger.info("=== 데이터 수집 파이프라인 완료 ===")
            return result
            
        except Exception as e:
            logger.error(f"파이프라인 실행 중 오류: {e}")
            return self._create_error_result(str(e))
    
    def crawl_products(self) -> List[Dict]:
        """제품 데이터를 크롤링합니다."""
        try:
            logger.info("크롤링 시작...")
            products = self.crawler.crawl_all_categories()
            
            # 데이터 검증
            valid_products = []
            for product in products:
                if self._validate_product(product):
                    valid_products.append(product)
                else:
                    logger.warning(f"유효하지 않은 제품 데이터: {product.get('name', 'Unknown')}")
            
            logger.info(f"크롤링 완료: {len(valid_products)}개 유효한 제품")
            return valid_products
            
        except Exception as e:
            logger.error(f"크롤링 실패: {e}")
            raise
    
    def save_products(self, products: List[Dict]) -> int:
        """제품 데이터를 데이터베이스에 저장합니다."""
        try:
            logger.info(f"데이터베이스 저장 시작: {len(products)}개 제품")
            saved_count = self.database.upsert_products(products)
            logger.info(f"데이터베이스 저장 완료: {saved_count}개 제품 처리")
            return saved_count
            
        except Exception as e:
            logger.error(f"데이터베이스 저장 실패: {e}")
            raise
    
    def cleanup_database(self):
        """데이터베이스를 정리합니다."""
        try:
            logger.info("데이터베이스 정리 시작")
            self.database.cleanup_old_data(days=30)
            logger.info("데이터베이스 정리 완료")
            
        except Exception as e:
            logger.error(f"데이터베이스 정리 실패: {e}")
    
    def _validate_product(self, product: Dict) -> bool:
        """제품 데이터의 유효성을 검증합니다."""
        required_fields = ['product_id', 'name', 'brand', 'rank', 'main_category', 'sub_category']
        
        for field in required_fields:
            if field not in product or not product[field]:
                return False
        
        # product_id가 정수인지 확인
        try:
            int(product['product_id'])
        except (ValueError, TypeError):
            return False
        
        # rank가 양수인지 확인
        try:
            rank = int(product['rank'])
            if rank <= 0:
                return False
        except (ValueError, TypeError):
            return False
        
        return True
    
    def _create_success_result(self, products: List[Dict], saved_count: int) -> Dict:
        """성공 결과를 생성합니다."""
        duration = (self.end_time - self.start_time).total_seconds()
        
        return {
            'status': 'success',
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'duration_seconds': duration,
            'crawled_count': len(products),
            'saved_count': saved_count,
            'error': None
        }
    
    def _create_error_result(self, error_message: str) -> Dict:
        """오류 결과를 생성합니다."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds() if self.start_time else 0
        
        return {
            'status': 'error',
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'crawled_count': 0,
            'saved_count': 0,
            'error': error_message
        }
    
    def get_statistics(self) -> Dict:
        """현재 데이터베이스 통계를 반환합니다."""
        try:
            return self.database.get_statistics()
        except Exception as e:
            logger.error(f"통계 조회 실패: {e}")
            return {}

def main():
    """메인 실행 함수"""
    try:
        # 파이프라인 실행
        pipeline = DataCollectionPipeline()
        result = pipeline.run()
        
        # 결과 출력
        print("\n" + "="*50)
        print("데이터 수집 결과")
        print("="*50)
        print(f"상태: {result['status']}")
        print(f"시작 시간: {result['start_time']}")
        print(f"종료 시간: {result['end_time']}")
        print(f"소요 시간: {result['duration_seconds']:.2f}초")
        print(f"크롤링된 제품: {result['crawled_count']}개")
        print(f"저장된 제품: {result['saved_count']}개")
        
        if result['error']:
            print(f"오류: {result['error']}")
        
        # 통계 정보 출력
        if result['status'] == 'success':
            stats = pipeline.get_statistics()
            print(f"\n데이터베이스 통계:")
            print(f"전체 제품 수: {stats.get('total_products', 0)}개")
            print(f"최근 업데이트: {stats.get('last_update', 'N/A')}")
        
        print("="*50)
        
        # 성공/실패에 따른 종료 코드
        sys.exit(0 if result['status'] == 'success' else 1)
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"예상치 못한 오류: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
