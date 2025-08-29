"""
화해(Hwahae) 뷰티 제품 랭킹 데이터 스크래핑 모듈
20년차 개발자의 실용적 접근: API 기반 안정적 크롤링
"""

import logging
import requests
import time
from typing import List, Dict, Optional
from urllib.parse import urljoin

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HwahaeAPICrawler:
    """화해 Gateway API 기반 크롤러 클래스"""
    
    def __init__(self):
        self.gateway_api_base = "https://gateway.hwahae.co.kr/v14/rankings"
        
        # 실제 화해 카테고리 매핑 (API ID 기반) - 스킨케어 + 클렌징
        self.categories = {
            # 클렌징 카테고리 (실제 존재하는 ID들)
            "4200": "클렌징 > 클렌징폼 > 기본",
            "4201": "클렌징 > 클렌징폼 > 수분",
            "4202": "클렌징 > 클렌징폼 > 진정",
            "4203": "클렌징 > 클렌징폼 > 보습",
            "4204": "클렌징 > 클렌징폼 > 모공",
            "4205": "클렌징 > 클렌징폼 > 브라이트닝",
            "4206": "클렌징 > 클렌징폼 > 안티에이징",
            "4207": "클렌징 > 클렌징폼 > 트러블",
            "4208": "클렌징 > 클렌징폼 > 각질",
            
            "4209": "클렌징 > 클렌징워터 > 기본",
            "4211": "클렌징 > 클렌징오일 > 기본",
            
            # 스킨케어 카테고리 (기존)
            # 스킨/토너 카테고리 (8개)
            "4157": "스킨케어 > 스킨/토너 > 수분",
            "4158": "스킨케어 > 스킨/토너 > 진정", 
            "4159": "스킨케어 > 스킨/토너 > 보습",
            "4160": "스킨케어 > 스킨/토너 > 모공",
            "4161": "스킨케어 > 스킨/토너 > 브라이트닝",
            "4162": "스킨케어 > 스킨/토너 > 안티에이징",
            "4163": "스킨케어 > 스킨/토너 > 트러블",
            "4164": "스킨케어 > 스킨/토너 > 각질",
            
            # 에센스/앰플/세럼 카테고리 (9개)
            "4174": "스킨케어 > 에센스/앰플/세럼 > 수분",
            "4175": "스킨케어 > 에센스/앰플/세럼 > 진정",
            "4176": "스킨케어 > 에센스/앰플/세럼 > 진정",
            "4177": "스킨케어 > 에센스/앰플/세럼 > 보습",
            "4178": "스킨케어 > 에센스/앰플/세럼 > 보습",
            "4179": "스킨케어 > 에센스/앰플/세럼 > 리페어",
            "4180": "스킨케어 > 에센스/앰플/세럼 > 리페어",
            "4181": "스킨케어 > 에센스/앰플/세럼 > 트러블",
            "4182": "스킨케어 > 에센스/앰플/세럼 > 브라이트닝",
            
            # 크림 카테고리 (8개)
            "4184": "스킨케어 > 크림 > 수분",
            "4185": "스킨케어 > 크림 > 수분",
            "4186": "스킨케어 > 크림 > 진정",
            "4187": "스킨케어 > 크림 > 진정",
            "4188": "스킨케어 > 크림 > 모공",
            "4189": "스킨케어 > 크림 > 보습",
            "4190": "스킨케어 > 크림 > 아이케어",
            "4191": "스킨케어 > 크림 > 트러블"
        }
        
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json",
            "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
            "Referer": "https://www.hwahae.co.kr/",
            "Origin": "https://www.hwahae.co.kr"
        }
    
    def safe_get(self, data: Dict, key: str, default='N/A'):
        """안전하게 데이터에서 값을 가져오는 함수"""
        if data is None:
            return default
        return data.get(key, default)
    
    def fetch_ranking_data(self, category_id: str, page: int = 1, page_size: int = 20) -> Optional[Dict]:
        """Gateway API에서 랭킹 데이터 가져오기"""
        url = f"{self.gateway_api_base}/{category_id}/details"
        params = {
            "page": page,
            "page_size": page_size
        }
        
        try:
            logger.info(f"📡 API 호출: {url} (페이지 {page})")
            response = requests.get(url, params=params, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                details_count = len(self.safe_get(data, 'data', {}).get('details', []))
                total_count = self.safe_get(self.safe_get(data, 'meta', {}), 'pagination', {}).get('total_count', 0)
                logger.info(f"✅ 성공: {details_count}개 제품 (총 {total_count}개)")
                return data
            else:
                logger.error(f"❌ 실패: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ 오류: {e}")
            return None
    
    def extract_products_from_api_data(self, api_data: Dict, category_id: str, page_offset: int = 0) -> List[Dict]:
        """API 응답에서 제품 정보 추출"""
        products = []
        
        try:
            # API 응답 구조: data.details 배열
            data_section = self.safe_get(api_data, 'data', {})
            details = self.safe_get(data_section, 'details', [])
            
            logger.info(f"📊 발견된 제품 수: {len(details)}")
            
            # 각 제품 정보 추출
            for i, detail in enumerate(details):
                if detail is None:
                    continue
                
                # 제품 정보 (안전하게 추출)
                product_info = self.safe_get(detail, 'product', {})
                brand_info = self.safe_get(detail, 'brand', {})
                goods_info = self.safe_get(detail, 'goods', {})
                
                # 실제 순위 계산 (페이지 오프셋 + 현재 인덱스)
                actual_rank = page_offset + i + 1
                
                # 필수 정보 확인
                brand_name = self.safe_get(brand_info, 'name', '')
                product_name = self.safe_get(product_info, 'name', '')
                product_id = self.safe_get(product_info, 'id', '')
                
                # 빈 값이 아닌 경우만 추가
                if brand_name and product_name and product_id and brand_name != 'N/A' and product_name != 'N/A':
                    # 카테고리 분류 (수정된 최종 로직)
                    category_string = self.categories.get(category_id, '기타 > 기타 > 기타')
                    category_parts = category_string.split(' > ')
                    main_category = category_parts[0]
                    middle_category = category_parts[1]
                    sub_category = category_parts[2]
                    
                    # goods_id 추출 (goods_info에서)
                    goods_id = self.safe_get(goods_info, 'id', product_id)
                    
                    # 제품명을 URL 친화적으로 변환
                    url_safe_name = product_name.replace(' ', '-').replace('/', '-').replace('[', '').replace(']', '')
                    
                    product = {
                        'product_id': int(product_id),
                        'name': product_name,
                        'brand': brand_name,
                        'image_url': self.safe_get(product_info, 'image_url', ''),
                        'product_url': f"https://www.hwahae.co.kr/goods/{url_safe_name}/{goods_id}?goods_tab=review_ingredients",
                        'rank': actual_rank,
                        'main_category': main_category,
                        'middle_category': middle_category,
                        'sub_category': sub_category,
                        'rating': self.safe_get(product_info, 'review_rating', 0),
                        'review_count': self.safe_get(product_info, 'review_count', 0),
                        'price': self.safe_get(product_info, 'price', 0),
                        'commerce_price': self.safe_get(goods_info, 'price', 0),
                        'discount_rate': self.safe_get(goods_info, 'discount_rate', 0),
                        'package_info': self.safe_get(product_info, 'package_info', ''),
                        'is_rank_new': self.safe_get(detail, 'is_rank_new', False),
                        'rank_delta': self.safe_get(detail, 'rank_delta', 0)
                    }
                    
                    products.append(product)
                    logger.info(f"✅ 제품 {actual_rank}위: {product['brand']} - {product['name']} (평점: {product['rating']})")
            
        except Exception as e:
            logger.error(f"❌ 제품 정보 추출 오류: {e}")
        
        return products
    
    def crawl_category_complete(self, category_id: str, category_name: str) -> List[Dict]:
        """한 카테고리의 전체 제품 수집"""
        logger.info(f"🎯 카테고리: {category_name} (ID: {category_id})")
        
        category_products = []
        page = 1
        page_size = 20
        total_products = 0
        
        # 첫 번째 페이지로 총 제품 수 확인
        first_page_data = self.fetch_ranking_data(category_id, page, page_size)
        if first_page_data:
            meta_section = self.safe_get(first_page_data, 'meta', {})
            pagination_section = self.safe_get(meta_section, 'pagination', {})
            total_count = self.safe_get(pagination_section, 'total_count', 0)
            
            logger.info(f"📊 총 {total_count}개 제품 발견")
            
            # 첫 번째 페이지 제품 추출
            products = self.extract_products_from_api_data(first_page_data, category_id, 0)
            if products:
                category_products.extend(products)
                total_products += len(products)
                logger.info(f"📊 페이지 {page}: {len(products)}개 제품 추가")
            
            # 나머지 페이지들 크롤링 (최대 100개까지만)
            max_products = min(total_count, 100)
            while total_products < max_products:
                page += 1
                page_offset = (page - 1) * page_size
                
                # API 호출
                api_data = self.fetch_ranking_data(category_id, page, page_size)
                
                if api_data:
                    # 제품 정보 추출
                    products = self.extract_products_from_api_data(api_data, category_id, page_offset)
                    
                    if products:
                        category_products.extend(products)
                        total_products += len(products)
                        logger.info(f"📊 페이지 {page}: {len(products)}개 제품 추가 (누적: {total_products}/{max_products})")
                        
                        # API 호출 간격
                        time.sleep(1)
                    else:
                        logger.warning(f"⚠️ 페이지 {page}: 제품 정보 없음")
                        break
                else:
                    logger.error(f"❌ 페이지 {page}: API 호출 실패")
                    break
        else:
            logger.error(f"❌ 첫 페이지 로드 실패")
        
        logger.info(f"📈 총 {len(category_products)}개 제품 수집 완료")
        return category_products
    
    def crawl_all_categories(self) -> List[Dict]:
        """모든 카테고리의 랭킹 데이터 수집"""
        all_products = []
        
        # 카테고리별 통계
        category_stats = {
            '클렌징': {'count': 0, 'products': []},
            '스킨/토너': {'count': 0, 'products': []},
            '에센스/앰플/세럼': {'count': 0, 'products': []},
            '크림': {'count': 0, 'products': []}
        }
        
        logger.info(f"🚀 화해 뷰티 랭킹 크롤링 시작")
        logger.info(f"📋 수집 대상: {len(self.categories)}개 카테고리")
        
        for category_id, category_name in self.categories.items():
            # 카테고리 분류 (클렌징과 스킨케어 구분)
            if '클렌징' in category_name:
                category_type = '클렌징'
            elif '스킨/토너' in category_name:
                category_type = '스킨/토너'
            elif '에센스/앰플/세럼' in category_name:
                category_type = '에센스/앰플/세럼'
            elif '크림' in category_name:
                category_type = '크림'
            else:
                category_type = '기타'
            
            # 카테고리 크롤링
            category_products = self.crawl_category_complete(category_id, category_name)
            
            if category_products:
                all_products.extend(category_products)
                
                # 통계 업데이트
                if category_type in category_stats:
                    category_stats[category_type]['count'] += 1
                    category_stats[category_type]['products'].extend(category_products)
            
            # 카테고리 간 대기
            time.sleep(2)
        
        # 통계 출력
        logger.info(f"📊 카테고리별 수집 통계:")
        for category_type, stats in category_stats.items():
            if stats['count'] > 0:
                logger.info(f"- {category_type}: {stats['count']}개 카테고리, {len(stats['products'])}개 제품")
        
        logger.info(f"🎉 전체 크롤링 완료: {len(all_products)}개 제품")
        return all_products

def main():
    """크롤러 테스트 실행"""
    crawler = HwahaeAPICrawler()
    products = crawler.crawl_all_categories()
    
    # 결과 출력
    print(f"\n=== 크롤링 결과 ===")
    print(f"총 {len(products)}개 제품 수집 완료")
    
    # 상위 5개 제품 출력
    for i, product in enumerate(products[:5]):
        print(f"{i+1}. {product['brand']} - {product['name']} (랭킹: {product['rank']}위, 평점: {product['rating']})")
    
    # 카테고리별 통계
    category_counts = {}
    for product in products:
        main_cat = product['main_category']
        category_counts[main_cat] = category_counts.get(main_cat, 0) + 1
    
    print(f"\n=== 카테고리별 통계 ===")
    for category, count in category_counts.items():
        print(f"{category}: {count}개 제품")

if __name__ == "__main__":
    main()
