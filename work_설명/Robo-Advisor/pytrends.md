### 기본 객체 생성
```py
from pytrends.request import TrendReq
pytrends = TrendReq(hl='en-US', tz=360) # hl=host language, tz=time zone (hl='ko', tz=540)
```

<br><br>

```py
pytrends.build_payload(kw_list, cat=0, timeframe='today 12-m', geo='', gprop='')
```
- kw_list : 분석할 키워드 리스트. 예: `['Python']`, `['Java', 'Spring']`

- cat : 카테고리 코드. 0은 전체 카테고리. 세부 카테고리는 Google Categories 참고

- timeframe : 검색 시작날짜와 종료날짜. default는 today 5-y(5년전부터 오늘까지) 

  - 예: 'today 5-y', 'today 12-m', '2022-01-01 2022-12-31'

  - today 1-m (지난 30일), now 7-d (지난 7일)

- geo : 국가 코드 (예: 'KR', 'US', 'JP')

- gprop :

  - '' : 웹 검색(구글 검색) 

  - 'images' : 이미지 검색

  - 'news' : 뉴스 검색

  - 'youtube' : 유튜브 검색

  - 'froogle' : 구글 쇼핑
 
<br><br>

```py
pytrends.interest_over_time()
```
- 설정한 키워드들의 시간별 관심도 데이터프레임을 반환

- 인덱스: 날짜, 컬럼: 키워드별 관심도 점수 (0~100)


<br><br>

### 지역별 관심도 데이터 반환
```py
pytrends.interest_by_region(resolution='COUNTRY', inc_low_vol=True, inc_geo_code=False)
```
- resolution : 'COUNTRY', 'REGION', 'CITY', 'DMA', 'METRO'

- inc_low_vol : True이면 검색량이 적은 지역도 포함

- inc_geo_code : True이면 ISO 국가 코드 포함


### 시간 단위로 관심도 가져오기 (정밀 분석용)
```py
df = pytrends.get_historical_interest(
    kw_list=['Python'],
    year_start=2022, month_start=1, day_start=1, hour_start=0,
    year_end=2022, month_end=12, day_end=10, hour_end=0,
    cat=0, geo='', gprop='', sleep=0
)
```


### 키워드 관련 인기 검색어와 급상승 검색어 반환 (딕셔너리)
```py
related = pytrends.related_queries()
```

### 오늘 + 어제 급상승 검색어 (최대 20개)
```py
pytrends.trending_searches(pn='south_korea')
```

### 실시간 트렌드 키워드 (국가 선택 가능, 한국은 불가)
```py
pytrends.realtime_trending_searches(pn='US')
```


### 추천 키워드 리스트 반환
```py
suggest = pytrends.suggestions('python') 
```
