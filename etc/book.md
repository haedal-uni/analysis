```py
import requests
url = api 인증키를 넣은 호출 url
r = requests.get(url)
data = r.json() # 서버로부터 받은 JSON 문자열을 파이썬 객체로 변환하여 반환 
print(data)
```

```python
books=[]
for d in data['response']['docs']
    books.append(d['doc'])

###############################

books=[d['doc'] for d in data['response']['docs']]
```

```py
df=pd.DataFrame(books)
df
df.to_json('books.json') # json 파일로 저장 
```

```py
import pandas as pd
df = pd.read_json('books.json')
df.head()

df_books = df[['col1', 'col2', 'col3']]
df_books.head()
```


```py
df_books.loc[[0,1], ['col1', 'col2']]
df_books.loc[0:1,'col1':'col2']
df.loc[:, 'col2':'col3']
df.loc[::2, 'col2':'col3'].head()
```

```py
import requests
isbn = 123456789
url = 'http://www.yes24.com/Product/Search?domain=BOOK&query={}'
r = requests.get(url.format(isbn))
print(r.text) # html 출력
```

```py
# html 안에 있는 내용을 찾을 때는 Beautiful Soup
from bs4 import BeautifulSoup
soup = BeautifulSoup(r.text, 'html.parser')
prd_link = soup.find('a', attrs={'class':'gd_name'}) # a 태그의 class 속성이 'gd_name'
print(prd_link['href']) # href 속성의 값을 얻을 수 있
```

```py
url = 'http://www.yes24.com'+prd_link['href']
r = requests.get(url)
print(r.text) # html 출력

soup = BeautifulSoup(r.text, 'html.parser')
prd_detail = soup.find('div', attrs={'id':'infoset_specific'})
print(prd_detail)

# find_all() : 특정 html 태그를 모두 찾아서 리스트로 반환
prd_tr_list = prd_detail.find_all('tr')
print(prd_tr_list)
```

```py
# axis 매개변수에 1을 지정하면 행에, 기본값 0을 사용하면 열에 적용한다. 
page_count = top10_books.apply(get_page_cnt2, axis=1)
print(page_count)
```


```py
x.mean()
x.median()
x.drop_duplicates().media() # 중복된 값을 가진 행 제거
x.min()
x.max()
x.quantile(0.25)
x.quantile([0.25, 0.5, 0.75])
pd.Series([1,2,3,4,5]).quantile(0.9) # 시리즈 객체를 정의한 후 분위수를 구함
x.var()
x.std()
x.mode() # 최빈값

# 기술 통계 _ 수치형 열만 연산할 수 있기 때문에 → numeric_only=True
x.mean(number_only=True) 
x.loc[:, '도서명':].mode()
```

```py
import numpy as np
import numpy as np
np.mean(df['대출건수')
np.average(df['대출건수'], weights=1/df['도서권수'])
np.median(df['대출건수'])
np.min(df['대출건수'])
np.max(df['대출건수'])
np.quantile(df['대출건수'], [0.25,0.5,0.75])
np.var(df['대출건수'])
np.std(df['대출건수'])

# np는 최빈값 계산 함수x
values, counts = np.unique(df['대출건수'], return_counts=True) # 고유한 값(values)과 등장 횟수(counts) 배열

```

```py
# csv 파일로 저장
x.to_csv('book7.csv', index=False)
```
