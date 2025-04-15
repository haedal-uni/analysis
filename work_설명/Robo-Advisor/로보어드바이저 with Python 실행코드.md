### 기본 문법
```py
def calculate_return(ohlcv_data: pd.DataFrame):
```
이 코드에서 ohlcv_data: pd.DataFrame 부분은 **"타입 힌트(Type Hint)"** 라고 불리는 문법

이 함수에 들어오는 ohlcv_data라는 변수는 pandas의 DataFrame 형식일 거야" 라고 미리 알려주는 역할

<br><br>

| 표현 | 의미 |
| --- | --- |
| `변수명: 타입` | 이 변수는 이런 타입이야 (힌트) |
| `-> 반환타입` | 이 함수는 이런 타입의 결과를 돌려줄 거야 |

```py
import pandas as pd

def calculate_return(ohlcv_data: pd.DataFrame) -> pd.Series:
    return ohlcv_data['close'].pct_change()
```
이 함수는 ohlcv_data는 pandas의 DataFrame이어야 하고

함수 결과는 pandas의 Series일 거라고 말해주는 것이다.(강제로 쓰는건 아님) 

<br><br>

