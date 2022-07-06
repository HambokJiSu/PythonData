"""
pandas dataframe 객체는 기본적으로 아래와 같은 클래스로 생성이 됩니다.
class pandas.DataFrame(data=None, index=None, columns=None, copy=None)

pandas.DataFrame으로 생성된 인스턴스는 크기의 변경이 가능한 2차원 배열입니다. (Series는 1차원)
데이터 구조에는 레이블이 지정된 축인 행과 열까지 포함되며, 클래스 매서드를 통해 레이블의 수정이 가능합니다.

Parameter
data : ndarray, Iterable, dict, DataFrame
dict에는 Series, 배열 등등 list와 유사한 오브젝트가 올 수 있습니다.
데이터가 dict인 경우 열(Columns)의 순서는 삽입 순서를 따릅니다.

index : 인덱스 또는 배열형태의 객체
인스턴스에 설정되는 행 레이블입니다. 입력하지 않으면 기본 인덱스가 설정됩니다. (0, 1, 2, 3...)
columns : 인덱스 또는 배열형태의 객체
인스턴스에 설정되는 열 레이블입니다. 입력하지 않으면 기본 인덱스가 설정됩니다. (0, 1, 2, 3...)
dtype : dtype 데이터 유형을 강제하고자 할때 값입니다. 기본값은 None이며 None일경우 type이 자동으로 추론됩니다.

copy : bool
Ture 일 경우 Dataframe의 원본 데이터를 수정하더라도 인스턴스가 변경되지 않지만
False일 경우 원본데이터를 수정할 시 인스턴스의 값도 바뀌게 됩니다.
"""
import numpy as np
import pandas as pd

#	dictionary로 만들기
data = {'A' : [1,2], 'B': [3,4]}
df = pd.DataFrame(data=data)
print(df)

#	index, column 설정
data = np.array([[1,2], [3,4]])
df = pd.DataFrame(data=data, index=['row1','row2'], columns=['col1', 'col2'])
print(df)