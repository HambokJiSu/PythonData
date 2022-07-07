#	레이블기반_스칼라 (at)
"""
값 가져오기 : result = df.at['행', '열']
값 설정하기 : df.at['행', '열'] = value
"""
import numpy as np
import pandas as pd

df = pd.DataFrame([[1,2], [3,4]], index=['row1', 'row2'], columns=['col1', 'col2'])
print(df)

print("{0:=^25}".format("값 가져오기"))
result = df.at['row1', 'col2']
print(result)

print("{0:=^25}".format("값 설정하기"))
df.at['row2', 'col1'] = '변경'
print(df)

print("{0:=^40}".format("loc 메서드로 Series 추출, 스칼라 값 얻는 방식"))
df2 = df.loc['row2'].at['col2']
print(df2)