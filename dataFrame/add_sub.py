"""
기본 사용법
df.add(other, axis='columns', level=None, fill_value=None)
other : 데이터프레임이나, Series, 스칼라 등 데이터가 올 수 있습니다. 더할 값입니다.
axis : 더할 레이블을 설정합니다. 0은 행(index), 1은 열 입니다. ※Series일 경우 Index와 일치시킬 축
level : multiIndex에서 계산할 Index의 레벨입니다.
fill_value : NaN 값등의 누락 요소를 계산 전에 이 값으로 대체합니다.
"""
import numpy as np
import pandas as pd

data = [[1,10,100],[2,20,200],[3,30,300]]
col = ['col1','col2','col3']
row = ['row1','row2','row3']
df = pd.DataFrame(data=data,index=row,columns=col)
print(df)

print("{0:=^25}".format("스칼라 값 더하기"))
result = df.add(1)
print(result)

print("{0:=^25}".format("다른 DataFrame객체를 더하기"))
data2  = [[3],[4],[5]]
df2 = pd.DataFrame(data=data2,index=['row1','row2','row3'],columns=['col1'])
print(df2)

#	col2, col3 데이터가 없어 Nan으로 반환
result = df.add(df2)
print(result)
#	fill_value 지정으로 빈 값에 대한 처리 추가
result = df.add(df2, fill_value=0)
print(result)

print("{0:=^25}".format("스칼라 값 빼기"))
result = df.sub(1)
print(result)

print("{0:=^25}".format("다른 DataFrame객체를 빼기"))
#	col2, col3 데이터가 없어 Nan으로 반환
result = df.sub(df2)
print(result)
#	fill_value 지정으로 빈 값에 대한 처리 추가
result = df.sub(df2, fill_value=0)
print(result)