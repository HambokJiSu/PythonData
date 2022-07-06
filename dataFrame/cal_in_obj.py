#	객체 내 연산
import numpy as np
import pandas as pd

#	반올림
print("{0:=^25}".format("반올림"))
col = ['col1','col2','col3']
row = ['row1','row2','row3']
data = np.random.rand(3,3)*100
df = pd.DataFrame(data=data, index=row, columns=col)
print(df)

print("{0:=^25}".format("반올림 0"))
print(df.round(0))

print("{0:=^25}".format("반올림 1"))
print(df.round(1))

print("{0:=^25}".format("반올림 -1"))
print(df.round(-1))

#	합계
print("{0:=^25}".format("합계"))
col = ['col1','col2','col3']
row = ['row1','row2','row3']
data = [[1,2,3],[4,5,6],[7,np.NaN,9]]
df = pd.DataFrame(data=data,index=row,columns=col)
print(df)

print("{0:=^25}".format("합계 axis 0"))
print(df.sum(axis=0))

#	skipna : NaN이 포함된 경우를 무시할지 여부
print("{0:=^25}".format("합계 axis 0, skipna=False"))
print(df.sum(axis=0, skipna=False))

print("{0:=^25}".format("합계 axis 1"))
print(df.sum(axis=1))

#	min_count : 계산에 필요한 숫자의 최소 갯수
print("{0:=^25}".format("합계 axis 1, min_count=3"))
print(df.sum(axis=1, min_count=3))

#	곱
print("{0:=^25}".format("합계"))
col = ['col1','col2','col3']
row = ['row1','row2','row3']
data = [[1,2,3],[4,5,6],[7,np.NaN,9]]
df = pd.DataFrame(data=data,index=row,columns=col)
print(df)

print("{0:=^25}".format("prod axis=0"))
print(df.prod(axis=0))

print("{0:=^25}".format("prod axis=1"))
print(df.prod(axis=1))

#	절대값
print("{0:=^25}".format("절대값"))
col = ['col1','col2','col3']
row = ['row1','row2','row3']
data = [[-1,2,-3],[-4,-5,6],[7,np.NaN,-9]]
df = pd.DataFrame(data=data,index=row,columns=col)
print(df)
print(df.abs())

#	전치 : 행과 열을 바꾸기 ex) 2:3 => 3:2
print("{0:=^25}".format("전치"))
print(df.transpose())

#	순위 : rank
#	DataFrame.rank(axis=0, method='average', numeric_only=None, na_option='keep', ascending=True, pct=False)
"""
axis : {0 : index / 1 : columns} 순위를 매길 레이블입니다.
method : {'average' / 'min' / 'max' / 'first' / 'dense'} 동순위 일때 처리 방법입니다.
average는 평균, min은 낮은순위, max는 높은순위, first는 나타나는순서대로
dense의 경우는 min과 같지만 그룹간 순위는 항상 1씩 증가합니다.
numeric_only : {True / False} 숫자만 순위를 매길지 여부 입니다.
na_option : {'keep' / 'top' / 'bottom'} NaN값의 처리 방법입니다.
keep의 경우 NaN순위 할당, top의 경우 낮은순위 할당, bottom의 경우 높은 순위를 할당합니다.
ascending : {True / False} 오름차순으로 할지의 여부 입니다.
pct : {True / False} 순위를 백분위수형식으로 할지 여부입니다.
"""
print("{0:=^25}".format("순위"))
data = [[5],[5],[pd.NA],[3],[-3.1],[5],[0.4],[6.7],[3]]
row = ['A★','B★','C','D☆','E','F★','G','H','I☆']
df = pd.DataFrame(data=data, index=row, columns=['Value'])
print(df)

"""
average : D와 I의 경우 각각 3등 4등이기때문에 3.5 출력
min : A, B, F의 경우 각각 5등 6등 7등으로 가장 낮은등수인 5 출력
max : A, B, F의 경우 각각 5등 6등 7등으로 가장 높등수인 7 출력
first : 동점일경우 위에서부터 매김 D와 I 각각 3등 4등
dense : min처럼 동작하지만 등수가 순차적으로 증가
"""

df['average']=df['Value'].rank(method='average')
df['min']=df['Value'].rank(method='min')
df['max']=df['Value'].rank(method='max')
df['first']=df['Value'].rank(method='first')
df['dense']=df['Value'].rank(method='dense')
print(df)

print("{0:=^25}".format("차이 diff"))
"""
df.diff(periods=1, axis=0)
axis : 비교할 축을 지정합니다. axis=0 인 경우 행끼리 비교하고 axis=1인 경우 열 끼리 비교합니다.
periods : 비교할 간격을 지정합니다. 기본은 +1로 바로 이전 값과 비교합니다.
"""
a = [1,2,3,4,5,6,7,8]
b = [1,2,4,8,16,32,64,128]
c = [8,7,6,5,4,3,2,1]
data = {"col1":a,"col2":b,"col3":c}
df = pd.DataFrame(data)
print(df)

print("{0:=^25}".format("diff axis=0"))
#	행 - 바로전 행 의 값을 출력
print(df.diff(axis=0))

print("{0:=^25}".format("diff axis=1"))
#	열 - 바로전 열의 값을 출력
print(df.diff(axis=1))
"""
periods의 사용
periods의 경우 기본값은 +1로 +1인 경우 바로 이전 값과의 차를 출력합니다.
+3인 경우 3칸 이전 값과 비교하고 -2인 경우 2칸 후의 값과 비교하게 됩니다.
"""
print("{0:=^25}".format("periods 3"))
print(df.diff(periods=3))

print("{0:=^25}".format("periods -2"))
print(df.diff(periods=-2))

print("{0:=^25}".format("차이 pct_change"))
"""
pct_change는 한 객체 내에서 행과 행의 차이를 현재값과의 백분율로 출력하는 메서드 입니다. 
즉, (다음행 - 현재행)÷현재행 을 의미합니다.

기본 사용법
※ 자세한 내용을 아래 예시를 참고 바랍니다.
df.pct_change(periods=1, fill_method='pad', limit=None, freq=None, kwargs)
periods : 비교할 간격을 지정합니다. 기본은 +1로 바로 이전 값과 비교합니다.
fill_method : {ffill : 앞의 값으로 채움 / bfill : 뒤의 값으로 채움} 결측치를 대체할 값입니다.
limit : 결측값을 몇개나 대체할지 정할 수 있습니다.
freq : 시계열 API에서 사용할 증분을 지정합니다. (예: 'M' 또는 BDay( ))
"""
a = [1,1,4,4,1,1]
b = [1,2,4,8,16,32]
c = [1,np.NaN,np.NaN,None,16,64]
data = {"col1":a,"col2":b,"col3":c}
df = pd.DataFrame(data)
print(df)
print("{0:=^25}".format("pct_change"))
print(df.pct_change())
#	periods 인수 사용
#	periods인수는 계산할 간격을 나타냅니다. 기본은 1로 +1을 의미하며 마이너스일 경우 반대방향으로 계산합니다.
print("{0:=^25}".format("periods 2"))
print(df.pct_change(periods=2))

print("{0:=^25}".format("periods -1"))
print(df.pct_change(periods=-1))

#	fill_method='ffill'인 경우는 기본값으로 바로 윗값으로 결측치를 대체합니다.
#	fill_method='bfill'인 경우는 바로 아랫값으로 결측치를 대체합니다.
print("{0:=^25}".format("fill_method='bfill'"))
print(df.pct_change(fill_method='bfill'))

print("{0:=^25}".format("fill_method='ffill'"))
print(df.pct_change(fill_method='ffill'))

print("{0:=^25}".format("limit=2"))
print(df.pct_change(limit=2))

#	누적 계산 (expending)
import numba	#	pip install numba
print("{0:=^25}".format("누적 계산 (expending)"))
data = {'col1':[1,2,3,4],'col2':[3,7,5,6]}
idx = ['row1','row2','row3','row4']
df = pd.DataFrame(data = data, index = idx)
print(df)

print("{0:=^25}".format("sum"))
print(df.expanding().sum())

print("{0:=^25}".format("sum min_periods=4"))
print(df.expanding(min_periods=4).sum())

print("{0:=^25}".format("sum min_periods=4"))
print(df.expanding(axis=1).sum())

print("{0:=^25}".format("sum method='table' engine='numba'"))
print(df.expanding(method='table').sum(engine='numba'))

#	기간이동 계산 (rolling)
print("{0:=^25}".format("기간이동 계산 (rolling)"))

period = pd.period_range(start='2022-01-13 00:00:00',end='2022-01-13 02:30:00',freq='30T')
data = {'col1':[1,2,3,4,5,6],'col2':period}
idx = ['row1','row2','row3','row4','row5','row6']
df = pd.DataFrame(data= data, index = idx)
print(df)

#	window 크기를 지정해주면, 현재 행 이전으로 window 크기 만큼의 계산을 수행합니다.
print("{0:=^25}".format("df.rolling(window=3).sum()"))
print(df.rolling(window=3).sum()) # 뒤에 추가 메서드를 이용하여 연산을 지정해주어야합니다.

print("{0:=^25}".format("df.rolling(window=3, closed='left').sum()"))
print(df.rolling(window=3, closed='left').sum())

print("{0:=^25}".format("df.rolling(window=3, closed='right').sum()"))
print(df.rolling(window=3, closed='right').sum())

#	그룹화 계산 (groupby)
print("{0:=^25}".format("그룹화 계산 (groupby)"))
"""
df.groupby(by=None, axis=0, level=None, as_index=True, sort=True, group_keys=True, squeeze=NoDefault.no_default, observed=False, dropna=True)
by : 그룹화할 내용입니다. 함수, 축, 리스트 등등이 올 수 있습니다.
axis : 그룹화를 적용할 축입니다.
level : 멀티 인덱스의 경우 레벨을 지정할 수 있습니다.
as_index : 그룹화할 내용을 인덱스로 할지 여부입니다. False이면 기존 인덱스가 유지됩니다.
sort : 그룹키를 정렬할지 여부입니다.
group_keys : apply메서드 사용시 결과에따라 그룹화 대상인 열이 인덱스와 중복(group key)이 될 수 있습니다. 이 때, group_keys=False로 인덱스를 기본값으로 지정할 수 있습니다.
squeeze : 결과가 1행 or 1열짜리 데이터일 경우 Series로, 1행&1열 짜리 데이터일 경우 스칼라로 출력합니다.
observed : Categorical로 그룹화 할 경우 Categorical 그룹퍼에 의해 관찰된 값만 표시할 지 여부입니다.
dropna : 결측값을 계산에서 제외할지 여부입니다.
"""
idx=['A','A','B','B','B','C','C','C','D','D','D','D','E','E','E']
col=['col1','col2','col3']
data = np.random.randint(0,9,(15,3))
df = pd.DataFrame(data=data, index=idx, columns=col).reset_index()
print(df)

#	추가 메서드 없이 groupby 메서드를 실행하면 DataFrameGroupBy 오브젝트가 생성이 됩니다.
print("{0:=^25}".format("df.groupby('index').mean()"))
print(df.groupby('index').mean()) # index 컬럼에 대해서 groupby 수행
print("{0:=^25}".format("df.groupby('index').count()"))
print(df.groupby('index').count())

#	group_keys
def top (df,n=2,col='col1'):
    return df.sort_values(by=col)[-n:] #상위 n개 열을 반환하는 함수 top 생성
print("{0:=^25}".format("df.groupby('index').apply(top)"))
print(df.groupby('index').apply(top))

print("{0:=^25}".format("df.groupby('index',group_keys=False).apply(top)"))
print(df.groupby('index',group_keys=False).apply(top))

#	observed 인수의 사용
#	Categorical 객체를 생성할 때, 그룹화(groupby)할 열에 있는 값이 아닌 값을 포함하게되면, 그룹화 할 때 해당 값을 표시할지 여부를 선택할 수 있습니다.
print("{0:=^25}".format("pd.Categorical"))
df_cat = pd.Categorical(df['index'], categories=['A','B','C','D','E','F']) # df의 index열에 대해서 A,B,C,D,E,F 로 Categorical을 하여 df_cat 생성
print(df_cat)

#	위 catrory 객체에 대해서 col1열을 groupby 하면 아래와 같이 카테고리에만 존재하는 F에대한 groupby 값이 출력됩니다.
print("{0:=^25}".format("groupby count"))
print(df['col1'].groupby(df_cat).count())

#	observed=True로 할경우 관찰되지 않는값 (카테고리에만 존재하는값)은 표시되지 않습니다.
print("{0:=^25}".format("as_index"))
print(df.groupby(['index'],as_index=False).sum())

#	dropna인수의 사용
#	dropna인수를 통해 결측값(NaN)이 포함된 경우 그룹화에서 제외할지 여부를 정할 수 있습니다.
#	먼저 index열의 6번행을 결측값(NaN)으로 변경해보겠습니다.
print("{0:=^25}".format("dropna"))
df.loc[6,'index'] = np.NaN
print(df)

#	일반적인 사용(dropna=True)시 NaN은 계산에서 제외되어 인덱스에 표시되지 않은것을 확인할 수 있습니다.
print(df.groupby('index').sum())

#	dropna=False인 경우 인덱스에 NaN이 포함되어 계산된 것을 알 수 있습니다.
print(df.groupby('index',dropna=False).sum())

print("{0:=^25}".format("level인수의 사용 (Multi Index)"))
idx = [['idx1','idx1','idx2','idx2','idx2'],['row1','row2','row1','row2','row3']]
col = ['col1','col2','col2']
data = np.random.randint(0,9,(5,3))
df = pd.DataFrame(data=data, index = idx, columns = col).rename_axis(index=['lv0','lv1'])
print(df)

#	level을 int로 지정해주는 경우
print("{0:=^25}".format("level을 int"))
print(df.groupby(level=1).sum())

#	level을 str로 지정해주는경우 + (여러개 지정시 순차적으로 groupby 됩니다.)
print("{0:=^25}".format("level을 str"))
print(df.groupby(['lv1','lv0']).sum())