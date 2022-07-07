"""
df.apply(func, axis=0, raw=False, result_type=None, args=(), kwargs)

function : 각 행이나 열에 적용할 함수 입니다.
axis : {0 : Index / 1 : columns} 함수를 적용할 축 입니다.
row : {True : ndarray / False : Series} 함수에 전달할 축의 형식입니다.
True면 ndarray형태로 전달하고 False면 Series형태로 전달합니다. 기본적으로 Series입니다.
result_type : {expand / reduce / broadcast} 반환값의 형태를 결정합니다. expand이면 배열 형태를
기준으로 열을 확장합니다.(기본 인덱스로), reduce인 경우는 그대로 Serise형태로 반환합니다.
broadcase인 경우 기존 열 형식대로 확장하여 반환합니다.(열의 수가 같아야합니다.)
"""
import numpy as np
import pandas as pd

#	축 기준 (apply)
print("{0:=^25}".format("축 기준 (apply)"))

col = ['col1','col2','col3']
row = ['row1','row2','row3']
data = [[1,2,3],[4,5,6],[7,8,9]]
df = pd.DataFrame(data=data,index=row,columns=col)
print(df)

#	np.sqrt	func항목이 np.sqrt처럼 축에대해 계산할 수 없는 형식이라면 아래와 같이 각 요소에 적용됩니다.
print("{0:=^25}".format("np.sqrt"))
print(df.apply(np.sqrt))

#	np.sum처럼 축에대해 적용이 가능한경우라면 축 기준으로 연산을 수행합니다.
print("{0:=^25}".format("np.sum"))
print(df.apply(np.sum))

#	axis가 0인경우 Index(행)에 대해 연산을 수행하고, 1인경우는 columns(열)에 대해 연산을 수행합니다.
print("{0:=^25}".format("np.prod, axis=0"))
print(df.apply(np.prod,axis=0))	#	Index(행)에 대해 연산

print("{0:=^25}".format("np.prod, axis=1"))
print(df.apply(np.prod,axis=1))	#	columns(열)에 대해 연산을 수행

#	result_type에 따른 차이
#	lamba를 사용하여 기존 DataFrame에 [1,2,3]객체를 apply
print("{0:=^25}".format("apply(lambda x : [1,2,3])"))
print(df.apply(lambda x : [1,2,3]))

print("{0:=^25}".format("lambda x : [1,2,3], axis=1,result_type='expand')"))
print(df.apply(lambda x : [1,2,3], axis=1,result_type='expand'))	#	func를 기준으로 확장하여 columns를 지정

print("{0:=^25}".format("lambda x : [1,2,3], axis=1,result_type='reduce')"))
print(df.apply(lambda x : [1,2,3], axis=1,result_type='reduce'))	#	func를 기준으로 축소하여 columns없이 Series 객체로 반환하는것을 확인

print("{0:=^25}".format("lambda x : [1,2,3], axis=1,result_type='broadcast')"))
print(df.apply(lambda x : [1,2,3], axis=1,result_type='broadcast'))	#	func를 기준으로 확장하되, columns는 기존 DataFrame의 것을 사용

#	요소별 (applymap)
print("{0:=^25}".format("요소별 (applymap)"))
"""
df.apply(func, axis=0, raw=False, result_type=None, args=(), kwargs)
func : 단일 값을 반환하는 함수 입니다.
na_action : {None / 'ignore} NaN의 무시 여부입니다. 'ignore'이면 NaN을 함수로 전달하지 않습니다.
"""
col = ['col1','col2','col3']
row = ['row1','row2','row3']
data = [[1,2,3],[4,5,6],[7,pd.NA,9]]
df = pd.DataFrame(data=data,index=row,columns=col)
print(df)

print("{0:=^25}".format("applymap(lambda x : x**2,na_action='ignore')"))
print(df.applymap(lambda x : x**2,na_action='ignore'))

#	함수내 함수 연속적용 (pipe)
print("{0:=^25}".format("함수내 함수 연속적용 (pipe)"))

"""
df.pipe(func, args, kwargs)
func : 함수입니다.
arg : 함수의 인수입니다.
kwargs : dict 형태의 함수의 인수입니다.

만약 함수 3개가 아래와 같이 있다고 해봅니다.
f1(data, arg1), f2(data, arg1, arg2, f3(data, arg3)
f1 > f2 > f3 순서로 포함되게 함수를 사용한다고 하면 아래와 같이 함수를 사용해야 합니다.
df=f1( f2( f3( data,arg3='c' ),arg2='b1',arg3='b2' ),arg1='a' )
이는 어떤 arg가 어떤함수인지 직관적으로 볼 수 없습니다. 이때, pipe함수를 사용할 수 있습니다.
df=data.pipe(f3, arg3='c').pipe(f2, arg2='b1', arg3='b2').pipe(f3, arg3='c')
"""

org_data = pd.DataFrame({'info':['삼성전자/3/70000','SK하이닉스/2/100000']})
print(org_data)

def code_name(data):
    result=pd.DataFrame(columns=['name','count','price']) 
    df = pd.DataFrame(list(data['info'].str.split('/'))) # '/ ' 로 구분하여 문자열을 나누어 리스트에 넣음
    result['name'] = df[0] # 여기엔 첫번째 값인 이름이 입력
    result['count']= df[1] # 여기엔 두번째 값인 수량이 입력
    result['price']= df[2] # 여기엔 세번째 값인 가격이 입력
    result = result.astype({'count':int,'price':int}) # count와 price를 int로 바꿈(기존str)
    return result

def value_cal(data,unit=''):
    result = pd.DataFrame(columns=['name','value']) 
    result['name'] =data['name'] # 이름은 기존거를 가져옴
    result['value']=data['count']*data['price'] # value는 count * price를 입력함
    result = result.astype({'value':str}) # value를 str로 변경(단위를 붙이기 위함)
    result['value']=result['value']+unit # 단위를 붙임
    return(result)

input = code_name(org_data)
print("{0:=^25}".format("value_cal(input,'원')"))
print(value_cal(input,'원'))

print("{0:=^25}".format("pipe 메서드를 사용하지 않는경우"))
print(value_cal(code_name(org_data),'원'))

print("{0:=^25}".format("pipe 메서드를 사용"))
print(org_data.pipe(code_name).pipe(value_cal,'원'))

print("{0:=^25}".format("함수연속적용_축별 (aggregate, agg)"))
"""
df.agg(func=None, axis=0, args, kwargs)
func : 함수입니다.
axis :{0 : index(row) / 1 : columns} 축입니다 0은 행, 1은 열 입니다. arg : 함수의 인수 입니다..
kwargs : dict 형태의 함수의 인수입니다.
"""
df = pd.DataFrame([[1,4,7],[2,5,8],[3,6,9]])
print(df)

print("{0:=^25}".format("np함수 df.agg(np.prod)"))
print(df.agg(np.prod))

print("{0:=^25}".format("문자열 df.agg('prod')"))
print(df.agg('prod'))

def func_sub(input):
    return max(input)-min(input)

print("{0:=^25}".format("사용자 정의 함수 df.agg([func_sub,'sum'])"))
ex4 = df.agg([func_sub,'sum'])	#	사용자 정의함수 사용 시 기본적으로 함수명으로 열 이름 설정
print(ex4)

print("{0:=^25}".format("여러 함수 동시 적용 df.agg(['min','max','sum','prod'])"))
ex6 = df.agg(['min','max','sum','prod'])
print(ex6)

#	axis인수를 변경할 경우
print("{0:=^25}".format("axis인수를 변경할 경우 df.agg('prod', axis=0)"))
ex7 = df.agg('prod', axis=0)
print(ex7)

print("{0:=^25}".format("axis인수를 변경할 경우 df.agg('prod', axis=1)"))
ex8 = df.agg('prod', axis=1)
print(ex8)

#	함수연속적용_요소별 (transform)
print("{0:=^25}".format("함수연속적용_요소별 (transform)"))
"""
df.transform(func, axis=0, args, kwargs)
func : 함수입니다.
axis :{0 : index(row) / 1 : columns} 축입니다 0은 행, 1은 열 입니다.
arg : 함수의 인수 입니다.
kwargs : dict 형태의 함수의 인수입니다.
"""
col = ['col1','col2','col3']
row = ['row1','row2','row3']
df = pd.DataFrame(data=[[10,40,70],[20,50,80],[30,60,90]],index=row,columns=col)

print("{0:=^25}".format("df.transform(np.sqrt)"))
ex1 = df.transform(np.sqrt)
print(ex1)

print("{0:=^25}".format("df.transform('sqrt')"))
ex2 = df.transform('sqrt')
print(ex2)

print("{0:=^25}".format("여러 함수 동시 적용 df.transform(['exp','sqrt'])"))
ex4 = df.transform(['exp','sqrt'])
print(ex4)