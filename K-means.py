
#%%

# 라이브러리
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import seaborn  as sb
import matplotlib.pyplot as plt
'exec(%matplotlib inline)'

# x,y 배열(데이터베이스) 생성
df=pd.DataFrame(columns=['x','y'])

# 데이터 추가
df.loc[0]=[2,3]
df.loc[1]=[2,11]
df.loc[2]=[2,18]
df.loc[3]=[4,5]
df.loc[4]=[4,7]
df.loc[5]=[5,3]
df.loc[6]=[5,15]
df.loc[7]=[6,6]
df.loc[8]=[6,8]
df.loc[9]=[6,9]
df.loc[10]=[7,2]
df.loc[11]=[7,4]
df.loc[12]=[7,5]
df.loc[13]=[7,17]
df.loc[14]=[7,18]
df.loc[15]=[8,5]
df.loc[16]=[8,4]
df.loc[17]=[9,10]
df.loc[18]=[9,11]
df.loc[19]=[9,15]
df.loc[20]=[9,19]
df.loc[21]=[10,5]
df.loc[22]=[10,8]
df.loc[23]=[10,18]
df.loc[24]=[12,6]
df.loc[25]=[13,5]
df.loc[26]=[14,11]
df.loc[27]=[15,6]
df.loc[28]=[15,18]
df.loc[29]=[18,12]

# 데이터베이스 (30)개 출력
df.head(30)

# 이름이 K-means Example이고
# x 좌표 이름이 x 이고
# y 좌표 이름이 y 인
# 그래프 출력
sb.lmplot('x','y',data=df,fit_reg=False,scatter_kws={"s":100})
plt.title("K-means Example")
plt.xlabel('x')
plt.ylabel('y')
#%%
# 각 포인트 값을 가져옴
points=df.values
#%%
# 각 포인트를 n_clusters 개수만큼 그룹화
kmeans=KMeans(n_clusters=3).fit(points)
#%%
# 그룹화 방식
kmeans.cluster_centers_
#%%
kmeans.labels_
#%%
# 그룹 이름을 붙여서 데이터베이스 (30)개 출력
df['cluster']=kmeans.labels_
df.head(30)

# 이름이 K-means Example2인
# 그룹화 된 그래프 출력
sb.lmplot('x','y',data=df,fit_reg=False,scatter_kws={"s":100},hue="cluster")
plt.title('K-means Example2')
