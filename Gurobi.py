# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 22:16:27 2021

@author: 叶子
"""


from tqdm import tqdm
from scipy import optimize
import pandas as pd
import numpy as np
import math
from gurobipy import *
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False


method='max(min)' #'max(sum)'
multiple=2


# In[1] 

'''
  # 11/ 利用Guroby 最优化求解n个乘客间距离的最大值
        
        有关距离的定义：11.1 max（min）——> optimize_max_min(n,multiple,seat,distance_seats)
                       11.2 max（sum）——> optimize_max_sum(n,multiple,seat,distance_seats)
                       11.2 max（avg）待完成
        
'''


## distance（）距离定义   seat表示座位集合   distance_seats表示两两位置间距离dij
def distance(multiple,x1,y1,x2,y2):
    delta_x=abs(x1-x2)
    delta_y=abs(y1-y2)
    distance=math.sqrt((multiple*delta_x)**2+delta_y**2)
    distance=round(distance,2)
    return distance

seat,x,y=multidict({
    'x_1A':[1,1],'x_1B':[1,2],'x_1D':[1,4],'x_1E':[1,5],
    'x_2A':[2,1],'x_2B':[2,2],'x_2D':[2,4],'x_2E':[2,5],
    'x_3A':[3,1],'x_3B':[3,2],'x_3D':[3,4],'x_3E':[3,5],
    'x_4A':[4,1],'x_4B':[4,2],'x_4D':[4,4],'x_4E':[4,5],
    'x_5A':[5,1],'x_5B':[5,2],'x_5D':[5,4],'x_5E':[5,5],
    'x_6D':[6,4],'x_6E':[6,5],
    'x_7D':[7,4],'x_7E':[7,5],
    'x_8A':[8,1],'x_8B':[8,2],'x_8D':[8,4],'x_8E':[8,5],
    'x_9A':[9,1],'x_9B':[9,2],'x_9D':[9,4],'x_9E':[9,5],
    'x_10A':[10,1],'x_10B':[10,2],'x_10D':[10,4],'x_10E':[10,5],
    'x_11A':[11,1],'x_11B':[11,2],'x_11D':[11,4],'x_11E':[11,5],
    'x_6A':[12,1],'x_6B':[12,2],'x_C':[12,3],'x_7A':[12,4],'x_7B':[12,5]
        })
      
distance_seats={}
for i in seat:
    for j in seat:
       distance_seats[('{}'.format(i),'{}'.format(j))]=distance(multiple,x[i],y[i],x[j],y[j])
distance_seats=tupledict(distance_seats)    
    



# In[2]         11.1 max（min）

def optimize_max_min(n,multiple,seat,distance_seats):

    m=Model('Maxdist_select_seat')
    # create variables
    var=m.addVars(seat,vtype=GRB.BINARY,name='select')  ### var为最终要求解的变量，表示某位置上有无乘客（0-1）
    aux_mult1 = m.addVars(seat, vtype=GRB.CONTINUOUS, name='aux_mult1')  ###aux_mult1 辅助变量1，表示常数乘以决策变量的结果
    
    # create objective
    m.setObjective((aux_mult1.sum('*'))/n,GRB.MAXIMIZE)
    m.Params.timelimit = 1000
    
    # create constraints n表示选座人数
    m.addConstr((var.sum('*')==n),name='con0')
    m.addConstrs((aux_mult1.select(i)[0]<=var.select(i)[0]*(1+100*(1-var.select(j)[0]))*distance_seats.select(i,j)[0] for i in seat for j in seat if i!=j),name='con1')
    
    m.update()
    m.optimize()
    
    ### 获取需要的优化变量结果
    result_var=[v.varName[9:-1] for v in m.getVars()[:45] if v.x==1]

    return m.objVal,result_var


ObjVal=pd.DataFrame(columns=['优化人数n','优化目标最大值'])
Var=pd.DataFrame(columns=['优化人数n','优化变量结果'])
for i in tqdm(range(30,46)): #46
    objVal,var=optimize_max_min(i,multiple,seat,distance_seats)
    ObjVal=ObjVal.append({'优化人数n':i,'优化目标最大值':objVal},ignore_index=True) 
    Var=Var.append({'优化人数n':i,'优化变量结果':var},ignore_index=True) 
ObjVal.to_excel(r'D:\大四下\毕业设计\ObjVal（method=max(min)  multiple={}）.xlsx'.format(multiple),index=False)
Var.to_excel(r'D:\大四下\毕业设计\Var（method=max(min)  multiple={}）.xlsx'.format(multiple),index=False)



# In[3]       11.2 max（sum）



def optimize_max_sum(n,multiple,seat,distance_seats):
    
    # Model
    m=Model('Maxdist_select_seat')
    # create variables
    result=m.addVars(seat,vtype=GRB.BINARY,name='select')
    result2=m.addVars(seat,seat,vtype=GRB.BINARY,name='select')
    
    # create objective
    m.setObjective((result2.prod(distance_seats)/2),GRB.MAXIMIZE)
    m.Params.timelimit = 1000
    
    # create constraints n表示选座人数
    m.addConstr((result.sum('*')==n),name='con0')
    m.addConstrs((result2.select(i,j)[0]==and_(result.select(i)[0],result.select(j)[0]) for i in seat for j in seat),name='con1')
    #m.addConstrs((result2.select(i,j)[0]<=result.select(i)[0] for i in seat for j in seat),name='con1')
    #m.addConstrs((result2.select(i,j)[0]<=result.select(j)[0] for i in seat for j in seat),name='con2')
    #m.addConstrs((result2.select(i,j)[0]>=result.select(i)[0]+result.select(j)[0]-1 for i in seat for j in seat),name='con3')
    m.update()
    m.optimize()
    return m.objVal


ObjVal=pd.DataFrame(columns=['优化人数n','优化目标最大值'])
for i in tqdm(range(2,46)):
    ObjVal=ObjVal.append({'优化人数n':i,'优化目标最大值':optimize_max_sum(i,multiple,seat,distance_seats)},ignore_index=True) 
ObjVal.to_excel(r'D:\大四下\毕业设计\ObjVal（method=max(sum)  multiple={}）.xlsx'.format(multiple),index=False)




# In[4]       11.3 max（avg）



def optimize_max_avg(n,multiple,seat,distance_seats):
    
    # Model
    m=Model('Maxdist_select_seat')
    # create variables
    result=m.addVars(seat,vtype=GRB.BINARY,name='select')
    result2=m.addVars(seat,seat,vtype=GRB.BINARY,name='select')
    
    # create objective
    m.setObjective((result2.prod(distance_seats))/(n*(n-1)),GRB.MAXIMIZE)
    m.Params.timelimit = 1000
    
    # create constraints n表示选座人数
    m.addConstr((result.sum('*')==n),name='con0')
    m.addConstrs((result2.select(i,j)[0]==and_(result.select(i)[0],result.select(j)[0]) for i in seat for j in seat),name='con1')
    #m.addConstrs((result2.select(i,j)[0]<=result.select(i)[0] for i in seat for j in seat),name='con1')
    #m.addConstrs((result2.select(i,j)[0]<=result.select(j)[0] for i in seat for j in seat),name='con2')
    #m.addConstrs((result2.select(i,j)[0]>=result.select(i)[0]+result.select(j)[0]-1 for i in seat for j in seat),name='con3')
    m.update()
    m.optimize()
    
    ### 获取需要的优化变量结果
    result_var=[v.varName[9:-1] for v in m.getVars()[:45] if v.x==1]

    return m.objVal,result_var


method='max(avg)' #'max(sum)'
multiple=2


ObjVal=pd.DataFrame(columns=['优化人数n','优化目标最大值'])
Var=pd.DataFrame(columns=['优化人数n','优化变量结果'])
for i in tqdm(range(30,46)): #46
    objVal,var=optimize_max_avg(i,multiple,seat,distance_seats)
    ObjVal=ObjVal.append({'优化人数n':i,'优化目标最大值':objVal},ignore_index=True) 
    Var=Var.append({'优化人数n':i,'优化变量结果':var},ignore_index=True) 
ObjVal.to_excel(r'D:\ObjVal（method=max(avg)  multiple={}）.xlsx'.format(multiple),index=False)
Var.to_excel(r'D:\Var（method=max(avg)  multiple={}）.xlsx'.format(multiple),index=False)



# In[5]       实际选座距离计算，比值指标计算

'''

  # 12/ 求解实际情况下不同上座率下的乘客间距离
      12.1 Real_total_distance_sum 计算n名乘客间总距离
      12.2 Real_total_distance_min 计算n名乘客最小距离的均值
  
'''
month='2020-10'

data_type=pd.read_excel(r'D:\大四下\毕业设计\驿动数据\购票数据处理结果\{}购票数据.xlsx'.format(month),index=False)
data_type=data_type.drop_duplicates(['路线','上/下行','发车日','发车时间','座位号']).reset_index(drop=True,inplace=False)  ## 剔除重复行

'''
data_type_route0=data_type[data_type['路线']=='汽车城（黄渡）---虹桥快线'].reset_index(drop=True, inplace=False)
data_type_route0_time0=data_type_route0[data_type_route0['发车时间']=='17:00'].reset_index(drop=True, inplace=False)

data_type_route0_time0_day0=data_type_route0_time0[data_type_route0_time0['发车日期']==pd.to_datetime('2020-10-9 00:00:00')].reset_index(drop=True, inplace=False)
data_type_route0_time0_day0.sort_values(by='预定时间',inplace=True, ascending=True)
data_type_route0_time0_day0.reset_index(drop=True, inplace=True)
'''

def Real_total_distance_min(data,ObjVal,route,direction,day,time):   ### 统计每一趟行程中上座人数为n时的乘客最小距离的均值，及与最优距离的比值
   n=len(data)
   RealVal=pd.DataFrame(columns=['路线','发车日期','发车时间','上座人数n','姓名','乘客间最小距离的均值','与最优距离比值','归一化比值'])
   for i in range(1,n):
       data0=data.loc[0:i]
       min_dist_sum=0
       for j in range(i+1):
           dist_j=[]         ### 对于每一名乘客求相对它的最小距离
           for k in range(i+1):
               if j!=k:
                 dist_j.append(distance_seats[('x_{}'.format(data0['座位号'][j]), 'x_{}'.format(data0['座位号'][k]))])
           min_dist_j=min(dist_j)
           min_dist_sum=min_dist_sum+min_dist_j
       min_dist_avg=round(min_dist_sum/(i+1),2)
       percent=(min_dist_avg)/ObjVal['优化目标最大值'][ObjVal[ObjVal['优化人数n']==i+1].index.tolist()[0]]
       normalized_percent=1 if (ObjVal['优化目标最大值'][ObjVal[ObjVal['优化人数n']==i+1].index.tolist()[0]]-1)==0 else (min_dist_avg-1)/(ObjVal['优化目标最大值'][ObjVal[ObjVal['优化人数n']==i+1].index.tolist()[0]]-1)
       RealVal=RealVal.append({'路线':route,'发车方向':direction,'发车日期':day,'发车时间':time,'上座人数n':i+1,'姓名':data0['姓名'][i],'乘客间最小距离的均值':min_dist_avg,'与最优距离比值':percent,'归一化比值':normalized_percent},ignore_index=True) 
   return RealVal

#ObjVal=pd.read_excel(r'C:\Users\叶子\Desktop\毕业设计\ObjVal（method={}  multiple={}）.xlsx'.format(method,multiple))
#RealVal=Real_total_distance_min(data_type_route0_time0_day0,ObjVal,'汽车城（黄渡）---虹桥快线','上行',pd.to_datetime('2019-12-11 00:00:00'),'17:00')



def Real_total_distance_sum(data,ObjVal,route,direction,day,time):   ### 统计每一趟行程中上座人数为n时的乘客间总距离，及与最优距离的比值
   n=len(data)
   RealVal=pd.DataFrame(columns=['路线','发车日期','发车时间','上座人数n','乘客间总距离','与最优距离比值'])
   for i in range(1,n):
       data0=data.loc[0:i]
       dist=0
       for j in range(i+1):
           for k in range(i+1):
               dist=dist+distance_seats[('x_{}'.format(data0['座位号'][j]), 'x_{}'.format(data0['座位号'][k]))]
       percent=(dist/2)/ObjVal['优化目标最大值'][ObjVal[ObjVal['优化人数n']==i+1].index.tolist()[0]]
       RealVal=RealVal.append({'路线':route,'发车方向':direction,'发车日期':day,'发车时间':time,'上座人数n':i+1,'乘客间总距离':dist/2,'与最优距离比值':percent},ignore_index=True) 
   return RealVal

#ObjVal=pd.read_excel(r'C:\Users\叶子\Desktop\毕业设计\ObjVal（method={}  multiple={}）.xlsx'.format(method,multiple))
#RealVal=Real_total_distance_sum(data_type_route0_time0_day0,ObjVal,'汽车城（黄渡）---虹桥快线','上行',pd.to_datetime('2019-12-11 00:00:00'),'17:00')



def Real_total_distance_avg(data,ObjVal,route,direction,day,time):   ### 统计每一趟行程中上座人数为n时的乘客间总距离，及与最优距离的比值
   n=len(data)
   RealVal=pd.DataFrame(columns=['路线','发车日期','发车时间','上座人数n','乘客间总距离','与最优距离比值'])
   for i in range(1,n):
       data0=data.loc[0:i]
       dist=0
       for j in range(i+1):
           for k in range(i+1):
               dist=dist+distance_seats[('x_{}'.format(data0['座位号'][j]), 'x_{}'.format(data0['座位号'][k]))]
       percent=(dist/((i+1)*i))/ObjVal['优化目标最大值'][ObjVal[ObjVal['优化人数n']==i+1].index.tolist()[0]]
       RealVal=RealVal.append({'路线':route,'发车方向':direction,'发车日期':day,'发车时间':time,'上座人数n':i+1,'乘客间总距离':dist/2,'与最优距离比值':percent},ignore_index=True) 
   return RealVal

#ObjVal=pd.read_excel(r'C:\Users\叶子\Desktop\毕业设计\ObjVal（method={}  multiple={}）.xlsx'.format(method,multiple))
#RealVal=Real_total_distance_sum(data_type_route0_time0_day0,ObjVal,'汽车城（黄渡）---虹桥快线','上行',pd.to_datetime('2019-12-11 00:00:00'),'17:00')



def target_analysis(data,method,ObjVal):
    result=pd.DataFrame(columns=['路线','发车日期','发车时间','上座人数n','乘客间总距离','与最优距离比值','归一化比值'])
    routes=list(set(data['路线']))
    for i in tqdm(range(len(routes))):
        data1=data[data['路线']==routes[i]].reset_index(drop=True,inplace=False)
        directions=list(set(data1['上/下行']))
        for j in range(len(directions)):
            data2=data1[data1['上/下行']==directions[j]].reset_index(drop=True,inplace=False)
            days=list(set(data2['发车日期']))
            for k in range(len(days)):
                data3=data2[data2['发车日期']==days[k]].reset_index(drop=True,inplace=False)
                times=list(set(data3['发车时间']))
                for l in range(len(times)):
                    data4=data3[data3['发车时间']==times[l]].reset_index(drop=True,inplace=False)
                    data4.sort_values(by='预定时间',inplace=True, ascending=True)
                    data4.reset_index(drop=True, inplace=True)
                    if method=='max(sum)':
                       RealVal=Real_total_distance_sum(data4,ObjVal,routes[i],directions[j],days[k],times[l])
                    elif method=='max(min)':
                       RealVal=Real_total_distance_min(data4,ObjVal,routes[i],directions[j],days[k],times[l])
                    elif method=='max(avg)':
                       RealVal=Real_total_distance_avg(data4,ObjVal,routes[i],directions[j],days[k],times[l])
                    result=result.append(RealVal,ignore_index=True)   
    
    return result



ObjVal=pd.read_excel(r'D:\大四下\毕业设计\驿动数据\优化求解结果\优化目标max值\ObjVal（method={}  multiple={}）.xlsx'.format(method,multiple))

months=['2019-1','2019-2','2019-3','2019-4','2019-5','2019-6','2019-7','2019-8','2019-9','2019-10','2019-11','2019-12',\
        '2020-1','2020-2','2020-3','2020-4','2020-5','2020-6','2020-7','2020-8','2020-9','2020-10','2020-11','2020-12']

for i in range(len(months)):
    month=months[i]
    data_type=pd.read_excel(r'D:\大四下\毕业设计\驿动数据\购票数据处理结果\{}购票数据.xlsx'.format(month))
    data_type=data_type.drop_duplicates(['路线','上/下行','发车日','发车时间','座位号']).reset_index(drop=True,inplace=False)  ## 剔除重复行

    result=target_analysis(data_type,method,ObjVal)    
    result.to_excel(r'D:\大四下\毕业设计\驿动数据\指标计算结果\{}指标计算结果(method={} multiple={}).xlsx'.format(month,method,multiple),index=False)




#### 重新计算归一化比值


