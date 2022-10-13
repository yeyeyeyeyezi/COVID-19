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


method='max(min)' #'max(avg)'
multiple=2


# In[1] 

'''
  # Guroby——the optimal allocation of the n passengers in the vehicle
        
        Different distance definitions：  
                       1 max（min）——> optimize_max_min(n,multiple,seat,distance_seats)
                       2 max（avg）——> optimize_max_sum(n,multiple,seat,distance_seats)
        
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
    



# In[2]         1. max（min）

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


ObjVal=pd.DataFrame(columns=['n','objVal'])
Var=pd.DataFrame(columns=['n','result_var'])
for i in tqdm(range(30,46)): #46
    objVal,var=optimize_max_min(i,multiple,seat,distance_seats)
    ObjVal=ObjVal.append({'n':i,'objVal':objVal},ignore_index=True) 
    Var=Var.append({'n':i,'result_var':var},ignore_index=True) 
ObjVal.to_excel(r'..\ObjVal（method=max(min)  multiple={}）.xlsx'.format(multiple),index=False)
Var.to_excel(r'..\Var（method=max(min)  multiple={}）.xlsx'.format(multiple),index=False)




# In[3]       2. max（avg）



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
    m.update()
    m.optimize()
    
    ### 获取需要的优化变量结果
    result_var=[v.varName[9:-1] for v in m.getVars()[:45] if v.x==1]

    return m.objVal,result_var


method='max(avg)' #'max(sum)'
multiple=2


ObjVal=pd.DataFrame(columns=['n','objVal'])
Var=pd.DataFrame(columns=['n','result_var'])
for i in tqdm(range(30,46)): #46
    objVal,var=optimize_max_avg(i,multiple,seat,distance_seats)
    ObjVal=ObjVal.append({'n':i,'objVal':objVal},ignore_index=True) 
    Var=Var.append({'n':i,'result_var':var},ignore_index=True) 
ObjVal.to_excel(r'..\ObjVal（method=max(avg)  multiple={}）.xlsx'.format(multiple),index=False)
Var.to_excel(r'..\Var（method=max(avg)  multiple={}）.xlsx'.format(multiple),index=False)





# In[5]       Calculation of actual seat selection distance, and the  distancing indicators In

'''

  # 12/ 求解实际情况下不同上座率下的乘客间距离
      12.1 Real_total_distance_sum 计算n名乘客间总距离
      12.2 Real_total_distance_min 计算n名乘客最小距离的均值
  
'''
month='2020-10'

data_type=pd.read_excel(r'..\{}_processsing.xlsx'.format(month),index=False)
data_type=data_type.drop_duplicates(['Line_name','Up/Down','Day','Time','Seat_number']).reset_index(drop=True,inplace=False)  ## 剔除重复行



def Real_total_distance_min(data,ObjVal,route,direction,day,time):   ### 统计每一趟行程中上座人数为n时的乘客最小距离的均值，及与最优距离的比值
   n=len(data)
   RealVal=pd.DataFrame(columns=['Line_name','Up/Down','Date','Time','Number_of_passengers','Distance','Indicators ratio','Normalized indicators ratio'])
   for i in range(1,n):
       data0=data.loc[0:i]
       min_dist_sum=0
       for j in range(i+1):
           dist_j=[]         ### 对于每一名乘客求相对它的最小距离
           for k in range(i+1):
               if j!=k:
                 dist_j.append(distance_seats[('x_{}'.format(data0['Seat_number'][j]), 'x_{}'.format(data0['Seat_number'][k]))])
           min_dist_j=min(dist_j)
           min_dist_sum=min_dist_sum+min_dist_j
       min_dist_avg=round(min_dist_sum/(i+1),2)
       percent=(min_dist_avg)/ObjVal['objVal'][ObjVal[ObjVal['n']==i+1].index.tolist()[0]]
       normalized_percent=1 if (ObjVal['objVal'][ObjVal[ObjVal['n']==i+1].index.tolist()[0]]-1)==0 else (min_dist_avg-1)/(ObjVal['优化目标最大值'][ObjVal[ObjVal['优化人数n']==i+1].index.tolist()[0]]-1)
       RealVal=RealVal.append({'Line_name':route,'Up/Down':direction,'Date':day,'Time':time,'Number_of_passengers':i+1,'Name':data0['姓名'][i],'Distance':min_dist_avg,'Indicators ratio':percent,'Normalized indicators ratio':normalized_percent},ignore_index=True) 
   return RealVal






def target_analysis(data,method,ObjVal):
    result=pd.DataFrame(columns=['Line_name','Up/Down','Date','Time','Number_of_passengers','Distance','Indicators ratio','Normalized indicators ratio'])
    routes=list(set(data['Line_name']))
    for i in tqdm(range(len(routes))):
        data1=data[data['Line_name']==routes[i]].reset_index(drop=True,inplace=False)
        directions=list(set(data1['Up/Down']))
        for j in range(len(directions)):
            data2=data1[data1['Up/Down']==directions[j]].reset_index(drop=True,inplace=False)
            days=list(set(data2['Date']))
            for k in range(len(days)):
                data3=data2[data2['Date']==days[k]].reset_index(drop=True,inplace=False)
                times=list(set(data3['Time']))
                for l in range(len(times)):
                    data4=data3[data3['Time']==times[l]].reset_index(drop=True,inplace=False)
                    data4.sort_values(by='Booking_time',inplace=True, ascending=True)
                    data4.reset_index(drop=True, inplace=True)
                    RealVal=Real_total_distance_min(data4,ObjVal,routes[i],directions[j],days[k],times[l])
                    result=result.append(RealVal,ignore_index=True)   
    
    return result



ObjVal=pd.read_excel(r'..\ObjVal（method={}  multiple={}）.xlsx'.format(method,multiple))

months=['2019-1','2019-2','2019-3','2019-4','2019-5','2019-6','2019-7','2019-8','2019-9','2019-10','2019-11','2019-12',\
        '2020-1','2020-2','2020-3','2020-4','2020-5','2020-6','2020-7','2020-8','2020-9','2020-10','2020-11','2020-12']

for i in range(len(months)):
    month=months[i]
    data_type=pd.read_excel(r'..\{}_processsing.xlsx'.format(month))
    data_type=data_type.drop_duplicates(['Line_name','Up/Down','Day','Time','Seat_number']).reset_index(drop=True,inplace=False)  ## 剔除重复行

    result=target_analysis(data_type,method,ObjVal)    
    result.to_excel(r'..\{} Distancing indicators results (method={} multiple={}).xlsx'.format(month,method,multiple),index=False)





# In[5]       Statistics of distancing indicator results monthly 

months=['2019-1','2019-2','2019-3','2019-4','2019-5','2019-6','2019-7','2019-8','2019-9','2019-10','2019-11','2019-12',\
        '2020-1','2020-2','2020-3','2020-4','2020-5','2020-6','2020-7','2020-8','2020-9','2020-10','2020-11','2020-12']

def final_results_statistics(method,multiple):
    
    ## final_result  ——> distancing indicator results monthly 
    final_result={}
    for i in tqdm(range(24)):
        result=pd.read_excel(r'..\{} Distancing indicators results (method={} multiple={}).xlsx'.format(month,method,multiple))
        result_target_analysis=pd.DataFrame(columns=['n','Distance','Indicators ratio','Normalized indicators ratio'])
        n_list=list(set(result['n']))
        for j in range(len(n_list)):
            m=list(result[result['n']==n_list[j]]['Distance'])
            m2=list(result[result['n']==n_list[j]]['Indicators ratio'])
            m3=list(result[result['n']==n_list[j]]['Normalized indicators ratio'])
            result_target_analysis=result_target_analysis.append({'n':n_list[j],'Distance':np.mean(m3),'Indicators ratio':np.mean(m),'Normalized indicators ratio':np.mean(m2)},ignore_index=True) 
        final_result[months[i]]=result_target_analysis
    
    ## final_mean_result  ——> mean distancing indicator results  in stage Pre-COVID-19
    df=pd.DataFrame()
    final_mean_result={}
    for i in range(3,12):
        df1=pd.read_excel(r'..\{} Distancing indicators results (method={} multiple={}).xlsx'.format(months[i],method,multiple))
        df=df.append(df1,ignore_index=True)
    result_target_analysis=pd.DataFrame(columns=['n','Distance','Indicators ratio','Normalized indicators ratio'])
    n_list=list(set(df['n']))
    for j in range(len(n_list)):
        m=list(df[df['n']==n_list[j]]['Distance'])
        m2=list(df[df['n']==n_list[j]]['Indicators ratio'])
        m3=list(df[df['n']==n_list[j]]['Normalized indicators ratio'])
        result_target_analysis=result_target_analysis.append({'n':j+2,'Distance':np.mean(m3),'Indicators ratio':np.mean(m),'Normalized indicators ratio':np.mean(m2)},ignore_index=True) 
    final_mean_result=result_target_analysis
    
    return final_result,final_mean_result

multiple1,multiple1_5,multiple2=1,1.5,2
final_result_multiple1,final_mean_result_multiple1=final_results_statistics(method,multiple1)[0],final_results_statistics(method,multiple1)[1]
final_result_multiple1_5,final_mean_result_multiple1_5=final_results_statistics(method,multiple1_5)[0],final_results_statistics(method,multiple1_5)[1]
final_result_multiple2,final_mean_result_multiple2=final_results_statistics(method,multiple2)[0],final_results_statistics(method,multiple2)[1]





### draw 'Normalized indicators ratio' results
fig=plt.figure(figsize=(14,8.7))
ax=fig.add_subplot(1,1,1)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

#plt.title('Curve of each month after the epidemic & Curve of the mean value before the epidemic',size=30,pad=55)
for i in range(12,24):
    ax=fig.add_subplot(3,4,i-11)
    max_count_n=len(final_result_multiple2[months[i]])-5 if len(final_result_multiple2[months[i]])==44 else len(final_result_multiple2[months[i]])
    x1,x2=final_result_multiple2[months[i]]['n'][0:max_count_n],final_mean_result_multiple2['n'][0:max_count_n]
    y1,y2=final_result_multiple2[months[i]]['Normalized indicators ratio'][0:max_count_n],final_mean_result_multiple2['Normalized indicators ratio'][0:max_count_n]
    plt.plot(x1,y1,linestyle='--',linewidth=2.4)   
    plt.plot(x2,y2,linestyle='-',linewidth=2.8)   
    plt.title('{}'.format(figure_months[i]),size=18,pad=4)
    plt.legend(['Value in {}'.format(figure_months2[i])]+['Mean value in 2019'],loc='center left', bbox_to_anchor=(0.29, 0.2),ncol=1,fontsize=11.5)
    plt.ylabel('Distancing indicator',size=14,labelpad=3)
    plt.xlabel('Number of passengers',size=14,labelpad=3)
    plt.yticks(fontsize=13)
    plt.xticks(fontsize=13)
    plt.grid(True,linewidth=0.4,linestyle='--')
    plt.ylim(0,0.9)
    plt.xlim(0,42)
    plt.tight_layout()
    ax.fill_between(x1,y1,y2[:len(x1)],color='y',interpolate=True,alpha=0.75)  # 在y1和y2封闭区间内填充


