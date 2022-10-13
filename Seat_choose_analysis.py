#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 08:51:51 2022

@author: yezi
"""


from collections import Counter
from tqdm import tqdm
import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False


month='2020-11'
vehicle_type='C364D5CFCE27A115'



'''

  # 0/  准备所需的数据 
      data: 对应month的交易数据
      data_vehicle：驿动车型数据
      vehicle_list：目标研究车型对应的车牌号信息
      number_xy:目标车型对应的座位编号表
'''


data=pd.read_excel(r'..\{}.xlsx'.format(month))
data_vehicle=pd.read_excel(r'..\Bus model.xlsx')
vehicle_list=data_vehicle[data_vehicle['Bus_model']=='{}'.format(vehicle_type)]['Bus_id'].tolist()

# Seat distribution corresponding to the research Bus model
number_xy=pd.DataFrame(columns=['Seat_number','x','y'])
for i in range(1,12):
    for j in ['A','B','D','E']:
        if '{}{}'.format(i,j) in ['7B','7A','C','6B','6A']:
            continue
        else:
            number_xy=number_xy.append({'Seat_number':'{}{}'.format(i,j),'x':i,'y':{'A':1,'B':2,'D':4,'E':5}[j]},ignore_index=True)
others=pd.DataFrame({'Seat_number':['6A','6B','C','7A','7B'],'x':[12,12,12,12,12],'y':[1,2,3,4,5]},columns=['Seat_number','x','y'])
number_xy=number_xy.append(others, ignore_index = True)




# In[2] 

'''

  # 1/  data processing


'''

def data_deal(data,vehicle_list,number_xy):  ### 针对某月/某车型的数据进行分析

    #由于预定时间为str类型，需先转换成datetime类型再排序
    book_time0=list(data['Booking_time'])
    book_time=[pd.to_datetime(i) for i in book_time0]
    data['Booking_time']=book_time
    
    # 1/排除已取消订单  
    # 2/找到 中国中车-46车型对应车牌号
    data_type=data[data['Bus_id'].isin(vehicle_list)].reset_index(drop=True, inplace=False)
    
    
    # 3/对于一条预定记录里包含多名乘客的情况，按乘客数列为不同记录
    data_seat0=list(data_type['Booking_seat'])
    data_seat=[i[1:-1].split('|') for i in data_seat0]
    for i in tqdm(range(len(data_seat))):
        l=len(data_seat[i])
        if l>1:
           df1 = data_type.loc[:i]
           df1.loc[i,'Booking_seat']='|'+data_seat[i][0]+'|'
           df2=data_type.loc[i+1:] if (i+1) !=len(data_seat) else pd.DataFrame()   ### 避免最后一行数据内有多名乘客时报错
           df3 = pd.DataFrame()
           for j in range(l-1):
               df3=df3.append(data_type.loc[i],ignore_index=True)
               df3.loc[j,'Booking_seat']='|'+data_seat[i][j+1]+'|'
           data_type = df1.append(df3).append(df2)   
    data_type.reset_index(drop=True, inplace=True)
    
    
    # 4/原数据集中增加一列坐标编号信息
    data_seat0=list(data_type['Booking_seat'])
    data_seat=[i[1:-1].split('|') for i in data_seat0]
    
    ### 由于数据有出入，给的车型表有误，因此要排除非给定车型的车辆
    a=[]
    for i in data_seat:
       a=a+i
    b=list(set(a))
    c=list(number_xy['Seat_number'])
    d=['|{}|'.format(x) for x in b if x not in c]
    e=list(data_type[data_type['Booking_seat'].isin(d)]['Seat_number'])   ## 找到包含不应出现座位号的车辆，删除对应信息
    data_type=data_type[-data_type['Seat_number'].isin(e)].reset_index(drop=True, inplace=False)
    data_seat0=list(data_type['Booking_seat'])
    data_seat=[i[1:-1].split('|') for i in data_seat0]
    
    
    number_x,number_y=[],[]
    for i in tqdm(range(len(data_seat))):
        number_x.append(int(number_xy[number_xy['Seat_number']==data_seat[i][0]]['x']))
        number_y.append(int(number_xy[number_xy['Seat_number']==data_seat[i][0]]['y']))
    data_type['Seat_number']=[i[0] for i in data_seat]
    data_type['x']=number_x
    data_type['y']=number_y
    data_type=data_type.drop_duplicates().reset_index(drop=True,inplace=False)     ## 剔除重复行
    data_type.to_excel(r'..\{}_processsing.xlsx'.format(month),index=False)

    return data_type

data_type=data_deal(data,vehicle_list,number_xy)


# In[3] 

  
'''

  # 2/  Visualized Analysis of Seat Choices
    2.1/ data statistic         
    2.2/ draw heapmap

'''    



###  Save the seat selection results of the first n passengers
def seat_select_n(data,n):
    new_data=pd.DataFrame()
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
                    if len(data4)>n:
                        data4=data4[:n]
                    new_data=new_data.append(data4,ignore_index=True)
    new_data.to_excel(r'..\{} fist {} passengers.xlsx'.format(month,n),index=False)
    return new_data
data_type=seat_select_n(data_type,4)





###  Draw the heatmap of the results of the first n passengers' seat choose results monthly 
def seat_select_heatmap_n(data,n=45):
    
    #对乘客们在车厢中的选座结果进行统计
    x_choose,y_choose=[],[]
    for i in range(len(data)):
         x_choose.append(list(data['x'])[i])
         y_choose.append(list(data['y'])[i])
    xy_choose=[(x_choose[i],y_choose[i]) for i in range(len(x_choose))]
    
    result_Counter=Counter(xy_choose)
    result_statistics=pd.DataFrame(columns=[5,4,3,2,1],index=[1,2,3,4,5,6,7,8,9,10,11,12])         ### 座位选择次数统计
    result_statistics_percent=pd.DataFrame(columns=[5,4,3,2,1],index=[1,2,3,4,5,6,7,8,9,10,11,12]) ### 座位选择百分比计算
    for i in [5,4,3,2,1]:
        for j in range(1,13):
            result_statistics[i][j]=result_Counter[(j,i)]
            result_statistics_percent[i][j]=round(result_Counter[(j,i)]/len(xy_choose),2)
    result_statistics.to_excel(r'..\Seat statistics-{} first {} passengers.xlsx'.format(month,n),index=False)
    result_statistics_percent.to_excel(r'..\Seat percent statistics-{} first {} passengers.xlsx'.format(month,n),index=False)
   
    #绘制乘客选座位置空间分布热力图
    data0,data1=[],[]     
    for i in range(1,13):
      data0.append(list(result_statistics.loc[i]))   
      data1.append(list(result_statistics_percent.loc[i])) 
      
    sns.set()
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    f, ax = plt.subplots(figsize=(5, 6))      # cmap='YlOrRd'
    sns.heatmap(data1, ax=ax,vmin=0,vmax=0.08,cmap='YlOrRd',annot=True,annot_kws={'size':13}, fmt='.2g',linewidths=2,cbar=True,mask=(data1==0))
    ax.set_title('{}'.format(month),fontsize=19,weight='bold',pad=10) #plt.title('热图'),均可设置图片标题
    ax.set_ylabel('Row',fontsize=16,labelpad=6)  #设置纵轴标签
    ax.set_xlabel('Column',fontsize=16,labelpad=6)  #设置横轴标签
    plt.yticks([i+0.5 for i in range(12)],[i for i in range(1,13)],size=14.5)  #设置纵轴标签
    plt.xticks([0+0.5,1+0.5,2+0.5,3+0.5,4+0.5],['Window','Aisle','Aisle','Aisle','Window'],size=14)  #设置横轴标签
    plt.show()

### 每个月画一张热力图
n=4
months=['2019-1','2019-2','2019-3','2019-4','2019-5','2019-6','2019-7','2019-8','2019-9','2019-10','2019-11','2019-12',\
        '2020-1','2020-2','2020-3','2020-4','2020-5','2020-6','2020-7','2020-8','2020-9','2020-10','2020-11','2020-12']
for i in range(len(months)):
   month=months[i]
   data_type=pd.read_excel(r'..\{} fist {} passengers.xlsx'.format(month,n))    
   seat_select_heatmap_n(data_type,n)






# In[4] 


### Draw the heatmap of the results of the first n passengers' seat choose results each stage

def stage_seat_select_heatmap_n(data,n=45):
    
    #对乘客们在车厢中的选座结果进行统计
    x_choose,y_choose=[],[]
    for i in range(len(data)):
         x_choose.append(list(data['x'])[i])
         y_choose.append(list(data['y'])[i])
    xy_choose=[(x_choose[i],y_choose[i]) for i in range(len(x_choose))]
    
    result_Counter=Counter(xy_choose)
    result_statistics=pd.DataFrame(columns=[5,4,3,2,1],index=[1,2,3,4,5,6,7,8,9,10,11,12])         ### 座位选择次数统计
    result_statistics_percent=pd.DataFrame(columns=[5,4,3,2,1],index=[1,2,3,4,5,6,7,8,9,10,11,12]) ### 座位选择百分比计算
    for i in [5,4,3,2,1]:
        for j in range(1,13):
            result_statistics[i][j]=result_Counter[(j,i)]
            result_statistics_percent[i][j]=round(result_Counter[(j,i)]/len(xy_choose),2)
       
    #绘制乘客选座位置空间分布热力图
    data0,data1=[],[]     
    for i in range(1,13):
      data0.append(list(result_statistics.loc[i]))   
      data1.append(list(result_statistics_percent.loc[i])) 
          
    sns.set()
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    f, ax = plt.subplots(figsize=(5, 6))      # cmap='YlOrRd'
    sns.heatmap(data1, ax=ax,vmin=0,vmax=0.08,cmap='YlOrRd',annot=True,annot_kws={'size':13}, fmt='.2g',linewidths=2,cbar=True,mask=(data1==0))
    ax.set_title('Stage 3',fontsize=19,weight='bold',pad=15,alpha=0) #plt.title('热图'),均可设置图片标题
    ax.set_ylabel('Row',fontsize=16,labelpad=6)  #设置纵轴标签
    ax.set_xlabel('Column',fontsize=16,labelpad=6)  #设置横轴标签
    plt.yticks([i+0.5 for i in range(12)],[i for i in range(1,13)],size=14.5)  #设置纵轴标签
    plt.xticks([0+0.5,1+0.5,2+0.5,3+0.5,4+0.5],['Window \n seat','Aisle \n seat','Aisle','Aisle \n seat','Window \n seat'],size=14)  #设置横轴标签
    plt.show()


n=4

months=[['2019-1','2019-2','2019-3','2019-4','2019-5','2019-6','2019-7','2019-8','2019-9','2019-10','2019-11','2019-12',\
        '2020-1'],['2020-2','2020-3'],['2020-4','2020-5','2020-6'],['2020-7','2020-8','2020-9','2020-10','2020-11','2020-12']]
stage_names=['Pre-COVID-19','Outbreak','Containment','Ongoing p&c']
for i in range(1):
    data_type=pd.DataFrame()
    stage_num=i+1
    stage_name=stage_names[i]
    for j in range(len(months[i])):
        month=months[i][j]
        data_type=data_type.append(pd.read_excel(r'..\{} fist {} passengers.xlsx'.format(month,n)),ignore_index=True)    
    stage_seat_select_heatmap_n(data_type,n)






# In[5] 

'''

  # 3/  Spatial Statistics of Seat Choices

'''  

### Morans_I

def ARCGIS_analysis(data1,data2,n):
    ARCGIS_data=pd.DataFrame(columns=['x','y','n','p'])
    x,y,data_n,data_p=[],[],[],[]
    for i in range(12):
        data_n=data_n+list(data1.loc[i])[::-1]
        data_p=data_p+list(data2.loc[i])[::-1]
        x=x+[i+1]*5
        y=y+[1,2,3,4,5]
    ARCGIS_data['x'],ARCGIS_data['y'],ARCGIS_data['n'],ARCGIS_data['p']=x,y,data_n,data_p
    ARCGIS_data=ARCGIS_data.drop([2,7,12,17,22,25,26,27,30,31,32,37,42,47,52]).reset_index(drop=True,inplace=False)
    ARCGIS_data.to_excel(r'..\Spatial Statistics-{} first {} passengers.xlsx'.format(month,n),index=False)
    return ARCGIS_data

n=4
months=['2019-1','2019-2','2019-3','2019-4','2019-5','2019-6','2019-7','2019-8','2019-9','2019-10','2019-11','2019-12','2020-1','2020-2','2020-3','2020-4','2020-5','2020-6','2020-7','2020-8','2020-9','2020-10','2020-11','2020-12']
for i in range(len(months)):
    month=months[i]
    data1=pd.read_excel(r'..\Seat statistics-{} first {} passengers.xlsx'.format(month,n)) 
    data2=pd.read_excel(r'..\Seat percent statistics-{} first {} passengers.xlsx'.format(month,n)) 
    ARCGIS_data=ARCGIS_analysis(data1,data2,n)
    




# In[6] 

def distance(multiple,x1,y1,x2,y2):
    delta_x=abs(x1-x2)
    delta_y=abs(y1-y2)
    distance=math.sqrt((multiple*delta_x)**2+delta_y**2)
    distance=round(distance,2)
    return distance


def Morans_I(data,weight_method,multiple):
    part1,part2=0,0 
    for i in range(45):
        for j in range(45): 
            dist_os=distance(multiple,data['x'][i],data['y'][i],data['x'][j],data['y'][j])
            if weight_method=='os_dist':
                part1=part1+(1/dist_os*(data['p'][i]-np.mean(data['p']))*(data['p'][j]-np.mean(data['p'])) if dist_os!=0 else 0)
                part2=part2+(1/dist_os if dist_os!=0 else 0)
            elif weight_method=='Rook':   ### 共享边
                part1=part1+(1*(data['p'][i]-np.mean(data['p']))*(data['p'][j]-np.mean(data['p'])) if (dist_os==1 or dist_os==multiple) else 0)
                part2=part2+(1 if (dist_os==1 or dist_os==multiple) else 0)
            elif weight_method=='Queen':  ### 共享点和边
                part1=part1+(1*(data['p'][i]-np.mean(data['p']))*(data['p'][j]-np.mean(data['p'])) if (dist_os==1 or dist_os==multiple or dist_os==round(math.sqrt(1+multiple**2),2)) else 0)
                part2=part2+(1 if (dist_os==1 or dist_os==multiple or dist_os==round(math.sqrt(1+multiple**2),2)) else 0)
    part3=0
    for i in range(45):
        part3=part3+(data['p'][i]-np.mean(data['p']))**2
    I=part1*45/(part3*part2)
    
    return I

I=Morans_I(ARCGIS_data,'os_dist',2)


### 求解各月空间自相关系数指标I
multiple=2
n=4
Is_os_dist,Is_Rook,Is_Queen=[],[],[]
months=['2019-1','2019-2','2019-3','2019-4','2019-5','2019-6','2019-7','2019-8','2019-9','2019-10','2019-11','2019-12','2020-1','2020-2','2020-3','2020-4','2020-5','2020-6','2020-7','2020-8','2020-9','2020-10','2020-11','2020-12']
for i in range(len(months)):
    month=months[i]
    data=pd.read_excel(r'..\Spatial Statistics-{} first {} passengers.xlsx'.format(month,n))
    Is_os_dist.append(Morans_I(data,'os_dist',multiple))
    Is_Rook.append(Morans_I(data,'Rook',multiple))
    Is_Queen.append(Morans_I(data,'Queen',multiple))



### 制图1 曲线图
#plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = 'Times New Roman'
fig=plt.figure(figsize=(7.8,4.2))
ax=fig.add_subplot(1,1,1)

plt.plot(months[3:],Is_os_dist[3:],alpha=0.8,linewidth=2)
#plt.scatter(months[3:],Is_os_dist[3:],alpha=0.8,s=15)
plt.plot(months[3:],Is_Rook[3:],alpha=0.8,linewidth=2)
#plt.scatter(months[3:],Is_Rook[3:],alpha=0.8,marker='x',s=15)
plt.plot(months[3:],Is_Queen[3:],alpha=0.8,linewidth=2)
#plt.scatter(months[3:],Is_Queen[3:],alpha=0.8,marker='v',s=15)
#plt.title('Morans’I ——the spatial autocorrelation index of seat selection results in each month',fontsize=24,pad=15)

plt.plot(['2020-2']*2,[0.12,0.85],linestyle='--')
#plt.text('2019-4',0.41,"Mean value in pre-COVID-19 stage",fontdict={'size':'14.2','color':'black'})

plt.text(4,0.514,"Stage 1",fontdict={'size':'14.2','color':'black'})
plt.text(9.9,0.514,"Stage 2",fontdict={'size':'14.2','color':'black'})
plt.text(12.5,0.514,"Stage 3",fontdict={'size':'14.2','color':'black'})
plt.text(16.5,0.514,"Stage 4",fontdict={'size':'14.2','color':'black'})

ax.add_patch(
    patches.Rectangle(
        xy=(0, 0.1),  # point of origin.
        width=10,
        height=0.9,
        linewidth=1,
        color='green',
        alpha=0.025,
        fill=True,
    )
)
ax.add_patch(
    patches.Rectangle(
        xy=(10, 0.1),  # point of origin.
        width=2,
        height=0.9,
        linewidth=1,
        color='pink',
        alpha=0.10,
        fill=True,
    )
)
ax.add_patch(
    patches.Rectangle(
        xy=(12, 0.1),  # point of origin.
        width=3,
        height=0.9,
        linewidth=1,
        color='blue',
        alpha=0.028,
        fill=True,
    )
)
ax.add_patch(
    patches.Rectangle(
        xy=(15, 0.1),  # point of origin.
        width=5,
        height=0.9,
        linewidth=1,
        color='gold',
        alpha=0.03,
        fill=True,
    )
)


plt.legend(['Euclidean inverse distance','Rook Contiguity','Queen Contiguity'],fontsize=8.9,loc=(0.66,0.79),framealpha=0.3)
plt.xlabel('Month',fontsize=18,labelpad=6)
plt.ylabel('Moran’s I',fontsize=18,labelpad=8)
plt.xticks(fontsize=11.5,rotation=45)
plt.yticks(fontsize=16)
plt.ylim(0.1,0.9)
plt.grid(True,linewidth=0.4,linestyle='--')
plt.savefig(r'..\Morans_I.png',dpi=500,bbox_inches='tight')#保存图片


