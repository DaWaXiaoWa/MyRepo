# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 21:25:49 2021

@author: David
"""

from nd2reader import ND2Reader
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import scipy.signal as signal
import pandas as pd
import matplotlib
import math
from matplotlib.pyplot import MultipleLocator
#from scipy.misc import imsave

 
def function_try(x):
    y=0.00372276*x**3-0.08175541*x**2+0.60201624*x-1.57996052
    return y

def function_try2(a,x):
    y=a[3]*np.power(np.fabs(x-a[0]),1.0/a[1])*np.sign(x-a[0])+a[2]+a[4]*x
    return y

def calibration_RSME2(a,log10_Ratio):
    y_target=np.array(range(4,11))
    y=np.zeros(len(log10_Ratio))
    for j in range(len(y)):
        # y[j]=a[0]*np.power((x[j]-a[1]),2)+a[2]+a[3]/(x[j]-a[4])
        y[j]=function_try2(a,log10_Ratio[j])
    delta_y=(y_target-y)
    RSME=np.dot(delta_y,delta_y)+2.0*delta_y[0]*delta_y[0]+2.0*delta_y[1]*delta_y[1]+2.0*delta_y[-1]*delta_y[-1]+2.0*delta_y[-2]*delta_y[-2]
    RSME=np.power(RSME,0.5)
    return RSME

def expose_calibration_parameters(filename_0):    
    ratio=np.zeros(10-4+1)
    for j in range(len(ratio)):
        ############1. 读取定标数据    
        #filename="RongLi20190715/pH"+str(j+4)+".nd2"
        filename=filename_0+str(j+4)+".nd2"
        with ND2Reader(filename) as images:
            x_count=images.metadata['width']
            y_count=images.metadata['height']    
            #print (filename,x_count, y_count)
            FITC_frame=np.zeros([y_count,x_count])
            FITC_frame=images.get_frame_2D(c=0, t=0, z=0, x=0, y=0, v=0)+0.0001
    
            TRITC_frame=np.zeros([y_count,x_count])
            TRITC_frame=images.get_frame_2D(c=1, t=0, z=0, x=0, y=0, v=0)+0.0001
    
        ############2. take average
            
            FITC_Intensity=np.mean(FITC_frame)       
            TRITC_Intensity=np.mean(TRITC_frame)        
            #FITC_Intensity=np.mean(FITC_frame[int(y_count/2)-100:int(y_count/2)+100,int(x_count/2)-100:int(x_count/2)+100])       
            #TRICT_Intensity=np.mean(TRICT_frame[int(y_count/2)-100:int(y_count/2)+100,int(x_count/2)-100:int(x_count/2)+100])    
        ############3. calculate the ratio
        ratio[j]=FITC_Intensity/TRITC_Intensity
        
    #plt.plot(ratio)
    #plt.plot(np.log10(ratio),range(4,11))
    #plt.scatter(np.log10(ratio),range(4,11), marker='o')
    plt.scatter(range(4,11),np.log10(ratio), marker='o')
    #plt.show()
    print(np.log10(ratio))
    log10_Ratio=np.log10(ratio)
    
    
    a0=[0.0,3.0,6.5,3.0,1.0]
    calibration_RSME2(a0,log10_Ratio)
    res=minimize(fun=calibration_RSME2, x0=a0, args=log10_Ratio,bounds=[(log10_Ratio[1],log10_Ratio[4]), (1.0, None), (5.0, 8.0),(0,3),(0,None)])
    print (res)
    print ('y=',res.x)
    
    xx=np.linspace(min(log10_Ratio),max(log10_Ratio),100,endpoint=True)
    yy=function_try2(res.x,xx)
    
    #plt.plot(xx,yy)
    plt.plot(yy,xx)
    plt.xlabel('pH')
    plt.ylabel('lg(FITC/TRITC)')
    #plt.show()
    return res.x

def concentration_calibration_parameters_2D(filename_1,filename_2):
    filename=filename_1+str(4)+".nd2"
    with ND2Reader(filename) as images:
        x_count=images.metadata['width']
        y_count=images.metadata['height']    
        #print (filename,x_count, y_count)
        FITC_frame=np.zeros([y_count,x_count])
        FITC_frame=images.get_frame_2D(c=0, t=0, z=0, x=0, y=0, v=0)+0.0001

        TRITC_frame=np.zeros([y_count,x_count])
        TRITC_frame=images.get_frame_2D(c=1, t=0, z=0, x=0, y=0, v=0)+0.0001
        
        FITC_Intensity=np.mean(FITC_frame)       
        TRITC_Intensity=np.mean(TRITC_frame)        
        #FITC_Intensity=np.mean(FITC_frame[int(y_count/2)-100:int(y_count/2)+100,int(x_count/2)-100:int(x_count/2)+100])       
        #TRICT_Intensity=np.mean(TRICT_frame[int(y_count/2)-100:int(y_count/2)+100,int(x_count/2)-100:int(x_count/2)+100])    
    ############3. calculate the ratio
    ratio_calib=FITC_Intensity/TRITC_Intensity
    
    filename=filename_2
    with ND2Reader(filename) as images:
        x_count=images.metadata['width']
        y_count=images.metadata['height']    
        #print (filename,x_count, y_count)
        FITC_frame=np.zeros([y_count,x_count])
        FITC_frame=images.get_frame_2D(c=0, t=0, z=0, x=0, y=0, v=0)+0.0001

        TRITC_frame=np.zeros([y_count,x_count])
        TRITC_frame=images.get_frame_2D(c=1, t=0, z=0, x=0, y=0, v=0)+0.0001
        
        FITC_Intensity=np.mean(FITC_frame[:,-25:])       
        TRITC_Intensity=np.mean(TRITC_frame[:,-25:])        
        #FITC_Intensity=np.mean(FITC_frame[int(y_count/2)-100:int(y_count/2)+100,int(x_count/2)-100:int(x_count/2)+100])       
        #TRICT_Intensity=np.mean(TRICT_frame[int(y_count/2)-100:int(y_count/2)+100,int(x_count/2)-100:int(x_count/2)+100])    
    ############3. calculate the ratio
    ratio_real=FITC_Intensity/TRITC_Intensity
    deltaLog10Intensity=np.log10(ratio_real/ratio_calib)
    return deltaLog10Intensity     


#filename_0="RongLi20210309/pH"
filename_0="Zhenggexi20210722/PH/PH"
filename_1="RongLi20210309/pH"
res=expose_calibration_parameters(filename_1)     
#标曲1对应filename_0,标曲2对应filename_1 

#filename_2="RongLi20210311/marble1_largeimage_3825_5ul.nd2"
filename_2="Zhenggexi20211009/new_calcite_largeimage_4395_4X.nd2"
deltaLog10Intensity=concentration_calibration_parameters_2D(filename_1,filename_2)  #filename_2是标曲根据入口pH强制平移的计算程序，每块样品（不同时间点）要选取同一个文件，尽量选取远离矿物表面的平面
print (deltaLog10Intensity)

#filename="RongLi20210311/marble1_largeimage_3825_5ul.nd2"
filename="Zhenggexi20211009/new_calcite_largeimage_4395_4X.nd2"
with ND2Reader(filename) as images:
    x_count=images.metadata['width']
    y_count=images.metadata['height']    
    #print (filename,x_count, y_count)
    FITC_frame=np.zeros([y_count,x_count])
    FITC_frame=1.0*images.get_frame_2D(c=0, t=0, z=0, x=0, y=0, v=0)+0.0001
    operator = np.ones((2,2))
    FITC_frame=signal.convolve2d(FITC_frame,operator,mode="same")

    TRITC_frame=np.zeros([y_count,x_count])
    TRITC_frame=1.0*images.get_frame_2D(c=1, t=0, z=0, x=0, y=0, v=0)+0.0001
    #operator = np.ones((5,5))
    TRITC_frame=signal.convolve2d(TRITC_frame,operator,mode="same")   
    #ratio2D=np.log10(FITC_frame/TRITC_frame) -deltaLog10Intensity
    ratio2D=np.log10(FITC_frame/TRITC_frame) - deltaLog10Intensity
    pH2D=function_try2(res,ratio2D)

x_count=ND2Reader(filename).metadata['width']
y_count=ND2Reader(filename).metadata['height'] 
real_x=round(x_count*3.08/1000,1)
real_y=round(y_count*3.08/1000,1)
ax=plt.gca()
#x_major_locator=MultipleLocator(2)
#y_major_locator=MultipleLocator(1)
#ax.xaxis.set_major_locator(x_major_locator)
#ax.yaxis.set_major_locator(y_major_locator)
norm=matplotlib.colors.Normalize(vmin=4, vmax=10)
tupian=plt.imshow(pH2D,norm=norm,cmap='tab20c')
#tupian=plt.imshow(pH2D,norm=norm)
#plt.xlim(0,real_x)
#plt.ylim(0,real_y)
x_step=np.round(np.linspace(0,real_x,math.ceil(x_count/500)),2)
y_step=np.round(np.linspace(0,real_y,math.ceil(y_count/250)),2)
plt.xticks(np.arange(0,x_count,500),x_step)
plt.yticks(np.arange(0,y_count,250),y_step)
#tupian.set_xticks(np.arange(0,real_x,2))
#tupian.set_yticks(np.arange(0,real_y,1))
plt.xlabel('Length/mm')
plt.ylabel('Width/mm')
cbar = plt.colorbar(tupian)  
cbar.set_ticks(np.linspace(4,10,7))  
cbar.set_label("pH")
cbar.set_ticklabels( ('4', '5', '6', '7', '8',  '9',  '10'))
#plt.savefig("4newpH_"+filename.replace("/", "_").replace("nd2","jpg"),dpi=1600)
plt.show()   ###############图片的格式化输出需要适当编写程序和调整，FITC_frame和TRITC_frame的原图片也要输出，ndviewer输出jpg
#plt.imshow(pH2D)
#plt.colorbar()   ###############图片的格式化输出需要适当编写程序和调整，FITC_frame和TRITC_frame的原图片也要输出，ndviewer输出jpg
#plt.imsave("pH_"+filename.replace("/", "_").replace("nd2","jpg"), pH2D)


# data_df = pd.DataFrame(pH2D)   #矩阵数据存储为Excel，可用于后续处理，运行较慢
# writer = pd.ExcelWriter("pH_"+filename.replace("/", "_").replace("nd2","xlsx"))
# data_df.to_excel(writer,"pH",float_format='%.7f') # float_format 控制精度
# writer.save()