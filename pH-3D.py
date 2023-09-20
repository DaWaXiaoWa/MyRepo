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
#from scipy.misc import imsave

 

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
    plt.scatter(np.log10(ratio),range(4,11), marker='o')
    print(np.log10(ratio))
    log10_Ratio=np.log10(ratio)
    
    
    a0=[0.0,3.0,6.5,3.0,1.0]
    calibration_RSME2(a0,log10_Ratio)
    res=minimize(fun=calibration_RSME2, x0=a0, args=log10_Ratio,bounds=[(log10_Ratio[1],log10_Ratio[4]), (1.0, None), (5.0, 8.0),(0,3),(0,None)])
    print (res)
    
    xx=np.linspace(min(log10_Ratio),max(log10_Ratio),100,endpoint=True)
    yy=function_try2(res.x,xx)
    
    plt.plot(xx,yy)
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

def plot_slices(matrix_3D, filename, slice_direction, slices_position):
    norm=matplotlib.colors.Normalize(vmin=4, vmax=10)
    if slice_direction=="x_slice":
        plt.imshow(matrix_3D[:,:,slices_position],norm=norm,cmap='tab20c')
    elif slice_direction=="y_slice":
        plt.imshow(matrix_3D[:,slices_position,:],norm=norm,cmap='tab20c')
    elif slice_direction=="z_slice":
        plt.imshow(matrix_3D[slices_position,:,:],norm=norm,cmap='tab20c')   
    cbar = plt.colorbar()  
    cbar.set_ticks(np.linspace(4,10,7))  
    cbar.set_ticklabels( ('4', '5', '6', '7', '8',  '9',  '10'))
    cbar.set_label("pH")
    plt.xticks(np.arange(0,x_count,200),np.linspace(0,real_x,math.ceil(x_count/200)))
    plt.yticks(np.arange(0,z_count,30),np.arange(0,z_count,30))
    
    plt.xlabel('Length/mm',fontsize=7)
    plt.ylabel('Height/(*3μm)',fontsize=7)
    plt.tick_params(labelsize=7)
    plt.savefig("new3DpH_"+slice_direction+"="+str(slices_position)+"_"+filename.replace("/", "_").replace("nd2","jpg"),dpi=500)
    plt.show()
    return 0

filename_0="RongLi20210309/pH"
#filename_0="Zhenggexi20210722/PH/PH"
#filename_1="RongLi20190715/pH"
filename_1="Zhenggexi20210824/pH"
res=expose_calibration_parameters(filename_0)    #标曲1对应filename_0,标曲2对应filename_1 

#filename_2="RongLi20210316/marble1_largeimage_3825_5ul.nd2"
filename_2="Zhenggexi20211009/new_calcite_largeimage_4395_4X.nd2"
deltaLog10Intensity=concentration_calibration_parameters_2D(filename_0,filename_2)  #filename_2是标曲根据入口pH强制平移的计算程序，每块样品（不同时间点）要选取同一个文件，尽量选取远离矿物表面的平面
print (deltaLog10Intensity)

#filename="RongLi20210316/marble1_inletdown_3700-3995_5ul.nd2"
#filename="Zhenggexi20210727/marble2_inlet_up_4190-4390_3um_4X_test.nd2"
filename="Zhenggexi20211009/new_calcite_outlet_down_4270-4660_3um_4X.nd2"



with ND2Reader(filename) as images:
    x_count=images.metadata['width']
    y_count=images.metadata['height']
    z_count=len(images.metadata['z_levels'])    
    #print (filename,x_count, y_count)
    
    real_x=round(x_count*3.08/1000,1)
    real_z=z_count*3

    dx=images.metadata['pixel_microns']
    dy=dx
    TRITC_frames=np.zeros([z_count,y_count,x_count])
    FITC_frames=np.zeros([z_count,y_count,x_count])
    fluid_flag=np.zeros([z_count,y_count,x_count])     ###给定节点是否为流体

    for k in range(z_count):
        FITC_frames[k]=1.0*images.get_frame_2D(c=0, t=0, z=k, x=0, y=0, v=0)+0.0001
        TRITC_frames[k]=1.0*images.get_frame_2D(c=1, t=0, z=k, x=0, y=0, v=0)+0.0001


    ratio3D=np.log10(FITC_frames/TRITC_frames) - deltaLog10Intensity
        
    pH3D=function_try2(res,ratio3D)
    
    
    cutoff_TRITC=256    #################TRITC的边界判定标准
    pH3D[TRITC_frames<cutoff_TRITC]=0


plot_slices(pH3D, filename, "y_slice", int(y_count/10))   ####画图调用参数，"x_slice"/"y_slice"/"z_slice"



