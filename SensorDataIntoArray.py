#-*- coding: utf-8 -*-
import string
import numpy as np
import scipy as sp  #惯例
from scipy.optimize import leastsq #这里就是我们要使用的最小二乘的函数
import matplotlib.pyplot as plt

#定义读取一个表格并计算的函数
def TransTxtIntoArray(Filename,direction):
        #通过文件名判断解耦时分母大小
        if Filename[1]=="z":
                FRefer=20
        else:
                FRefer=40
                
        f=file(Filename,'r')#打开文件
        LineIndex=0
        #声明list变量，用来存放item,Ux~Uz的值
        listFx=[]
        listFy=[]
        listFz=[]
    
        listUx=[]
        listUy=[]
        listUz=[]
      
        
        listArray=(listFy,listFz,listFx,listUy,listUz,listUx)
    
        #读取文件，并将数据加入list
        while True:   #或者写成for eachLine in f:
                strLine=f.readline()
                if len(strLine)==0: # Zero length indicates EOF
                        break
                LineIndex+=1
                if (LineIndex==1)|(LineIndex==2):
                        continue    #第一行是txt文件的说明，第二行是表头，忽略
                
                Valuelist=strLine.split()
                for i in range(1,7):
                        num=string.atof(Valuelist[i])
                        listArray[i-1].append(num)       
        f.close()
        return listFy,listFz,listFx,listUy,listUz,listUx,FRefer

     
#在其它模块调用方式
listFy,listFz,listFx,listUy,listUz,listUx,FRefer=TransTxtIntoArray("Fx+Test1Avr.txt",0)






