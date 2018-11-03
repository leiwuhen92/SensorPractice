#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from SensorDataIntoArray import TransTxtIntoArray

np.seterr(divide='ignore', invalid='ignore')#RuntimeWarning: invalid value encountered in divide 解决除法中分母为0无效问题

def logsig(x):
    return 1/(1+np.exp(-x))

listFy1,listFz1,listFx1,listUy1,listUz1,listUx1,FRefer=TransTxtIntoArray("Fx+test1Avr.txt",0)
listFy2,listFz2,listFx2,listUy2,listUz2,listUx2,FRefer=TransTxtIntoArray("Fx-test1Avr.txt",1)
listFy3,listFz3,listFx3,listUy3,listUz3,listUx3,FRefer=TransTxtIntoArray("Fy+test1Avr.txt",0)
listFy4,listFz4,listFx4,listUy4,listUz4,listUx4,FRefer=TransTxtIntoArray("Fy-test1Avr.txt",1)
listFy5,listFz5,listFx5,listUy5,listUz5,listUx5,FRefer=TransTxtIntoArray("Fz+test1Avr.txt",0)

listFy=np.hstack((listFy1,listFy2,listFy3,listFy4,listFy5))  #(110,)  即(samnum,)
listFz=np.hstack((listFz1,listFz2,listFz3,listFz4,listFz5))  #(110,)
listFx=np.hstack((listFx1,listFx2,listFx3,listFx4,listFx5))  #(110,)
listUy=np.hstack((listUy1,listUy2,listUy3,listUy4,listUy5))  #(110,)
listUz=np.hstack((listUz1,listUz2,listUz3,listUz4,listUz5))  #(110,)
listUx=np.hstack((listUx1,listUx2,listUx3,listUx4,listUx5))  #(110,)


samplein = np.mat([listUy,listUz,listUx]) #3*samnum
#print samplein
#print samplein.min(axis=1).T.tolist()[0]# 1*3 所有项的最小值
sampleinminmax = np.array([samplein.min(axis=1).T.tolist()[0],samplein.max(axis=1).T.tolist()[0]]).transpose()#3*2，对应最大值最小值
sampleout = np.mat([listFy,listFz,listFx])#3*samnum
sampleoutminmax = np.array([sampleout.min(axis=1).T.tolist()[0],sampleout.max(axis=1).T.tolist()[0]]).transpose()#3*2，对应最大值最小值
#print sampleinminmax
#print sampleout

#数据标准化 X=2*(x-xmin)/(xmax-xmin)-1
#3*samnum
sampleinnorm = (2*(np.array(samplein.T)-sampleinminmax.transpose()[0])/(sampleinminmax.transpose()[1]-sampleinminmax.transpose()[0])-1).transpose() #(x-xmin)/(xmax-xmin)
#3*samnum
sampleoutnorm = (2*(np.array(sampleout.T).astype(float)-sampleoutminmax.transpose()[0])/sampleoutminmax.transpose()[1]-sampleoutminmax.transpose()[0]-1).transpose()
where_are_nan = np.isnan(sampleoutnorm)
sampleoutnorm[where_are_nan] = 0

maxepochs = 60000  #训练步数
learnrate = 0.001  #学习速率
errorfinal = 0.65*10**(-3)
samnum = len(listFx)    #训练样本多少行数据
numLine= len(listFy1)   ####读取的txt文本行数
indim = 3
outdim = 3
hiddenunitnum = 8

w1 = 0.5*np.random.rand(hiddenunitnum,indim)-0.1  #8*3 权重
b1 = 0.5*np.random.rand(hiddenunitnum,1)-0.1      #8*1 偏置
w2 = 0.5*np.random.rand(outdim,hiddenunitnum)-0.1 #3*8
b2 = 0.5*np.random.rand(outdim,1)-0.1             #3*1

errhistory = []

for i in range(maxepochs): #训练步数
    ##Oj=sigmod(Ij)=1/(1+e−Ij),其中#Ij=∑i(Wij*Oi)
    hiddenout = logsig((np.dot(w1,sampleinnorm).transpose()+b1.transpose())).transpose() #8*samnum
    networkout = (np.dot(w2,hiddenout).transpose()+b2.transpose()).transpose() #3*samnum

    #Tj−Oj，Ej代表神经元j的误差，Oj表示神经元j的输出， Tj表示当前训练样本的参考输出
    err = sampleoutnorm - networkout #3*samnum
    sse = sum(sum(err**2)) #一个数 ∑j((Tj−Oj)^2)

    errhistory.append(sse)
    if sse < errorfinal:
        break

    delta2 = err  #1*samnum 输出层误差

    delta1 = np.dot(w2.transpose(),delta2)*hiddenout*(1-hiddenout)#隐含层误差#8*samnum

    dw2 = np.dot(delta2,hiddenout.transpose())     # Oi*Ej 
    db2 = np.dot(delta2,np.ones((samnum,1)))       

    dw1 = np.dot(delta1,sampleinnorm.transpose())  
    db1 = np.dot(delta1,np.ones((samnum,1)))       

    w2 += learnrate*dw2#更新权重与偏置，#Wij=Wij+λ*Oi*Ej
    b2 += learnrate*db2

    w1 += learnrate*dw1
    b1 += learnrate*db1
    
# 误差曲线图
minerr = min(errhistory)          #最小误差 
plt.plot(errhistory)              #近似绿色直线                              
plt.plot(range(0,i+1000,1000),[minerr]*len(range(0,i+1000,1000)))#蓝色曲线 多次迭代后误差趋于一点

ax=plt.gca()
ax.set_ylim(0,1500)
ax.set_xlabel('iteration')  #轴标签
ax.set_ylabel('error')
ax.set_title('Error History')
plt.savefig('errorhistory.png',dpi=500)#dpi每英寸点数即分辨率
plt.show()
plt.close()

# 仿真输出和实际输出对比图
#Oj=sigmod(Ij)=1/(1+e−Ij),其中#Ij=∑i(Wij*Oi)
hiddenout = logsig((np.dot(w1,sampleinnorm).transpose()+b1.transpose())).transpose()#8*samnum
networkout = (np.dot(w2,hiddenout).transpose()+b2.transpose()).transpose()#3*samnum  标准化后的实际输出

#将标准化数据还原为实际数据
diff = sampleoutminmax[:,1]-sampleoutminmax[:,0] #标准化公式X=2*(x-xmin)/(xmax-xmin)-1
networkout2 = (networkout+1)/2 #得到(x-xmin)/(xmax-xmin)
networkout2[0] = networkout2[0]*diff[0]+sampleoutminmax[0][0]#Fy计算输出 x=(X+1)/2*(xmax-xmin)+xmin
networkout2[1] = networkout2[1]*diff[1]+sampleoutminmax[1][0]#Fz计算输出
networkout2[2] = networkout2[2]*diff[2]+sampleoutminmax[2][0]#Fx计算输出

Fy1=networkout2[0][0:numLine]
Fy2=networkout2[0][numLine:2*numLine]
Fy3=networkout2[0][2*numLine:3*numLine]
Fy4=networkout2[0][3*numLine:4*numLine]
Fy5=networkout2[0][4*numLine:5*numLine]

Fz1=networkout2[1][0:numLine]
Fz2=networkout2[1][numLine:2*numLine]
Fz3=networkout2[1][2*numLine:3*numLine]
Fz4=networkout2[1][3*numLine:4*numLine]
Fz5=networkout2[1][4*numLine:5*numLine]

Fx1=networkout2[2][0:numLine]
Fx2=networkout2[2][numLine:2*numLine]
Fx3=networkout2[2][2*numLine:3*numLine]
Fx4=networkout2[2][3*numLine:4*numLine]
Fx5=networkout2[2][4*numLine:5*numLine]


################将预测输出、I类误差写入文件########################
#Fx+作用力
Filename="Fx+.txt"
Filename=Filename[0:-4]#去掉文件名后缀
Filename+="Error.txt"
f=file(Filename,'w')#打开文件
f.write("Item\tFy\tFz\tFx\tF'y\tF'z\tF'x\tErrorFy\tErrorFz\tErrorFx\n")# \t在excel中是一个单元格
for i in range(0,numLine):    #行数
    f.write(str(i)+"\t"+str(listFy1[i])+"\t"+str(listFz1[i])+"\t"+str(listFx1[i])+"\t"+
            str(Fy1[i])+"\t"+str(Fz1[i])+"\t"+str(Fx1[i])+"\t"+
            str(abs((listFy1[i]-Fy1[i])/FRefer))+"\t"+str(abs((listFz1[i]-Fz1[i])/FRefer))+"\t"+str(abs((listFx1[i]-Fx1[i])/FRefer))+ "\n")
f.close()

#Fx-作用力
Filename="Fx-.txt"
Filename=Filename[0:-4]#去掉文件名后缀
Filename+="Error.txt"
f=file(Filename,'w')#打开文件
f.write("Item\tFy\tFz\tFx\tF'y\tF'z\tF'x\tErrorFy\tErrorFz\tErrorFx\n")# \t在excel中是一个单元格
for i in range(0,numLine):    #行数
    f.write(str(i)+"\t"+str(listFy2[i])+"\t"+str(listFz2[i])+"\t"+str(listFx2[i])+"\t"+
            str(Fy2[i])+"\t"+str(Fz2[i])+"\t"+str(Fx2[i])+"\t"+
            str(abs((listFy2[i]-Fy2[i])/FRefer))+"\t"+str(abs((listFz2[i]-Fz2[i])/FRefer))+"\t"+str(abs((listFx2[i]-Fx2[i])/FRefer))+ "\n")
f.close()

#Fy+作用力
Filename="Fy+.txt"
Filename=Filename[0:-4]#去掉文件名后缀
Filename+="Error.txt"
f=file(Filename,'w')#打开文件
f.write("Item\tFy\tFz\tFx\tF'y\tF'z\tF'x\tErrorFy\tErrorFz\tErrorFx\n")# \t在excel中是一个单元格
for i in range(0,numLine):    #行数
    f.write(str(i)+"\t"+str(listFy3[i])+"\t"+str(listFz3[i])+"\t"+str(listFx3[i])+"\t"+
            str(Fy3[i])+"\t"+str(Fz3[i])+"\t"+str(Fx3[i])+"\t"+
            str(abs((listFy3[i]-Fy3[i])/FRefer))+"\t"+str(abs((listFz3[i]-Fz3[i])/FRefer))+"\t"+str(abs((listFx3[i]-Fx3[i])/FRefer))+ "\n")
f.close()

#Fy-作用力
Filename="Fy-.txt"
Filename=Filename[0:-4]#去掉文件名后缀
Filename+="Error.txt"
f=file(Filename,'w')#打开文件
f.write("Item\tFy\tFz\tFx\tF'y\tF'z\tF'x\tErrorFy\tErrorFz\tErrorFx\n")# \t在excel中是一个单元格
for i in range(0,numLine):    #行数
    f.write(str(i)+"\t"+str(listFy4[i])+"\t"+str(listFz4[i])+"\t"+str(listFx4[i])+"\t"+
            str(Fy4[i])+"\t"+str(Fz4[i])+"\t"+str(Fx4[i])+"\t"+
            str(abs((listFy4[i]-Fy4[i])/FRefer))+"\t"+str(abs((listFz4[i]-Fz4[i])/FRefer))+"\t"+str(abs((listFx4[i]-Fx4[i])/FRefer))+ "\n")
f.close()

#Fz+作用力
Filename="Fz+.txt"
Filename=Filename[0:-4]#去掉文件名后缀
Filename+="Error.txt"
f=file(Filename,'w')#打开文件
f.write("Item\tFy\tFz\tFx\tF'y\tF'z\tF'x\tErrorFy\tErrorFz\tErrorFx\n")# \t在excel中是一个单元格
for i in range(0,numLine):    #行数
    f.write(str(i)+"\t"+str(listFy5[i])+"\t"+str(listFz5[i])+"\t"+str(listFx5[i])+"\t"+
            str(Fy5[i])+"\t"+str(Fz5[i])+"\t"+str(Fx5[i])+"\t"+
            str(abs((listFy5[i]-Fy5[i])/FRefer))+"\t"+str(abs((listFz5[i]-Fz5[i])/FRefer))+"\t"+str(abs((listFx5[i]-Fx5[i])/FRefer))+ "\n")
f.close()


################XYZ方向分别受力时预测结果作图显示########################
title=['Fy','Fz','Fx']
item=['X+direction force','X-direction force','Y+direction force','Y-direction force','Z+direction force']
i=0

fig1, ax1 = plt.subplots(nrows=3,ncols=1, sharex=True)# 创建图表1
listF=[listFy1,listFz1,listFx1]
F=[Fy1,Fz1,Fx1]
for row in ax1:
    row.plot(range(numLine), listF[i],label="Theoretical data")
    row.plot(range(numLine),     F[i],label="Prediction data")
    row.set_title(title[i])
    row.set_xlabel('Item')
    row.set_ylabel('Force(N)')
    row.grid()
    i=i+1
    if i>=3:
        i=0
        break
plt.tight_layout()
fig1.savefig('Fitting curve of X+ direction force prediction.png',dpi=500,bbox_inches='tight')#bbox_inches检出当前图标周围的空白部分

fig2, ax2 = plt.subplots(nrows=3,ncols=1, sharex=True)# 创建图表2
listF=[listFy2,listFz2,listFx2]
F=[Fy2,Fz2,Fx2]
for row in ax2:
    row.plot(range(numLine), listF[i],label="Theoretical data")
    row.plot(range(numLine),     F[i],label="Prediction data")
    row.set_title(title[i])
    row.set_xlabel('Item')
    row.set_ylabel('Force(N)')
    row.grid()
    i=i+1
    if i>=3:
        i=0
        break
plt.tight_layout()
fig2.savefig('Fitting curve of X- direction force prediction.png',dpi=500,bbox_inches='tight')#bbox_inches检出当前图标周围的空白部分

fig3, ax3 = plt.subplots(nrows=3,ncols=1, sharex=True)# 创建图表3
listF=[listFy3,listFz3,listFx3]
F=[Fy3,Fz3,Fx3]
for row in ax3:
    row.plot(range(numLine), listF[i],label="Theoretical data")
    row.plot(range(numLine),     F[i],label="Prediction data")
    row.set_title(title[i])
    row.set_xlabel('Item')
    row.set_ylabel('Force(N)')
    row.grid()
    i=i+1
    if i>=3:
        i=0
        break

plt.tight_layout()
fig3.savefig('Fitting curve of Y+ direction force prediction.png',dpi=500,bbox_inches='tight')#bbox_inches检出当前图标周围的空白部分

fig4, ax4 = plt.subplots(nrows=3,ncols=1, sharex=True)# 创建图表4
listF=[listFy4,listFz4,listFx4]
F=[Fy4,Fz4,Fx4]
for row in ax4:
    row.plot(range(numLine), listF[i],label="Theoretical data")
    row.plot(range(numLine),     F[i],label="Prediction data")
    row.set_title(title[i])
    row.set_xlabel('Item')
    row.set_ylabel('Force(N)')
    row.grid()
    i=i+1
    if i>=3:
        i=0
        break
plt.tight_layout()
fig4.savefig('Fitting curve of Y- direction force prediction.png',dpi=500,bbox_inches='tight')#bbox_inches检出当前图标周围的空白部分

fig5, ax5 = plt.subplots(nrows=3,ncols=1, sharex=True)# 创建图表5
listF=[listFy5,listFz5,listFx5]
F=[Fy5,Fz5,Fx5]
for row in ax5:
    row.plot(range(numLine), listF[i],label="Theoretical data")
    row.plot(range(numLine),     F[i],label="Prediction data")
    row.set_title(title[i])
    row.set_xlabel('Item')
    row.set_ylabel('Force(N)')
    row.grid()
    i=i+1
    if i>=3:
        i=0
        break
plt.tight_layout()
fig5.savefig('Fitting curve of Z+ direction force prediction.png',dpi=500,bbox_inches='tight')#bbox_inches检出当前图标周围的空白部分

'''plt.legend(loc='best')
leg = plt.gca().get_legend()
ltext  = leg.get_texts()
plt.setp(ltext, fontsize='small')#设置图例大小'''
plt.show()
plt.close()


