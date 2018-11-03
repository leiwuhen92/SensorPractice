import matplotlib.pyplot as plt

item=['X+direction force','X-direction force','Y+direction force','Y-direction force','Z+direction force']
i=0
fig, big_axes = plt.subplots(figsize=(15.0, 15.0) , nrows=2, ncols=1, sharex=True) 

for row, big_ax in enumerate(big_axes):
    big_ax.set_title("%s \n\n" % item[i], fontsize=16)
    i+=1
 

    # Turn off axis lines and ticks of the big subplot 
    # obs alpha is 0 in RGBA string!
    big_ax.tick_params(labelcolor=(0,0,0,0), top='off', bottom='off', left='off', right='off')
    # removes the white frame
    big_ax._frameon = False

for i, j in enumerate(['Fy','Fz','Fx','Fy','Fz','Fx']):
    ax = fig.add_subplot(2,3,i+1)
    ax.set_title(j)




#����ͼ������Ե�Լ�ͼ���Ŀհ�����
plt.subplots_adjust(left=0.2, bottom=0, right=0.8, top=1.0, hspace=0.2, wspace=0.3)
fig.set_facecolor('w')
plt.tight_layout()
plt.savefig('X����������������Ԥ�����.png',dpi=500,bbox_inches='tight')#bbox_inches�����ǰͼ����Χ�Ŀհײ���
plt.show()    
