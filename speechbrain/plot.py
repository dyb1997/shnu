import matplotlib.pyplot as plt
import numpy as np

file = open('/mnt/data3/dyb/ffsvc/train_utt.txt')  #打开文档
data = file.readlines() #读取文档数据
para_1 = []  #新建列表，用于保存第一列数据
para_2 = []  #新建列表，用于保存第二列数据
i = 0
a=0
for num in data:
	# split用于将每一行数据用逗号分割成多个对象
    #取分割后的第0列，转换成float格式后添加到para_1列表中
    i=i+1
    para_1.append(float(i))
    #取分割后的第1列，转换成float格式后添加到para_1列表中
    para_2.append(float(num.split(':')[1]))
    a+=float(num.split(':')[1])
plt.figure()

max_indx=np.argmax(para_2)#max value index
min_indx=np.argmin(para_2)#min value index
plt.plot(max_indx,para_2[max_indx],'ks')
show_max='['+str(max_indx)+' '+str(para_2[max_indx])+']'
plt.annotate(show_max,xytext=(max_indx,para_2[max_indx]),xy=(max_indx,para_2[max_indx]))
plt.plot(min_indx,para_2[min_indx],'gs')
show_min='['+str(min_indx)+' '+str(para_2[min_indx])+']'
plt.annotate(show_min,xytext=(min_indx,para_2[min_indx]),xy=(min_indx,para_2[min_indx]))
plt.title('FFSVC2020DEV')
plt.plot(para_1, para_2)
plt.xlabel("spk-num")#横坐标名字
plt.ylabel("utt-num")#纵坐标名字
plt.legend(loc = "best")#图例
plt.show()