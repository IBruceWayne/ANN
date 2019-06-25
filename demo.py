import random
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
from numpy.matlib import repmat
import time

def get_circle_point(x, y, r, point_num):
	'''
	(x, y) 圆心
	r 半径
	point_num 随机点的个数
	'''
	p_list = []
	p0_min = x - r
	p0_max = x + r
	p1_min = y - r
	p1_max = y + r
	while True:
		if len(p_list) == int(point_num):
			return p_list
		p0 = random.uniform(p0_min, p0_max)
		p1 = random.uniform(p1_min, p1_max)
		if (p0 - x)**2 + (p1 - y)**2 <= r**2:
			p_list.append([p0, p1])


def get_arc_point(x, y, r1, r2, point_num):
	'''
	(x, y) 圆心
	r1 小半径 r2 大半径
	point_num 随机点的个数
	'''
	p_list = []
	p0_min = x - r2
	p0_max = x + r2
	p1_min = y - r2
	p1_max = y + r2
	r1 = r1 + 0.5
	while True:
		if len(p_list) == int(point_num):
			return p_list
		p0 = random.uniform(p0_min, p0_max)
		p1 = random.uniform(p1_min, p1_max)
		if (p0 - x)**2 + (p1 - y)**2 <= r2**2 and (p0 - x)**2 + (p1 - y)**2 > r1**2:
			p_list.append([p0, p1])

def g(x,deriv=False):
	# sigmoid激活函数
    '''
    a = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    if (deriv==True):
        return 1-np.multiply(a,a)
    return a
'''    
    z=1.0/(1+np.exp(-x))
    if(deriv==True):
        return np.multiply(z,(1-z))#sigmod导数
    return z#sigmod函数


def norm(x):
	mean = np.mean(x, axis=-1)
	std = np.std(x, axis=-1)
	x = x - mean
	x = x/std
	return x


def ANN(alpha,iters,x,y):
	# x shape: (feature, number)
	# y shape: (number, 1)
    n,m = x.shape
    num = 3 #隐藏层神经元的个数
    w1 = np.random.random((num,2))*0.001 #(4*2)
    b1 = np.random.random((num,1))*0.001
    w2 = np.random.random((1,num))*0.001
    b2 = np.random.random((1,1))*0.001
    #向量化实现：横向堆叠，横向代表样本，竖向代表节点
    for iter in range(iters):
        #正向传播四个公式
        z1 = w1*x+repmat(b1,1,x.shape[1])
        a1 = g(z1)
        z2 = w2*a1+repmat(b2,1,x.shape[1])
        a2 = g(z2) #预测值
        if (iter+1)%10000 == 0:
        	loss = np.sum(a2)/m
        	print('in step {}, loss is {}'.format(iter+1, loss))
        #反向传播六个公式
        dz2 = a2 - y.T#1,80
        dw2 = 1/m*dz2*a1.T#1,2
        db2 = 1/m*np.sum(dz2,axis=1)#1,1
        dz1 = np.multiply(w2.T*dz2,g(z1,True))
        dw1 = 1/m*dz1*x.T
        db1 = 1/m*np.sum(dz1,axis=1)
        #更新梯度
        w1 = w1 - alpha*dw1
        b1 = b1 - alpha*db1
        w2 = w2 - alpha*dw2
        b2 = b2 - alpha*db2
    return w1,b1,w2,b2


def test(p, x, y):
	w1,b1,w2,b2 = p
	z1 = w1*x+repmat(b1,1,x.shape[1])
	a1 = g(z1)
	z2 = w2*a1+repmat(b2,1,x.shape[1])
	a2 = g(z2) #预测值
	pre = a2.tolist()[0]
	y = y.T.tolist()[0]
	count = 0
	for i in range(len(pre)):
		if pre[i]>=0.5 and y[i] == 1:
			count+=1
		elif pre[i]<0.5 and y[i] == 0:
			count+=1
	print('准确率：', count/len(y))
	return count/len(y)


def get_mm_point(point, w1, b1, w2=None, b2=None):
	# point: list of list [[p1, p2], [p1, p2], ...]
	# w :weights
	# b :bias
	new_p = g(np.mat(w1)*np.mat(point).T+b1) # [out_dimen, point_number] 
	if w2 != None:
		new_p = g(np.mat(w2)*new_p.T+b2)
	return new_p



def main():
	start = time.time()
	num = 400  # 生成样本点的个数。
	# 点太多的话，神经网络拟合的会比较慢（因为自己用numpy写的神经网络，没有进行数据归一化、改进优化器等）
	point1 = get_circle_point(5, 10, 2, num)  # 圆里的数据点
	point2 = get_arc_point(5, 10, 2, 5,num)  # 圆弧里的数据点

	fig=plt.figure()
	ax1=fig.add_subplot(2,2,1)
	ax2=fig.add_subplot(2,2,2, projection='3d')
	ax3=fig.add_subplot(2,1,2, projection='3d')

	# 画变换前,二维数据分布
	x1 = [i for i, j in point1]
	y1 = [j for i, j in point1]
	x2 = [i for i, j in point2]
	y2 = [j for i, j in point2]

	ax1.scatter(x1, y1, c='r', label='origin')
	ax1.scatter(x2, y2, c='b')
	ax1.legend()
	# 线性变换
	w = [[1, 2], [2, 3], [4, 1]]  # 随机权重
	b = [[1], [2], [3]]

	w2 = [[-0.43149021, -1.57661212],[ 1.12721405, -1.31048343],[-1.90766324, -0.53304404]]  # 两层神经网络,经过训练后得到的第一层权重（第一个线性变换，维度为2）
	b2 = [[20.07284971],[ 4.98916593],[11.97648772]]

	w3 = [[ 4.21699551, -2.29536064],[ 4.8615568 , -0.87519498]]  # # 两层神经网络,经过训练后得到的第一层权重（第一个线性变换，维度为3）
	b3 = [[-8.44099007],[-5.1810846 ]]

	n_p1 = get_mm_point(point1, w2, b2) # [3, 400] 
	n1_x = n_p1[0,:].tolist()[0]
	n1_y = n_p1[1,:].tolist()[0]
	n1_z = n_p1[2,:].tolist()[0]

	n_p2 = get_mm_point(point2, w2, b2)  # [3, 400]
	n2_x = n_p2[0,:].tolist()[0]
	n2_y = n_p2[1,:].tolist()[0]
	n2_z = n_p2[2,:].tolist()[0]

	ax2.scatter(n1_x, n1_y, n1_z, c='r', label='default nn')
	ax2.scatter(n2_x, n2_y, n1_z, c='b')
	ax2.legend()
	# 神经网络
	x = []  # 存放所有数据点
	x.extend(point1)
	x.extend(point2)
	y = [0 if i<num else 1 for i in range(len(x))]  # 构造标签
	x = np.mat(x).T
	y = np.mat(y).T
	# 训练神经网络，返回参数
	parameters = ANN(0.001, 10000000, x, y)  
	test(parameters, x, y)  # 测试正确率
	print(parameters)
	w4,b4,_,_ = parameters
	print(parameters)
	n_p1 = get_mm_point(point1, w4, b4) # [3, 400] 
	n1_x = n_p1[0,:].tolist()[0]
	n1_y = n_p1[1,:].tolist()[0]
	n1_z = n_p1[2,:].tolist()[0]

	n_p2 = g(np.mat(w4)*np.mat(point2).T+b4)  # [3, 400]
	n2_x = n_p2[0,:].tolist()[0]
	n2_y = n_p2[1,:].tolist()[0]
	n2_z = n_p2[2,:].tolist()[0]
	ax3.scatter(n1_x, n1_y, n1_z, c='r', label='trained nn')
	ax3.scatter(n2_x, n2_y, n2_z, c='b')  # 如果第三幅图效果不好，可以增加ann迭代次数
	ax3.legend()

	end = time.time()
	print('time cost(s):', end - start)
	plt.show()

main()

