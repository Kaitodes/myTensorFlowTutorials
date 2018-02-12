#!usr/bin/env python
#-*- coding:utf-8 -*-
import os  
import numpy as np  
import tensorflow as tf  
import input_data     
import model  
import argparse
  
N_CLASSES = 2  # 2个输出神经元，［1，0］ 或者 ［0，1］猫和狗的概率
IMG_W = 208  # 重新定义图片的大小，图片如果过大则训练比较慢  
IMG_H = 208  
BATCH_SIZE = 4  #每批数据的大小
CAPACITY = 256  
MAX_STEP = 500 # 训练的步数，应当 >= 10000
learning_rate = 0.001 # 学习率，建议刚开始的 learning_rate <= 0.0001
  

def run_training():  
    #get current path
    path = os.getcwd()
    #get upper path
    # os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
	# 数据集
	train_dir = path + '/imgs/'
	if os.path.exists('saveNet') == False:
		os.mkdir('saveNet')
	#logs_train_dir 存放训练模型的过程的数据，在tensorboard 中查看 
	logs_train_dir = path + '/saveNet/'

	# 获取图片和标签集
	print 'train_dir', train_dir
	train, train_label = input_data.get_files(train_dir)
	# 生成批次
	train_batch, train_label_batch = input_data.get_batch(train,  
														  train_label,  
														  IMG_W,  
														  IMG_H,  
														  BATCH_SIZE,   
														  CAPACITY)
	# 进入模型
	train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES) 
	# 获取 loss 
	train_loss = model.losses(train_logits, train_label_batch)
	# 训练 
	train_op = model.trainning(train_loss, learning_rate)
	# 获取准确率 
	train__acc = model.evaluation(train_logits, train_label_batch)  
	# 合并 summary
	summary_op = tf.summary.merge_all()  

	gpu_options = tf.GPUOptions()
	config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement = True,allow_soft_placement = False)
	sess = tf.Session(config=config)
	# 保存summary
	train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)  
	saver = tf.train.Saver()  
	  
	sess.run(tf.global_variables_initializer())  
	coord = tf.train.Coordinator()  
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)  
      
	try:  
		for step in np.arange(MAX_STEP):  
			if coord.should_stop():  
					break  
			_, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])  
				 
			if step % 50 == 0:  
				print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))  
				summary_str = sess.run(summary_op)  
				train_writer.add_summary(summary_str, step)  
			  
			if step % 250 == 0 or (step + 1) == MAX_STEP:  
				# 每隔2000步保存一下模型，模型保存在 checkpoint_path 中
				print '保存模型'
				checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')  
				saver.save(sess, checkpoint_path, global_step=step)  
				  
	except tf.errors.OutOfRangeError:  
		print('Done training -- epoch limit reached')  
	finally:  
		coord.request_stop()
	coord.join(threads)  
	sess.close()  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default="YOLO_small.ckpt", type=str)
    parser.add_argument('--data_dir', default="data", type=str)
    parser.add_argument('--threshold', default=0.2, type=float)
    parser.add_argument('--iou_threshold', default=0.5, type=float)
    parser.add_argument('--gpu', default='0', type=str)
    args = parser.parse_args()

    if args.gpu is not None:
        GPU = args.gpu

    os.environ['CUDA_VISIBLE_DEVICES'] = GPU

    print('Start training ...')
    run_training()
    print('Done training.')

if __name__ == '__main__':

    # python train.py 
    main()
 