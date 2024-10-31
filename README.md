<data>
	<trn_input_proc>
		处理后的UNET输入数据，
		*.npz为压缩的训练集和标签集；训练集为[200,200,14]的s2遥感影像数组；标签集为[200,200]的二值PV标记，0代表非PV，1代表PV
  		records.pickle为pickle打包上述数据的输出，使用pickle.load打开
		
 	<trn_input_raw>
		原始遥感影像数据，大小不确定，为16通道；其中最后两个通道没有被用于训练，每个通道在[200,200]之外的部分被舍去，之内没有数据的部分补0
  
  	<trn_target_raw>
		原始PV标注，同上
  
   	<tst_output>
		UNET的预测输出，为【200,200,2】的数组，第3维，即长度为2的维度上有[0,1]与[1,0]两种情况，代表有无PV的预测
  
<s2>
	<training>
		UNET原始架构
  
 	[deploy_s2.py]
  		UNET进行预测

  	[readtif.py]
   		tif文件的预处理

   	[train_S2_unet.py]
    		UNET的训练  	

其它：网络文件太大，无法上传


2024.10.23
	yangzhichao@master 
		(1)cloned codes of PV
		(2)add testing module at 0MLHW/s22024.10.23
	yangzhichao@master 
		(1)cloned codes of PV
		(2)add testing module at 0MLHW/s2
2024.10.31
	yangzhichao@master
 		(1)add training and testing code to run unet
   		(2)add raw data processing code <readtif.py>
