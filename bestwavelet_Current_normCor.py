import os 
import sys 
from pathlib import Path 
from datetime import datetime
from itertools import combinations 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.feature_selection import SelectKBest, f_regression

from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout, Input, Activation, Flatten, TimeDistributed, RepeatVector
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import Model 
from tensorflow.keras import Sequential
import tensorflow as tf 

#訓練時不使用GPU，因為使用GPU 訓練的速度和使用CPU 訓練的速度差不多，
#而且CPU 可以同時訓練數個模型

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


#把數據匯入的程式
def load_data(filepwd):
	df = pd.read_csv(filepwd, header = [0])
	return df 

#最大最小值正規化的程式
def CJ_minmaxfunV3_reg(train_data, train_label, test_data, test_label):

	column_name = train_data.columns

	MinMax = MinMaxScaler()

	MinMax.fit(train_data)

	train_data_minmax = MinMax.transform(train_data)
	test_data_minmax = MinMax.transform(test_data)

	train_data_minmax_pd = pd.DataFrame(train_data_minmax, columns = column_name)
	test_data_minmax_pd = pd.DataFrame(test_data_minmax, columns = column_name)

	label_column_name = train_label.name

	label_minmax = MinMaxScaler()

	train_label_numpy = train_label.to_numpy()
	test_label_numpy = test_label.to_numpy()

	train_label_reshape = np.reshape(train_label_numpy, (len(train_label_numpy), 1))
	test_label_reshape = np.reshape(test_label_numpy, (len(test_label_numpy), 1))

	label_minmax.fit(train_label_reshape)

	train_label_minmax = label_minmax.transform(train_label_reshape)
	test_label_minmax = label_minmax.transform(test_label_reshape)

	train_label_minmax_pd = pd.Series(train_label_minmax.flatten(), name = label_column_name)
	test_label_minmax_pd = pd.Series(test_label_minmax.flatten(), name = label_column_name)

	return train_data_minmax_pd, train_label_minmax_pd, test_data_minmax_pd, test_label_minmax_pd, label_minmax

#反正規化的程式
def remove_ScalerFunction(predict_label_scaler, Scaler):

	predict_label_scaler_numpy = predict_label_scaler

	predict_label_scaler_reshape = np.reshape(predict_label_scaler_numpy, (len(predict_label_scaler_numpy), 1))

	predict_label_reshape = Scaler.inverse_transform(predict_label_scaler_reshape)

	predict_label_reshape_pd = pd.Series(predict_label_reshape.flatten())

	return predict_label_reshape_pd

#將沒有常數特徵移除
def remove_constant(train_data, test_data):
	
	remove_constact_engine = VarianceThreshold(threshold = 0)
	remove_constact_engine.fit(train_data)
	contant = train_data.columns[~remove_constact_engine.get_support()]
	train_data.drop(labels = contant, axis = 1, inplace = True)
	test_data.drop(labels = contant, axis = 1, inplace = True)

	return train_data, test_data

#將非數值的特徵移除
def remove_object(train_data, test_data):

	train_data = train_data.select_dtypes(exclude = ['object'])
	use_columns = train_data.columns
	test_data_col = test_data[use_columns]
	test_data = test_data_col.copy()

	return train_data, test_data

#將標準差小的特徵移除
def remove_QC(train_data, test_data, threshold_value):
	
	remove_constact_engine = VarianceThreshold(threshold = threshold_value)
	remove_constact_engine.fit(train_data)
	contant = train_data.columns[~remove_constact_engine.get_support()]
	train_data.drop(labels = contant, axis = 1, inplace = True)
	test_data.drop(labels = contant, axis = 1, inplace = True)

	return train_data, test_data

#對資料進行KFold 的切割
def CJ_kfold(use_kf_number, data, group_number):

	kf_data = KFold(n_splits = group_number, shuffle = False)
	kf_data.get_n_splits(data)

	kf_number = 1
	for train_index, test_index in kf_data.split(data):

		if kf_number == use_kf_number:
			break

		kf_number = kf_number+1

	return train_index, test_index

def CJ_random_kfold(use_kf_number, data, group_number):
	kf_data = KFold(n_splits = group_number, shuffle = True)
	kf_data.get_n_splits(data)

	kf_number = 1
	for train_index, test_index in kf_data.split(data):
		if kf_number == use_kf_number:
			break

		kf_number = kf_number + 1

	return train_index, test_index

#創建一個CSV的文件，自由命名
def creatCsvlog4(run):
	csvFilePWD =  str(run) +".csv"
	with open(csvFilePWD, 'w') as f:
		f.close()
	print("creat csv:csvFilePWD3")
	return csvFilePWD

#將資料寫入CSV 文件中
def logwrite(write, filepwd):
	with open(filepwd, 'a') as f:
		f.write(write)
		f.write('\n')
		f.close()

def Adjust_r2_LSTM(test_value, predict_value, train_x):
	r2 = r2_score(test_value, predict_value)
	datapoint = len(test_value)
	number_feature = train_x.shape[2]
	abj_r2 = 1 - ((1 - r2)*(datapoint - 1))/(datapoint - number_feature - 1)
	#print(r2)
	#print(datapoint)
	#print(number_feature)
	return abj_r2

#這裡計算模型的MSE MAE MAPE R2 和 adjr2 的分數
def model_performaceV2(test_value_minmax, predict_value_minmax, Scaler, train_x):

	test_value = remove_ScalerFunction(test_value_minmax, Scaler)
	predict_value = remove_ScalerFunction(predict_value_minmax, Scaler)

	mse = mean_squared_error(test_value, predict_value)
	mae = mean_absolute_error(test_value, predict_value)
	mape = mean_absolute_percentage_error(test_value, predict_value)
	r2 = r2_score(test_value, predict_value)
	abjr2 = Adjust_r2_LSTM(test_value, predict_value, train_x)

	return mse, mae, mape, r2, abjr2

#這裡將相關度高於threshold 的特徵移除
def correlation(dataset, threshold):

    col_corr = set()

    corr_matrix = dataset.corr()

    for i in range(len(corr_matrix.columns)):
        
        for j in range(i):
        
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                
                #print(abs(corr_matrix.iloc[i, j]), corr_matrix.columns[i], corr_matrix.columns[j])
                colname = corr_matrix.columns[j]
                # and add it to our correlated set
                col_corr.add(colname)
                
    return col_corr

#這裡將相關度高的特徵移除
def remove_cor(train_data, test_data, threshold):

    corr_feature = correlation(train_data, threshold)

    run_train = train_data.copy()
    run_test = test_data.copy()

    run_train.drop(labels = corr_feature, axis = 1, inplace = True)
    run_test.drop(labels = corr_feature, axis = 1, inplace = True)

    return run_train, run_test

#這裡將數據創建成LSTM 3維時間序列的程式
def LSTM_Data(data, label, time_step, menberOfGroup, numberofG):
	#print("input data shape:" + str(data.shape))
	#print("input label shape:" + str(label.shape))
	#print("time step:" + str(time_step))
	data = data.to_numpy()
	data_reshape = data.reshape(numberofG, menberOfGroup, data.shape[1])
	#print(data_reshape.shape)
	
	for a in range(numberofG):
		for b in range(menberOfGroup - time_step + 1):
			if b == 0:
				input_d = data_reshape[a,b:b+time_step]
				continue
			input_d = np.append(input_d, data_reshape[a, b:b+time_step])
		
		input_d = input_d.reshape(menberOfGroup - time_step + 1, time_step, data.shape[1])
		if a == 0:
			input_data_all = input_d
			continue
		input_data_all = np.append(input_data_all, input_d)
	
	input_data_all_reshape = input_data_all.reshape((menberOfGroup - time_step + 1)*numberofG, time_step, data.shape[1])
	
	#print("LSTM input data shape: " + str(input_data_all_reshape.shape))

	label = label.to_numpy()
	label_reshape = label.reshape(numberofG,menberOfGroup,1)
	#print(label.shape)
	#print(label_reshape.shape)
	label_input = []
	for c in range(numberofG):
		for d in range(menberOfGroup - time_step + 1):
			if d == 0:
				label_input = label_reshape[c,d+time_step-1]
				continue
			label_input = np.append(label_input, label_reshape[c, d + time_step - 1])

		if c == 0:
			all_label = label_input
			continue

		all_label = np.append(all_label, label_input)
	#print(all_label.shape)

	return input_data_all_reshape, all_label

#這裡建立模型
def buildManyToOneModel_model9(shape):
    model = Sequential()
    model.add(LSTM(100, input_length=shape[1], input_dim = shape[2]))
    #model.add(GRU(250, input_length=shape[1], input_dim = shape[2]))
    model.add(Dropout(0.2))
    #model.add(Dense(100))
    #model.add(Dropout(0.25))
    #model.add(Dense(50))
    #model.add(Dropout(0.05))
    model.add(Dense(1))
    model.compile(loss = "mse", optimizer = "adam")
    model.summary()
    return model

def buildManyToOneModel_with_v2(shape, lr_in, lstm_node_1, lstm_node_2, dropout_number_1, dropout_number_2):
	model = Sequential()
	model.add(LSTM(lstm_node_1, return_sequences = True, input_length = shape[1], input_dim = shape[2]))
	model.add(Dropout(dropout_number_1))
	model.add(LSTM(lstm_node_2))
	model.add(Dropout(dropout_number_2))
	model.add(Dense(1))
	opt = Adam(learning_rate = lr_in)
	model.compile(loss = "mse", optimizer = opt)
	model.summary()
	return model

def LSTM_dataV2(data, label, gnumber):

	data_numpy = data.to_numpy()
	reshape_data = data_numpy.reshape(len(label), gnumber, data.shape[1])

	return reshape_data

def lstm_data_to_2d(LSTM_data_3D):
	lstm_data_2d = LSTM_data_3D.reshape(LSTM_data_3D.shape[0], LSTM_data_3D.shape[1]*LSTM_data_3D.shape[2])
	return lstm_data_2d

def lstm_data_invers_3d(LSTM_data_3D, LSTM_data_2D):
	re_lstm_data_3d = LSTM_data_2D.reshape(LSTM_data_2D.shape[0], LSTM_data_3D.shape[1], LSTM_data_3D.shape[2])
	return re_lstm_data_3d


#這裡定義要使用那個window 的空間
windowlist =  ['haar', 'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10', 'db11', 'db12', 'db13', \
'db14', 'db15', 'db16', 'db17' ,'db18', 'db19', 'db20', 'db21', 'db22', 'db23','db24', 'db25', 'db26', 'db27', 'db28',\
'db29','db30','db31','db32','db33','db34','db35','db36','db37','db38', 'sym2','sym3','sym4','sym5','sym6','sym7','sym8', \
'sym9','sym10','sym11','sym12','sym13','sym14','sym15','sym16','sym17','sym18','sym19','sym20', 'coif1','coif2','coif3','coif4',\
'coif5']
#windowlist =  ['coif6','coif7','coif8','coif9','coif10','coif11','coif12','coif13','coif14','coif15','coif16','coif17','bior1.1','bior1.3',\
#'bior1.5','bior2.2','bior2.4','bior2.6','bior2.8','bior3.1','bior3.3','bior3.5','bior3.7','bior3.9','bior4.4','bior5.5','bior6.8',\
#'rbio1.1','rbio1.3','rbio1.5','rbio2.2','rbio2.4','rbio2.6','rbio2.8','rbio3.1','rbio3.3','rbio3.5','rbio3.7','rbio3.9','rbio4.4','rbio5.5','rbio6.8']

print(len(windowlist))

#這裡確保要測試的wavelet 都有
for win in windowlist:

	staticFilepwd = "./waveletStatic/wlStatic-" + str(win) + "-TorAccCur-ABCDEFHI.csv"
	print(staticFilepwd)
	df = load_data(staticFilepwd)
	df_2 = load_data("./wlTorAccCurstatic4-ABCDEFHI-Ra.csv")

lr_in = 0.001
lstm_node_1_in = 20
dropout_number_1_in = 0.1
lstm_node_2_in = 10
dropout_number_2_in = 0.1
batch_in = 9
epoc_in = 300


for win in windowlist:

	pwd = creatCsvlog4("./log-result/"+str(win))
	logwrite("kf,mse,mae,mape,r2", pwd)

	staticFilepwd = "./waveletStatic/wlStatic-" + str(win) + "-TorAccCur-ABCDEFHI.csv"

	print(staticFilepwd)

	df = load_data(staticFilepwd)
	df_2 = load_data("./wlTorAccCurstatic4-ABCDEFHI-Ra.csv")

	data = df.copy()

	#這裡將加速規 3 軸的特徵和機械手臂扭矩的特徵進行移除只保留研磨機電流的訊號特徵
	axlist = ["acc1","acc2","acc3", "ax1", "ax2", "ax3", "ax4", "ax5", "ax6"]
	wtname = ["or", "wa", "wd", "waa", "wad", "wda", "wdd"]
	staticlist = ["sg_mean", "sg_std", "sg_var", "sg_skew", "sg_kur", "sg_max", "sg_min", "sg_mad", "sg_iqr", "sg_rms", "sg_n5", "sg_n25", "sg_n50", "sg_n75", "sg_n95"]
	colname = []
	for a in axlist:
	#   print(a)
	    for b in wtname:
	#       print(b)
	        for c in staticlist:
	#           print(a + "-" + b +"-" + c)
	            data.pop(a + "-" + b +"-" + c)

	label = df_2.pop('Ra')
	callback = EarlyStopping(monitor = "loss", patience = 100, verbose = 1, mode = "auto")
	for a in range(1,9):

		train_index, test_index = CJ_kfold(a, data, 8)
		train_index_label, test_index_label = CJ_kfold(a, label, 8)

		train_data = data.loc[train_index, ]
		train_label = label.loc[train_index_label, ]

		test_data = data.loc[test_index, ]
		test_label = label.loc[test_index, ]

		train_data_minmax, train_label_minmax, \
		test_data_minmax, test_label_minmax, \
		minmax_scaler = CJ_minmaxfunV3_reg(train_data, train_label, test_data, test_label)

		#train_data_lstm = LSTM_dataV2(train_data_minmax, train_label_minmax, 6)
		#test_data_lstm = LSTM_dataV2(test_data_minmax, test_label_minmax, 6)

		train_data_minmax, test_data_minmax = remove_QC(train_data_minmax, test_data_minmax, 0.01)

		#train_data_minmax, test_data_minmax = remove_cor(train_data_minmax, test_data_minmax, 0.8)

		train_data_lstm, train_label_lstm = LSTM_Data(train_data_minmax, train_label_minmax, 3, 25, 7)

		test_data_lstm, test_label_lstm = LSTM_Data(test_data_minmax, test_label_minmax, 3, 25, 1)

		train_data_lstm_2d = lstm_data_to_2d(train_data_lstm)

		for b in range(5):

			train_index_model, valid_index_model = CJ_random_kfold(b, train_data_lstm_2d, 5)

			train_data_lstm_model_2d = train_data_lstm_2d[train_index_model]
			train_label_lstm_model_2d = train_label_lstm[train_index_model]

			valid_data_lstm_model_2d = train_data_lstm_2d[valid_index_model]
			valid_label_lstm_model_2d = train_label_lstm[valid_index_model]

			print(train_data_lstm_model_2d.shape)
			print(train_label_lstm_model_2d.shape)

			train_data_lstm_model_3d = lstm_data_invers_3d(train_data_lstm, train_data_lstm_model_2d)
			valid_data_lstm_model_3d = lstm_data_invers_3d(train_data_lstm, valid_data_lstm_model_2d)

			print(train_data_lstm_model_3d.shape)
			print(valid_data_lstm_model_3d.shape)

			model = buildManyToOneModel_with_v2(train_data_lstm_model_3d.shape, lr_in, lstm_node_1_in, lstm_node_2_in, dropout_number_1_in, dropout_number_2_in)

			model.fit(train_data_lstm_model_3d, train_label_lstm_model_2d, epochs = epoc_in, batch_size = batch_in, validation_data = (valid_data_lstm_model_3d, valid_label_lstm_model_2d))

			predict_value = model.predict(test_data_lstm)

			mse, mae, mape, r2, adjr2 = model_performaceV2(test_label_lstm, predict_value, minmax_scaler, train_data_lstm_model_3d)

			logwrite(str(a) + ',' + str(mse) + ',' + str(mae) + ',' + str(mape) + ',' + str(r2), pwd)
