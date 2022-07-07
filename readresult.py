import os 
import csv
from pathlib import Path
import numpy as np  

def readresult(filePWD):

	readmse = []
	readmae = []
	readmape = []
	readr2 = [] 
	readadjr2 = []

	with open(filePWD, newline = '') as csvfile:
		rows = csv.DictReader(csvfile)
		for row in rows:
			readmse.append(float(row['mse']))
			readmae.append(float(row['mae']))
			readmape.append(float(row['mape']))
			readr2.append(float(row['r2']))

	return readmse, readmae, readmape, readr2

windowlist = ['haar', 'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10', 'db11', 'db12', 'db13', \
'db14', 'db15', 'db16', 'db17' ,'db18', 'sym2','sym3','sym4','sym5','sym6','sym7','sym8', \
'sym9','sym10','sym11','sym12','sym13','sym14','sym15','sym16','sym17','sym18', 'coif1','coif2','coif3','coif4',\
'coif5','coif6','bior1.1','bior1.3','bior1.5','bior2.2','bior2.4','bior2.6','bior2.8','bior3.1','bior3.3','bior3.5','bior3.7','bior3.9','bior4.4','bior5.5','bior6.8',\
'rbio1.1','rbio1.3','rbio1.5','rbio2.2','rbio2.4','rbio2.6','rbio2.8','rbio3.1','rbio3.3','rbio3.5','rbio3.7','rbio3.9','rbio4.4','rbio5.5','rbio6.8']

def creatCsvlog3(run):
    csvFilePWD = "./" + str(Path(__file__).stem) + str(run) +".csv"
    with open(csvFilePWD, 'w') as f:
        f.close()
    print("creat csv:csvFilePWD3")
    return csvFilePWD

def logwrite(write, filepwd):
    with open(filepwd, 'a') as f:
        f.write(write)
        f.write('\n')
        f.close()

pwd = creatCsvlog3('abc')
logwrite("win, avgmse,stdmse,avgmae,stdmae,avgmape,stdmape,avgr2,stdr2", pwd)

for win in windowlist:

	re_pwd = "./log-result/" + str(win) + ".csv"
	print(re_pwd)
	mse, mae, mape, r2 = readresult(re_pwd)

	avgmse = np.mean(mse)
	avgmae = np.mean(mae)
	avgmape = np.mean(mape)
	avgr2 = np.mean(r2)

	stdmse = np.std(mse)
	stdmae = np.std(mae)
	stdmape = np.std(mape)
	stdr2 = np.std(r2)

	logwrite(str(win) + "," + str(avgmse) + "," + str(stdmse) + "," + str(avgmae) + "," + str(stdmae) + "," + str(avgmape) + "," + str(stdmape) + "," + str(avgr2) + "," + str(stdr2), pwd)

















