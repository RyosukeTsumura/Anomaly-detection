import numpy as np
from numpy import arange, sin, pi
import pandas as pd
import os







batch_size = 128#データ数を割り切れる最大の自然数にしている

#model_parameter
#LL = 100 #length_learn予測に使う学習データの個数:T
LL = batch_size
TP = LL + 1 #起点から学習を行う地点までの時間time_predict:⊿t(>=T)
min_read_row = 0#CSVでどの列から読み始めるか

train_position = ['./excel/train/read','train',0]
validation_position  = ['./excel/validation/read','validation',0]
check_position = ['./excel/check/read','check',2]
test_mode0 = ['./excel/test','test0',0]
test_mode1 = ['./excel/test','test1',1]


model_read_position = './model/read'

model_path_cl = "./learn_model_cl.h5"
model_path_re = "./learn_model_re.h5"
read_mode = 0
save_mode = 0#1なら処理したcsvを保存
csv_nan_check = 0#1ならcsv内のnanをチェック


def get_array(csv_position):
    print('\n' + csv_position[1])
    files = os.listdir(csv_position[0])
    read_mode = csv_position[2]
    X = []
    y = []
    csv_lengths = []

    for i in range(len(files)):#フォルダ内のファイル数だけ繰り返す
        
        print(files[i])
        #csvからデータを抽出
        if(read_mode == 0):
            temp_X = pd.read_csv(csv_position[0] + '/' + files[i],usecols=['Fz'])
            temp_y = pd.read_csv(csv_position[0] + '/' + files[i],usecols=['normal_state','abnormal_state1','abnormal_state2','abnormal_state3'])

        elif(read_mode == 1):
            temp_X = pd.read_csv(csv_position[0] + '/' + files[i],usecols=['Fz'])
            temp_y = pd.read_csv(csv_position[0] + '/' + files[i],usecols=['Fz'])               
            
        elif(read_mode == 2):
            temp_X = pd.read_csv(csv_position[0] + '/' + files[i],usecols=['Fz'])
            temp_y = pd.read_csv(csv_position[0] + '/' + files[i],usecols=['normal_state','abnormal_state1','abnormal_state2','abnormal_state3'])
            real_force = temp_X.values
            np.savetxt('realforce.csv',real_force[TP:,:],delimiter=',')

        
        if(csv_nan_check == 1):
            #print('csv_checking:',i)
            print(temp_X.isnull().any())
            print(temp_y.isnull().any())            

        temp_X = temp_X.values#ndarrayに変換
        temp_y = temp_y.values#ndarrayに変換
      
        for j in range(temp_X.shape[0] - TP):#ファイル内で畳み込むための行列に分割するループ
            X.append(temp_X[j:j+LL,:])

        for j in range(temp_y.shape[0] - TP):#ファイル内で畳み込むための行列に分割するループ
            y.append(temp_y[j+TP,:])

        csv_lengths.append(temp_X.shape[0])
        
    # 最小のCSVの長さに合わせて後ろを捨てる
    for i in range(len(X)):
        X[i] = X[i][:min(csv_lengths)]
        y[i] = y[i][:min(csv_lengths)]
    
    X = np.array(X)#ndarrayに変換
    y = np.array(y)#ndarrayに変換
    X = np.abs(X)#絶対値を取る
    y = np.abs(y)

    X_dim = X.shape[2]
    y_dim = y.shape[1]
        
    print('X_shape:',X.shape)
    print('y_shape:',y.shape)   

    return X,y,X_dim,y_dim

print('model_mode:check_csv')
csv_nan_check = 1
X_train, y_train, indim_train, outdim_train = get_array(train_position)
X_validation, y_validation, indim_validation, outdim_validation = get_array(validation_position)
X_check, y_check, indim_check, outdim_check = get_array(check_position)

