import matplotlib.pyplot as plt
import numpy as np
from numpy import arange, sin, pi, random
import pandas as pd
import os
import json

from keras.layers import TimeDistributed
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.recurrent import GRU
from keras.models import Sequential
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, TensorBoard, Callback
from keras.utils import plot_model

from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#model_mode = 0
predict_mode = 0
recurrent_mode = 'no_models'#モデルのフラグ管理関数

#batch_size = 32#20190125
batch_size = 128#データ数を割り切れる最大の自然数にしている,デフォルト：128

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
threshold_position  = ['./excel/threshold/read','threshold',0]


model_read_position = './model/read'

n_epoch = '0'
#model_path_cl = "./learn_model_cl.h5"
model_path_cl = "./learn_model_cl"
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

            temp_X = temp_X.dropna()
            temp_y = temp_y.dropna()


        elif(read_mode == 1):
            temp_X = pd.read_csv(csv_position[0] + '/' + files[i],usecols=['Fz'])
            temp_y = pd.read_csv(csv_position[0] + '/' + files[i],usecols=['Fz']) 

            temp_X = temp_X.dropna()
            temp_y = temp_y.dropna()
                       
            
        elif(read_mode == 2):
            temp_X = pd.read_csv(csv_position[0] + '/' + files[i],usecols=['Fz'])
            temp_y = pd.read_csv(csv_position[0] + '/' + files[i],usecols=['normal_state','abnormal_state1','abnormal_state2','abnormal_state3'])
            
            temp_X = temp_X.dropna()
            temp_y = temp_y.dropna()
            
            real_force = temp_X.values
            np.savetxt('realforce.csv',real_force[TP:,:],delimiter=',')

        
        if(csv_nan_check == 1):
            #print('csv_checking:',i)
            
            print(temp_X.isnull().any())
            print(temp_y.isnull().any())
            print(np.isnan(temp_X))
            print(np.isnan(temp_y))            

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
    

    #X = np.array(X)#ndarrayに変換
    #y = np.array(y)#ndarrayに変換

    #print(X.dtype)
    #print(y.dtype)

    X_dim = X.shape[2]
    y_dim = y.shape[1]

    if(save_mode == 1):
        print('save_mode')
        save_y = np.reshape(y,(y.shape[0],y.shape[1]))
        np.savetxt(csv_position[1] + 'y.csv',save_y,delimiter=',')

        save_X0 = np.reshape(X[:,:,0],(X.shape[0],X.shape[1]))
        np.savetxt(csv_position[1] + 'X0.csv',save_X0,delimiter=',')
        
    print('X_shape:',X.shape)
    print('y_shape:',y.shape)   

    return X,y,X_dim,y_dim

def build_model_cl(indim, outdim):#分類学習用モデル

    n_hidden1 = 300

    #indim = 4
    #outdim = 3

    model = Sequential()

    if(recurrent_mode == 'lstm'):
        model.add(LSTM(
        input_length = LL,
        input_dim = indim,
        output_dim = n_hidden1,
        return_sequences=False,
        stateful = False))

    elif(recurrent_mode == 'gru'):
        model.add(GRU(
        input_length = LL,
        input_dim = indim,
        output_dim = n_hidden1,
        return_sequences=False,
        stateful = False))
        
    model.add(Dropout(0.2))
    model.add(Dense(
        output_dim = outdim))
    #model.add(Activation("sigmoid"))
    model.add(Activation('softmax'))#nan問題が未解決
    #optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999,clipnorm = 1)
    #model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer=optimizer)
    model.compile(loss='mse',metrics=['accuracy'],optimizer=optimizer)
    return model

def train(data = None,model = None):
    if data is None:
        print ('Loading data... ')
        X_train, y_train, indim_train, outdim_train = get_array(train_position)
        X_validation, y_validation, indim_validation, outdim_validation = get_array(validation_position)
    else:
        X_train, y_train, X_validation, y_validation = data
        print ('\nData Loaded. Compiling...\n')

    if model is None:
        print("Getting model...")
        model = build_model_cl(indim_train,outdim_train)

    hist = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data = (X_validation, y_validation)
        )
    
    
    
    model.save_weights(model_path_cl)

    loss1 = hist.history['loss']
    val_loss1 = hist.history['val_loss']
    np.savetxt('loss.csv',loss1,delimiter=',')
    np.savetxt('val_loss.csv',val_loss1,delimiter=',')

    acc1 = hist.history['acc']
    val_acc1 = hist.history['val_acc']
                
    np.savetxt('acc.csv',acc1,delimiter=',')
    np.savetxt('val_acc.csv',val_acc1,delimiter=',')

    '''
    loss1 = np.reshape(loss1,(loss1.shape[0],1))
    val_loss1 = np.reshape(val_loss1,(val_loss1.shape[0],1))
    acc1 = np.reshape(acc1,(acc1.shape[0],1))
    val_acc1 = np.reshape(val_acc1,(val_acc1.shape[0],1))
    model_analyze = np.concatenate((loss1,val_loss1), axis = 1)
    model_analyze = np.concatenate((model_analyze,acc1), axis = 1)
    model_analyze = np.concatenate((model_analyze,val_acc1), axis = 1)
    '''

    #np.savetxt('model_analyze.csv',model_analyze,delimiter=',')

def predict(model_position):
    print("\nPredicting sequence...")
    #csv_files = os.listdir(check_position[0])
    #print(model_position)
    model_files = os.listdir(model_position)
    model_predict = []
    #predict_mode = 1
    #plot_color = ['red','yellow','green','blue','orangered','greenyellow','cyan','magenta','marron','lime','darkslategray','midnightblue','purple']

    
    X_check, y_check, indim_check, outdim_check = get_array(check_position)
    model_predict = build_model_cl(indim_check,outdim_check)
    model_predict.load_weights(model_position + '/' + model_files[0], by_name=True)
    print('mode_weight__load _complete')

    seq_predicted = model_predict.predict(X_check)
    print('predict_complete')

    np.savetxt('actual.csv',y_check,delimiter=',')
    np.savetxt('predicted.csv',seq_predicted,delimiter=',')


    print('save_complete')

    return()

def predict_thre(model_position):
    print("\nPredicting sequence...")

    model_files = os.listdir(model_position)
    model_predict = []
    number_of_split = 100
    correct_count = []
    temp_PD = 0 
    temp_count1 = 0
    temp_count2 = 0
    temp_count3 = 0
    temp_count4 = 0
    temp_recall = 0
    beta = 0.5
    
    X_check, y_check, indim_check, outdim_check = get_array(check_position)
    model_predict = build_model_cl(indim_check,outdim_check)
    model_predict.load_weights(model_position + '/' + model_files[0], by_name=True)
    print('mode_weight__load _complete')

    seq_predicted = model_predict.predict(X_check)
    print('predict_complete')

    for i in range(number_of_split):
        threshold = (i+1) / number_of_split
        #seq_predicted[j,0]⇒（予想）正常確率のステータス
        #y_check[j,0]⇒（実際）正常状態のステータス（1⇒正常）
        for j in range(y_check.shape[0]):
            temp_PD = 1 - seq_predicted[j,0]        
            
            if(temp_PD < threshold and y_check[j,0] == 1):#予想⇒正常、実際⇒正常
                
                temp_count1 = temp_count1 + 1#真陰性TN

            elif(temp_PD >= threshold and y_check[j,0] == 0):#予想⇒異常、実際⇒異常
                temp_count2 = temp_count2 + 1#真陽性TP

            elif(temp_PD < threshold and y_check[j,0] == 0):#予想⇒正常、実際⇒異常
                temp_count3 = temp_count3 + 1#偽陰性FN

            elif(temp_PD >= threshold and y_check[j,0] == 1):#予想⇒異常、実際⇒正常
                temp_count4 = temp_count4 + 1#偽陽性FP

        if (temp_count2 == 0 and temp_count3 == 0):
            temp_recall = 0
        else:
            temp_recall = temp_count2/(temp_count2 + temp_count3)
        
        if (temp_count2 == 0 and temp_count4 == 0):
            temp_recall = 0
        else:   
            temp_precision = temp_count2/(temp_count2 + temp_count4)

        if (temp_recall + beta**2*temp_precision == 0):
            F_measure = 0
        else:
            F_measure = (1+beta**2)*temp_recall*temp_precision/(temp_recall + beta**2*temp_precision)        



        correct_count.append([threshold,temp_count1,temp_count2,temp_count3,temp_count4,threshold,temp_recall,temp_precision,F_measure])
        #print(threshold,temp_count)
        temp_count1 = 0
        temp_count2 = 0
        temp_count3 = 0
        temp_count4 = 0
        temp_PD = 0

    correct_count = np.array(correct_count)#ndarrayに変換   

    np.savetxt('actual.csv',y_check,delimiter=',')
    np.savetxt('predicted.csv',seq_predicted,delimiter=',')
    np.savetxt('correct.csv',correct_count,delimiter=',')
  
    print('save_complete')

    return()

def get_array_for_threshold(csv_position):
    print('\n' + csv_position[1])
    files = os.listdir(csv_position[0])

    X_list = []
    y_list = []
    X_dim_list = []
    y_dim_list = []

    for i in range(len(files)):#フォルダ内のファイル数だけ繰り返す
        
        X = []
        y = []
        print(files[i])
        #csvからデータを抽出                
            
        temp_X = pd.read_csv(csv_position[0] + '/' + files[i],usecols=['Fz'])
        temp_y = pd.read_csv(csv_position[0] + '/' + files[i],usecols=['normal_state','abnormal_state1','abnormal_state2','abnormal_state3'])
            
        temp_X = temp_X.dropna()
        temp_y = temp_y.dropna()
            
        temp_X = temp_X.values#ndarrayに変換
        temp_y = temp_y.values#ndarrayに変換
      
        for j in range(temp_X.shape[0] - TP):#ファイル内で畳み込むための行列に分割するループ
            X.append(temp_X[j:j+LL,:])

        for j in range(temp_y.shape[0] - TP):#ファイル内で畳み込むための行列に分割するループ
            y.append(temp_y[j+TP,:])

        X = np.array(X)#ndarrayに変換
        y = np.array(y)#ndarrayに変換

        X = np.abs(X)#絶対値を取る
        y = np.abs(y)
 
        X_dim = X.shape[2]
        y_dim = y.shape[1]

        X_list.append(X)
        y_list.append(y)
        X_dim_list.append(X_dim)
        y_dim_list.append(y_dim)

    return X_list,y_list,X_dim_list,y_dim_list,i

def calculate_threshold(model_position):
    print("\nPredicting sequence...")

    model_files = os.listdir(model_position)
    X_check_list = []
    y_check_list = []
    indim_check_list = []
    outdim_check_list = []
    model_predict = []

    number_of_split = 100
    
    temp_PD = 0 
    temp_count1 = 0
    temp_count2 = 0
    temp_count3 = 0
    temp_count4 = 0
    temp_recall = 0
    beta = 0.5
    
    X_check_list, y_check_list, indim_check_list, outdim_check_list,loop_number = get_array_for_threshold(threshold_position)
    for h in range(loop_number+1):

        correct_count = []
        X_check = X_check_list[h]
        y_check = y_check_list[h]
        indim_check = indim_check_list[h]
        outdim_check = outdim_check_list[h]

        model_predict = build_model_cl(indim_check,outdim_check)
        model_predict.load_weights(model_position + '/' + model_files[0], by_name=True)
        print('mode_weight__load _complete')

        seq_predicted = model_predict.predict(X_check)
        print('predict_complete')

        for i in range(number_of_split):
            threshold = (i+1) / number_of_split
            #seq_predicted[j,0]⇒（予想）正常確率のステータス
            #y_check[j,0]⇒（実際）正常状態のステータス（1⇒正常）
            for j in range(y_check.shape[0]):
                temp_PD = 1 - seq_predicted[j,0]        
                
                if(temp_PD < threshold and y_check[j,0] == 1):#予想⇒正常、実際⇒正常
                    
                    temp_count1 = temp_count1 + 1#真陰性TN

                elif(temp_PD >= threshold and y_check[j,0] == 0):#予想⇒異常、実際⇒異常
                    temp_count2 = temp_count2 + 1#真陽性TP

                elif(temp_PD < threshold and y_check[j,0] == 0):#予想⇒正常、実際⇒異常
                    temp_count3 = temp_count3 + 1#偽陰性FN

                elif(temp_PD >= threshold and y_check[j,0] == 1):#予想⇒異常、実際⇒正常
                    temp_count4 = temp_count4 + 1#偽陽性FP

            if (temp_count2 == 0 and temp_count3 == 0):
                temp_recall = 0
            else:
                temp_recall = temp_count2/(temp_count2 + temp_count3)
            
            if (temp_count2 == 0 and temp_count4 == 0):
                temp_recall = 0
            else:   
                temp_precision = temp_count2/(temp_count2 + temp_count4)

            if (temp_recall + beta**2*temp_precision == 0):
                F_measure = 0
            else:
                F_measure = (1+beta**2)*temp_recall*temp_precision/(temp_recall + beta**2*temp_precision)        

            correct_count.append([threshold,temp_recall,temp_precision,F_measure])

            temp_count1 = 0
            temp_count2 = 0
            temp_count3 = 0
            temp_count4 = 0
            temp_PD = 0

        correct_count = np.array(correct_count)#ndarrayに変換   

        np.savetxt('actual'+ str(h)+'.csv',y_check,delimiter=',')
        np.savetxt('predicted'+ str(h)+'.csv',seq_predicted,delimiter=',')
        np.savetxt('correct'+ str(h)+'.csv',correct_count,delimiter=',')
    
        print('save_complete'+str(h))

    return()

model_mode = input('model_mode_number:')
model_mode = int(model_mode)


if (model_mode == 0):
    print('model_mode:train')
    epochs = input('number_of_epochs:')
    n_epochs =epochs
    epochs = int(epochs)
    
    recurrent_mode = input('choose recurrent_layer. lstm or gru:')

    model_path_cl = model_path_cl + '_' + recurrent_mode + '_' + n_epochs + 'epochs.h5'
    print(model_path_cl)
    #epochs = 1
    train()
elif(model_mode == 1):
    print('model_mode:predict')
    recurrent_mode = input('choose recurrent_layer. lstm or gru:')
    predict(model_read_position)
elif (model_mode == 2):
    print('model_mode:check_csv')
    csv_nan_check = 1
    X_train, y_train, indim_train, outdim_train = get_array(train_position)
    X_validation, y_validation, indim_validation, outdim_validation = get_array(validation_position)
    X_check, y_check, indim_check, outdim_check = get_array(check_position)

elif(model_mode == 3):
    print('model_mode:predict_threshold')
    recurrent_mode = input('choose recurrent_layer. lstm or gru:')
    predict_thre(model_read_position)

elif(model_mode == 4):
    print('model_mode:calculate_threshold')
    recurrent_mode = input('choose recurrent_layer. lstm or gru:')
    calculate_threshold(model_read_position)