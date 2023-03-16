from code.stage_5_code.Method_GCN import MethodGCN
from code.stage_5_code.Method_GCN_ResNet import MethodDeepGCNResNet
from code.stage_5_code.Result_Saver import Result_Saver
from code.stage_5_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from code.stage_5_code.Setting_Train_Test import Setting_Train_Test
from code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.stage_5_code.Dataset_Loader import Dataset_Loader
from SettingCV import SettingCV
import numpy as np
import torch

# ---- Multi-Layer Perceptron script ----
if 0:
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------

    # ---- objection initialization setction ---------------

    # data_obj.dataset_source_file_name = 'train.csv'
    # data_obj.dataset5_source_folder_path = '../../data/stage_5_data/'
    # data_obj.dataset5_source_file_name = 'test.csv'

    data_obj.load()
# --- GCN Model ---
dataset_name = 'citeseer'

if dataset_name == 'cora':
    nclass = 7
    nfeature = 1433
elif dataset_name == 'citeseer':
    nclass = 6
    nfeature = 3703
elif dataset_name == 'pubmed':
    nclass = 3
    nfeature = 500

if 1:
    for depth in [1, 2, 3, 4, 5, 6, 7]:
        for residual_type in ['naive', 'raw', 'graph_naive', 'graph_raw']:
            acc_test_list = []
            print('Method: GCN, dataset: ' + dataset_name + ', depth: ' + str(depth) + ', residual type: ' + residual_type)
            for iter in range(1):
                #---- parameter section -------------------------------
                lr = 0.01
                epoch = 1000
                weight_decay = 5e-4
                c = 0.1
                seed = iter
                dropout = 0.5
                nhid = 16
                depth = 3
                #------------------------------------------------------

                #---- objection initialization setction ---------------
                print('Start')

                data_obj = Dataset_Loader('citeseer', '')
                data_obj.dataset_source_folder_path = '../../data/citeseer'
                data_obj.dataset_name = 'citeseer'
                data_obj.c = c
                data_obj.method_type = 'GCN'

                # method_obj = MethodGCN(nfeature, nhid, nclass, dropout, seed)
                method_obj = MethodDeepGCNResNet(nfeature, nhid, nclass, dropout, seed, depth)
                method_obj.lr = lr
                method_obj.lr = lr
                method_obj.epoch = epoch
                method_obj.residual_type = residual_type

                result_obj = Result_Saver('', '')
                result_obj.result_destination_folder_path = './result/GResNet/'
                result_obj.result_destination_file_name = 'DeepGCNResNet_' + dataset_name + '_' + residual_type + '_depth_' + str(depth) + '_iter_' + str(iter)

                setting_obj = SettingCV('', '')

                evaluate_obj = Evaluate_Accuracy('', '')
                #------------------------------------------------------

                #---- running section ---------------------------------
                setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
                acc_test = setting_obj.load_run_save_evaluate()
                print('Testing Acc: ', acc_test)
                acc_test_list.append(acc_test)
                print('*******************************')
                #------------------------------------------------------
            print(acc_test_list)
            print(np.mean(acc_test_list), np.std(acc_test_list))
            print('Finished')
