from code.stage_5_code.Method_GCN import GCN
from code.stage_5_code.Result_Saver import Result_Saver
from code.stage_5_code.Setting_KFold_CV import Setting_KFold_CV
from code.stage_5_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from code.stage_5_code.Setting_Train_Test import Setting_Train_Test
from code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch
from code.stage_5_code.Dataset_Loader import load

# ---- Multi-Layer Perceptron script ----
if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = load()
    data_obj.dataset_source_folder_path = '../../data/citeseer'
    # data_obj.dataset_source_file_name = 'train.csv'
    # data_obj.dataset5_source_folder_path = '../../data/stage_5_data/'
    # data_obj.dataset5_source_file_name = 'test.csv'

    data_obj.load()

    method_obj = GCN()

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_5_result/GCN_'
    result_obj.result_destination_file_name = 'prediction_result'

    # setting_obj = Setting_KFold_CV('k fold cross validation', '')
    setting_obj = Dataset_Loader('train test split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # # ---- running section ---------------------------------
    # print('************ Start ************')
    # setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    # setting_obj.print_setup_summary()
    # mean_score, std_score = setting_obj.load_run_save_evaluate()
    # print('************ Overall Performance ************')
    # print('MLP Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    # print('************ Finish ************')
    # # ------------------------------------------------------

    # Train model
    t_total = time.time()
    for epoch in range(args.epochs):
        train(epoch)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    test()

