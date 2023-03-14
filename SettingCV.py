from code.base_class.setting import setting
from sklearn.model_selection import KFold
import numpy as np


class SettingCV(setting):
    fold = None

    def load_run_save_evaluate(self):
        # load dataset
        loaded_data = self.dataset.load()

        # run MethodModule
        self.method.data = loaded_data
        learned_result, test_acc = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        return test_acc
