
from typing import Dict, Optional

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from nvflare.apis.executor import Executor
from nvflare.apis.shareable import Shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.apis.dxo import DXO, DataKind

class SVMTrainer(Executor):
    def __init__(self, data_path, train_task_name):
        self.data_path = data_path
        self.train_task_name = train_task_name
        self.svm = None
        self.kernel = None
        self.x_train = None
        self.y_train = None
        self.params = {}
        super().__init__()

    def load_data(self) -> Dict[str, pd.DataFrame]:
        print("********Loading data******** ")
        data = pd.read_csv(self.data_path)
        self.x_train = data.drop(["label"], axis=1)
        self.y_train = data["label"]
        
    
    def train(self, curr_round,global_param: Optional[dict], fl_ctx: FLContext):
        if curr_round == 0:
            print("**************************Training round 0*****************")
            if global_param is None or "kernel" not in global_param:
                self.system_panic("Kernel not specified in global parameters.", fl_ctx)
            # only perform training on the first round
            self.kernel = global_param["kernel"]
            self.svm = SVC(kernel=self.kernel, gamma=global_param["gamma"], class_weight=global_param["class_weight"])
            # train model
            self.svm.fit(self.x_train, self.y_train)
            # get support vectors
            index = self.svm.support_
            local_support_x = self.x_train.iloc[index]
            local_support_y = self.y_train.iloc[index]
            self.params = {
                "support_x": local_support_x.to_numpy().tolist(),  # Convert to list
                "support_y": local_support_y.to_numpy().tolist()   # Convert to list
            }
        elif curr_round > 1:
            self.system_panic("Federated SVM only performs training for one round, system exiting.", fl_ctx)
        return self.params
    
    def execute(self, task_name:str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        global_param = shareable["DXO"]
        curr_round =shareable.get_header("current_round")
        print("********************************* global param , curr_round ********************************")
        print(global_param)
        print(curr_round)
        print("*********************************************************************")
        ctx = fl_ctx
        if task_name == self.train_task_name:
            self.load_data()
            params = self.train(curr_round, global_param=global_param["data"], fl_ctx=ctx)
            dxo = DXO(data_kind= DataKind.WEIGHTS, data=params)
            dxo.update_shareable(shareable)
        return shareable