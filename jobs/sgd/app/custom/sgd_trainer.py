
from typing import Dict, Optional

import pandas as pd
import numpy as np

from sklearn.linear_model import SGDClassifier
from nvflare.apis.executor import Executor
from nvflare.apis.shareable import Shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.apis.dxo import DXO, DataKind

class SGDTrainer(Executor):
    def __init__(self, data_path, train_task_name):
        self.data_path = data_path
        self.train_task_name = train_task_name
        self.sgd = None
        self.x_train = None
        self.y_train = None
        self.n_features= None
        self.random_state = 42
        self.params = {}
        super().__init__()

    def load_data(self) -> Dict[str, pd.DataFrame]:
        print("********Loading data******** ")
        data = pd.read_csv(self.data_path)
        self.x_train = data.drop(["label"], axis=1)
        self.y_train = data["label"]
        self.n_features = len(self.x_train.columns)
    
    def train(self, curr_round: int, global_param: Optional[dict], fl_ctx: FLContext):
        print("********Training******** ")
        if curr_round == 0:
            # initialize model with global_param
            # and set to all zero
            fit_intercept = bool(global_param["fit_intercept"])
            self.sgd= SGDClassifier(
                loss=global_param["loss"],
                penalty=global_param["penalty"],
                fit_intercept=fit_intercept,
                learning_rate=global_param["learning_rate"],
                eta0=global_param["eta0"],
                max_iter=100,
                warm_start=True,
                random_state=self.random_state,
            )
            n_classes = global_param["n_classes"]
            self.sgd.classes_ = np.array(list(range(n_classes)))
            self.sgd.coef_ = np.zeros((1, self.n_features))
            if fit_intercept:
                self.sgd.intercept_ = np.zeros((1,))
            # Training starting from global model
            # Note that the parameter update using global model has been performed
            # during global model evaluation
            self.sgd.fit(self.x_train, self.y_train)
            if self.sgd.fit_intercept:
                self.params = {
                    "coef": self.sgd.coef_,
                    "intercept": self.sgd.intercept_,
                }
            else:
                self.params = {"coef": self.sgd.coef_}
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