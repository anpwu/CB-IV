from __future__ import annotations
from typing import Dict, Any, List, NamedTuple, TYPE_CHECKING, Optional
import os
import copy
import logging
import torch
from torch import nn
import numpy as np
from pathlib import Path
from utils import log, set_seed

from sklearn.model_selection import train_test_split
import numpy as np

class TrainDataSet(NamedTuple):
    treatment: np.ndarray
    instrumental: np.ndarray
    covariate: Optional[np.ndarray]
    outcome: np.ndarray
    structural: np.ndarray

class TestDataSet(NamedTuple):
    treatment: np.ndarray
    covariate: Optional[np.ndarray]
    structural: np.ndarray
    instrumental: Optional[np.ndarray]
    outcome: Optional[np.ndarray]

class TrainDataSetTorch(NamedTuple):
    treatment: torch.Tensor
    instrumental: torch.Tensor
    covariate: torch.Tensor
    outcome: torch.Tensor
    structural: torch.Tensor

    @classmethod
    def from_numpy(cls, train_data: TrainDataSet):
        covariate = None
        if train_data.covariate is not None:
            covariate = torch.tensor(train_data.covariate, dtype=torch.float32)
        return TrainDataSetTorch(treatment=torch.tensor(train_data.treatment, dtype=torch.float32),
                                 instrumental=torch.tensor(train_data.instrumental, dtype=torch.float32),
                                 covariate=covariate,
                                 outcome=torch.tensor(train_data.outcome, dtype=torch.float32),
                                 structural=torch.tensor(train_data.structural, dtype=torch.float32))

    def to_gpu(self):
        covariate = None
        if self.covariate is not None:
            covariate = self.covariate.cuda()
        return TrainDataSetTorch(treatment=self.treatment.cuda(),
                                 instrumental=self.instrumental.cuda(),
                                 covariate=covariate,
                                 outcome=self.outcome.cuda(),
                                 structural=self.structural.cuda())

class TestDataSetTorch(NamedTuple):
    treatment: torch.Tensor
    instrumental: torch.Tensor
    covariate: torch.Tensor
    outcome: torch.Tensor
    structural: torch.Tensor

    @classmethod
    def from_numpy(cls, test_data: TestDataSet):
        covariate = None
        instrumental = None
        outcome = None
        if test_data.covariate is not None:
            covariate = torch.tensor(test_data.covariate, dtype=torch.float32)
        if test_data.instrumental is not None:
            instrumental = torch.tensor(test_data.instrumental, dtype=torch.float32)
        if test_data.outcome is not None:
            outcome = torch.tensor(test_data.outcome, dtype=torch.float32)
        return TestDataSetTorch(treatment=torch.tensor(test_data.treatment, dtype=torch.float32),
                                covariate=covariate,
                                instrumental=instrumental,
                                outcome=outcome,
                                structural=torch.tensor(test_data.structural, dtype=torch.float32))
    def to_gpu(self):
        covariate = None
        instrumental = None
        outcome = None
        if self.covariate is not None:
            covariate = self.covariate.cuda()
        if self.instrumental is not None:
            instrumental = self.instrumental.cuda()
        if self.outcome is not None:
            outcome = self.outcome.cuda()
        return TestDataSetTorch(treatment=self.treatment.cuda(),
                                covariate=covariate,
                                instrumental=instrumental,
                                outcome=outcome,
                                structural=self.structural.cuda())

def inv_logit_np(x):
    return np.log(x / (1-x))

def logit_np(x):
    return 1 / (1 + np.exp(-x))

def inv_logit(x):
    return torch.log(x / (1-x))

def logit(x):
    return 1 / (1 + torch.exp(-x))

def linear_log_loss(target: torch.Tensor,
                    feature: torch.Tensor,
                    reg: float):
    weight = fit_linear(target, feature, reg)
    pred = linear_reg_pred(feature, weight)
    
    labels = logit(target)
    logits = logit(pred)
    return (-(torch.log(logits) * labels+torch.log(1-logits) * (1-labels))).sum() + reg * torch.norm(weight) ** 2

def linear_reg_loss(target: torch.Tensor,
                    feature: torch.Tensor,
                    reg: float):
    weight = fit_linear(target, feature, reg)
    pred = linear_reg_pred(feature, weight)
    return torch.norm((target - pred)) ** 2 + reg * torch.norm(weight) ** 2

def fit_linear(target: torch.Tensor,
               feature: torch.Tensor,
               reg: float = 0.0):
    assert feature.dim() == 2
    assert target.dim() >= 2
    nData, nDim = feature.size()
    A = torch.matmul(feature.t(), feature)
    device = feature.device
    A = A + reg * torch.eye(nDim, device=device)

    A_inv = torch.inverse(A)
    if target.dim() == 2:
        b = torch.matmul(feature.t(), target)
        weight = torch.matmul(A_inv, b)
    else:
        b = torch.einsum("nd,n...->d...", feature, target)
        weight = torch.einsum("de,d...->e...", A_inv, b)
    return weight

def linear_reg_pred(feature: torch.Tensor, weight: torch.Tensor):
    assert weight.dim() >= 2
    if weight.dim() == 2:
        return torch.matmul(feature, weight)
    else:
        return torch.einsum("nd,d...->n...", feature, weight)

def outer_prod(mat1: torch.Tensor, mat2: torch.Tensor):
    mat1_shape = tuple(mat1.size())
    mat2_shape = tuple(mat2.size())
    assert mat1_shape[0] == mat2_shape[0]
    nData = mat1_shape[0]
    aug_mat1_shape = mat1_shape + (1,) * (len(mat2_shape) - 1)
    aug_mat1 = torch.reshape(mat1, aug_mat1_shape)
    aug_mat2_shape = (nData,) + (1,) * (len(mat1_shape) - 1) + mat2_shape[1:]
    aug_mat2 = torch.reshape(mat2, aug_mat2_shape)
    return aug_mat1 * aug_mat2

def add_const_col(mat: torch.Tensor):
    assert mat.dim() == 2
    n_data = mat.size()[0]
    device = mat.device
    return torch.cat([mat, torch.ones((n_data, 1), device=device)], dim=1)

class DFLMonitor:
    train_data_t: TrainDataSetTorch
    test_data_t: TestDataSetTorch
    validation_data_t: TrainDataSetTorch

    def __init__(self, t_loss, y_loss, dump_folder: Path, trainer: DFLTrainer):

        self.t_loss = t_loss
        self.y_loss = y_loss
        self.metrics = {"stage1_insample_loss": [],
                        "stage1_outsample_loss": [],
                        "stage2_insample_loss": [],
                        "stage2_outsample_loss": [],
                        "test_loss": []}

        self.dump_folder: Path = dump_folder
        self.trainer: DFLTrainer = trainer
        
        self.val_best = 99999
        self.pred_ate_train_best = 99999
        self.pred_ate_test_best = 99999

        self.pred_ate_train_final = 99999
        self.pred_ate_test_final = 99999

    def configure_data(self, train_data_t: TrainDataSetTorch,
                       test_data_t: TestDataSetTorch,
                       validation_data_t: TrainDataSetTorch):

        self.train_data_t = train_data_t
        self.test_data_t = test_data_t
        self.validation_data_t = validation_data_t

    def record(self, logfile: str, _logfile: str, verbose: int):
        self.trainer.treatment_net.train(False)
        if self.trainer.covariate_net is not None:
            self.trainer.covariate_net.train(False)

        n_train_data = self.train_data_t.treatment.size()[0]
        n_val_data = self.validation_data_t.treatment.size()[0]
        n_test_data = self.test_data_t.treatment.size()[0]
        with torch.no_grad():
            treatment_train_feature = self.trainer.treatment_net(self.train_data_t.treatment)
            treatment_val_feature = self.trainer.treatment_net(self.validation_data_t.treatment)
            treatment_test_feature = self.trainer.treatment_net(self.test_data_t.treatment)

            covariate_train_feature = None
            covariate_val_feature = None
            covariate_test_feature = None
            if self.trainer.covariate_net is not None:
                covariate_train_feature = self.trainer.covariate_net(self.train_data_t.covariate)
                covariate_val_feature = self.trainer.covariate_net(self.validation_data_t.covariate)
                covariate_test_feature = self.trainer.covariate_net(self.test_data_t.covariate)

            # stage2
            feature = DFIVModel.augment_stage2_feature(treatment_train_feature,
                                                       covariate_train_feature,
                                                       self.trainer.add_intercept)

            weight = fit_linear(self.train_data_t.outcome, feature, self.trainer.lam)
            insample_pred = linear_reg_pred(feature, weight)
            if self.y_loss == 'bin':
                labels = logit(self.train_data_t.outcome)
                logits = logit(insample_pred)
                insample_loss = (-(torch.log(logits) * labels+torch.log(1-logits) * (1-labels))).sum() / n_train_data
            else:
                insample_loss = torch.norm(self.train_data_t.outcome - insample_pred) ** 2 / n_train_data

            val_feature = DFIVModel.augment_stage2_feature(treatment_val_feature,
                                                           covariate_val_feature,
                                                           self.trainer.add_intercept)
            outsample_pred = linear_reg_pred(val_feature, weight)
            if self.y_loss == 'bin':
                labels = logit(self.validation_data_t.outcome)
                logits = logit(outsample_pred)
                outsample_loss = (-(torch.log(logits) * labels+torch.log(1-logits) * (1-labels))).sum() / n_val_data
            else:
                outsample_loss = torch.norm(self.validation_data_t.outcome - outsample_pred) ** 2 / n_val_data

            # eval for test
            test_feature = DFIVModel.augment_stage2_feature(treatment_test_feature,
                                                            covariate_test_feature,
                                                            self.trainer.add_intercept)
            test_pred = linear_reg_pred(test_feature, weight)
            if self.y_loss == 'bin':
                labels = logit(self.test_data_t.structural)
                logits = logit(test_pred)
                test_loss = (-(torch.log(logits) * labels+torch.log(1-logits) * (1-labels))).sum() / n_test_data
            else:
                test_loss = torch.norm(self.test_data_t.structural - test_pred) ** 2 / n_test_data

            if verbose >= 1:
                log(logfile, f"insample_loss:{insample_loss.item()}", False)
                log(logfile, f"outsample_loss:{outsample_loss.item()}", False)
                log(logfile, f"test_loss:{test_loss.item()}", False)
                log(_logfile, f"insample_loss:{insample_loss.item()}", False)
                log(_logfile, f"outsample_loss:{outsample_loss.item()}", False)
                log(_logfile, f"test_loss:{test_loss.item()}", False)

            treatment0_train_feature = self.trainer.treatment_net(self.train_data_t.treatment-self.train_data_t.treatment)
            treatment0_test_feature = self.trainer.treatment_net(self.test_data_t.treatment-self.test_data_t.treatment)
            treatment1_train_feature = self.trainer.treatment_net(self.train_data_t.treatment-self.train_data_t.treatment+1)
            treatment1_test_feature = self.trainer.treatment_net(self.test_data_t.treatment-self.test_data_t.treatment+1)

            test_feature1 = DFIVModel.augment_stage2_feature(treatment1_test_feature,
                                                        covariate_test_feature,
                                                        self.trainer.add_intercept)
            test_pred1 = linear_reg_pred(test_feature1, weight)
            test_feature0 = DFIVModel.augment_stage2_feature(treatment0_test_feature,
                                                        covariate_test_feature,
                                                        self.trainer.add_intercept)
            test_pred0 = linear_reg_pred(test_feature0, weight)

            train_feature1 = DFIVModel.augment_stage2_feature(treatment1_train_feature,
                                                        covariate_train_feature,
                                                        self.trainer.add_intercept)
            train_pred1 = linear_reg_pred(train_feature1, weight)
            train_feature0 = DFIVModel.augment_stage2_feature(treatment0_train_feature,
                                                        covariate_train_feature,
                                                        self.trainer.add_intercept)
            train_pred0 = linear_reg_pred(train_feature0, weight)

            if outsample_loss < self.val_best:
                log(logfile, f"val_best from {self.val_best} to {outsample_loss}.")
                log(_logfile, f"val_best from {self.val_best} to {outsample_loss}.", False)
                self.val_best = outsample_loss

                self.pred_ate_test_best = test_pred1.mean() - test_pred0.mean()
                self.pred_ate_train_best = train_pred1.mean() - train_pred0.mean()

                log(logfile, f"train_ate_best: {self.pred_ate_train_best.item()}; test_ate_best: {self.pred_ate_test_best.item()}.")
                log(_logfile, f"train_ate_best: {self.pred_ate_train_best.item()}; test_ate_best: {self.pred_ate_test_best.item()}.", False)

            self.pred_ate_test_final = test_pred1.mean() - test_pred0.mean()
            self.pred_ate_train_final = train_pred1.mean() - train_pred0.mean()

            log(logfile, f"train_ate_final: {self.pred_ate_train_final.item()}; test_ate_final: {self.pred_ate_test_final.item()}.")
            log(_logfile, f"train_ate_final: {self.pred_ate_train_final.item()}; test_ate_final: {self.pred_ate_test_final.item()}.", False)

class DFLModel:
    weight_mat: torch.Tensor

    def __init__(self,
                 treatment_net: nn.Module,
                 covariate_net: Optional[nn.Module],
                 add_intercept: bool
                 ):
        self.treatment_net = treatment_net
        self.covariate_net = covariate_net
        self.add_intercept = add_intercept

    @staticmethod
    def augment_feature(treatment_feature: torch.Tensor,
                        covariate_feature: Optional[torch.Tensor],
                        add_intercept: bool):
        feature = treatment_feature
        if add_intercept:
            feature = add_const_col(feature)

        if covariate_feature is not None:
            feature_tmp = covariate_feature
            if add_intercept:
                feature_tmp = add_const_col(feature_tmp)
            feature = outer_prod(feature, feature_tmp)
            feature = torch.flatten(feature, start_dim=1)

        return feature

    @staticmethod
    def fit_dfl(treatment_feature: torch.Tensor,
                covariate_feature: Optional[torch.Tensor],
                outcome_t: torch.Tensor,
                lam: float, add_intercept: bool
                ):

        # stage1
        feature = DFLModel.augment_feature(treatment_feature,
                                           covariate_feature,
                                           add_intercept)

        weight = fit_linear(outcome_t, feature, lam)
        pred = linear_reg_pred(feature, weight)
        loss = torch.norm((outcome_t - pred)) ** 2 + lam * torch.norm(weight) ** 2

        labels = logit(outcome_t)
        logits = logit(pred)
        log_loss = (-(torch.log(logits) * labels+torch.log(1-logits) * (1-labels))).sum() + lam * torch.norm(weight) ** 2

        return dict(weight=weight, loss=loss, log_loss=log_loss)

    def fit_t(self, train_data_t: TrainDataSetTorch, lam: float):
        treatment_feature = self.treatment_net(train_data_t.treatment)
        outcome_t = train_data_t.outcome
        covariate_feature = None
        if self.covariate_net is not None:
            covariate_feature = self.covariate_net(train_data_t.covariate)

        res = DFLModel.fit_dfl(treatment_feature, covariate_feature, outcome_t, lam, self.add_intercept)
        self.weight_mat = res["weight"]

    def fit(self, train_data: TrainDataSet, lam: float):
        train_data_t = TrainDataSetTorch.from_numpy(train_data)
        self.fit_t(train_data_t, lam)

    def predict_t(self, treatment: torch.Tensor, covariate: Optional[torch.Tensor]):
        treatment_feature = self.treatment_net(treatment)
        covariate_feature = None
        if self.covariate_net:
            covariate_feature = self.covariate_net(covariate)

        feature = DFLModel.augment_feature(treatment_feature, covariate_feature, self.add_intercept)
        return linear_reg_pred(feature, self.weight_mat)

    def predict(self, treatment: np.ndarray, covariate: Optional[np.ndarray]):
        treatment_t = torch.tensor(treatment, dtype=torch.float32)
        covariate_t = None
        if covariate is not None:
            covariate_t = torch.tensor(covariate, dtype=torch.float32)
        return self.predict_t(treatment_t, covariate_t).data.numpy()

    def evaluate_t(self, y_loss: str, test_data: TestDataSetTorch):
        target = test_data.structural
        with torch.no_grad():
            pred = self.predict_t(test_data.treatment, test_data.covariate)
        if y_loss == 'bin':
            return (torch.norm((target - pred)) ** 2) / target.size()[0]
        else:
            return (torch.norm((target - pred)) ** 2) / target.size()[0]

    def evaluate(self, y_loss: str, test_data: TestDataSet):
        return self.evaluate_t(y_loss, TestDataSetTorch.from_numpy(test_data)).data.item()

class DFIVModel:
    stage1_weight: torch.Tensor
    stage2_weight: torch.Tensor

    def __init__(self,
                 treatment_net: nn.Module,
                 instrumental_net: nn.Module,
                 covariate_net: Optional[nn.Module],
                 add_stage1_intercept: bool,
                 add_stage2_intercept: bool
                 ):
        self.treatment_net = treatment_net
        self.instrumental_net = instrumental_net
        self.covariate_net = covariate_net
        self.add_stage1_intercept = add_stage1_intercept
        self.add_stage2_intercept = add_stage2_intercept

    @staticmethod
    def augment_stage1_feature(instrumental_feature: torch.Tensor,
                               add_stage1_intercept: bool):

        feature = instrumental_feature
        if add_stage1_intercept:
            feature = add_const_col(feature)
        return feature

    @staticmethod
    def augment_stage2_feature(predicted_treatment_feature: torch.Tensor,
                               covariate_feature: Optional[torch.Tensor],
                               add_stage2_intercept: bool):
        feature = predicted_treatment_feature
        if add_stage2_intercept:
            feature = add_const_col(feature)

        if covariate_feature is not None:
            feature_tmp = covariate_feature
            if add_stage2_intercept:
                feature_tmp = add_const_col(feature_tmp)
            feature = outer_prod(feature, feature_tmp)
            feature = torch.flatten(feature, start_dim=1)

        return feature

    @staticmethod
    def fit_2sls(treatment_1st_feature: torch.Tensor,
                 instrumental_1st_feature: torch.Tensor,
                 instrumental_2nd_feature: torch.Tensor,
                 covariate_2nd_feature: Optional[torch.Tensor],
                 outcome_2nd_t: torch.Tensor,
                 lam1: float, lam2: float,
                 add_stage1_intercept: bool,
                 add_stage2_intercept: bool,
                 ):

        # stage1
        feature = DFIVModel.augment_stage1_feature(instrumental_1st_feature, add_stage1_intercept)
        stage1_weight = fit_linear(treatment_1st_feature, feature, lam1)

        # predicting for stage 2
        feature = DFIVModel.augment_stage1_feature(instrumental_2nd_feature,
                                                   add_stage1_intercept)
        predicted_treatment_feature = linear_reg_pred(feature, stage1_weight)

        # stage2
        feature = DFIVModel.augment_stage2_feature(predicted_treatment_feature,
                                                   covariate_2nd_feature,
                                                   add_stage2_intercept)

        stage2_weight = fit_linear(outcome_2nd_t, feature, lam2)
        pred = linear_reg_pred(feature, stage2_weight)
        stage2_loss = torch.norm((outcome_2nd_t - pred)) ** 2 + lam2 * torch.norm(stage2_weight) ** 2

        labels = logit(outcome_2nd_t)
        logits = logit(pred)
        stage2_log_loss = (-(torch.log(logits) * labels+torch.log(1-logits) * (1-labels))).sum() + lam2 * torch.norm(stage2_weight) ** 2

        return dict(stage1_weight=stage1_weight,
                    predicted_treatment_feature=predicted_treatment_feature,
                    stage2_weight=stage2_weight,
                    stage2_loss=stage2_loss,
                    stage2_log_loss=stage2_log_loss)

    def fit_t(self,
              train_1st_data_t: TrainDataSetTorch,
              train_2nd_data_t: TrainDataSetTorch,
              lam1: float, lam2: float):

        treatment_1st_feature = self.treatment_net(train_1st_data_t.treatment)
        instrumental_1st_feature = self.instrumental_net(train_1st_data_t.instrumental)
        instrumental_2nd_feature = self.instrumental_net(train_2nd_data_t.instrumental)
        outcome_2nd_t = train_2nd_data_t.outcome
        covariate_2nd_feature = None
        if self.covariate_net is not None:
            covariate_2nd_feature = self.covariate_net(train_2nd_data_t.covariate)

        res = DFIVModel.fit_2sls(treatment_1st_feature,
                                 instrumental_1st_feature,
                                 instrumental_2nd_feature,
                                 covariate_2nd_feature,
                                 outcome_2nd_t,
                                 lam1, lam2,
                                 self.add_stage1_intercept,
                                 self.add_stage2_intercept)

        self.stage1_weight = res["stage1_weight"]
        self.stage2_weight = res["stage2_weight"]

    def fit(self, train_1st_data: TrainDataSet, train_2nd_data: TrainDataSet, lam1: float, lam2: float):
        train_1st_data_t = TrainDataSetTorch.from_numpy(train_1st_data)
        train_2nd_data_t = TrainDataSetTorch.from_numpy(train_2nd_data)
        self.fit_t(train_1st_data_t, train_2nd_data_t, lam1, lam2)

    def predict_t(self, treatment: torch.Tensor, covariate: Optional[torch.Tensor]):
        treatment_feature = self.treatment_net(treatment)
        covariate_feature = None
        if self.covariate_net:
            covariate_feature = self.covariate_net(covariate)

        feature = DFIVModel.augment_stage2_feature(treatment_feature,
                                                   covariate_feature,
                                                   self.add_stage2_intercept)
        return linear_reg_pred(feature, self.stage2_weight)

    def predict(self, treatment: np.ndarray, covariate: Optional[np.ndarray]):
        treatment_t = torch.tensor(treatment, dtype=torch.float32)
        covariate_t = None
        if covariate is not None:
            covariate_t = torch.tensor(covariate, dtype=torch.float32)
        return self.predict_t(treatment_t, covariate_t).data.numpy()

    def evaluate_t(self, y_loss: str, test_data: TestDataSetTorch):
        target = test_data.structural
        with torch.no_grad():
            pred = self.predict_t(test_data.treatment, test_data.covariate)
        if y_loss == 'bin':
            return (torch.norm((target - pred)) ** 2) / target.size()[0]
        else:
            return (torch.norm((target - pred)) ** 2) / target.size()[0]

    def evaluate(self, y_loss: str, test_data: TestDataSet):
        return self.evaluate_t(y_loss, TestDataSetTorch.from_numpy(test_data)).data.item()

class DFLTrainer(object):

    def __init__(self, t_loss: str, y_loss: str, logfile: str, _logfile: str, data_list: List, net_list: List, train_params: Dict[str, Any],
                 gpu_flg: bool = False, dump_folder: Optional[Path] = None):
        self.t_loss = t_loss
        self.y_loss = y_loss
        self.logfile = logfile
        self._logfile = _logfile

        self.data_list = data_list
        self.gpu_flg = gpu_flg and torch.cuda.is_available()
        if self.gpu_flg:
            log(logfile, "gpu mode")
            log(_logfile, "gpu mode", False)

        # configure training params
        self.n_epoch: int = train_params["n_epoch"]
        self.treatment_weight_decay = train_params["treatment_weight_decay"]
        self.covariate_weight_decay = train_params["covariate_weight_decay"]
        self.lam: float = train_params["lam"]
        self.n_iter_treatment = train_params["n_iter_treatment"]
        self.n_iter_covariate = train_params["n_iter_covariate"]
        self.add_intercept: bool = train_params["add_intercept"]

        # build networks
        networks = net_list
        self.treatment_net: nn.Module = networks[0]
        self.covariate_net: Optional[nn.Module] = networks[2]

        if self.gpu_flg:
            self.treatment_net.to("cuda:0")
            if self.covariate_net is not None:
                self.covariate_net.to("cuda:0")

        self.treatment_opt = torch.optim.Adam(self.treatment_net.parameters(),
                                              weight_decay=self.treatment_weight_decay)
        if self.covariate_net:
            self.covariate_opt = torch.optim.Adam(self.covariate_net.parameters(),
                                                  weight_decay=self.covariate_weight_decay)
        self.monitor = None
        if dump_folder is not None:
            self.monitor = DFLMonitor(t_loss, y_loss, dump_folder, self)

    def train(self, rand_seed: int = 42, verbose: int = 0, epoch_show: int = 20) -> float:
        """

        Parameters
        ----------
        rand_seed: int
            random seed
        verbose : int
            Determine the level of logging
        Returns
        -------
        oos_result : float
            The performance of model evaluated by oos
        """
        logfile=self.logfile
        _logfile=self._logfile
        train_data = self.data_list[0]
        test_data = self.data_list[2]
        train_data_t = TrainDataSetTorch.from_numpy(train_data)
        test_data_t = TestDataSetTorch.from_numpy(test_data)
        if self.gpu_flg:
            train_data_t = train_data_t.to_gpu()
            test_data_t = test_data_t.to_gpu()

        if self.monitor is not None:
            validation_data = self.data_list[1]
            validation_data_t = TrainDataSetTorch.from_numpy(validation_data)
            if self.gpu_flg:
                validation_data_t = validation_data_t.to_gpu()
            self.monitor.configure_data(train_data_t, test_data_t, validation_data_t)

        self.lam *= train_data_t[0].size()[0]

        for t in range(self.n_epoch):
            self.update_treatment(train_data_t, verbose)
            if self.covariate_net:
                self.update_covariate_net(train_data_t, verbose)

            if t % epoch_show == 0 or t == self.n_epoch - 1:
                if verbose >= 1:
                    log(logfile, f"Epoch {t} ended")
                    log(_logfile, f"Epoch {t} ended", False)
                if self.monitor is not None:
                    self.monitor.record(logfile, _logfile, verbose)

        mdl = DFLModel(self.treatment_net, self.covariate_net, self.add_intercept)
        mdl.fit_t(train_data_t, self.lam)
        if self.gpu_flg:
            torch.cuda.empty_cache()

        oos_loss: float = mdl.evaluate_t(self.y_loss, test_data_t).data.item()
        if verbose >= 1:
            log(logfile, f"test_loss:{oos_loss}")
            log(_logfile, f"test_loss:{oos_loss}", False)
        return oos_loss, self.monitor

    def update_treatment(self, train_data_t, verbose):
        logfile=self.logfile
        _logfile=self._logfile

        self.treatment_net.train(True)
        if self.covariate_net:
            self.covariate_net.train(False)

        # have covariate features
        covariate_feature = None
        if self.covariate_net:
            covariate_feature = self.covariate_net(train_data_t.covariate).detach()

        for i in range(self.n_iter_treatment):
            self.treatment_opt.zero_grad()
            treatment_feature = self.treatment_net(train_data_t.treatment)
            res = DFLModel.fit_dfl(treatment_feature, covariate_feature, train_data_t.outcome,
                                   self.lam, self.add_intercept)
            if self.y_loss == 'bin':
                loss = res["log_loss"]
            else:
                loss = res["loss"]
            loss.backward()
            if verbose >= 2:
                log(logfile, f"treatment learning: {loss.item()}")
                log(_logfile, f"treatment learning: {loss.item()}", False)
            self.treatment_opt.step()

    def update_covariate_net(self, train_data_t: TrainDataSetTorch, verbose: int):
        logfile=self.logfile
        _logfile=self._logfile

        self.treatment_net.train(False)
        treatment_feature = self.treatment_net(train_data_t.treatment).detach()
        self.covariate_net.train(True)
        for i in range(self.n_iter_covariate):
            self.covariate_opt.zero_grad()
            covariate_feature = self.covariate_net(train_data_t.covariate)
            res = DFLModel.fit_dfl(treatment_feature, covariate_feature, train_data_t.outcome,
                                   self.lam, self.add_intercept)
            if self.y_loss == 'bin':
                loss = res["log_loss"]
            else:
                loss = res["loss"]
            loss.backward()
            if verbose >= 2:
                log(logfile, f"update covariate: {loss.item()}")
                log(_logfile, f"update covariate: {loss.item()}", False)
            self.covariate_opt.step()


def build_net(t_input_dim, v_input_dim, x_input_dim):
    treatment_net = nn.Sequential(nn.Linear(t_input_dim, 16),
                                  nn.ReLU(),
                                  nn.Linear(16, 1))

    instrumental_net = nn.Sequential(nn.Linear(v_input_dim, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 32),
                                     nn.BatchNorm1d(32))

    covariate_net = nn.Sequential(nn.Linear(x_input_dim, 128),
                                  nn.ReLU(),
                                  nn.Linear(128, 32),
                                  nn.BatchNorm1d(32),
                                  nn.ReLU(),
                                  nn.Linear(32, 16),
                                  nn.ReLU())
    
    return treatment_net, instrumental_net, covariate_net

def run(exp, args, dataDir, resultDir, train, val, test, device):
    set_seed(args.seed)
    
    logfile = f'{resultDir}/log.txt'
    _logfile = f'{resultDir}/DFL.txt'

    try:
        train.to_numpy()
        val.to_numpy()
        test.to_numpy()
    except:
        pass
    
    if args.mode == 'v':
        t_input_dim = 1
        v_input_dim = args.mV
        x_input_dim = args.mX
        train_v = train.v
        train_x = train.x
        val_v = val.v
        val_x = val.x
        test_v = test.v
        test_x = test.x
    elif args.mode == 'x':
        t_input_dim = 1
        v_input_dim = args.mX
        x_input_dim = args.mX
        train_v = train.x
        train_x = train.x
        val_v = val.x
        val_x = val.x
        test_v = test.x
        test_x = test.x
    elif args.mode == 'vx':
        t_input_dim = 1
        v_input_dim = args.mV + args.mX
        x_input_dim = args.mX
        train_v = np.concatenate((train.v,train.x),1)
        train_x = train.x
        val_v = np.concatenate((val.v,val.x),1)
        val_x = val.x
        test_v = np.concatenate((test.v,test.x),1)
        test_x = test.x
    elif args.mode == 'xx':
        t_input_dim = 1
        v_input_dim = args.mV + args.mX
        x_input_dim = args.mV + args.mX
        train_v = np.concatenate((train.v,train.x),1)
        train_x = np.concatenate((train.v,train.x),1)
        val_v = np.concatenate((val.v,val.x),1)
        val_x = np.concatenate((val.v,val.x),1)
        test_v = np.concatenate((test.v,test.x),1)
        test_x = np.concatenate((test.v,test.x),1)
        
    # please set the config
    train_config = {"n_epoch": 100, 
                    "lam": 0.1, 
                    'n_iter_treatment': 20,
                    'n_iter_covariate': 20,
                    'treatment_weight_decay': 0.0,
                    'covariate_weight_decay': 0.1,
                    "add_intercept": True}   
    
    if args.dfl_t_loss == 'bin':
        train_t = train.t
        train_t[train_t==0] = -6.9068 # ln(1/999), y = 0.001
        train_t[train_t==1] = 6.9068  # ln(999), y = 0.999
        val_t = val.t
        val_t[val_t==0] = -6.9068 # ln(1/999), y = 0.001
        val_t[val_t==1] = 6.9068  # ln(999), y = 0.999
        test_t = test.t
        test_t[test_t==0] = -6.9068 # ln(1/999), y = 0.001
        test_t[test_t==1] = 6.9068  # ln(999), y = 0.999
    else:
        train_t = train.t
        val_t   = val.t
        test_t  = test.t

    if args.dfl_y_loss == 'bin':
        train_y = train.y
        train_y[train_y==0] = -6.9068 # ln(1/999), y = 0.001
        train_y[train_y==1] = 0.999 
        val_y = val.y
        val_y[val_y==0] = -6.9068 # ln(1/999), y = 0.001
        val_y[val_y==1] = 6.9068  # ln(999), y = 0.999
        test_y = test.y
        test_y[test_y==0] = -6.9068 # ln(1/999), y = 0.001
        test_y[test_y==1] = 6.9068  # ln(999), y = 0.999
    else:
        train_y = train.y
        val_y   = val.y
        test_y  = test.y

    train_data = TrainDataSet(treatment=train_t,
                              instrumental=train_v,
                              covariate=train_x,
                              outcome=train_y,
                              structural=train_y)
    val_data = TrainDataSet(treatment=val_t,
                              instrumental=val_v,
                              covariate=val_x,
                              outcome=val_y,
                              structural=val_y)
    test_data = TestDataSet(treatment=test_t,
                              instrumental=test_v,
                              covariate=test_x,
                              structural=test_y,
                              outcome=None)
    data_list = [train_data, val_data, test_data]

    treatment_net, instrumental_net, covariate_net = build_net(t_input_dim, v_input_dim, x_input_dim)
    net_list = [treatment_net, None, covariate_net]

    trainer = DFLTrainer(args.dfl_t_loss, args.dfl_y_loss, logfile, _logfile, data_list, net_list, train_config, args.use_gpu, resultDir)
    test_loss, monitor = trainer.train(rand_seed=args.seed, verbose=args.verbose, epoch_show=args.epoch_show)
    
    return monitor