import torch
import math
from torch import Tensor, nn
import pytorch_lightning as pl
from gmsl.model import BaseModel, SEGNNModel
from torch_geometric.data import Batch
from typing import Tuple

class MultiTaskModel(pl.LightningModule):
    def __init__(
        self,
        args,
        sdim: int = 128,
        vdim: int = 16,
        depth: int = 5,
        r_cutoff: float = 5.0,
        num_radial: int = 32,
        model_type: str = "egnn",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        patience_scheduler: int = 10,
        factor_scheduler: float = 0.75,
        max_epochs: int = 30,
        use_norm: bool = True,
        aggr: str = "mean",
        enhanced: bool = True,
        offset_strategy: int = 0,
        task = 'multi',
        readout = 'vanilla'
    ):
        if not model_type.lower() in ["painn", "eqgat", "schnet", "segnn", "egnn", "egnn_edge", "gearnet"]:
            print("Wrong select model type")
            print("Exiting code")
            exit()

        super(MultiTaskModel, self).__init__()
        self.save_hyperparameters(args)
        print("Initializing MultiTask Model...")
        self.sdim = sdim
        self.vdim = vdim
        self.depth = depth
        self.r_cutoff = r_cutoff
        self.num_radial = num_radial
        self.model_type = model_type.lower()
        self.learning_rate = learning_rate
        self.patience_scheduler = patience_scheduler
        self.factor_scheduler = factor_scheduler
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        # 4 Multilabel Binary Classification Tasks
        # 之后可以做到Config中
        self.property_info = {'ec': 538, 'mf': 490, 'bp': 1944, 'cc': 321, 'reaction':384, 'fold': 1195}
        # self.property_info = {'ec': 3615, 'mf': 490, 'bp': 1944, 'cc': 321}
        # self.property_info = {'ec': 3615, 'mf': 5348, 'bp': 10285, 'cc': 1901}
        self.affinity_info = {'lba': 1, 'ppi': 1}
        # Weight of loss for each task
        # 之后可以变成可学习的版本
        self.property_alphas = [1, 1, 1, 1, 1, 1]
        self.affinity_alphas = [1, 1]
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.l2_aux = nn.MSELoss()
        self.bce = nn.BCELoss()
        self.task = task
        self.training_step_outputs =[]
        self.validation_step_outputs =[]
        self.test_step_outputs = []
        if offset_strategy == 0:
            num_elements = 10
        elif offset_strategy == 1:
            num_elements = 10 + 9
        elif offset_strategy == 2:
            num_elements = 10 + 9 * 20
        else:
            raise NotImplementedError
        # print("Num Elements:", num_elements)
        if model_type != "segnn":
            self.model = BaseModel(sdim=sdim,
                                   vdim=vdim,
                                   depth=depth,
                                   r_cutoff=r_cutoff,
                                   num_radial=num_radial,
                                   model_type=model_type,
                                   graph_level=True,
                                   num_elements=num_elements,
                                   out_units=1,
                                   dropout=0.0,
                                   use_norm=use_norm,
                                   aggr=aggr,
                                   cross_ablate=args.cross_ablate,
                                   no_feat_attn=args.no_feat_attn,
                                   task = task,
                                   readout=readout
                                # protein_function_class_dims = class_dims
                                   )
        else:
            self.model = SEGNNModel(num_elements=num_elements,
                                    out_units=1,
                                    hidden_dim=sdim + 3 * vdim + 5 * vdim // 2,
                                    lmax=2,
                                    depth=depth,
                                    graph_level=True,
                                    use_norm=False)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.factor_scheduler,
            patience=self.patience_scheduler,
            min_lr=1e-7,
            verbose=True,
        )

        schedulers = [
            {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
            }
        ]

        return [optimizer], schedulers
    def cal_fmax(self, preds, y_true):
        thresholds = torch.linspace(0, 0.95, 20)
        fs = []
        for thres in thresholds:
            # y_pred = torch.zeros(preds.shape).to(preds.device)
            # y_pred[preds>=thres] = 1
            # y_pred[preds<thres] = 0
            y_pred = torch.where(preds>=thres, 1, 0)
            # has_pred = torch.zeros(y_pred.shape[0])
            # has_pred[predict_rows >= 1] = 1
            predict_rows = torch.sum(y_pred, dim=1)
            has_pred = torch.where(predict_rows >= 1, 1, 0)

            pred_new = y_pred[has_pred == 1, :]
            true_new = y_true[has_pred == 1, :]
            # class labels = 538+490+1944+321
            if pred_new.shape[0] == 0:
                continue
            # print("precision:", (torch.sum((pred_new == true_new), dim=1) / torch.sum(pred_new, dim=1)).shape)
            # curr_position = 0
            # fs_thres = []
            # for task in self.class_nums:
            #     precision = torch.mean(torch.sum(((pred_new[:, curr_position: curr_position + task] == true_new[:, curr_position: curr_position + task]) * (true_new[:, curr_position: curr_position + task] == 1)), dim=1) / torch.sum(pred_new[:, curr_position: curr_position + task], dim=1))
            #     recall = torch.mean(torch.sum(((y_pred[:, curr_position: curr_position + task] == y_true[:, curr_position: curr_position + task]) * (y_true[:, curr_position: curr_position + task] == 1)), dim=1)/ torch.sum(y_true[:, curr_position: curr_position + task], dim=1))
            #     f = (2 * precision * recall) / (precision + recall)
            #     if not torch.isnan(f):
            #         fs_thres.append(f.item())
            #     else:
            #         fs_thres.append(-1)
            #     curr_position += task
            # # 计算总的fmax
            precision = torch.mean(torch.sum(((pred_new == true_new) * (true_new == 1)), dim=1) / torch.sum(pred_new, dim=1))
            recall = torch.mean(torch.sum(((y_pred == y_true) * (y_true == 1)), dim=1)/ torch.sum(y_true, dim=1))
            f = (2 * precision * recall) / (precision + recall)
            if not torch.isnan(f):
                f = f.item()
            else:
                f = -1
            # fs_thres.append(f)
            fs.append(f)
        fs = torch.tensor(fs)
        f_max = torch.max(fs, dim=0)[0]
        return f_max.item()
            
    def forward(self, data: Batch) -> Tuple[Tensor, Tensor]:
        # if self.model_type == 'egnn_edge':
        #     y_true = data.y.view(-1, )
        #     y_aux_true = data.external_edge_dist.view(-1)
        #     y_pred, y_aux_pred = self.model(data=data)
        #     y_pred = y_pred.view(-1, )
        #     y_aux_pred = y_aux_pred.view(-1, )
        #     # print("Y_pred:", y_pred, y_true)
        #     # print("Y_aux_pred:", y_aux_pred, y_aux_true)
        #     return y_pred, y_true, y_aux_pred, y_aux_true
        self.datatype = data.type
        y_affinity_pred, y_property_pred = self.model(data=data)[0], self.model(data=data)[1]
        y_affinity_true = data.affinities
        y_affinity_mask = data.affinity_mask
        y_property_true = data.functions
        y_property_mask = data.valid_masks

        # Predict Affinities
        y_affinity_preds = []
        y_affinity_trues = []
        for j, (affinity_name, affinity_num) in enumerate(self.affinity_info.items()):
            curr_pred = y_affinity_pred[j].view(-1, )
            # print(y_affinity_true.shape)
            # print(y_affinity_true[:, j].shape)
            curr_true = y_affinity_true[:, j].view(-1, )
            curr_mask = y_affinity_mask[:, j].view(-1, )
            # print("Affinity Before:", affinity_name, curr_pred.shape, curr_true.shape, curr_mask.shape)
            curr_pred = curr_pred[curr_mask == 1]
            curr_true = curr_true[curr_mask == 1]
            # print("Affinity After:", affinity_name, curr_pred.shape, curr_true.shape)
            y_affinity_preds.append(curr_pred)
            y_affinity_trues.append(curr_true)

        # Predict Properties
        left = 0
        y_property_preds = []
        y_property_trues = []
        for i, (property_name, class_num) in enumerate(self.property_info.items()):
            right = left + class_num
            curr_pred = y_property_pred[i]
            curr_true = y_property_true[:, left: right]
            curr_mask = y_property_mask[:, i]
            # print("Property Before:", property_name, curr_pred.shape, curr_true.shape, curr_mask.shape)
            curr_pred = curr_pred[curr_mask == 1]
            curr_true = curr_true[curr_mask == 1]
            # print("Property After:", property_name, curr_pred.shape, curr_true.shape)
            y_property_preds.append(curr_pred)
            y_property_trues.append(curr_true)
            left = right
        
        return y_affinity_preds, y_property_preds, y_affinity_trues, y_property_trues


    def training_step(self, batch, batch_idx):
        # if self.model_type == "egnn_edge":
        #     y_pred, y_true, y_aux_pred, y_aux_true = self(batch)
        #     loss1 = self.l2(y_pred, y_true)
        #     loss2 = self.l2_aux(y_aux_pred, y_aux_true)
        #     # alpha = 0.1
        #     loss = loss1 + 0.2 * loss2
        #     # print("Loss:", loss)
        #     self.log("train_l2", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        #     aux_l2_loss = self.l2(y_aux_pred, y_aux_true)
        # self.log("aux_loss", aux_l2_loss, on_step=True, on_epoch=False, prog_bar=True)
        y_affinity_preds, y_property_preds, y_affinity_trues, y_property_trues = self(batch)
        bce_losses = 0
        l2_losses = 0
        # This output contains bce and l2 loss for each category as well as in total.
        training_step_output = {}
        for i, (property_name, class_num) in enumerate(self.property_info.items()):  
        # print("Shape:", y_property_pred.shape==y_property_true.shape)
        # print("Label", torch.isnan(y_property_true).any(), torch.isnan(y_property_pred).any())
        # 这是因为这里会预测出一个nan，不知道为什么，之后看一看
            y_property_pred = y_property_preds[i]
            y_property_true = y_property_trues[i]
            if torch.isnan(y_property_pred).any():
                print("Wrong Prediction of Properties!", batch.prot_id, batch)
            if not torch.isnan(y_property_pred).any():
                bce_loss = self.bce(y_property_pred, y_property_true)
            else:
                bce_loss = torch.tensor([torch.nan])
            loss_name = 'bce_' + property_name
            training_step_output[loss_name] = bce_loss
            self.log(loss_name, bce_loss, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
            if not torch.isnan(bce_loss):
                bce_losses += self.property_alphas[i] * bce_loss
        for i, (affinity_name, affinity_num) in enumerate(self.affinity_info.items()):
            y_affinity_pred = y_affinity_preds[i]
            y_affinity_true = y_affinity_trues[i]
            if len(y_affinity_pred) != 0:
                l2_loss= self.l2(y_affinity_pred, y_affinity_true)
            else:
                l2_loss = torch.tensor([torch.nan])
            loss_name = 'l2_' + affinity_name
            training_step_output[loss_name] = l2_loss
            self.log(loss_name, l2_loss, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
            if not torch.isnan(l2_loss):
                l2_losses += self.affinity_alphas[i] * l2_loss
        if not bce_losses == 0 and not l2_losses == 0:
            # print("BCE Loss:", bce_loss, "L2 Loss:", l2_loss)
            loss = 10 * bce_losses + l2_losses
        elif not bce_losses == 0:
            # print("BCE Loss:", bce_loss)
            loss = 10 * bce_losses
        elif not l2_losses == 0:
            # print("L2 Loss:", l2_loss)
            loss = l2_losses
        else:
            print("Found a batch with no annotations!")
            raise RuntimeError
        training_step_output['loss'] = loss
        training_step_output['bce_total'] = bce_losses if bce_losses != 0 else -1
        training_step_output['l2_total'] = l2_losses if l2_losses != 0 else -1
        if torch.isnan(loss):
            print(bce_losses, l2_losses)
            print(y_affinity_pred.shape, y_affinity_true.shape)
            print(y_property_pred.shape, y_property_true.shape)

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log("train_l2", l2_losses, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log("train_bce", bce_losses, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.training_step_outputs.append(training_step_output)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        # if self.model_type == "egnn_edge":
        #     y_pred, y_true, _, _ = self(batch)
        # else:
        y_affinity_preds, y_property_preds, y_affinity_trues, y_property_trues = self(batch)
        for i, (affinity_name, affinity_num) in enumerate(self.affinity_info.items()):
            curr_pred = y_affinity_preds[i]
            curr_true = y_affinity_trues[i]
            l2_loss = self.l2(curr_pred, curr_true)
            log_name = 'val_loss_' + affinity_name
            self.log(log_name, l2_loss, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        
        self.validation_step_outputs.append({"affinity_true": y_affinity_trues, 'affinity_pred': y_affinity_preds, 'property_true': y_property_trues, "property_pred": y_property_preds})
        
        
    def test_step(self, batch, batch_idx):
        # if self.model_type == "egnn_edge":
        #     y_pred, y_true, _, _ = self(batch)
        y_affinity_preds, y_property_preds, y_affinity_trues, y_property_trues = self(batch)
        self.test_step_outputs.append({"affinity_true": y_affinity_trues,"affinity_pred": y_affinity_preds,"property_true": y_property_trues,"property_pred": y_property_preds})

    def on_train_epoch_end(self):
        avg_loss = torch.stack([x["loss"] for x in self.training_step_outputs]).mean().item()
        mse_loss = torch.stack([x["l2_total"] for x in self.training_step_outputs if x['l2_total'] != -1]).mean().item()
        bce_loss = torch.stack([x["bce_total"] for x in self.training_step_outputs if x['bce_total'] != -1]).mean().item()
        to_print = (
            f"{self.current_epoch:<10}: "
            f"TOTAL: {round(avg_loss, 4)}, "
            f"MSE: {round(mse_loss, 4)}, "
            f"RMSE: {round(math.sqrt(mse_loss), 4)}"
            f"BCE: {round(bce_loss, 4)}"
        )

        print("TRAIN", to_print)
        self.training_step_outputs.clear()
        self.log("train_eloss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train_el2", mse_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train_bce", bce_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

    def on_validation_epoch_end(self):
        valid_res = {}
        l2_losses = 0
        bce_losses = 0
        for i, (affinity_name, affinity_num) in enumerate(self.affinity_info.items()):
            affinity_pred = torch.concat([x["affinity_pred"][i] for x in self.validation_step_outputs])
            affinity_true = torch.concat([x["affinity_true"][i] for x in self.validation_step_outputs])
            mse = self.l2(affinity_pred, affinity_true).item()
            mae = self.l1(affinity_pred, affinity_true).item()
            rmse = math.sqrt(mse)
            l2_losses += mse
            log_l2 = 'val_mse_' + affinity_name
            log_rmse = 'val_rmse_' + affinity_name
            log_l1 = 'val_mae_' + affinity_name
            self.log(log_l1, round(mae, 4), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log(log_l2, round(mse, 4), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log(log_rmse, round(rmse, 4), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            valid_res[log_l2] = round(mse, 4)
            # valid_res[log_rmse] = round(rmse, 4)
            valid_res[log_l1] = round(mae, 4)
    
        for i, (property_name, property_num) in enumerate(self.property_info.items()):
            property_pred = torch.concat([x["property_pred"][i] for x in self.validation_step_outputs])
            property_true = torch.concat([x["property_true"][i] for x in self.validation_step_outputs])
            bce_loss = self.bce(property_pred, property_true).item()
            bce_losses += bce_loss
            f_max = self.cal_fmax(property_pred, property_true)
            log_bce = 'val_bce_' + property_name
            log_fmax = 'val_fmax_' + property_name
            self.log(log_bce, round(bce_loss, 4), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log(log_fmax, round(f_max, 4), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            # valid_res[log_bce] = round(bce_loss, 4)
            valid_res[log_fmax] = round(f_max, 4)
        
        # 计算reaction的acc
        property_pred = torch.concat([x["property_pred"][4] for x in self.validation_step_outputs])
        property_true = torch.concat([x["property_true"][4] for x in self.validation_step_outputs])
        predicted = torch.argmax(property_pred,dim=1)
        labels = torch.argmax(property_true,dim=1)
        # print("labels:",labels)
        correct = (predicted==labels).sum().item()
        total = labels.size(0)
        acc = correct / total
        # print("acc:{},total:{}".format(acc,total))
        self.log("val_acc_reaction", round(acc,4),on_step=False,on_epoch=True,prog_bar=False, sync_dist=True)
        valid_res["val_acc_reaction"] = round(acc,4)
        
        # 计算 fold 的 acc
        property_pred = torch.concat([x["property_pred"][5] for x in self.validation_step_outputs])
        property_true = torch.concat([x["property_true"][5] for x in self.validation_step_outputs])
        predicted = torch.argmax(property_pred,dim=1)
        labels = torch.argmax(property_true,dim=1)
        # print("labels:",labels)
        correct = (predicted==labels).sum().item()
        total = labels.size(0)
        acc = correct / total
        # print("acc:{},total:{}".format(acc,total))
        self.log("val_acc_fold", round(acc,4),on_step=False,on_epoch=True,prog_bar=False, sync_dist=True)
        valid_res["val_acc_fold"] = round(acc,4)
            
        val_loss = l2_losses + 10 * bce_losses
        valid_res['val_loss'] = val_loss
        valid_res['epoch'] = self.current_epoch
        print("VALID:", valid_res)
        self.log('val_loss', round(val_loss, 4), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        test_res = {}
        for i, (affinity_name, affinity_num) in enumerate(self.affinity_info.items()):
            affinity_pred = torch.concat([x["affinity_pred"][i] for x in self.test_step_outputs])
            affinity_true = torch.concat([x["affinity_true"][i] for x in self.test_step_outputs])
            mse = self.l2(affinity_pred, affinity_true).item()
            mae = self.l1(affinity_pred, affinity_true).item()
            rmse = math.sqrt(mse)
            log_l2 = 'test_mse_' + affinity_name
            log_rmse = 'test_rmse_' + affinity_name
            log_l1 = 'test_mae_' + affinity_name
            self.log(log_l1, round(mae, 4), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log(log_l2, round(mse, 4), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log(log_rmse, round(rmse, 4), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            test_res[log_l2] = round(mse, 4)
            # test_res[log_rmse] = round(rmse, 4)
            test_res[log_l1] = round(mae, 4)
        for i, (property_name, property_num) in enumerate(self.property_info.items()):
            property_pred = torch.concat([x["property_pred"][i] for x in self.test_step_outputs])
            property_true = torch.concat([x["property_true"][i] for x in self.test_step_outputs])
            bce_loss = self.bce(property_pred, property_true).item()
            f_max = self.cal_fmax(property_pred, property_true)
            log_bce = 'test_bce_' + property_name
            log_fmax = 'test_fmax_' + property_name
            self.log(log_bce, round(bce_loss, 4), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log(log_fmax, round(f_max, 4), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            # test_res[log_bce] = round(bce_loss, 4)
            test_res[log_fmax] = round(f_max, 4)
        
        # 计算reaction的acc
        property_pred = torch.concat([x["property_pred"][4] for x in self.validation_step_outputs])
        property_true = torch.concat([x["property_true"][4] for x in self.validation_step_outputs])
        predicted = torch.argmax(property_pred,dim=1)
        labels = torch.argmax(property_true,dim=1)
        # print("labels:",labels)
        correct = (predicted==labels).sum().item()
        total = labels.size(0)
        acc = correct / total
        # print("acc:{},total:{}".format(acc,total))
        self.log("test_acc_reaction", round(acc,4),on_step=False,on_epoch=True,prog_bar=False, sync_dist=True)
        test_res["test_acc_reaction"] = round(acc,4)
        
        # 计算reaction的acc
        property_pred = torch.concat([x["property_pred"][5] for x in self.validation_step_outputs])
        property_true = torch.concat([x["property_true"][5] for x in self.validation_step_outputs])
        predicted = torch.argmax(property_pred,dim=1)
        labels = torch.argmax(property_true,dim=1)
        # print("labels:",labels)
        correct = (predicted==labels).sum().item()
        total = labels.size(0)
        acc = correct / total
        # print("acc:{},total:{}".format(acc,total))
        self.log("test_acc_fold", round(acc,4),on_step=False,on_epoch=True,prog_bar=False, sync_dist=True)
        test_res["test_acc_fold"] = round(acc,4)
            
        test_res['epoch'] = self.current_epoch
        self.res = test_res
        print("TEST:", test_res)
        self.test_step_outputs.clear()
        # return mse, mae

class AffinityModel(pl.LightningModule):
    def __init__(
        self,
        args,
        sdim: int = 128,
        vdim: int = 16,
        depth: int = 5,
        r_cutoff: float = 5.0,
        num_radial: int = 32,
        model_type: str = "eqgat",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        patience_scheduler: int = 10,
        factor_scheduler: float = 0.75,
        max_epochs: int = 30,
        use_norm: bool = True,
        aggr: str = "mean",
        enhanced: bool = True,
        offset_strategy: int = 0,
        task = 'affinity',
        readout = 'vanilla',
    ):
        if not model_type.lower() in ["painn", "eqgat", "schnet", "segnn", "egnn", "egnn_edge", "gearnet"]:
            print("Wrong select model type")
            print("Exiting code")
            exit()

        super(AffinityModel, self).__init__()
        print("Initializing Affinity Model...")
        self.save_hyperparameters(args)

        self.sdim = sdim
        self.vdim = vdim
        self.depth = depth
        self.r_cutoff = r_cutoff
        self.num_radial = num_radial
        self.model_type = model_type.lower()
        self.learning_rate = learning_rate
        self.patience_scheduler = patience_scheduler
        self.factor_scheduler = factor_scheduler
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.task = task
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.training_step_outputs =[]
        self.validation_step_outputs =[]
        self.test_step_outputs = []
        if offset_strategy == 0:
            num_elements = 10
        elif offset_strategy == 1:
            num_elements = 10 + 9
        elif offset_strategy == 2:
            num_elements = 10 + 9 * 20
        else:
            raise NotImplementedError
        # print("Num Elements:", num_elements)
        if model_type != "segnn":
            self.model = BaseModel(sdim=sdim,
                                   vdim=vdim,
                                   depth=depth,
                                   r_cutoff=r_cutoff,
                                   num_radial=num_radial,
                                   model_type=model_type,
                                   graph_level=True,
                                   num_elements=num_elements,
                                   out_units=1,
                                   dropout=0.0,
                                   use_norm=use_norm,
                                   aggr=aggr,
                                   cross_ablate=args.cross_ablate,
                                   no_feat_attn=args.no_feat_attn,
                                   task=task,
                                   readout=readout,
                                #    protein_function_class_dims = class_dims
                                   )
        else:
            self.model = SEGNNModel(num_elements=num_elements,
                                    out_units=1,
                                    hidden_dim=sdim + 3 * vdim + 5 * vdim // 2,
                                    lmax=2,
                                    depth=depth,
                                    graph_level=True,   
                                    use_norm=False)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.factor_scheduler,
            patience=self.patience_scheduler,
            min_lr=1e-7,
            verbose=True,
        )

        schedulers = [
            {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
            }
        ]

        return [optimizer], schedulers
            
    def forward(self, data: Batch) -> Tuple[Tensor, Tensor]:
        y_affinity_pred = self.model(data)[0].view(-1, )
        # print(type(data))
        # y = self.model(data)
        # print(y.shape)
        # print(type(self.model(data=data)))
        y_affinity_pred = self.model(data=data)[0]
        # print("?")
        y_affinity_true = data.y.view(-1, )
        y_affinity_mask = data.affinity_mask.view(-1, )
        # print("Affinity Shape:", y_affinity_mask.shape, y_affinity_true.shape, y_affinity_pred.shape)
        y_affinity_true = y_affinity_true[y_affinity_mask == 1]
        y_affinity_pred = y_affinity_pred[y_affinity_mask == 1]
        return y_affinity_pred, y_affinity_true


    def training_step(self, batch, batch_idx):
        y_affinity_pred, y_affinity_true = self(batch)
        loss= self.l2(y_affinity_pred, y_affinity_true)
        self.log("train_l2", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        with torch.no_grad():
            l1_loss = self.l1(y_affinity_pred, y_affinity_true)
        self.log("train_mae", l1_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.training_step_outputs.append({"loss": loss, "l1": l1_loss})
        return {"loss": loss, "l1": l1_loss}
        

    def validation_step(self, batch, batch_idx):
        
        y_affinity_pred, y_affinity_true = self(batch)
        l2_loss= self.l2(y_affinity_pred, y_affinity_true)
        l1_loss = self.l1(y_affinity_pred, y_affinity_true)
        self.log("val_l2", l2_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log("val_mae", l1_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.validation_step_outputs.append({"affinity_true": y_affinity_true, "affinity_pred": y_affinity_pred,})
        return {
                "affinity_true": y_affinity_true,
                "affinity_pred": y_affinity_pred,
                }

    def test_step(self, batch, batch_idx):
        
        y_affinity_pred, y_affinity_true = self(batch)
        self.test_step_outputs.append({"affinity_true": y_affinity_true, "affinity_pred": y_affinity_pred})
        return {
            "affinity_true": y_affinity_true,
            "affinity_pred": y_affinity_pred,
            }

    def on_train_epoch_end(self):
        avg_loss = torch.stack([x["loss"] for x in self.training_step_outputs]).mean().item()
        to_print = (
            f"{self.current_epoch:<10}: "
            f"BCE_LOSS: {round(avg_loss, 4)}, "
        )

        print(" TRAIN", to_print)
        self.training_step_outputs.clear()
        self.log("train_eloss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

    def on_validation_epoch_end(self):
        # print("Validation Outputs:", self.validation_step_outputs)
        y_pred = torch.concat([x["affinity_pred"] for x in self.validation_step_outputs])
        y_true = torch.concat([x["affinity_true"] for x in self.validation_step_outputs])
        mse = self.l2(y_pred, y_true).item()
        mae = self.l1(y_pred, y_true).item()
        rmse = math.sqrt(mse)
        to_print = (
            f"{self.current_epoch:<10}: "
            f"MAE: {round(mae, 4)}, "
            f"MSE: {round(mse, 4)}, "
            f"RMSE: {round(rmse, 4)}"
        )
        self.res = {"mse": mse, "rmse": rmse, "mae": mae}
        # print("Self.Res:", self.res)
        print("VALID:", to_print)
        self.validation_step_outputs.clear()
        self.log("val_loss", mse, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val_el1", mae, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val_ermse", rmse, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

    def on_test_epoch_end(self):
        y_pred = torch.concat([x["affinity_pred"] for x in self.test_step_outputs])
        y_true = torch.concat([x["affinity_true"] for x in self.test_step_outputs])
        mse = self.l2(y_pred, y_true).item()
        mae = self.l1(y_pred, y_true).item()
        rmse = math.sqrt(mse)
        to_print = (
            f"{self.current_epoch:<10}: "
            f"MAE: {round(mae, 4)}, "
            f"MSE: {round(mse, 4)}, "
            f"RMSE: {round(rmse, 4)}"
        )

        print(" TEST", to_print)
        self.res = {"mse": mse, "rmse": rmse, "mae": mae}
        print("Self.Res:", self.res)
        self.test_step_outputs.clear()
        self.log("test_mse", mse, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("test_rmse", rmse, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("test_el1", mae, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return mse, mae
    




class PropertyModel(pl.LightningModule):
    def __init__(
        self,
        args,
        sdim: int = 128,
        vdim: int = 16,
        depth: int = 5,
        r_cutoff: float = 5.0,
        num_radial: int = 32,
        model_type: str = "eqgat",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        patience_scheduler: int = 10,
        factor_scheduler: float = 0.75,
        max_epochs: int = 30,
        use_norm: bool = True,
        aggr: str = "mean",
        enhanced: bool = True,
        offset_strategy: int = 0,
        task = 'go'
    ):
        if not model_type.lower() in ["painn", "eqgat", "schnet", "segnn", "egnn", "egnn_edge", "gearnet"]:
            print("Wrong select model type")
            print("Exiting code")
            exit()

        super(PropertyModel, self).__init__()
        print("Initializing Property Model...")
        self.save_hyperparameters(args)

        self.sdim = sdim
        self.vdim = vdim
        self.depth = depth
        self.r_cutoff = r_cutoff
        self.num_radial = num_radial
        self.model_type = model_type.lower()
        self.learning_rate = learning_rate
        self.patience_scheduler = patience_scheduler
        self.factor_scheduler = factor_scheduler
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.task = task
        self.bce = nn.BCELoss()
        self.training_step_outputs =[]
        self.validation_step_outputs =[]
        self.test_step_outputs = []
        if offset_strategy == 0:
            num_elements = 10
        elif offset_strategy == 1:
            num_elements = 10 + 9
        elif offset_strategy == 2:
            num_elements = 10 + 9 * 20
        else:
            raise NotImplementedError
        # print("Num Elements:", num_elements)
        if model_type != "segnn":
            self.model = BaseModel(sdim=sdim,
                                   vdim=vdim,
                                   depth=depth,
                                   r_cutoff=r_cutoff,
                                   num_radial=num_radial,
                                   model_type=model_type,
                                   graph_level=True,
                                   num_elements=num_elements,
                                   out_units=1,
                                   dropout=0.0,
                                   use_norm=use_norm,
                                   aggr=aggr,
                                   cross_ablate=args.cross_ablate,
                                   no_feat_attn=args.no_feat_attn,
                                   task=task
                                #    protein_function_class_dims = class_dims
                                   )
        else:
            self.model = SEGNNModel(num_elements=num_elements,
                                    out_units=1,
                                    hidden_dim=sdim + 3 * vdim + 5 * vdim // 2,
                                    lmax=2,
                                    depth=depth,
                                    graph_level=True,
                                    use_norm=False)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.factor_scheduler,
            patience=self.patience_scheduler,
            min_lr=1e-7,
            verbose=True,
        )

        schedulers = [
            {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
            }
        ]

        return [optimizer], schedulers
    def cal_fmax(self, preds, y_true):
        # print("cal fmax:", y_pred.shape, y_true.shape)
        thresholds = torch.linspace(0, 0.95, 20)
        fs = []
        # print("Thres:", thresholds)
        if self.task == 'ec':
            classes = [538]
            # classes = [3615]
        elif self.task == 'go':
            classes = [490, 1944, 321]
            # classes = [5348, 10285, 1901]
        elif self.task == 'mf':
            classes = [490]
            # classes = 5348
        elif self.task == 'bp':
            classes = [1944]
            # classes = [10285]
        elif self.task == 'cc':
            classes = [321]
            # classes = [1901]
        elif self.task == 'reaction':
            classes = [384]
        elif self.task == 'fold':
            classes = [1195]

        for thres in thresholds:
            y_pred = torch.zeros(preds.shape).to(preds.device)
            y_pred[preds>=thres] = 1
            y_pred[preds<thres] = 0
            # print("Pred:", y_pred.shape)
            has_pred = torch.zeros(y_pred.shape[0])
            predict_rows = torch.sum(y_pred, dim=1)
            has_pred[predict_rows >= 1] = 1
            pred_new = y_pred[has_pred == 1, :]
            true_new = y_true[has_pred == 1, :]
            # class labels = 538+490+1944+321
            # print("Pred New:", pred_new.shape)
            if pred_new.shape[0] == 0:
                continue
            # print("precision:", (torch.sum((pred_new == true_new), dim=1) / torch.sum(pred_new, dim=1)).shape)
            curr_position = 0
            fs_thres = []
            for task in classes:
                precision = torch.mean(torch.sum(((pred_new[:, curr_position: curr_position + task] == true_new[:, curr_position: curr_position + task]) * (true_new[:, curr_position: curr_position + task] == 1)), dim=1) / torch.sum(pred_new[:, curr_position: curr_position + task], dim=1))
                recall = torch.mean(torch.sum(((y_pred[:, curr_position: curr_position + task] == y_true[:, curr_position: curr_position + task]) * (y_true[:, curr_position: curr_position + task] == 1)), dim=1)/ torch.sum(y_true[:, curr_position: curr_position + task], dim=1))
                f = (2 * precision * recall) / (precision + recall)
                if not torch.isnan(f):
                    fs_thres.append(f.item())
                else:
                    fs_thres.append(-1)
                curr_position += task
            # 计算总的fmax
            precision = torch.mean(torch.sum(((pred_new == true_new) * (true_new == 1)), dim=1) / torch.sum(pred_new, dim=1))
            recall = torch.mean(torch.sum(((y_pred == y_true) * (y_true == 1)), dim=1)/ torch.sum(y_true, dim=1))
            f = (2 * precision * recall) / (precision + recall)
            if not torch.isnan(f):
                f = f.item()
            else:
                f = -1
            fs_thres.append(f)
            fs.append(fs_thres)
        fs = torch.tensor(fs)
        f_max = torch.max(fs, dim=0)[0]
        # print(f_max)
        if self.task in ['ec', 'mf', 'bp', 'cc', 'reaction', 'fold']:
            return f_max[0].item()
        elif self.task == 'go':
            return f_max[0].item(), f_max[1].item(), f_max[2].item(), f_max[3].item()
            
    def forward(self, data: Batch) -> Tuple[Tensor, Tensor]:
        if self.task == 'go':
            y_property_pred = torch.hstack(self.model(data))
        else:
            y_property_pred = self.model(data)[0]
        y_property_true = data.functions
        y_property_mask = data.valid_masks
        y_property_true = y_property_true[(y_property_mask == 1).sum(dim=1) > 0, :]
        y_property_pred = y_property_pred[(y_property_mask == 1).sum(dim=1) > 0, :]
        return y_property_pred, y_property_true


    def training_step(self, batch, batch_idx):
        y_property_pred, y_property_true = self(batch)
        # print(y_property_pred.shape)
        bce_loss = self.bce(y_property_pred, y_property_true) * int(len(y_property_pred)!=0)
        loss = 10 * bce_loss
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log("train_bce", bce_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.training_step_outputs.append({"loss": loss, "bce": bce_loss})
        return {"loss": loss, "bce": bce_loss}

    def validation_step(self, batch, batch_idx):
        y_property_pred, y_property_true = self(batch)
        bce_loss = self.bce(y_property_pred, y_property_true) * int(len(y_property_pred)!=0)
        self.log("val_bce", bce_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log("val_loss", 10 * bce_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.validation_step_outputs.append({"property_true": y_property_true, "property_pred": y_property_pred})
        return {
                "property_true": y_property_true,
                "property_pred": y_property_pred
                }

    def test_step(self, batch, batch_idx):
        y_property_pred, y_property_true = self(batch)
        self.test_step_outputs.append({"property_true": y_property_true,"property_pred": y_property_pred})
        return {
            "property_true": y_property_true,
            "property_pred": y_property_pred
            }

    def on_train_epoch_end(self):
        avg_loss = torch.stack([x["loss"] for x in self.training_step_outputs]).mean().item()
        to_print = (
            f"{self.current_epoch:<10}: "
            f"BCE_LOSS: {round(avg_loss, 4)}, "
        )
        self.training_step_outputs.clear()
        print(" TRAIN", to_print)
        self.log("train_eloss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

    def on_validation_epoch_end(self):
        property_pred = torch.concat([x["property_pred"] for x in self.validation_step_outputs])
        property_true = torch.concat([x["property_true"] for x in self.validation_step_outputs])
        bce_loss = self.bce(property_pred, property_true)
        if self.task== "reaction":
            # 计算acc
            predicted = torch.argmax(property_pred,dim=1)
            labels = torch.argmax(property_true,dim=1)
            correct = (predicted==labels).sum().item()
            total = labels.size(0)
            acc = correct / total
            print("acc:{},total:{}".format(acc,total))
            self.log("val_acc_reaction", acc, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if self.task== "fold":
            # 计算acc
            predicted = torch.argmax(property_pred,dim=1)
            labels = torch.argmax(property_true,dim=1)
            correct = (predicted==labels).sum().item()
            total = labels.size(0)
            acc = correct / total
            print("acc:{},total:{}".format(acc,total))
            self.log("val_acc_fold", acc, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if self.task in ['ec', 'bp', 'mf', 'cc','reaction', 'fold']:
            fmax_all = self.cal_fmax(property_pred, property_true)
            val_loss = 10 * bce_loss
            to_print = (
            f"{self.current_epoch:<10}: "
            f"FMAX_{self.task}: {round(fmax_all, 4)}"
            )
            self.res = {'fmax_all'.format(self.task): fmax_all}
            # print("Self.Res:", self.res)
            print("VALID:", to_print)
            self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("val_fmax_all", fmax_all, on_step=False, on_epoch=True, prog_bar=False,sync_dist=True)
        else:
            
            fmax_mf, fmax_bp, fmax_cc, fmax_all = self.cal_fmax(property_pred, property_true)
            val_loss = 10 * bce_loss
            to_print = (
                f"{self.current_epoch:<10}: "
                f"FMAX_MF: {round(fmax_mf, 4)}"
                f"FMAX_BP: {round(fmax_bp, 4)}"
                f"FMAX_CC: {round(fmax_cc, 4)}"
                f"FMAX_ALL: {round(fmax_all, 4)}"
            )
            self.res = {"fmax_mf": fmax_mf, "fmax_bp": fmax_bp, "fmax_cc": fmax_cc, 'fmax_all': fmax_all}
            # print("Self.Res:", self.res)
            print("VALID:", to_print)
            self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("val_fmax_mf", fmax_mf, on_step=False, on_epoch=True, prog_bar=False,sync_dist=True)
            self.log("val_fmax_bp", fmax_bp, on_step=False, on_epoch=True, prog_bar=False,sync_dist=True)
            self.log("val_fmax_cc", fmax_cc, on_step=False, on_epoch=True, prog_bar=False,sync_dist=True)
            self.log("val_fmax_all", fmax_all, on_step=False, on_epoch=True, prog_bar=False,sync_dist=True)
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        property_pred = torch.concat([x["property_pred"] for x in self.test_step_outputs])
        property_true = torch.concat([x["property_true"] for x in self.test_step_outputs])
        if self.task== "reaction":
            # 计算acc
            predicted = torch.argmax(property_pred,dim=1)
            labels = torch.argmax(property_true,dim=1)
            correct = (predicted==labels).sum().item()
            total = labels.size(0)
            acc = correct / total
            print("acc:{},total:{}".format(acc,total))
            self.log("test_acc_reaction", acc, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if self.task== 'fold':
            # 计算acc
            predicted = torch.argmax(property_pred,dim=1)
            labels = torch.argmax(property_true,dim=1)
            correct = (predicted==labels).sum().item()
            total = labels.size(0)
            acc = correct / total
            print("acc:{},total:{}".format(acc,total))
            self.log("test_acc_fold", acc, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if self.task in ['ec', 'bp', 'mf', 'cc', 'reaction', 'fold']:
            fmax_all = self.cal_fmax(property_pred, property_true)
            to_print = (
            f"{self.current_epoch:<10}: "
            f"FMAX_{self.task}: {round(fmax_all, 4)}"
            )
            self.res = {'fmax_all'.format(self.task): fmax_all}
            # print("Self.Res:", self.res)
            print("TEST:", to_print)
            self.log("test_fmax_all", fmax_all, on_step=False, on_epoch=True, prog_bar=False,sync_dist=True)
        else:
            fmax_mf, fmax_bp, fmax_cc, fmax_all = self.cal_fmax(property_pred, property_true)
            to_print = (
                f"{self.current_epoch:<10}: "
                f"FMAX_BP: {round(fmax_bp, 4)}"
                f"FMAX_MF: {round(fmax_mf, 4)}"
                f"FMAX_CC: {round(fmax_cc, 4)}"
                f"FMAX_ALL: {round(fmax_all, 4)}"
            )

            print(" TEST", to_print)
            self.res = {"fmax_bp": fmax_bp, "fmax_mf": fmax_mf, "fmax_cc": fmax_cc, "fmax_all": fmax_all}
            print("Self.Res:", self.res)
            self.log("test_fmax_bp", fmax_bp, on_step=False, on_epoch=True, prog_bar=False,sync_dist=True)
            self.log("test_fmax_mf", fmax_mf, on_step=False, on_epoch=True, prog_bar=False,sync_dist=True)
            self.log("test_fmax_cc", fmax_cc, on_step=False, on_epoch=True, prog_bar=False,sync_dist=True)
            self.log("test_fmax_all", fmax_all, on_step=False, on_epoch=True, prog_bar=False,sync_dist=True)
        self.test_step_outputs.clear()
        return fmax_all
    






