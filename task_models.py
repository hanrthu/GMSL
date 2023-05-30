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
        task = 'multi'
    ):
        if not model_type.lower() in ["painn", "eqgat", "schnet", "segnn", "egnn", "egnn_edge"]:
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

        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.l2_aux = nn.MSELoss()
        self.bce = nn.BCELoss()
        self.task = task
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
                                   task = task
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
        thresholds = torch.linspace(0, 0.95, 20)
        fs = []
        for thres in thresholds:
            y_pred = torch.zeros(preds.shape).to(preds.device)
            y_pred[preds>=thres] = 1
            y_pred[preds<thres] = 0
            has_pred = torch.zeros(y_pred.shape[0])
            predict_rows = torch.sum(y_pred, dim=1)
            has_pred[predict_rows >= 1] = 1
            pred_new = y_pred[has_pred == 1, :]
            true_new = y_true[has_pred == 1, :]
            # class labels = 538+490+1944+321
            classes = [538, 490, 1944, 321]
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
        return f_max[0].item(), f_max[1].item(), f_max[2].item(), f_max[3].item(), f_max[4].item()
            
    def forward(self, data: Batch) -> Tuple[Tensor, Tensor]:
        if self.model_type == 'egnn_edge':
            y_true = data.y.view(-1, )
            y_aux_true = data.external_edge_dist.view(-1)
            y_pred, y_aux_pred = self.model(data=data)
            y_pred = y_pred.view(-1, )
            y_aux_pred = y_aux_pred.view(-1, )
            # print("Y_pred:", y_pred, y_true)
            # print("Y_aux_pred:", y_aux_pred, y_aux_true)
            return y_pred, y_true, y_aux_pred, y_aux_true
        self.datatype = data.type
        if data.type == 'multi':
            y_affinity_pred, y_property_pred = self.model(data=data)[0].view(-1, ), self.model(data=data)[1]
            y_affinity_true = data.y.view(-1, )
            y_affinity_mask = data.affinity_mask.view(-1, )
            y_property_true = data.functions
            y_property_mask = data.valid_masks
            y_affinity_true = y_affinity_true[y_affinity_mask == 1]
            y_affinity_pred = y_affinity_pred[y_affinity_mask == 1]
            y_property_true = y_property_true[(y_property_mask == 1).sum(dim=1) > 0, :]
            y_property_pred = y_property_pred[(y_property_mask == 1).sum(dim=1) > 0, :]
            return y_affinity_pred, y_property_pred, y_affinity_true, y_property_true
        elif data.type in ['ec', 'go', 'bp', 'mf', 'cc']:
            y_property_pred = self.model(data)
            y_property_true = data.functions
            y_property_mask = data.valid_masks
            y_property_true = y_property_true[(y_property_mask == 1).sum(dim=1) > 0, :]
            y_property_pred = y_property_pred[(y_property_mask == 1).sum(dim=1) > 0, :]
            return y_property_pred, y_property_true
        elif data.type == 'affinity':
            y_affinity_pred = self.model(data=data).view(-1, )
            y_affinity_true = data.y.view(-1, )
            y_affinity_mask = data.affinity_mask.view(-1, )
            y_affinity_true = y_affinity_true[y_affinity_mask == 1]
            y_affinity_pred = y_affinity_pred[y_affinity_mask == 1]
            return y_affinity_pred, y_affinity_true
        else:
            print("Unknown data type!")
            raise RuntimeError


    def training_step(self, batch, batch_idx):
        # # print("Before train step: ", torch.cuda.memory_allocated(0) / 1024 / 1024)
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
        if batch.type == 'multi':
            y_affinity_pred, y_property_pred, y_affinity_true, y_property_true = self(batch)
            bce_loss = self.bce(y_property_pred, y_property_true) * int(len(y_property_pred)!=0)
            l2_loss= self.l2(y_affinity_pred, y_affinity_true) * int(len(y_affinity_pred) != 0)
            if not torch.isnan(bce_loss) and not torch.isnan(l2_loss):
                loss = 10 * bce_loss + l2_loss
            elif not torch.isnan(bce_loss):
                loss = 10 * bce_loss
            elif not torch.isnan(l2_loss):
                loss = l2_loss
            else:
                print("Found a batch with no annotations!")
                raise RuntimeError
            if torch.isnan(loss):
                print(bce_loss, l2_loss)
                print(y_affinity_pred.shape, y_affinity_true.shape)
                print(y_property_pred.shape, y_property_true.shape)
            self.log("train_l2", l2_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

            with torch.no_grad():
                l1_loss = self.l1(y_affinity_pred, y_affinity_true) * int(len(y_affinity_pred) != 0)
            self.log("train_mae", l1_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
            self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
            self.log("train_bce", bce_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
            return {"loss": loss, "l1": l1_loss, "bce": bce_loss}
        elif batch.type in ['ec', 'go', 'bp', 'mf', 'cc']:
            y_property_pred, y_property_true = self(batch)
            bce_loss = self.bce(y_property_pred, y_property_true) * int(len(y_property_pred)!=0)
            loss = 10 * bce_loss
            self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
            self.log("train_bce", bce_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
            return {"loss": loss, "bce": bce_loss}
        elif batch.type == 'affinity':
            y_affinity_pred, y_affinity_true = self(batch)
            l2_loss= self.l2(y_affinity_pred, y_affinity_true) * int(len(y_affinity_pred) != 0)
            self.log("train_l2", l2_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
            with torch.no_grad():
                l1_loss = self.l1(y_affinity_pred, y_affinity_true) * int(len(y_affinity_pred) != 0)
            self.log("train_mae", l1_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
            self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
            return {"loss": loss, "l1": l1_loss}
        else:
            print("Unknown data type!")
            raise RuntimeError

    def validation_step(self, batch, batch_idx):
        # if self.model_type == "egnn_edge":
        #     y_pred, y_true, _, _ = self(batch)
        # else:
        if batch.type == 'multi':
            y_affinity_pred, y_property_pred, y_affinity_true, y_property_true = self(batch)

        # calculate batch-loss
            l2_loss = self.l2(y_affinity_pred, y_affinity_true) * int(len(y_affinity_pred) != 0)
            l1_loss = self.l1(y_affinity_pred, y_affinity_true) * int(len(y_affinity_pred) != 0)

            self.log("val_loss", l2_loss, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
            self.log("val_mae", l1_loss, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
            return {
                    "affinity_true": y_affinity_true,
                    "affinity_pred": y_affinity_pred,
                    "property_true": y_property_true,
                    "property_pred": y_property_pred
                    }
        elif batch.type in ['ec', 'go', 'bp', 'mf', 'cc']:
            y_property_pred, y_property_true = self(batch)
            bce_loss = self.bce(y_property_pred, y_property_true) * int(len(y_property_pred)!=0)
            self.log("val_bce", bce_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
            self.log("val_loss", 10 * bce_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
            return {
                    "property_true": y_property_true,
                    "property_pred": y_property_pred
                    }
        elif batch.type == 'affinity':
            y_affinity_pred, y_affinity_true = self(batch)
            l2_loss= self.l2(y_affinity_pred, y_affinity_true) * int(len(y_affinity_pred) != 0)
            l1_loss = self.l1(y_affinity_pred, y_affinity_true) * int(len(y_affinity_pred) != 0)
            self.log("val_l2", l2_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
            self.log("val_mae", l1_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
            return {
                    "affinity_true": y_affinity_true,
                    "affinity_pred": y_affinity_pred,
                    }
        else:
            print("Unknown data type!")
            raise RuntimeError
    def test_step(self, batch, batch_idx):
        # if self.model_type == "egnn_edge":
        #     y_pred, y_true, _, _ = self(batch)
        if batch.type == 'multi':
            y_affinity_pred, y_property_pred, y_affinity_true, y_property_true = self(batch)
            return {
                "affinity_true": y_affinity_true,
                "affinity_pred": y_affinity_pred,
                "property_true": y_property_true,
                "property_pred": y_property_pred
            }
        elif batch.type in ['ec', 'go', 'bp', 'mf', 'cc']:
            y_property_pred, y_property_true = self(batch)
            return {
                "property_true": y_property_true,
                "property_pred": y_property_pred
                }
        elif batch.type == 'affinity':
            y_affinity_pred, y_affinity_true = self(batch)
            return {
                "affinity_true": y_affinity_true,
                "affinity_pred": y_affinity_pred,
                }
        else:
            print("Unknown data type!")
            raise RuntimeError
    # def training_step_end(self, step_output):
    #     for name, parms in self.model.named_parameters():
    #         print('-->name:', name)
    #         print('-->para:', parms)
    #         print('-->grad_requirs:',parms.requires_grad)
    #         print('-->grad_value:',parms.grad)
    #         print("===")

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean().item()
        mae_loss = torch.stack([x["l1"] for x in outputs]).mean().item()
        to_print = (
            f"{self.current_epoch:<10}: "
            f"MAE: {round(mae_loss, 4)}, "
            f"MSE: {round(avg_loss, 4)}, "
            f"RMSE: {round(math.sqrt(avg_loss), 4)}"
        )

        print(" TRAIN", to_print)
        self.log("train_eloss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train_el1", mae_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

    def validation_epoch_end(self, outputs):

        y_pred = torch.concat([x["affinity_pred"] for x in outputs])
        y_true = torch.concat([x["affinity_true"] for x in outputs])
        property_pred = torch.concat([x["property_pred"] for x in outputs])
        property_true = torch.concat([x["property_true"] for x in outputs])
        mse = self.l2(y_pred, y_true).item()
        mae = self.l1(y_pred, y_true).item()
        bce_loss = self.bce(property_pred, property_true)
        fmax_ec, fmax_mf, fmax_bp, fmax_cc, fmax_all = self.cal_fmax(property_pred, property_true)
        rmse = math.sqrt(mse)
        val_loss = mse + 10 * bce_loss
        to_print = (
            f"{self.current_epoch:<10}: "
            f"MAE: {round(mae, 4)}, "
            f"MSE: {round(mse, 4)}, "
            f"RMSE: {round(rmse, 4)}"
            f"FMAX_EC: {round(fmax_ec, 4)}"
            f"FMAX_BP: {round(fmax_bp, 4)}"
            f"FMAX_MF: {round(fmax_mf, 4)}"
            f"FMAX_CC: {round(fmax_cc, 4)}"
            f"FMAX_ALL: {round(fmax_all, 4)}"
        )
        self.res = {"mse": mse, "rmse": rmse, "mae": mae, "fmax_ec": fmax_ec, "fmax_bp": fmax_bp, "fmax_mf": fmax_mf, "fmax_cc": fmax_cc, 'fmax_all': fmax_all}
        # print("Self.Res:", self.res)
        print("VALID:", to_print)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val_eloss", mse, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val_el1", mae, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val_ermse", rmse, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val_fmax_ec", fmax_ec, on_step=False, on_epoch=True, prog_bar=False,sync_dist=True)
        self.log("val_fmax_bp", fmax_bp, on_step=False, on_epoch=True, prog_bar=False,sync_dist=True)
        self.log("val_fmax_mf", fmax_mf, on_step=False, on_epoch=True, prog_bar=False,sync_dist=True)
        self.log("val_fmax_cc", fmax_cc, on_step=False, on_epoch=True, prog_bar=False,sync_dist=True)
        self.log("val_fmax_all", fmax_all, on_step=False, on_epoch=True, prog_bar=False,sync_dist=True)

    def test_epoch_end(self, outputs):
        y_pred = torch.concat([x["affinity_pred"] for x in outputs])
        y_true = torch.concat([x["affinity_true"] for x in outputs])
        property_pred = torch.concat([x["property_pred"] for x in outputs])
        property_true = torch.concat([x["property_true"] for x in outputs])
        mse = self.l2(y_pred, y_true).item()
        mae = self.l1(y_pred, y_true).item()
        rmse = math.sqrt(mse)
        fmax_ec, fmax_mf, fmax_bp, fmax_cc, fmax_all = self.cal_fmax(property_pred, property_true)
        to_print = (
            f"{self.current_epoch:<10}: "
            f"MAE: {round(mae, 4)}, "
            f"MSE: {round(mse, 4)}, "
            f"RMSE: {round(rmse, 4)}"
            f"FMAX_EC: {round(fmax_ec, 4)}"
            f"FMAX_BP: {round(fmax_bp, 4)}"
            f"FMAX_MF: {round(fmax_mf, 4)}"
            f"FMAX_CC: {round(fmax_cc, 4)}"
            f"FMAX_ALL: {round(fmax_all, 4)}"
        )

        print(" TEST", to_print)
        self.res = {"mse": mse, "rmse": rmse, "mae": mae, "fmax_ec": fmax_ec, "fmax_bp": fmax_bp, "fmax_mf": fmax_mf, "fmax_cc": fmax_cc, "fmax_all": fmax_all}
        print("Self.Res:", self.res)
        self.log("test_mse", mse, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("test_rmse", rmse, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("test_el1", mae, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("test_fmax_ec", fmax_ec, on_step=False, on_epoch=True, prog_bar=False,sync_dist=True)
        self.log("test_fmax_bp", fmax_bp, on_step=False, on_epoch=True, prog_bar=False,sync_dist=True)
        self.log("test_fmax_mf", fmax_mf, on_step=False, on_epoch=True, prog_bar=False,sync_dist=True)
        self.log("test_fmax_cc", fmax_cc, on_step=False, on_epoch=True, prog_bar=False,sync_dist=True)
        self.log("test_fmax_all", fmax_all, on_step=False, on_epoch=True, prog_bar=False,sync_dist=True)
        return mse, mae

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
        task = 'affinity'
    ):
        if not model_type.lower() in ["painn", "eqgat", "schnet", "segnn", "egnn", "egnn_edge"]:
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
            
    def forward(self, data: Batch) -> Tuple[Tensor, Tensor]:
        y_affinity_pred = self.model(data=data).view(-1, )
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
        return {"loss": loss, "l1": l1_loss}
        

    def validation_step(self, batch, batch_idx):
        
        y_affinity_pred, y_affinity_true = self(batch)
        l2_loss= self.l2(y_affinity_pred, y_affinity_true)
        l1_loss = self.l1(y_affinity_pred, y_affinity_true)
        self.log("val_l2", l2_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log("val_mae", l1_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        return {
                "affinity_true": y_affinity_true,
                "affinity_pred": y_affinity_pred,
                }

    def test_step(self, batch, batch_idx):
        
        y_affinity_pred, y_affinity_true = self(batch)
        return {
            "affinity_true": y_affinity_true,
            "affinity_pred": y_affinity_pred,
            }

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean().item()
        to_print = (
            f"{self.current_epoch:<10}: "
            f"BCE_LOSS: {round(avg_loss, 4)}, "
        )

        print(" TRAIN", to_print)
        self.log("train_eloss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

    def validation_epoch_end(self, outputs):
        y_pred = torch.concat([x["affinity_pred"] for x in outputs])
        y_true = torch.concat([x["affinity_true"] for x in outputs])
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
        self.log("val_loss", mse, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val_el1", mae, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val_ermse", rmse, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

    def test_epoch_end(self, outputs):
        y_pred = torch.concat([x["affinity_pred"] for x in outputs])
        y_true = torch.concat([x["affinity_true"] for x in outputs])
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
        if not model_type.lower() in ["painn", "eqgat", "schnet", "segnn", "egnn", "egnn_edge"]:
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
        elif self.task == 'go':
            classes = [490, 1944, 321]
        elif self.task == 'mf':
            classes = [490]
        elif self.task == 'bp':
            classes = [1944]
        elif self.task == 'cc':
            classes = [321]

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
        if self.task in ['ec', 'mf', 'bp', 'cc']:
            return f_max[0].item()
        elif self.task == 'go':
            return f_max[0].item(), f_max[1].item(), f_max[2].item(), f_max[3].item()
            
    def forward(self, data: Batch) -> Tuple[Tensor, Tensor]:
        y_property_pred = self.model(data)
        y_property_true = data.functions
        y_property_mask = data.valid_masks
        y_property_true = y_property_true[(y_property_mask == 1).sum(dim=1) > 0, :]
        y_property_pred = y_property_pred[(y_property_mask == 1).sum(dim=1) > 0, :]
        return y_property_pred, y_property_true


    def training_step(self, batch, batch_idx):
        y_property_pred, y_property_true = self(batch)
        bce_loss = self.bce(y_property_pred, y_property_true) * int(len(y_property_pred)!=0)
        loss = 10 * bce_loss
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log("train_bce", bce_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        return {"loss": loss, "bce": bce_loss}

    def validation_step(self, batch, batch_idx):
        y_property_pred, y_property_true = self(batch)
        bce_loss = self.bce(y_property_pred, y_property_true) * int(len(y_property_pred)!=0)
        self.log("val_bce", bce_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log("val_loss", 10 * bce_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        return {
                "property_true": y_property_true,
                "property_pred": y_property_pred
                }

    def test_step(self, batch, batch_idx):
        y_property_pred, y_property_true = self(batch)
        return {
            "property_true": y_property_true,
            "property_pred": y_property_pred
            }

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean().item()
        to_print = (
            f"{self.current_epoch:<10}: "
            f"BCE_LOSS: {round(avg_loss, 4)}, "
        )

        print(" TRAIN", to_print)
        self.log("train_eloss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

    def validation_epoch_end(self, outputs):
        property_pred = torch.concat([x["property_pred"] for x in outputs])
        property_true = torch.concat([x["property_true"] for x in outputs])
        bce_loss = self.bce(property_pred, property_true)
        if self.task in ['ec', 'bp', 'mf', 'cc']:
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

    def test_epoch_end(self, outputs):
        property_pred = torch.concat([x["property_pred"] for x in outputs])
        property_true = torch.concat([x["property_true"] for x in outputs])
        if self.task in ['ec', 'bp', 'mf', 'cc']:
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
        return fmax_all
    



