import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Any, Dict
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning import loggers as pl_loggers
from base_model_codebrain import Model, LightweightRawEEGDecoder
from utils import grad_logger, get_data, CosineWDSchedule
import os
from dataset.seedv_dataset import LoadSeedvDataset
from CTC import ChannelTokenCompressor

bcic2a_channels = [
    'FZ',
    'FC3', 'FC1', 'FCZ', 'FC2', 'FC4',
    'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6',
    'CP3', 'CP1', 'CPZ', 'CP2', 'CP4',
    'P1', 'PZ', 'P2',
    'POZ'
]

bcic2b_channels = ["C3", "Cz", "C4"]

seedv_channels = [
    'FP1', 'FPZ', 'FP2',
    'AF3', 'AF4',
    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO3', 'POZ', 'PO4', 'PO8',
    'O1', 'OZ', 'O2'
]

class VectorQuantizer(nn.Module):
    """
    Standard VQ layer.

    Input:
        z_e: [B, S, D]
    Output:
        z_q_st: [B, S, D], straight-through quantized embeddings
        indices: [B, S], discrete code ids
        loss_dict: codebook_loss, commit_loss, vq_loss, perplexity, used_codes
    """
    def __init__(
        self,
        num_codes: int = 1024,
        code_dim: int = 768,
        beta: float = 0.25,
        init_std: float = 0.02,
        use_cosine: bool = True,
        hard: bool  = True
    ):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.beta = beta
        self.use_cosine = use_cosine
        self.hard = hard

        self.codebook = nn.Embedding(num_codes, code_dim)
        nn.init.normal_(self.codebook.weight, mean=0.0, std=init_std)

    def forward(self, z_e: torch.Tensor, temperature: float = 1):
        """
        z_e: [B, S, D]
        """
        B, S, D = z_e.shape
        assert D == self.code_dim, f"z_e dim {D} != code_dim {self.code_dim}"

        z_flat = z_e.reshape(-1, D)  # [B*S, D]
        code = self.codebook.weight  # [K, D]
        tau = temperature

        if self.use_cosine:
            z_norm = F.normalize(z_flat, dim=-1)
            code_norm = F.normalize(code, dim=-1)
            # cosine distance = 1 - cosine similarity
            sim = torch.matmul(z_norm, code_norm.t())  # [B*S, K]
            dist = 1.0 - sim
            logits = sim
        else:
            # L2 distance: ||z||^2 - 2 z e + ||e||^2
            dist = (
                z_flat.pow(2).sum(dim=1, keepdim=True)
                - 2 * torch.matmul(z_flat, code.t())
                + code.pow(2).sum(dim=1).unsqueeze(0)
            )  # [B*S, K]
            logits = -dist

        if self.training:
            # [B*S, K]
            soft_one_hot = F.gumbel_softmax(
                logits,
                tau=tau,
                hard=self.hard,
                dim=-1,
            )
            z_q = torch.matmul(soft_one_hot, code).view(B, S, D)
            indices = torch.argmax(soft_one_hot, dim=-1)

        else:
            indices = torch.argmin(dist, dim=1)
            z_q = self.codebook(indices).view(B, S, D)
            one_hot = F.one_hot(indices, num_classes=self.num_codes).float()
            soft_one_hot = one_hot

        # VQ-VAE losses
        if self.use_cosine:
            z_e_loss = F.normalize(z_e, dim=-1)
            z_q_loss = F.normalize(z_q, dim=-1)

            codebook_loss = F.mse_loss(z_q_loss, z_e_loss.detach())
            commit_loss = F.mse_loss(z_e_loss, z_q_loss.detach())
        else:
            codebook_loss = F.mse_loss(z_q, z_e.detach())
            commit_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + self.beta * commit_loss

        # straight-through estimator
        # z_q_st = z_e + (z_q - z_e).detach()

        # codebook statistics
        with torch.no_grad():
            hard_indices = indices.detach()
            hard_one_hot = F.one_hot(hard_indices, num_classes=self.num_codes).float()
            avg_probs = hard_one_hot.mean(dim=0)

            perplexity = torch.exp(
                -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
            )
            used_codes = (avg_probs > 0).sum()

        loss_dict = {
            "vq_loss": vq_loss,
            "codebook_loss": codebook_loss,
            "commit_loss": commit_loss,
            "perplexity": perplexity,
            "used_codes": used_codes.float(),
        }

        return z_q, indices.view(B, S), loss_dict

class LitCodebook(pl.LightningModule):

    def __init__(self, load_path="./DAREEG_3_large@epoch=193-valid_loss=0.6080.ckpt", codebook_size=[512, 256], 
                 beta=0.25, freeze_encoder=True, save_dir="./saved_codebook", save_name="best_codebook.pt"):
        super().__init__()    
        
        self.beta = beta
        self.freeze_encoder = freeze_encoder
        self.save_dir = save_dir
        self.save_name = save_name
        self.temperature = 1.0
        os.makedirs(save_dir, exist_ok=True)

        self.encoder = Model(dropout=0, num_res_layers=8, foundation_dir=load_path)
        if self.freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        # Codebook
        num_codes, code_dim = codebook_size
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.encoder_dim = 200

        self.to_code_dim = nn.Linear(self.encoder_dim, code_dim)

        self.primary_vq = VectorQuantizer(
            num_codes=num_codes,
            code_dim=code_dim,
            beta=beta,
            init_std=0.02,
            use_cosine=False,
            hard=True
        )
        
        self.residual_vq = VectorQuantizer(
            num_codes=num_codes,
            code_dim=code_dim,
            beta=beta,
            init_std=0.02,
            use_cosine=False,
            hard=True
        )

        self.patch_size = 200

        self.decoder = LightweightRawEEGDecoder(
            code_dim=code_dim,
            patch_size=self.patch_size,
            hidden_dim=128,
            refine_channels=16,
            kernel_size=7,
            dropout=0.0,
        )

        self.loss_fn        = torch.nn.MSELoss()
        self.register_buffer("best_valid_loss", torch.tensor(float("inf")))
        
    def encode_eeg(self, x):
        if self.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                feature = self.encoder(x)
        else:
            feature = self.encoder(x)
        return feature


    def forward(self, x):
        x = self.encoder.prepare_input(x)
        z = self.encode_eeg(x)  # [B, C, N, D]
        B, C, N, D = z.shape
        z = z.reshape(B, C * N, self.encoder_dim)

        z_e = self.to_code_dim(z)  # [B, C, code_dim]
        z_q_prim, token_ids, vq_stats_prim = self.primary_vq(z_e, self.temperature)  # [B, C, code_dim]
        residual_input = z_e - z_q_prim.detach()
        z_q_resi, _, vq_stats_resi = self.residual_vq(residual_input, self.temperature)  # [B, C, code_dim]
        z_q_sum = z_q_prim + z_q_resi
        z_q_sum = z_q_sum.reshape(B,C,N,self.code_dim)
        pred = self.decoder(z_q_sum) 

        return {
            "pred": pred,
            "target": x,
            "z_e": z_e,
            "z_q": z_q_sum,
            "token_ids": token_ids,
            "vq_stats": (vq_stats_prim, vq_stats_resi),
        }


    def _shared_step(self, batch, stage: str):
        x, _ = batch

        out = self.forward(x)

        pred = out["pred"]
        target = out["target"]

        recon_loss = self.loss_fn(pred, target)
        cosine_loss = 1.0 - F.cosine_similarity(pred, target, dim=-1,).mean()
        vq_loss_prim = out["vq_stats"][0]["vq_loss"]
        vq_loss_resi = out["vq_stats"][1]["vq_loss"]
        vq_loss = vq_loss_resi + vq_loss_prim
        # codebook_loss = out["vq_stats"]["codebook_loss"]
        # commit_loss = out["vq_stats"]["commit_loss"]

        loss = recon_loss + vq_loss + 0.1 * cosine_loss

        self.log(f"{stage}_loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log(f"{stage}_recon_loss", recon_loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log(f"{stage}_vq_loss", vq_loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log(f"{stage}_vq_loss_prim", vq_loss_prim, on_epoch=True, on_step=False, sync_dist=True)
        self.log(f"{stage}_vq_loss_resi", vq_loss_resi, on_epoch=True, on_step=False, sync_dist=True)
        # self.log(f"{stage}_codebook_loss", codebook_loss, on_epoch=True, on_step=False, sync_dist=True)
        # self.log(f"{stage}_commit_loss", commit_loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log(f"{stage}_perplexity_prim", out["vq_stats"][0]["perplexity"], on_epoch=True, on_step=False, sync_dist=True)
        self.log(f"{stage}_used_codes_prim", out["vq_stats"][0]["used_codes"], on_epoch=True, on_step=False, sync_dist=True)
        self.log(f"{stage}_perplexity_resi", out["vq_stats"][1]["perplexity"], on_epoch=True, on_step=False, sync_dist=True)
        self.log(f"{stage}_used_codes_resi", out["vq_stats"][1]["used_codes"], on_epoch=True, on_step=False, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="valid")
    
    def on_validation_epoch_end(self):
        """
        Save the codebook with the lowest valid_loss.
        This saves only the useful tokenizer components instead of the full checkpoint.
        """
        if self.trainer.sanity_checking:
            return
        self.log('temperature', self.temperature, on_epoch=True, on_step=False, sync_dist=False)
        self.temperature = max(0.5, self.temperature * 0.99)
        valid_loss = self.trainer.callback_metrics.get("valid_loss", None)

        if valid_loss is None:
            return

        valid_loss = valid_loss.detach()

        if valid_loss < self.best_valid_loss:
            self.best_valid_loss.copy_(valid_loss)

            if self.trainer.is_global_zero:
                save_path = os.path.join(self.save_dir, self.save_name)

                torch.save(
                    {
                        "epoch": self.current_epoch,
                        "valid_loss": valid_loss.cpu(),
                        "codebook_weight_prim": self.primary_vq.codebook.weight.detach().cpu(),
                        "codebook_weight_resi": self.residual_vq.codebook.weight.detach().cpu(),
                        "encoder": self.encoder.state_dict(),
                        "to_code_dim": self.to_code_dim.state_dict(),
                        "codebook_size": (self.num_codes, self.code_dim),
                    },
                    save_path,
                )

                # print(f"[Best Codebook Saved] epoch={self.current_epoch}, valid_loss={valid_loss.item():.6f}, path={save_path}")

        if self.trainer.is_global_zero:
            last_name = self.save_name.replace(".pt", "_last.pt")
            last_path = os.path.join(self.save_dir, last_name)
            torch.save(
                    {
                        "epoch": self.current_epoch,
                        "valid_loss": valid_loss.cpu(),
                        "codebook_weight_prim": self.primary_vq.codebook.weight.detach().cpu(),
                        "codebook_weight_resi": self.residual_vq.codebook.weight.detach().cpu(),
                        "to_code_dim": self.to_code_dim.state_dict(),
                        "encoder": self.encoder.state_dict(),
                        "codebook_size": (self.num_codes, self.code_dim),
                    },
                    last_path,
                )
    
    def on_train_batch_start(self, batch: Any, batch_idx: int):
        self.wd_scheduler.step()
        
        return super().on_train_batch_start(batch, batch_idx)
    
    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        grad_stats = grad_logger(self.decoder.named_parameters())
        self.log('grad_stats.first_layer', grad_stats.first_layer, on_epoch=True, on_step=False, sync_dist=True)
        self.log('grad_stats.last_layer', grad_stats.last_layer, on_epoch=True, on_step=False, sync_dist=True)
        self.log('grad_stats.min', grad_stats.min, on_epoch=True, on_step=False, sync_dist=True)
        self.log('grad_stats.max', grad_stats.max, on_epoch=True, on_step=False, sync_dist=True)
                
        return super().on_train_batch_end(outputs, batch, batch_idx)
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        res = super().on_load_checkpoint(checkpoint)

        self.configure_optimizers()
        return res
    
    def configure_optimizers(self):

        def decay_filter(name, param):
            return param.requires_grad and ("bias" not in name) and (len(param.shape) != 1)

        def no_decay_filter(name, param):
            return param.requires_grad and (("bias" in name) or (len(param.shape) == 1))

        modules = {
            "to_code_dim": self.to_code_dim,
            "vq_prim": self.primary_vq,
            "vq_resi": self.residual_vq,
            "decoder": self.decoder,
        }

        if not self.freeze_encoder:
            modules["encoder"] = self.encoder

        decay_params = []
        no_decay_params = []

        for module_name, module in modules.items():
            for n, p in module.named_parameters():
                full_name = f"{module_name}.{n}"
                if decay_filter(full_name, p):
                    decay_params.append(p)
                elif no_decay_filter(full_name, p):
                    no_decay_params.append(p)

        param_groups = [
            {
                "params": decay_params,
            },
            {
                "params": no_decay_params,
                "WD_exclude": True,
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(param_groups, lr=6e-5, weight_decay=1e-6)

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=max_epochs,
            div_factor=2,
            final_div_factor=8,
            pct_start=0.2,
        )

        lr_dict = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "monitor": "valid_loss",
            "strict": True,
            "name": None,
        }

        self.wd_scheduler = CosineWDSchedule(
            optimizer,
            ref_wd=1e-6,
            final_wd=1e-6,
            T_max=int(max_epochs * steps_per_epoch),
        )

        return {"optimizer": optimizer, "lr_scheduler": lr_dict}

if __name__ == "__main__":
    max_epochs = 50
    max_lr = 5e-4
    batch_size= 64
    devices=[0, 1, 2]
    data_path = "~/data/downtasks/BCIC_2a_0_38HZ"
    data_path_seedv = "~/data/downtasks/preprocessed_SEEDV2"
    for i in range(1, 2):
        '''
        train_dataset,valid_dataset,test_dataset = get_data(i,data_path,1,is_few_EA = True, target_sample=600)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
        test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, num_workers=4, shuffle=False)
        '''
        
        load_dataset = LoadSeedvDataset(data_path_seedv, batch_size)
        data_loader = load_dataset.get_data_loader()
        train_loader, valid_loader, test_loader = data_loader['train'], data_loader['val'], data_loader['test']
        
        global steps_per_epoch
        steps_per_epoch = len(train_loader)
        print(len(train_loader) * batch_size, len(valid_loader) * batch_size, len(test_loader) * batch_size)
        model = LitCodebook(load_path="~/code/NeuroCode-LLM/training/comparasive_methods/CodeBrain.pth", 
                            codebook_size=[512, 256], beta=0.25, 
                            freeze_encoder=False, 
                            save_dir="./saved_codebook", save_name=f"rcb_adapt_gumbel_seedv_codebrain.pt")
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')


        callbacks = [lr_monitor]
        trainer = pl.Trainer(accelerator="gpu", strategy='ddp_find_unused_parameters_true', devices=devices, max_epochs=max_epochs, callbacks=callbacks, 
                            logger=[pl_loggers.TensorBoardLogger('./logs/', name=f"cb_seedv_codebrain_tb", version=f'{i}_gumbel_residual'), 
                                    pl_loggers.CSVLogger('./logs/', name=f"cb_csv_seedv")])
        trainer.fit(model, train_loader, valid_loader)
