import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Any, Dict
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning import loggers as pl_loggers
from base_model import EEGTransformer, EEGTransformerReconstructor
from utils import grad_logger, get_data, CosineWDSchedule, get_bcic2020t3_data
import os
# from dataset.seedv_dataset import LoadSeedvDataset
# from dataset.tueg_dataset import generate_dataset_tuab
# from dataset.seediv_dataset import get_seed4_data
from dataset.mmwm_dataset import get_wm_data_per_subject_split, temporal_interpolation
from dataset_info import mmwm_channels, bcic2a_channels, selected_bcic2020t3_channels, selected_bcic2020t3_idx
from torch.utils.data import ConcatDataset

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
                 beta=0.25, freeze_encoder=True, save_dir="./saved_codebook", save_name="best_codebook.pt", save_last=True):
        super().__init__()    
        
        self.beta = beta
        self.freeze_encoder = freeze_encoder
        self.save_dir = save_dir
        self.save_name = save_name
        self.patch_size = 64
        self.temperature = 1.0
        self.save_last = save_last
        os.makedirs(save_dir, exist_ok=True)

        encoder = EEGTransformer(
            img_size=[56, 512],
            patch_size=32*2,
            embed_num=4,
            embed_dim=256,
            depth=8,
            num_heads=4,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
        
        reconstructor = EEGTransformerReconstructor(
            num_patches=encoder.num_patches,
            patch_size=32*2,
            embed_num=4,
            embed_dim=codebook_size[1],
            depth=8,
            num_heads=4,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6))

        self.target_tokens = encoder.num_patches[0] * encoder.num_patches[1]

        pretrain_ckpt = torch.load(load_path, map_location="cpu", weights_only=False)
        
        target_encoder_stat = {}
        for k,v in pretrain_ckpt['state_dict'].items():
            if k.startswith("target_encoder."):
                target_encoder_stat[k[15:]]=v
            
        self.encoder        = encoder
        self.encoder.load_state_dict(target_encoder_stat)
        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        print(selected_bcic2020t3_channels)
        self.chans_id = encoder.prepare_chan_ids(selected_bcic2020t3_channels)
        self.TUAB_KEEP_IDX = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            10, 11, 12, 13, 14, 15,
            18, 19, 20,
        ]
        self.BCIC2020T3_KEEP_IDX = selected_bcic2020t3_idx

        # Codebook
        num_codes, code_dim = codebook_size
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.encoder_dim = self.encoder.embed_dim

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
        
        # Deocder
        self.decoder = reconstructor

        self.loss_fn        = torch.nn.MSELoss()
        self.closs = torch.nn.CrossEntropyLoss()
        self.register_buffer("best_valid_loss", torch.tensor(float("inf")))
        
        self.linear = nn.Sequential(
            nn.Linear(256, 16),
            nn.Dropout(0.3),
            nn.Flatten(start_dim=1),
            nn.Linear(64*16, 4)
        )

    def forward_target(self, x):
        with torch.no_grad():
            C, N = self.encoder.num_patches
            assert x.shape[-1]%N==0 and x.shape[-2]%C == 0
            block_size_c, block_size_n = x.shape[-2]//C, x.shape[-1]//N
            x = x.view(x.shape[0], C, block_size_c, N, block_size_n)
            x = x.permute(0, 3, 1, 2, 4).contiguous() # B, N, C, bc, bn
            x = x.view(x.shape[0], C, N, block_size_c * block_size_n)
            x = x.view(x.shape[0], C * N, block_size_c * block_size_n)
            y = F.layer_norm(x, (x.size(-1),))
            return y # [B, C*N, patch_size]
        
    def encode_eeg(self, x):
        if self.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                z = self.encoder(x, self.chans_id.to(x))
        else:
            z = self.encoder(x, self.chans_id.to(x))

        if z.dim() == 4:
            # [B, N, E, D] -> [B, N*E, D]
            B, N, E, D = z.shape
            z = z.reshape(B, N * E, D)
        elif z.dim() == 3:
            # [B, S, D]
            pass
        else:
            raise ValueError(f"Unexpected encoder output shape: {z.shape}")

        return z


    def forward(self, x):
        """
        x: [B, C, T]

        return dict:
            pred: [B, C*N, patch_size]
            target: [B, C*N, patch_size]
        """
        target = self.forward_target(x)  # [B, C*N, patch_size]
        z = self.encode_eeg(x)  # [B, S=N*E, D]
        z_e = self.to_code_dim(z)  # [B, S, code_dim]
        z_q_prim, token_ids, vq_stats_prim = self.primary_vq(z_e, self.temperature)  # [B, S, code_dim]
        residual_input = z_e - z_q_prim.detach()
        z_q_resi, _, vq_stats_resi = self.residual_vq(residual_input, self.temperature)  # [B, S, code_dim]
        z_q_st = z_q_prim + z_q_resi
        z_q_st = z_q_st.view(z_q_st.shape[0], self.encoder.num_patches[1], self.encoder.embed_num, self.code_dim)
        pred = self.decoder(z_q_st, self.chans_id.to(x))  # [B, C*N, patch_size]

        return {
            "pred": pred,
            "target": target,
            "encode": z,
            "z_e": z_e,
            "z_q": z_q_st,
            "token_ids": token_ids,
            "vq_stats": (vq_stats_prim, vq_stats_resi),
        }


    def _shared_step(self, batch, stage: str):
        x, y = batch
        x = x[:,self.BCIC2020T3_KEEP_IDX,:]
        # x = temporal_interpolation(x, 512)
        # x = x[:, self.TUAB_KEEP_IDX, :]
        # x = x[:,:,:960]
        # x = x.float()/100
        out = self.forward(x)

        pred = out["pred"]
        target = out["target"]

        # feature = out["encode"]
        # class_logit = self.linear(feature)
        # y = y.long().view(-1)
        # class_loss = self.closs(class_logit, y)

        recon_loss = self.loss_fn(pred, target)
        vq_loss_prim = out["vq_stats"][0]["vq_loss"]
        vq_loss_resi = out["vq_stats"][1]["vq_loss"]
        vq_loss =  vq_loss_prim + vq_loss_resi
        # codebook_loss = out["vq_stats"]["codebook_loss"]
        # commit_loss = out["vq_stats"]["commit_loss"]

        loss = recon_loss + vq_loss# + class_loss * 0.1

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
                        # "codebook_weight_resi": self.residual_vq.codebook.weight.detach().cpu(),
                        "to_code_dim": self.to_code_dim.state_dict(),
                        "encoder": self.encoder.state_dict(),
                        "encoder_num_patches": self.encoder.num_patches,
                        "patch_size": self.patch_size,
                        "target_tokens": self.target_tokens,
                        "codebook_size": (self.num_codes, self.code_dim),
                    },
                    save_path,
                )

                # print(f"[Best Codebook Saved] epoch={self.current_epoch}, valid_loss={valid_loss.item():.6f}, path={save_path}")

        if self.trainer.is_global_zero and self.save_last:
            last_name = self.save_name.replace(".pt", "_last.pt")
            last_path = os.path.join(self.save_dir, last_name)
            torch.save(
                    {
                        "epoch": self.current_epoch,
                        "valid_loss": valid_loss.cpu(),
                        "codebook_weight_prim": self.primary_vq.codebook.weight.detach().cpu(),
                        # "codebook_weight_resi": self.residual_vq.codebook.weight.detach().cpu(),
                        "to_code_dim": self.to_code_dim.state_dict(),
                        "encoder": self.encoder.state_dict(),
                        "encoder_num_patches": self.encoder.num_patches,
                        "patch_size": self.patch_size,
                        "target_tokens": self.target_tokens,
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
    '''
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        res = super().on_load_checkpoint(checkpoint)

        # self.configure_optimizers()
        return res
    '''
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
    max_epochs = 100
    max_lr = 5e-4
    batch_size= 64
    devices=[0, 1, 2]
    data_path = "~/data/downtasks/BCIC_2a_0_38HZ"
    data_path_seedv = "~/data/downtasks/preprocessed_SEEDV2"
    data_path_tuev = "~/data/downtasks/TUEV/tuh_eeg_events/edf/processed"
    data_path_seediv = "~/data/downtasks/SEED4/processed_data"
    data_path_MMWM = '~/data/multi_model/EEG/Sub_S1_single'
    data_path_bcic2020t3 = "~/data/downtasks/BCIC2020_T3/preprocessed_BCIC2020_T3"
    for i in [1]:
        print(f"####################### DIMEN = {256} ########################")
        '''
        # bcic
        train_dataset,valid_dataset,test_dataset = get_data(1,data_path,1,is_few_EA = True, target_sample=1024)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
        test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, num_workers=4, shuffle=False)
        '''
        '''
        # seedv
        load_dataset = LoadSeedvDataset(data_path_seedv, batch_size)
        data_loader = load_dataset.get_data_loader()
        train_loader, valid_loader, test_loader = data_loader['train'], data_loader['val'], data_loader['test']
        '''
        # tueg
        '''
        train_dataset, test_dataset, valid_dataset, ch_names, metrics = generate_dataset_tuab(dataset='TUEV', data_path=data_path_tuev,)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
        test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, num_workers=4, shuffle=False)
        '''
        #seediv
        '''
        train_dataset,valid_dataset,test_dataset = get_seed4_data(i, data_path_seediv, 0.8, False, use_channels=seediv_channels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
        test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, num_workers=4, shuffle=False)
        '''
        # MMWM
        # train_dataset,valid_dataset,test_dataset = get_wm_data_per_subject_split(data_path=data_path_MMWM, split=(0.8, 0., 0.2), use_channels=mmwm_channels)
        # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
        # valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
        # test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, num_workers=4, shuffle=False)

        # bcic2020t3
        train_dataset,valid_dataset,test_dataset = get_bcic2020t3_data(data_path_bcic2020t3)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
        test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, num_workers=4, shuffle=False)

        global steps_per_epoch
        steps_per_epoch = len(train_loader)
        size = 512 # 512
        dimension = 256 # 256
        print(len(train_loader) * batch_size, len(valid_loader) * batch_size, len(test_loader) * batch_size)
        model = LitCodebook(load_path="./DARE-EEG_base.ckpt", 
                            codebook_size=[size, dimension], beta=0.25, 
                            freeze_encoder=False, 
                            save_dir="./saved_codebook", save_name=f"rcb_adapt_gumbel_bcic2020t3.pt", save_last=False)
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

        callbacks = [lr_monitor]
        trainer = pl.Trainer(accelerator="gpu", strategy='ddp_find_unused_parameters_true', devices=devices, max_epochs=max_epochs, callbacks=callbacks, 
                            logger=[pl_loggers.TensorBoardLogger('./logs/', name=f"cb_bcic2020t3_tb", version=f'{i}'), 
                                    pl_loggers.CSVLogger('./logs/', name=f"rcb_csv_bcic2020t3")])
        trainer.fit(model, train_loader, valid_loader)
