import argparse
from collections import defaultdict
import os
import random
import numpy as np
import torch
from model import Model
from TimeLLM import Model as TimeLLMModel
from data_provider.data_factory import data_provider
import torch.nn as nn

import pytorch_lightning as pl
import copy
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from utils import get_metrics

def seed_everything(seed: int):
	random.seed(seed)
	# os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

class ClassHead(nn.Module):
    def __init__(self, N, E, D, num_classes, dropout_rate=0):
        super(ClassHead, self).__init__()
        # self.norm = nn.LayerNorm(D)
        self.linear = nn.Linear(D, 32)
        self.dropout = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear2 = nn.Linear(32 * N * E, num_classes)
        # self.linear2 = nn.Linear(6272, num_classes)

    def forward(self, x):
        # x = self.norm(x)
        x = self.linear(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.linear2(x)
        return x

class LitEEGClassifier(pl.LightningModule):
    def __init__(self, args, codebook_load_path, steps_per_epoch: int, log_path: str = None, sub: int = 1):
        super().__init__()
        self.args = args
        self.codebook_load_path = codebook_load_path
        self.steps_per_epoch = steps_per_epoch
        self.log_path = log_path
        self.sub = sub
        self.is_binary = args.is_binary
        self.save_cb_ids = False

        self.model = Model(args, codebook_load_path)
        # self.model = TimeLLMModel(args)
        # class_weights = [18.3945,  1.1158,  1.9996, 21.8434,  1.3869,  0.2644]
        self.criterion = nn.CrossEntropyLoss()

        self.save_hyperparameters(vars(args))

        self.all_logits = []
        self.all_y = []
        self.all_cb_ids = []
        self.all_y_train = []

        self.label_names = {
            0: "left_hand",
            1: "right_hand",
        }
        self.codebook_size = args.num_codes

        self.best_epoch = -1
        self.best_res = None
    def _compute_label_codebook_statistics(
        self,
        all_cb_ids,
        all_labels,
    ):
        all_cb_ids = all_cb_ids.long().cpu()
        all_labels = all_labels.long().reshape(-1).cpu()

        if all_cb_ids.ndim != 2:
            raise ValueError(
                f"all_cb_ids should have shape [N, S], "
                f"but received {tuple(all_cb_ids.shape)}"
            )

        if all_cb_ids.shape[0] != all_labels.shape[0]:
            raise ValueError(
                "The number of samples in all_cb_ids and all_labels "
                "must be identical."
            )

        unique_labels = torch.unique(all_labels).sort().values
        statistics = {}

        for label_tensor in unique_labels:
            label = int(label_tensor.item())

            class_mask = all_labels == label

            class_code_ids = all_cb_ids[class_mask].reshape(-1)

            counts = torch.bincount(
                class_code_ids,
                minlength=self.codebook_size,
            )

            total_tokens = int(counts.sum().item())

            if total_tokens == 0:
                probabilities = torch.zeros_like(
                    counts,
                    dtype=torch.float32,
                )
                perplexity = 0.0
            else:
                probabilities = counts.float() / total_tokens

                nonzero_probabilities = probabilities[
                    probabilities > 0
                ]

                entropy = -torch.sum(
                    nonzero_probabilities
                    * torch.log(nonzero_probabilities)
                )

                perplexity = float(torch.exp(entropy).item())

            used_codes = int((counts > 0).sum().item())

            top_k = min(20, self.codebook_size)

            top_counts, top_ids = torch.topk(
                counts,
                k=top_k,
            )

            valid_top_mask = top_counts > 0

            top_ids = top_ids[valid_top_mask]
            top_counts = top_counts[valid_top_mask]

            if total_tokens > 0:
                top_probabilities = (
                    top_counts.float() / total_tokens
                )
            else:
                top_probabilities = torch.zeros_like(
                    top_counts,
                    dtype=torch.float32,
                )

            statistics[label] = {
                "num_samples": int(class_mask.sum().item()),
                "num_tokens": total_tokens,
                "used_codes": used_codes,
                "perplexity": perplexity,
                "counts": counts.numpy(),
                "probabilities": probabilities.numpy(),
                "top_ids": top_ids.numpy(),
                "top_counts": top_counts.numpy(),
                "top_probabilities": top_probabilities.numpy(),
            }

        return statistics
    
    def _save_codebook_statistics(self,statistic,epoch,codebook_name="codebook",):
        labels = np.array(sorted(statistic.keys()),dtype=np.int64,)
        # [num_classes, num_codes]
        counts = np.stack([
            np.asarray(
                statistic[int(label)]["counts"],
                dtype=np.int64,
            )
            for label in labels
        ])

        # [num_classes, num_codes]
        probabilities = np.stack([
            np.asarray(
                statistic[int(label)]["probabilities"],
                dtype=np.float32,
            )
            for label in labels
        ])
        max_top_k = max(
            len(statistic[int(label)]["top_ids"])
            for label in labels
        )
        top_ids = np.full(
            shape=(len(labels), max_top_k),
            fill_value=-1,
            dtype=np.int64,
        )
        for row, label in enumerate(labels):
            ids = np.asarray(
                statistic[int(label)]["top_ids"],
                dtype=np.int64,
            )
            top_ids[row, :len(ids)] = ids
        save_dir = os.path.join(
            self.trainer.default_root_dir,
            "codebook_statistics",
        )
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(
            save_dir,
            f"{codebook_name}_epoch_{epoch:02d}.npz",
        )

        np.savez_compressed(
            save_path,
            labels=labels,
            counts=counts,
            probabilities=probabilities,
            top_ids=top_ids,
        )

        print(f"Codebook statistics saved to: {save_path}")

    def forward(self, x, y=None):
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        x, y = batch
        x = x.float()
        y = y.long().view(-1)
        # s = s.long().view(-1)
        logits, cb_ids = self.forward(x, y)

        # logits should be [B, num_class]
        loss = self.criterion(logits, y)
        # if stage == 'valid' or stage == 'test':
        #     subject_loss = 0.0
        # else:
        #     subject_loss = self.criterion(subject_logits, s)
        loss = loss # + subject_loss * 0.02

        if stage == 'valid' or stage == 'test':
            all_logits = self.all_gather(logits).detach()
            all_y = self.all_gather(y).detach()

            all_logits = all_logits.reshape(-1, logits.size(-1))
            all_y = all_y.reshape(-1)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        bs = x.size(0)

        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs,)
        # self.log(f"{stage}_subject_loss", subject_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=bs,)
        self.log(f"{stage}_acc", acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs,)

        if stage == 'valid' or stage == 'test':
            return all_logits, all_y, loss
        if self.save_cb_ids:
            gather_cb_ids = self.all_gather(cb_ids).detach()
            gather_cb_ids = gather_cb_ids.reshape(-1,cb_ids.shape[-1])
            self.all_cb_ids.append(gather_cb_ids.cpu())
            gather_y_train = self.all_gather(y).detach()
            gather_y_train = gather_y_train.reshape(-1)
            self.all_y_train.append(gather_y_train.cpu())
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, stage="train")
        return loss

    def validation_step(self, batch, batch_idx):
        logits, y, loss = self._shared_step(batch, stage="valid")
        self.all_logits.append(logits.cpu())
        self.all_y.append(y.cpu())
        return loss
    
    def on_validation_epoch_start(self):
        # if self.current_epoch == 4:
        #     self.model.save_mid = False
        self.all_logits = []
        self.all_y = []
    
    def on_train_epoch_start(self):
        # if self.current_epoch == 4:
        #     self.model.save_mid = True
        self.all_cb_ids = []
        self.all_y_train = []

    def on_train_epoch_end(self):
        if not self.save_cb_ids:
            return
        if self.current_epoch == 0 or self.current_epoch == 10:
            all_cb_ids = torch.cat(self.all_cb_ids)
            all_labels = torch.cat(self.all_y_train)
            print(all_cb_ids.shape)
            print(all_labels.shape)
            if all_cb_ids.shape[0] != all_labels.shape[0]:
                raise RuntimeError(
                    "The final number of code-ID samples does not match "
                    "the final number of labels: "
                    f"{all_cb_ids.shape[0]} vs {all_labels.shape[0]}"
                )
            if self.trainer.is_global_zero:
                statistic = self._compute_label_codebook_statistics(all_cb_ids, all_labels)
                codebook_name="bcic2b_cbinfo"
                self._save_codebook_statistics(
                    statistic=statistic,
                    epoch=self.current_epoch,
                    codebook_name=codebook_name,
                )
                save_dir = os.path.join(self.trainer.default_root_dir,"codebook_statistics",)
                save_path = os.path.join(save_dir,f"{codebook_name}_vector_epoch_{self.current_epoch:02d}.npz",)
                np.savez_compressed(
                    save_path,
                    labels=all_labels,
                    cb_ids=all_cb_ids
                )


    def on_validation_epoch_end(self):
        if len(self.all_logits) == 0:
            return

        logits = torch.cat(self.all_logits, dim=0)
        y = torch.cat(self.all_y, dim=0)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        if self.is_binary:
            metrics = ['balanced_acc', 'kappa', 'auroc', 'auprc'] # acc, kappa, weighted f1
            probs = torch.softmax(logits, dim=1)[:, 1]
        else:
            metrics = ['balanced_acc', 'f1', 'kappa']
            probs = torch.softmax(logits, dim=1)

        y = y.cpu().numpy()
        preds = preds.cpu().numpy()
        probs = probs.cpu().numpy()
        results = get_metrics(metrics=metrics, is_binary=self.is_binary, all_labels=y, all_preds=preds, all_probs=probs)

        for k, v in results.items():
            if v is None:
                continue
            if k == 'loss' or k == 'balanced_acc':
                self.log(f"valid_{k}", float(v), prog_bar=True, sync_dist=False)
            else:
                self.log(f"valid_{k}", float(v), prog_bar=False, sync_dist=False)

        if self.best_epoch == -1 or results['balanced_acc'] > self.best_res['balanced_acc']:
            self.best_epoch = self.current_epoch
            self.best_res = results
            self.chead_weight = copy.deepcopy(self.model.output_projection.state_dict())
    
    def on_fit_end(self):
        if self.log_path is not None and self.trainer.is_global_zero:
            with open(self.log_path, 'a') as f:
                f.write(f"{self.sub} Epoch:{self.best_epoch} {self.best_res}\n")
                # torch.save(self.chead_weight,os.path.join('./logs/checkpoints', f"bcic2b_sub{self.sub}_class_head.pth"))

    def test_step(self, batch, batch_idx):
        logits, y, loss = self._shared_step(batch, stage="test")
        return loss
    

    def configure_optimizers(self):
        # only optimize trainable params
        trainable_params = [p for p in self.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.args.learning_rate,
            weight_decay=getattr(self.args, "weight_decay", 5e-4),
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.args.learning_rate,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.args.train_epochs,
            pct_start=self.args.pct_start,
            div_factor=2,
            final_div_factor=8,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.args.train_epochs * self.steps_per_epoch, eta_min=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def summarize_trainable_by_level(model, level=2):
    stats = defaultdict(lambda: {"trainable": 0, "total": 0, "n_trainable": 0, "n_total": 0})

    for name, param in model.named_parameters():
        parts = name.split(".")
        module_name = ".".join(parts[:level]) if len(parts) >= level else name

        num = param.numel()
        stats[module_name]["total"] += num
        stats[module_name]["n_total"] += 1

        if param.requires_grad:
            stats[module_name]["trainable"] += num
            stats[module_name]["n_trainable"] += 1

    print("\n========== Trainable Summary by Module ==========")
    print(f"{'Module':45s} | {'Status':10s} | {'Trainable / Total Params':28s} | {'Trainable / Total Tensors'}")
    print("-" * 120)

    for module_name, s in sorted(stats.items()):
        trainable = s["trainable"]
        total = s["total"]
        ratio = 100 * trainable / total if total > 0 else 0.0

        if trainable == 0:
            status = "Frozen"
        elif trainable == total:
            status = "Trainable"
        else:
            status = "Partial"

        print(
            f"{module_name:45s} | {status:10s} | "
            f"{trainable:,} / {total:,} ({ratio:6.2f}%)      | "
            f"{s['n_trainable']} / {s['n_total']}"
        )

    print("=" * 120)

def main():
    parser = argparse.ArgumentParser(description=None)
    # basic config
    parser.add_argument("--log_dir", type=str, default="./logs/",
                        help="where to save the results")
    parser.add_argument("--model_comment", type=str, default="none",
                        help="prefix when saving test results")
    parser.add_argument("--model", type=str, default="EEGLLM",
                        help="model name, options: [Autoformer, DLinear, EEGLLM]")
    parser.add_argument("--codebook_load_path", type=str, default="./codebook/", help="path to load codebook")
    parser.add_argument("--load_path", type=str, default="./DARE-EEG_base.ckpt", help="Foundation model checkpoint path")
    parser.add_argument("--seed", type=int, default=2024, help="random seed")

    # data info
    parser.add_argument("--description", type=str, default="none", help="dataset description for prompt")
    parser.add_argument("--data", type=str, required=True,  help="dataset type")
    parser.add_argument("--data_path", type=str, default="eeg.npz", help="data file")
    parser.add_argument("--loader", type=str, default="eeg", help="dataset loader type")
    parser.add_argument("--checkpoints", type=str, default="./logs/checkpoints/", help="location of model checkpoints")
    parser.add_argument("--num_workers", type=int, default=8, help="data loader num workers")

    parser.add_argument("--seq_len", type=int, default=1000, help="EEG time length T")
    parser.add_argument("--enc_in", type=int, default=64, help="number of EEG channels C")
    parser.add_argument("--num_class", type=int, default=2, help="number of classes")
    parser.add_argument("--num_subjects", type=int, default=8, help="number of subjects for subject discriminator")
    parser.add_argument("--is_binary", action="store_true", help="binary classification")
    
    # Model info
    # eeg encoder
    parser.add_argument("--hidden_dim", type=int, default=64, help="hidden dim of eeg encoder")
    parser.add_argument("--depth", type=int, default=6, help="depth of eeg encoder")
    parser.add_argument("--num_heads", type=int, default=4, help="heads of eeg encoder")
    parser.add_argument("--mlp_ratio", type=float, default=2, help="mlp ratio of eeg encoder")
    parser.add_argument("--patch_size", type=int, default=64, help="embedding patch size of eeg encoder")
    parser.add_argument("--fusion_dim", type=int, default=128, help="fusion dimension")
    parser.add_argument("--embed_num", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    # codebook
    parser.add_argument("--num_codes", type=int, default=512)
    parser.add_argument("--code_dim", type=int, default=256)
    parser.add_argument("--beta", type=float, default=0.25)

    # llm info
    parser.add_argument("--prompt_domain", type=int, default=0, help="")
    parser.add_argument("--llm_model", type=str, default="QWEN", help="LLM model")
    parser.add_argument("--llm_dim", type=int, default=4096, help="LLM hidden dimension")
    parser.add_argument("--llm_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=8, help="reprogramming heads")
    parser.add_argument("--d_ff", type=int, default=128, help="reprogramming feedforward dimension")

    # optimization
    parser.add_argument("--itr", type=int, default=1, help="experiments times")
    parser.add_argument("--train_epochs", type=int, default=10, help="train epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size of train input data")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="batch size of model evaluation")
    parser.add_argument("--patience", type=int, default=10, help="early stopping patience")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="optimizer learning rate")
    parser.add_argument("--des", type=str, default="test", help="exp description")
    parser.add_argument("--pct_start", type=float, default=0.2, help="pct_start")

    # LoRa
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    # trainer
    parser.add_argument("--devices", type=str, default="0")
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--strategy", type=str, default="auto")

    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoints, exist_ok=True)

    device_list = [int(d) for d in args.devices.split(",")]

    # Training
    for i in range(1, 2):
        dimen = args.code_dim
        size = args.num_codes
        # print(f"##$##$##$##$## SIZE_BCIC2A_{size} ##$##$##$##$##")
        seed_everything(args.seed)
        codebook_load_path = args.codebook_load_path + f"rcb_adapt_gumbel_bcic2020t3.pt"
        # log_path = args.log_dir + args.data + '.txt'
        log_path = args.log_dir + args.data + '_ours.txt'
        print("LOG TO: ", log_path)
        print("LOAD CODEBOOK FROM: ", codebook_load_path)
        train_loader, val_loader, test_loader = data_provider(args=args, flag="ALL", sub=i)

        steps_per_epoch = len(train_loader)
        '''
        lit_model = LitEEGClassifier.load_from_checkpoint(
            checkpoint_path=(
                "./logs/checkpoints/seediv_sub1/best-epoch=001-valid_loss=1.2076-valid_acc=0.4969.ckpt"
            ),
            args=args,
            codebook_load_path=codebook_load_path,
            steps_per_epoch=steps_per_epoch,
            log_path=log_path,
            sub=i,
        )
        '''
        lit_model = LitEEGClassifier(
            args=args,
            codebook_load_path=codebook_load_path,
            steps_per_epoch=steps_per_epoch,
            log_path=log_path,
            sub=i,
        )
        tb_logger = pl_loggers.TensorBoardLogger(
            save_dir=args.log_dir,
            name=f"{args.data}_ours_tb",
            version=f"cb_sub{i}"
        )

        csv_logger = pl_loggers.CSVLogger(
            save_dir=args.log_dir,
            name=f"{args.data}_csv",
        )

        ckpt_callback = ModelCheckpoint(
            dirpath=os.path.join(args.checkpoints, f"{args.data}_sub{i}"),
            filename="best-{epoch:03d}-{valid_loss:.4f}-{valid_acc:.4f}",
            monitor="valid_balanced_acc",
            mode="max",
            save_top_k=1,
            save_last=True,
        )

        lr_monitor = LearningRateMonitor(logging_interval="step")

        callbacks = [
            # ckpt_callback,
            lr_monitor,
        ]

        trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=device_list if torch.cuda.is_available() else 1,
            # gradient_clip_val=5.0,
            # gradient_clip_algorithm="norm",
            strategy=args.strategy,
            # strategy="ddp_find_unused_parameters_true",
            max_epochs=args.train_epochs,
            callbacks=callbacks,
            logger=[tb_logger, csv_logger],
            precision=args.precision,
            log_every_n_steps=10,
        )
        # lit_model.model.output_projection = ClassHead(16, 4, 256, 2, 0.3)
        # lit_model.model.output_projection.load_state_dict(torch.load("./logs/checkpoints/bcic2b_sub7/bcic2b_sub7_class_head.pth"))
        
        trainer.fit(
            lit_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        
        # trainer.validate(lit_model, test_loader)

if __name__ == '__main__':
    main()
