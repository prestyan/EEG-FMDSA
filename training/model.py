from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
import transformers
from encoder import EEGTransformer, LinearWithConstraint
from peft import LoraConfig, get_peft_model, TaskType
from functools import partial
import torch.nn.functional as F
from datainfo_provider import *
from GRL import GradientReverseLayer, SubjectClassifier
from utils import temporal_interpolation, Conv1dWithConstraint
from transformers import AutoConfig, AutoModel, AutoTokenizer
import numpy as np
transformers.logging.set_verbosity_error()

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x
    

class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_codes: int = 1024,
        code_dim: int = 768,
        beta: float = 0.25,
        init_std: float = 0.02,
        use_cosine: bool = True,
    ):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.beta = beta
        self.use_cosine = use_cosine

        self.codebook = nn.Embedding(num_codes, code_dim)
        nn.init.normal_(self.codebook.weight, mean=0.0, std=init_std)

    def forward(self, z_e: torch.Tensor):
        """
        z_e: [B, S, D]
        """
        B, S, D = z_e.shape
        assert D == self.code_dim, f"z_e dim {D} != code_dim {self.code_dim}"

        z_flat = z_e.reshape(-1, D)  # [B*S, D]
        code = self.codebook.weight  # [K, D]

        if self.use_cosine:
            z_norm = F.normalize(z_flat, dim=-1)
            code_norm = F.normalize(code, dim=-1)
            # cosine distance = 1 - cosine similarity
            sim = torch.matmul(z_norm, code_norm.t())  # [B*S, K]
            dist = 1.0 - sim
        else:
            # L2 distance: ||z||^2 - 2 z e + ||e||^2
            dist = (
                z_flat.pow(2).sum(dim=1, keepdim=True)
                - 2 * torch.matmul(z_flat, code.t())
                + code.pow(2).sum(dim=1).unsqueeze(0)
            )  # [B*S, K]

        indices = torch.argmin(dist, dim=1)
        z_q = self.codebook(indices).view(B, S, D)

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
        z_q_st = z_e + (z_q - z_e).detach()

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

        return z_q_st, indices.view(B, S), loss_dict
    
class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape # [B*C, patch num, patch dim]
        S, _ = source_embedding.shape # [S, d_llm(768)]
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
    
class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)


class Model(nn.Module):

    def __init__(self, configs, cb_load_path):
        super(Model, self).__init__()
        self.num_classes = configs.num_class
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.cb_load_path = cb_load_path
        self.data = configs.data
        self.save_mid = False

        if configs.data == 'bcic2a':
            self.idx1 = C3_IDX
            self.idx2 = CZ_IDX
            self.idx3 = C4_IDX
        elif configs.data == 'bcic2b':
            self.idx1 = C3_IDX_BCIC2B
            self.idx2 = CZ_IDX_BCIC2B
            self.idx3 = C4_IDX_BCIC2B
        elif configs.data == 'seedv' or configs.data == 'seediv':
            self.idx1 = F3_IDX_SEEDV
            self.idx2 = FZ_IDX_SEEDV
            self.idx3 = F4_IDX_SEEDV
        elif configs.data == 'mmwm' or configs.data == 'mmwm_small':
            self.mmwm_frontal_idx = MMWM_FRONTAL_IDX
            self.mmwm_parietal_idx = MMWM_PARIETAL_IDX

        print('------------------LOADING LLM MODELS---------------------')
        if configs.llm_model == 'LLAMA':
            model_name = "~/data/LLM_Models/TinyLlama-1.1B"
            print("################ LOAD LLAMA #################")
            self.llama_config = AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=True,
            )

            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True

            self.llm_model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=True,
                config=self.llama_config,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=True,
            )

        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2', local_files_only=True)

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'QWEN3':
            model_name = "~/data/LLM_Models/Qwen3-0.6B-Base"

            self.qwen_config = AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=True
            )

            self.qwen_config.num_hidden_layers = configs.llm_layers
            self.qwen_config.output_attentions = True
            self.qwen_config.output_hidden_states = True

            self.llm_model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=True,
                config=self.qwen_config,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=True
            )
        else:
            raise Exception('LLM model is not defined')
        print('------------------LOADING SUCCESSFULLY---------------------')
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        if configs.use_lora:
            self._apply_lora(configs)
        else:
            for param in self.llm_model.parameters():
                param.requires_grad = False


        if configs.description != None:
            self.description = configs.description
        else:
            self.description = "Motor imagery EEG signals with two classes (left hand and right hand), represented as discrete token sequences."
        print(self.description)
        self.chan_conv = Conv1dWithConstraint(64, 58, 1, max_norm=1)
        self.encoder = EEGTransformer(
            img_size=(configs.enc_in, configs.seq_len),
            patch_size=configs.patch_size,
            embed_num=configs.embed_num,
            embed_dim=configs.hidden_dim,
            depth=configs.depth,
            num_heads=configs.num_heads,
            mlp_ratio=configs.mlp_ratio,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
        
        self.chans_id = self.encoder.prepare_chan_ids(TARGET_BCIC2020T3_CHANNELS)
        
        pretrain_ckpt = torch.load(configs.load_path, map_location="cpu", weights_only=False)

        target_encoder_stat = {}
        for k,v in pretrain_ckpt['state_dict'].items():
            if k.startswith("target_encoder."):
                 target_encoder_stat[k[15:]]=v

        self.encoder.load_state_dict(target_encoder_stat)
        # for p in self.encoder.parameters():
        #    p.requires_grad = False

        # Load Codebook
        self.to_code_dim = nn.Linear(configs.hidden_dim, configs.code_dim)
        self.primary_codebook = VectorQuantizer(
            num_codes=configs.num_codes,
            code_dim=configs.code_dim,
            beta=configs.beta,
            init_std=0.02,
            use_cosine=False,
        )
        # self.residual_codebook = VectorQuantizer(
        #     num_codes=configs.num_codes,
        #     code_dim=configs.code_dim,
        #     beta=configs.beta,
        #     init_std=0.02,
        #     use_cosine=False,
        # )

        cb_ckpt = torch.load(self.cb_load_path, map_location="cpu", weights_only=False)
        codebook_weight = cb_ckpt["codebook_weight_prim"]  # [K, D]
        # codebook_weight_resi = cb_ckpt["codebook_weight_resi"]  # [K, D]
        # encoder_weight = cb_ckpt["encoder"]  # [K, D]
        # self.encoder.load_state_dict(encoder_weight)
        # for p in self.encoder.parameters():
        #     p.requires_grad = False
        # codebook_weight_resi = cb_ckpt["codebook_weight_resi"]  # [K, D]
        self.to_code_dim.load_state_dict(cb_ckpt["to_code_dim"])

        assert codebook_weight.shape == self.primary_codebook.codebook.weight.shape, \
            f"Codebook shape mismatch: ckpt {codebook_weight.shape}, model {self.primary_codebook.codebook.weight.shape}"

        with torch.no_grad():
            self.primary_codebook.codebook.weight.copy_(codebook_weight)
            # self.residual_codebook.codebook.weight.copy_(codebook_weight_resi)

        for param in self.primary_codebook.parameters():
            param.requires_grad = False
        # for param in self.residual_codebook.parameters():
        #     param.requires_grad = False

        for param in self.to_code_dim.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(configs.dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1024
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.code_dim, configs.n_heads, self.d_ff, self.d_llm)
        # self.to_llm_dim = nn.Linear(configs.code_dim, self.d_llm)

        self.llm_projection = nn.Linear(self.d_llm, 256)

        self.output_projection = ClassHead(self.encoder.num_patches[1], configs.embed_num, 256, self.num_classes, configs.dropout)

    def forward(self, x_enc, y=None):
        # x_enc = x_enc[:, TUAB_KEEP_IDX, :]
        # x_enc = x_enc[:, :, 0:960]
        # x_enc = x_enc.float()/100
        if self.data == 'seedv' or self.data == 'seediv':
            alpha_power, beta_power = compute_alpha_beta_power(
                x_enc, sfreq=256.0, relative=True,
            )
        elif self.data == 'tuab':
            features, feature_names = compute_tuab_prompt_features(x_enc, sfreq=200.0)
            tuab_feature_idx = {name: i for i, name in enumerate(feature_names)}
            tuab_features_cpu = features.detach().float().cpu()
        elif self.data == 'tuev':
            features, feature_names = compute_tuev_prompt_features(x_enc,sfreq=200.0)
            tuev_feature_idx = {name: i for i, name in enumerate(feature_names)}
            tuev_features_cpu = features.detach().float().cpu()
        elif self.data == 'mmwm' or self.data == 'mmwm_small':
            theta_power, alpha_power, beta_power = compute_theta_alpha_beta_power(x_enc,sfreq=500,)
        elif self.data == 'bcic2020_T3':
            features, feature_names = (compute_bcic2020_t3_prompt_features(x_enc,sfreq=256.0,))
            bcic_t3_feature_idx = {name: i for i, name in enumerate(feature_names)}
            bcic_t3_features_cpu = (features.detach().float().cpu())
        else:
            mu_power, beta_power = compute_mu_beta_power(
                x_enc, sfreq=250.0, relative=False
            )
        if self.data == 'mmwm' or self.data == 'mmwm_small':
            x_enc = temporal_interpolation(x_enc, 512)
        if self.data == 'bcic2020_T3':
            # x_enc = x_enc[:, BCIC2020T3_KEEP_INDEX, :]
            x_enc = self.chan_conv(x_enc)
        x_tokens = self.encoder(x_enc) # [B, N, E, D] self.chans_id.to(x_enc)
        B, N, E, D = x_tokens.size()
        S = N * E
        x_tokens = x_tokens.permute(0, 2, 1, 3).contiguous() # B, E, N, D
        # Use codebook
        x_tokens = x_tokens.view(B, S, D) # [B, S, D]
        x_tokens_cb = self.to_code_dim(x_tokens) # [B, S, D_cb]
        x_tokens_cb, token_ids, cb_info = self.primary_codebook(x_tokens_cb)
        # x_tokens_rcb, _, _ = self.residual_codebook(x_tokens_cb - x_tokens_pcb)
        # x_tokens_cb = x_tokens_pcb + x_tokens_rcb
        x_tokens_cb = x_tokens_cb.view(B, S, -1)
        if self.save_mid:
            tokens_fmdsr_np = x_tokens_cb.detach().float().cpu().numpy()
            tokens_np = x_tokens.detach().float().cpu().numpy()
            labels_np = y.detach().long().cpu().numpy()
            x_np = x_enc.detach().float().cpu().numpy()
            np.savez_compressed(
                './logs/mid_output/mmwm_mid.npz',
                x=x_np,
                tokens_fmdsr=tokens_fmdsr_np,
                tokens=tokens_np,
                labels=labels_np,
            )
        # x_tokens_cb = x_tokens

        prompt = []
        for b in range(x_enc.shape[0]):
            if self.data == 'seedv' or self.data == 'seediv':
                alpha_global = alpha_power[b].mean().item()
                beta_global = beta_power[b].mean().item()
                f3_alpha = alpha_power[b, self.idx1].item()
                fz_beta = beta_power[b, self.idx2].item()
                f4_alpha = alpha_power[b, self.idx3].item()
                alpha_asym = (f4_alpha - f3_alpha) / (f4_alpha + f3_alpha + 1e-6)
                prompt_ = generate_prompt(self.data, self.description, self.num_classes, alpha_global, beta_global, f3_alpha, f4_alpha, fz_beta, alpha_asym)
            elif self.data == 'tuab':
                feat = tuab_features_cpu[b]
                def get_feat(name):
                    return feat[tuab_feature_idx[name]].item()
                global_slow = (get_feat("global_delta_rel") + get_feat("global_theta_rel"))
                global_alpha = get_feat("global_alpha_rel")
                frontal_slow = get_feat("frontal_slow_mean")
                temporal_slow = get_feat("temporal_slow_mean")
                posterior_alpha = get_feat("posterior_alpha_mean")
                delta_asymmetry = get_feat("delta_bilateral_asymmetry")
                spectral_entropy = get_feat("global_spectral_entropy")
                tuab_summary = {
                    "global_slow": global_slow,
                    "global_alpha": global_alpha,
                    "frontal_slow": frontal_slow,
                    "temporal_slow": temporal_slow,
                    "posterior_alpha": posterior_alpha,
                    "delta_asymmetry": delta_asymmetry,
                    "spectral_entropy": spectral_entropy,
                }
                prompt_ = generate_prompt(self.data, self.description, self.num_classes,tueg_summary=tuab_summary)
            elif self.data == 'tuev':
                feat = tuev_features_cpu[b]
                current_y = y[b] if y is not None else None # only for train for label aug
                def get_tuev_feat(name):
                    return feat[tuev_feature_idx[name]].item()
                tuev_summary = {
                    "line_length": get_tuev_feat("line_length"),
                    "transient_prominence": get_tuev_feat("transient_prominence"),
                    "temporal_concentration": get_tuev_feat("temporal_concentration"),
                    "spatial_concentration": get_tuev_feat("spatial_concentration"),
                    "cross_channel_sync": get_tuev_feat("cross_channel_sync"),
                    "periodicity": get_tuev_feat("periodicity"),
                    "delta_ratio": get_tuev_feat("delta_ratio"),
                    "fast_ratio": get_tuev_feat("fast_ratio"),
                    "spectral_entropy": get_tuev_feat("spectral_entropy"),
                }
                prompt_ = generate_prompt(self.data,self.description,self.num_classes,tueg_summary=tuev_summary, y=current_y)
            elif self.data == 'mmwm' or self.data == 'mmwm_small':
                theta_global = theta_power[b].mean()
                alpha_global = alpha_power[b].mean()
                beta_global = beta_power[b].mean()
                global_band_sum = theta_global + alpha_global + beta_global + 1e-6
                global_theta_rel = theta_global / global_band_sum
                global_alpha_rel = alpha_global / global_band_sum
                global_beta_rel = beta_global / global_band_sum
                frontal_theta = theta_power[b, self.mmwm_frontal_idx].mean()
                frontal_alpha = alpha_power[b, self.mmwm_frontal_idx].mean()
                frontal_beta = beta_power[b, self.mmwm_frontal_idx].mean()
                frontal_band_sum = frontal_theta + frontal_alpha + frontal_beta + 1e-6
                frontal_theta_rel = frontal_theta / frontal_band_sum
                parietal_theta = theta_power[b, self.mmwm_parietal_idx].mean()
                parietal_alpha = alpha_power[b, self.mmwm_parietal_idx].mean()
                parietal_beta = beta_power[b, self.mmwm_parietal_idx].mean()
                parietal_band_sum = parietal_theta + parietal_alpha + parietal_beta + 1e-6
                parietal_alpha_rel = parietal_alpha / parietal_band_sum
                frontal_theta_parietal_alpha_ratio = frontal_theta / (parietal_alpha + 1e-6)
                mmwm_summary = {
                    "global_theta_rel": global_theta_rel.item(),
                    "global_alpha_rel": global_alpha_rel.item(),
                    "global_beta_rel": global_beta_rel.item(),
                    "frontal_theta_rel": frontal_theta_rel.item(),
                    "parietal_alpha_rel": parietal_alpha_rel.item(),
                    "frontal_theta_parietal_alpha_ratio": frontal_theta_parietal_alpha_ratio.item(),
                }
                prompt_ = generate_prompt(self.data,self.description,self.num_classes,mmwm_summary=mmwm_summary,)
            elif self.data == 'bcic2020_T3':
                feat = bcic_t3_features_cpu[b]
                def get_bcic_t3_feat(name):
                    return feat[bcic_t3_feature_idx[name]].item()
                bcic_t3_summary = {
                    "global_delta_rel": get_bcic_t3_feat("global_delta_rel"),
                    "global_theta_rel": get_bcic_t3_feat("global_theta_rel"),
                    "global_alpha_rel": get_bcic_t3_feat("global_alpha_rel"),
                    "global_beta_rel": get_bcic_t3_feat("global_beta_rel"),
                    "global_gamma_rel": get_bcic_t3_feat("global_gamma_rel"),
                    "frontal_theta_rel": get_bcic_t3_feat("frontal_theta_rel"),
                    "central_beta_rel": get_bcic_t3_feat("central_beta_rel"),
                    "parietal_alpha_rel": get_bcic_t3_feat("parietal_alpha_rel"),
                    "temporal_beta_gamma_rel": get_bcic_t3_feat("temporal_beta_gamma_rel"),
                    "temporal_beta_gamma_asymmetry": get_bcic_t3_feat("temporal_beta_gamma_asymmetry"),
                    "global_spectral_entropy": get_bcic_t3_feat("global_spectral_entropy"),
                }
                prompt_ = generate_prompt(self.data,self.description,self.num_classes,bcic_t3_summary=bcic_t3_summary,)

            else:
                mu_global = mu_power[b].mean().item()
                beta_global = beta_power[b].mean().item()
                c3_mu = mu_power[b, self.idx1].item()
                c4_mu = mu_power[b, self.idx3].item()
                cz_beta = beta_power[b, self.idx2].item()
                mu_asym = (c3_mu - c4_mu) / (c3_mu + c4_mu + 1e-6)
                prompt_ = generate_prompt(self.data, self.description, self.num_classes, mu_global, beta_global, c3_mu, c4_mu, cz_beta, mu_asym)
            prompt.append(prompt_)
        x_tokens_cb = x_tokens_cb.view(B, E, N, -1)
        llm_out = self.llm_extractor(x_tokens_cb, prompt) # B, E*N, D_llm
        llm_out = self.llm_projection(llm_out) # B, E*N, 256
        # if you need pool
        # llm_out = llm_out.mean(dim=1)
        # x_tokens = x_tokens.mean(dim=1) 

        fused_feat = llm_out + x_tokens
        # fused_feat = torch.cat([llm_out, x_tokens], dim=-1)
        outputs = self.output_projection(fused_feat)

        # subject_feat = fused_feat.mean(dim=1)       # [B, 256]
        # subject_feat = self.grl(subject_feat)
        # subject_logits = self.subject_classifier(subject_feat)

        return outputs, token_ids
    
    def print_feat_stats(self, name, x):
        with torch.no_grad():
            print(
                f"{name:12s} | "
                f"shape={tuple(x.shape)} | "
                f"mean={x.mean().item():.6f} | "
                f"max={x.max().item():.6f} | "
            )

    def llm_extractor(self, x_enc, prompt=None):
        # Input: B, N, E, D_cb
        # x_enc = self.normalize_layers(x_enc, 'norm')
        B, E, N, D_cb = x_enc.size()
        x_enc = x_enc.view(B*E, N, D_cb) # [B*E, Patch Num, dim]


        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        enc_out = self.reprogramming_layer(x_enc, source_embeddings, source_embeddings)
        # enc_out = self.to_llm_dim(x_enc)
        _, N, D = enc_out.size()
        enc_out = enc_out.view(B, E, N, D)
        enc_out = enc_out.view(B, E*N, D)

        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        # llama_enc_out = enc_out
        llm_hidden = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        llm_hidden = llm_hidden[:, prompt_embeddings.shape[1]:, :]

        BCT, L_llm, D_llm = llm_hidden.shape

        return llm_hidden

    def _apply_lora(self, configs):
        if configs.llm_model == 'LLAMA':
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif configs.llm_model == 'QWEN3':
        # Qwen3 uses LLaMA-style attention projection names
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif configs.llm_model == 'GPT2':
            target_modules = ["c_attn", "c_proj"]
        elif configs.llm_model == 'BERT':
            target_modules = ["query", "key", "value"]
        else:
            raise ValueError(f"Unsupported llm_model for LoRA: {configs.llm_model}")

        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=getattr(configs, "lora_r", 8),
            lora_alpha=getattr(configs, "lora_alpha", 16),
            lora_dropout=getattr(configs, "lora_dropout", 0.1),
            bias="none",
            target_modules=target_modules,
        )

        self.llm_model = get_peft_model(self.llm_model, peft_config)
        self.llm_model.print_trainable_parameters()

    
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
    
'''
class ClassHead2(nn.Module):
    def __init__(self, D, num_classes, dropout_rate=0):
        super(ClassHead2, self).__init__()
        self.linear = nn.Linear(D, 32)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    

class ClassHead3(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        num_classes: int = 4,
        dropout: float = 0.1,
        return_feat: bool = False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.return_feat = return_feat

        self.fusion_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            LinearWithConstraint(hidden_dim, hidden_dim // 2, max_norm=1),
            # nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.classifier = LinearWithConstraint(hidden_dim // 2, num_classes, max_norm=0.25)

    def forward(self, x):
        fused_feat = self.fusion_proj(x)

        logits = self.classifier(fused_feat)

        if self.return_feat:
            return logits, fused_feat

        return logits
'''
