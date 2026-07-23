import torch
import torch.nn as nn
import math


# =========================================================
# BCIC IV 2a: common 22-channel order
# x is assumed to be [B, C, T] with this channel order
# =========================================================
BCIC2A_CHANNELS = [
    'FZ', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4',
    'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6',
    'CP3', 'CP1', 'CPZ', 'CP2', 'CP4',
    'P1', 'PZ', 'P2', 'POZ'
]

BCIC2A_CH_TO_IDX = {ch: i for i, ch in enumerate(BCIC2A_CHANNELS)}

C3_IDX = BCIC2A_CH_TO_IDX['C3']   # 7
CZ_IDX = BCIC2A_CH_TO_IDX['CZ']   # 9
C4_IDX = BCIC2A_CH_TO_IDX['C4']   # 11


# =========================================================
# BCIC IV 2b: common 3-channel order
# x is assumed to be [B, C, T] with this channel order
# =========================================================
BCIC2B_CHANNELS = [
    "C3", "CZ", "C4"
]

BCIC2B_CH_TO_IDX = {ch: i for i, ch in enumerate(BCIC2B_CHANNELS)}

C3_IDX_BCIC2B = BCIC2B_CH_TO_IDX["C3"]   # 0
CZ_IDX_BCIC2B = BCIC2B_CH_TO_IDX["CZ"]   # 1
C4_IDX_BCIC2B = BCIC2B_CH_TO_IDX["C4"]   # 2


# =========================================================
# SEED-V: representative 3-channel order for emotion EEG
# x is assumed to be [B, C, T] with SEEDV_CHANNELS order
# =========================================================
SEEDV_CHANNELS = [
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

SEEDV_CH_TO_IDX = {ch: i for i, ch in enumerate(SEEDV_CHANNELS)}

F3_IDX_SEEDV = SEEDV_CH_TO_IDX["F3"]   # 7
FZ_IDX_SEEDV = SEEDV_CH_TO_IDX["FZ"]   # 9
F4_IDX_SEEDV = SEEDV_CH_TO_IDX["F4"]   # 11

TARGET_TUEG_CHANNELS = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3',
    'P4', 'O1', 'O2', 'F7', 'F8',
    'T7', 'T8', 'P7', 'P8',
    'FZ', 'CZ', 'PZ'
]
TUAB_KEEP_IDX = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    10, 11, 12, 13, 14, 15,
    18, 19, 20,
]

MMWM_CHANNELS = [
        "FP1", "FP2",
        "F7", "F3", "FZ", "F4", "F8",
        "FC5", "FC1", "FC2", "FC6",
        "T7", "C3", "CZ", "C4", "T8",
        "CP5", "CP1", "CP2", "CP6",
        "P7", "P3", "PZ", "P4", "P8",
        "O1", "OZ", "O2"
]
MMWM_CH_TO_IDX = {ch.upper(): i for i, ch in enumerate(MMWM_CHANNELS)}

MMWM_FRONTAL_IDX = [
    MMWM_CH_TO_IDX['F3'],
    MMWM_CH_TO_IDX['FZ'],
    MMWM_CH_TO_IDX['F4'],
]

MMWM_PARIETAL_IDX = [
    MMWM_CH_TO_IDX['P3'],
    MMWM_CH_TO_IDX['PZ'],
    MMWM_CH_TO_IDX['P4'],
]

BCIC2020_T3_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6",
    "T7", "C3", "Cz", "C4", "T8",
    "TP9", "CP5", "CP1", "CP2", "CP6", "TP10",
    "P7", "P3", "Pz", "P4", "P8",
    "PO9", "O1", "Oz", "O2", "PO10",
    "AF7", "AF3", "AF4", "AF8",
    "F5", "F1", "F2", "F6",
    "FT9", "FT7", "FC3", "FC4", "FT8", "FT10",
    "C5", "C1", "C2", "C6",
    "TP7", "CP3", "CPz", "CP4", "TP8",
    "P5", "P1", "P2", "P6",
    "PO7", "PO3", "POz", "PO4", "PO8",
]

TARGET_BCIC2020T3_CHANNELS = [
    'FP1', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 
    'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 
    'FC1', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 
    'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 
    'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 
    'P6', 'P8', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2'
]

BCIC2020T3_KEEP_INDEX = [0, 1, 33, 34, 2, 36, 3, 37, 4, 38, 5, 39, 6, 41, 
                         7, 42, 8, 9, 43, 10, 44, 11, 
                         46, 12, 47, 13, 48, 14, 49, 15, 50, 17, 51, 18, 52, 19, 
                         53, 20, 54, 22, 55, 23, 56, 24, 57, 25, 58, 
                         26, 59, 60, 61, 62, 63, 28, 29, 30]

BCIC2020_T3_CH_TO_IDX = {
    ch.upper(): idx
    for idx, ch in enumerate(BCIC2020_T3_CHANNELS)
}


def bcic_t3_indices(channel_names):
    return [
        BCIC2020_T3_CH_TO_IDX[ch.upper()]
        for ch in channel_names
    ]


BCIC_T3_FRONTAL_IDX = bcic_t3_indices([
    "Fp1", "Fp2",
    "AF7", "AF3", "AF4", "AF8",
    "F7", "F5", "F3", "F1",
    "Fz",
    "F2", "F4", "F6", "F8",
])

BCIC_T3_CENTRAL_IDX = bcic_t3_indices([
    "FC5", "FC3", "FC1",
    "FC2", "FC4", "FC6",
    "C5", "C3", "C1",
    "Cz",
    "C2", "C4", "C6",
])

BCIC_T3_LEFT_TEMPORAL_IDX = bcic_t3_indices([
    "FT9", "FT7", "T7", "TP9", "TP7", "P7",
])

BCIC_T3_RIGHT_TEMPORAL_IDX = bcic_t3_indices([
    "FT10", "FT8", "T8", "TP10", "TP8", "P8",
])

BCIC_T3_PARIETAL_IDX = bcic_t3_indices([
    "CP5", "CP3", "CP1",
    "CPz",
    "CP2", "CP4", "CP6",
    "P5", "P3", "P1",
    "Pz",
    "P2", "P4", "P6",
])


def compute_bandpower_fft(
    x: torch.Tensor,
    sfreq: float = 250.0,
    fmin: float = 8.0,
    fmax: float = 13.0,
    eps: float = 1e-8,
    relative: bool = False,
    total_fmin: float = 4.0,
    total_fmax: float = 40.0,
) -> torch.Tensor:
    """
    Compute band power for EEG using FFT-based PSD estimation.

    Args:
        x: [B, C, T]
        sfreq: sampling frequency
        fmin, fmax: target frequency band
        eps: numerical stability
        relative: if True, return relative band power
        total_fmin, total_fmax: denominator band for relative power

    Returns:
        band_power: [B, C]
    """
    if x.dim() != 3:
        raise ValueError(f"x must be [B, C, T], but got {x.shape}")

    B, C, T = x.shape
    device = x.device
    dtype = x.dtype

    # Remove DC bias
    x = x - x.mean(dim=-1, keepdim=True)

    # Hann window
    window = torch.hann_window(T, device=device, dtype=dtype).view(1, 1, T)
    xw = x * window

    # rFFT
    fft = torch.fft.rfft(xw, dim=-1)                     # [B, C, F]
    psd = (fft.real ** 2 + fft.imag ** 2) / (T + eps)   # simple PSD-like estimate

    # Frequency bins
    freqs = torch.fft.rfftfreq(T, d=1.0 / sfreq).to(device)

    # Target band mask
    band_mask = (freqs >= fmin) & (freqs <= fmax)
    if band_mask.sum() == 0:
        raise ValueError(f"No frequency bins found in band [{fmin}, {fmax}] Hz")

    band_power = psd[..., band_mask].mean(dim=-1)        # [B, C]

    if relative:
        total_mask = (freqs >= total_fmin) & (freqs <= total_fmax)
        if total_mask.sum() == 0:
            raise ValueError(
                f"No frequency bins found in total band [{total_fmin}, {total_fmax}] Hz"
            )
        total_power = psd[..., total_mask].mean(dim=-1)  # [B, C]
        band_power = band_power / (total_power + eps)

    return band_power


def compute_mu_beta_power(
    x: torch.Tensor,
    sfreq: float = 250.0,
    mu_band=(8.0, 13.0),
    beta_band=(13.0, 30.0),
    relative: bool = False,
):
    """
    Args:
        x: [B, C, T]
        sfreq: sampling rate, BCIC2a is 250 Hz
        mu_band: mu frequency band
        beta_band: beta frequency band
        relative: whether to return relative band power

    Returns:
        mu_power:   [B, C]
        beta_power: [B, C]
    """
    mu_power = compute_bandpower_fft(
        x,
        sfreq=sfreq,
        fmin=mu_band[0],
        fmax=mu_band[1],
        relative=relative,
    )

    beta_power = compute_bandpower_fft(
        x,
        sfreq=sfreq,
        fmin=beta_band[0],
        fmax=beta_band[1],
        relative=relative,
    )

    return mu_power, beta_power

def compute_alpha_beta_power(
    x: torch.Tensor,
    sfreq: float = 256.0,
    alpha_band=(8.0, 14.0),
    beta_band=(14.0, 31.0),
    relative: bool = True,
):

    if x.dim() != 3:
        raise ValueError(
            f"x must have shape [B, C, T], but got {x.shape}"
        )

    alpha_power = compute_bandpower_fft(
        x=x,
        sfreq=sfreq,
        fmin=alpha_band[0],
        fmax=alpha_band[1],
        relative=relative,
    )

    beta_power = compute_bandpower_fft(
        x=x,
        sfreq=sfreq,
        fmin=beta_band[0],
        fmax=beta_band[1],
        relative=relative,
    )

    return alpha_power, beta_power

def compute_theta_alpha_beta_power(
    x: torch.Tensor,
    sfreq: float = 500.0,
    theta_band=(4.0, 8.0),
    alpha_band=(8.0, 14.0),
    beta_band=(14.0, 31.0),
    relative: bool = True,
):
    if x.dim() != 3:
        raise ValueError(
            f"x must have shape [B, C, T], but got {x.shape}"
        )

    theta_power = compute_bandpower_fft(
        x=x,
        sfreq=sfreq,
        fmin=theta_band[0],
        fmax=theta_band[1],
        relative=relative,
    )

    alpha_power = compute_bandpower_fft(
        x=x,
        sfreq=sfreq,
        fmin=alpha_band[0],
        fmax=alpha_band[1],
        relative=relative,
    )

    beta_power = compute_bandpower_fft(
        x=x,
        sfreq=sfreq,
        fmin=beta_band[0],
        fmax=beta_band[1],
        relative=relative,
    )

    return theta_power, alpha_power, beta_power

TUAB19_CHANNELS = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3',
    'P4', 'O1', 'O2', 'F7', 'F8',
    'T3', 'T4', 'T5', 'T6',
    'FZ', 'CZ', 'PZ'
]
TUAB19_CH_TO_IDX = {ch: i for i, ch in enumerate(TUAB19_CHANNELS)}
def compute_tuab_prompt_features(x: torch.Tensor, sfreq: float = 200.0, eps: float = 1e-8):
    assert x.ndim == 3, f"Expected [B, C, T], got {tuple(x.shape)}"
    B, C, T = x.shape
    assert C == 19, f"Expected 19 TUAB channels, got {C}"

    x = x.float()

    x = x - x.mean(dim=-1, keepdim=True)

    window = torch.hann_window(
        T,
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, T)

    x_win = x * window

    fft = torch.fft.rfft(x_win, dim=-1)
    psd = (fft.real.pow(2) + fft.imag.pow(2)) / (
        sfreq * window.pow(2).sum(dim=-1, keepdim=True)
    )

    freqs = torch.fft.rfftfreq(
        T,
        d=1.0 / sfreq,
        device=x.device,
    )

    def band_power(low, high):
        if high == 30.0:
            mask = (freqs >= low) & (freqs <= high)
        else:
            mask = (freqs >= low) & (freqs < high)

        return torch.trapz(
            psd[..., mask],
            freqs[mask],
            dim=-1,
        )

    delta = band_power(0.5, 4.0)    # [B, C]
    theta = band_power(4.0, 8.0)
    alpha = band_power(8.0, 13.0)
    beta = band_power(13.0, 30.0)

    total_power = delta + theta + alpha + beta + eps

    rel_delta = delta / total_power
    rel_theta = theta / total_power
    rel_alpha = alpha / total_power
    rel_beta = beta / total_power

    # ---------------------------------------------------------
    # 2. Global spectral features
    # ---------------------------------------------------------
    global_delta = rel_delta.mean(dim=1, keepdim=True)
    global_theta = rel_theta.mean(dim=1, keepdim=True)
    global_alpha = rel_alpha.mean(dim=1, keepdim=True)
    global_beta = rel_beta.mean(dim=1, keepdim=True)

    global_psd = psd.mean(dim=1)  # [B, F]

    entropy_mask = (freqs >= 0.5) & (freqs <= 30.0)
    entropy_psd = global_psd[:, entropy_mask]
    prob = entropy_psd / (entropy_psd.sum(dim=-1, keepdim=True) + eps)

    spectral_entropy = -(
        prob * torch.log(prob + eps)
    ).sum(dim=-1, keepdim=True)

    spectral_entropy = spectral_entropy / math.log(
        entropy_mask.sum().item()
    )

    slope_mask = (freqs >= 2.0) & (freqs <= 30.0)

    log_freq = torch.log(freqs[slope_mask] + eps)
    log_power = torch.log(global_psd[:, slope_mask] + eps)

    log_freq_centered = log_freq - log_freq.mean()
    denom = log_freq_centered.pow(2).sum() + eps

    spectral_slope = (
        log_power * log_freq_centered.unsqueeze(0)
    ).sum(dim=-1, keepdim=True) / denom

    # ---------------------------------------------------------
    # 3. Regional features
    # ---------------------------------------------------------
    regions = {
        "frontal": [
            TUAB19_CH_TO_IDX[ch]
            for ch in ['FP1', 'FP2', 'F3', 'F4', 'F7', 'F8', 'FZ']
        ],
        "central": [
            TUAB19_CH_TO_IDX[ch]
            for ch in ['C3', 'C4', 'CZ']
        ],
        "temporal": [
            TUAB19_CH_TO_IDX[ch]
            for ch in ['T3', 'T4', 'T5', 'T6']
        ],
        "posterior": [
            TUAB19_CH_TO_IDX[ch]
            for ch in ['P3', 'P4', 'O1', 'O2', 'PZ']
        ],
    }

    slow = rel_delta + rel_theta

    regional_slow_mean = []
    regional_alpha_mean = []
    regional_slow_peak = []

    for _, idxs in regions.items():
        idxs = torch.tensor(idxs, device=x.device)

        region_slow = slow[:, idxs]
        region_alpha = rel_alpha[:, idxs]

        regional_slow_mean.append(
            region_slow.mean(dim=1)
        )

        regional_alpha_mean.append(
            region_alpha.mean(dim=1)
        )

        regional_slow_peak.append(
            torch.quantile(region_slow, q=0.75, dim=1)
        )

    regional_slow_mean = torch.stack(regional_slow_mean, dim=1)
    regional_alpha_mean = torch.stack(regional_alpha_mean, dim=1)
    regional_slow_peak = torch.stack(regional_slow_peak, dim=1)

    # ---------------------------------------------------------
    # 4. Bilateral asymmetry
    # ---------------------------------------------------------
    left_idx = torch.tensor([
        TUAB19_CH_TO_IDX['FP1'],
        TUAB19_CH_TO_IDX['F3'],
        TUAB19_CH_TO_IDX['C3'],
        TUAB19_CH_TO_IDX['P3'],
        TUAB19_CH_TO_IDX['O1'],
        TUAB19_CH_TO_IDX['F7'],
        TUAB19_CH_TO_IDX['T3'],
        TUAB19_CH_TO_IDX['T5'],
    ], device=x.device)

    right_idx = torch.tensor([
        TUAB19_CH_TO_IDX['FP2'],
        TUAB19_CH_TO_IDX['F4'],
        TUAB19_CH_TO_IDX['C4'],
        TUAB19_CH_TO_IDX['P4'],
        TUAB19_CH_TO_IDX['O2'],
        TUAB19_CH_TO_IDX['F8'],
        TUAB19_CH_TO_IDX['T4'],
        TUAB19_CH_TO_IDX['T6'],
    ], device=x.device)

    def bilateral_asymmetry(power):
        left_power = power[:, left_idx]
        right_power = power[:, right_idx]

        diff = torch.abs(
            torch.log(left_power + eps) -
            torch.log(right_power + eps)
        )

        return diff.mean(dim=1, keepdim=True)

    delta_asym = bilateral_asymmetry(delta)
    alpha_asym = bilateral_asymmetry(alpha)
    beta_asym = bilateral_asymmetry(beta)

    # ---------------------------------------------------------
    # 5. Temporal morphology features
    # ---------------------------------------------------------
    x_std = x.std(dim=-1, keepdim=True) + eps
    z = (x - x.mean(dim=-1, keepdim=True)) / x_std

    line_length = torch.abs(
        z[..., 1:] - z[..., :-1]
    ).mean(dim=-1).mean(dim=1, keepdim=True)

    kurtosis = (
        z.pow(4).mean(dim=-1) - 3.0
    ).mean(dim=1, keepdim=True)

    # ---------------------------------------------------------
    # 6. Concatenate: [B, 23]
    # ---------------------------------------------------------
    features = torch.cat([
        global_delta,
        global_theta,
        global_alpha,
        global_beta,
        spectral_entropy,
        spectral_slope,

        regional_slow_mean,
        regional_alpha_mean,
        regional_slow_peak,

        delta_asym,
        alpha_asym,
        beta_asym,

        line_length,
        kurtosis,
    ], dim=1)

    feature_names = [
        "global_delta_rel",
        "global_theta_rel",
        "global_alpha_rel",
        "global_beta_rel",
        "global_spectral_entropy",
        "global_spectral_slope",

        "frontal_slow_mean",
        "central_slow_mean",
        "temporal_slow_mean",
        "posterior_slow_mean",

        "frontal_alpha_mean",
        "central_alpha_mean",
        "temporal_alpha_mean",
        "posterior_alpha_mean",

        "frontal_slow_p75",
        "central_slow_p75",
        "temporal_slow_p75",
        "posterior_slow_p75",

        "delta_bilateral_asymmetry",
        "alpha_bilateral_asymmetry",
        "beta_bilateral_asymmetry",

        "normalized_line_length",
        "excess_kurtosis",
    ]

    return features, feature_names

def _relative_band_power(psd, freqs, low_hz, high_hz, valid_mask, eps=1e-8):
    """
    psd:   [B, C, F]
    freqs: [F]
    return: [B, C]
    """
    band_mask = (
        (freqs >= low_hz)
        & (freqs < high_hz)
        & valid_mask
    )

    if band_mask.sum() == 0:
        return torch.zeros(
            psd.shape[0],
            psd.shape[1],
            device=psd.device,
            dtype=psd.dtype,
        )

    band_power = psd[..., band_mask].sum(dim=-1)
    total_power = psd[..., valid_mask].sum(dim=-1).clamp_min(eps)

    return band_power / total_power


def compute_tuev_prompt_features(x,sfreq=200.0,low_freq=1.0,high_freq=45.0,eps=1e-8):
    if x.ndim != 3:
        raise ValueError(
            f"Expected x with shape [B, C, T], but got {tuple(x.shape)}."
        )

    x = x.float()
    batch_size, num_channels, num_samples = x.shape

    if num_samples < 16:
        raise ValueError(
            f"TUEV prompt feature extraction requires at least 16 samples, "
            f"but received T={num_samples}."
        )
    x_centered = x - x.mean(dim=-1, keepdim=True)
    channel_std = x_centered.std(dim=-1, keepdim=True).clamp_min(eps)
    z = x_centered / channel_std

    # Normalized line length: larger values usually indicate faster waveform changes.
    line_length = z.diff(dim=-1).abs().mean(dim=(1, 2))

    # Transient prominence: ratio between extreme and median normalized amplitudes.
    abs_z_flat = z.abs().reshape(batch_size, -1)
    q50 = torch.quantile(abs_z_flat, 0.50, dim=1).clamp_min(eps)
    q99 = torch.quantile(abs_z_flat, 0.99, dim=1)
    transient_prominence = q99 / q50

    kurtosis = z.pow(4).mean(dim=(1, 2)).clamp_min(eps)
    log_kurtosis = torch.log1p(kurtosis)

    temporal_energy = z.pow(2).mean(dim=1)  # [B, T]
    topk_time = max(1, int(math.ceil(0.10 * num_samples)))
    top_energy = temporal_energy.topk(topk_time, dim=-1).values.sum(dim=-1)
    total_temporal_energy = temporal_energy.sum(dim=-1).clamp_min(eps)
    temporal_concentration = top_energy / total_temporal_energy

    channel_energy = z.pow(2).mean(dim=-1)  # [B, C]
    spatial_concentration = (
        channel_energy.max(dim=-1).values
        / channel_energy.sum(dim=-1).clamp_min(eps)
    )

    if num_channels > 1:
        z_unit = z / z.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(eps)
        corr = torch.bmm(z_unit, z_unit.transpose(1, 2))  # [B, C, C]
        abs_corr_sum = corr.abs().sum(dim=(1, 2)) - num_channels
        cross_channel_sync = abs_corr_sum / (num_channels * (num_channels - 1))
    else:
        cross_channel_sync = torch.zeros(batch_size,device=x.device,dtype=x.dtype,)

    window = torch.hann_window(
        num_samples,
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, -1)

    spectrum = torch.fft.rfft(z * window, dim=-1)
    psd = spectrum.abs().pow(2)

    freqs = torch.fft.rfftfreq(
        num_samples,
        d=1.0 / sfreq,
        device=x.device,
    )

    valid_mask = (
        (freqs >= low_freq)
        & (freqs <= min(high_freq, sfreq / 2.0))
    )

    if valid_mask.sum() == 0:
        raise ValueError(
            f"No valid frequency bins found in [{low_freq}, {high_freq}] Hz. "
            f"Check sfreq={sfreq} and input length={num_samples}."
        )

    delta_ratio = _relative_band_power(psd, freqs, 1.0, 4.0, valid_mask).mean(dim=1)
    alpha_ratio = _relative_band_power(psd, freqs, 8.0, 13.0, valid_mask).mean(dim=1)
    beta_ratio = _relative_band_power(psd, freqs, 13.0, 30.0, valid_mask).mean(dim=1)
    gamma_ratio = _relative_band_power(psd, freqs, 30.0, 45.0, valid_mask).mean(dim=1)
    fast_ratio = beta_ratio + gamma_ratio
    valid_psd = psd[..., valid_mask]
    spectral_prob = valid_psd / valid_psd.sum(dim=-1, keepdim=True).clamp_min(eps)
    spectral_entropy = -(spectral_prob * spectral_prob.clamp_min(eps).log()).sum(dim=-1)
    spectral_entropy = spectral_entropy / math.log(valid_psd.shape[-1])
    spectral_entropy = spectral_entropy.mean(dim=1)
    fft_len = 2 * num_samples
    ac_spectrum = torch.fft.rfft(z, n=fft_len, dim=-1)
    autocorr = torch.fft.irfft(
        ac_spectrum.abs().pow(2),
        n=fft_len,
        dim=-1,
    )[..., :num_samples]

    autocorr = autocorr / autocorr[..., :1].abs().clamp_min(eps)
    lag_start = max(1, int(round(0.2 * sfreq)))
    lag_end = min(num_samples, int(round(2.0 * sfreq)) + 1)
    if lag_end > lag_start:
        periodicity = autocorr[..., lag_start:lag_end].amax(dim=-1).mean(dim=1)
        periodicity = periodicity.clamp_min(0.0)
    else:
        periodicity = torch.zeros(batch_size,device=x.device,dtype=x.dtype,)
    features = torch.stack(
        [
            line_length,
            transient_prominence,
            log_kurtosis,
            temporal_concentration,
            spatial_concentration,
            cross_channel_sync,
            periodicity,
            delta_ratio,
            alpha_ratio,
            fast_ratio,
            spectral_entropy,
        ],
        dim=1,
    )

    feature_names = [
        "line_length",
        "transient_prominence",
        "log_kurtosis",
        "temporal_concentration",
        "spatial_concentration",
        "cross_channel_sync",
        "periodicity",
        "delta_ratio",
        "alpha_ratio",
        "fast_ratio",
        "spectral_entropy",
    ]

    return features, feature_names

def generate_prompt(data, description, num_classes=None, global1=None, global2=None, band1_num=None, band2_num=None, band3_num=None, asym_num=None, tueg_summary=None, mmwm_summary=None, bcic_t3_summary=None, y=None):
    # The code for generating prompt words is being improved and will be made public here soon
    prompt_ = None
    return prompt_


def compute_bcic2020_t3_prompt_features(x,sfreq=256.0,eps=1e-8,):
    if x.ndim != 3:
        raise ValueError(
            f"Expected x with shape [B, C, T], got {tuple(x.shape)}"
        )

    if x.shape[1] != 64:
        raise ValueError(
            f"BCIC2020-T3 expects 64 channels, got {x.shape[1]}"
        )

    x = x.float()

    batch_size, num_channels, num_timepoints = x.shape
    device = x.device
    dtype = x.dtype

    # Remove temporal DC offset from every channel.
    x = x - x.mean(dim=-1, keepdim=True)

    # Hann window.
    window = torch.hann_window(
        num_timepoints,
        device=device,
        dtype=dtype,
    )

    x_windowed = x * window.view(1, 1, -1)

    # One-sided FFT.
    fft_values = torch.fft.rfft(
        x_windowed,
        dim=-1,
    )

    # Periodogram-like PSD.
    window_energy = window.pow(2).sum().clamp_min(eps)

    psd = (
        fft_values.abs().pow(2)
        / (sfreq * window_energy)
    )

    freqs = torch.fft.rfftfreq(
        num_timepoints,
        d=1.0 / sfreq,
        device=device,
    )

    def band_power(low, high):
        """
        Sum PSD within [low, high).
        Output shape: [B, C]
        """
        mask = (freqs >= low) & (freqs < high)

        if mask.sum() == 0:
            raise ValueError(
                f"No frequency bins found in band {low}-{high} Hz"
            )

        return psd[..., mask].sum(dim=-1)

    # Use 1-45 Hz as the normalization interval.
    # Low gamma is restricted to 30-45 Hz to reduce sensitivity
    # to residual muscle and line-frequency contamination.
    delta_power = band_power(1.0, 4.0)
    theta_power = band_power(4.0, 8.0)
    alpha_power = band_power(8.0, 13.0)
    beta_power = band_power(13.0, 30.0)
    gamma_power = band_power(30.0, 45.0)

    total_power = (
        delta_power
        + theta_power
        + alpha_power
        + beta_power
        + gamma_power
        + eps
    )

    delta_rel = delta_power / total_power
    theta_rel = theta_power / total_power
    alpha_rel = alpha_power / total_power
    beta_rel = beta_power / total_power
    gamma_rel = gamma_power / total_power

    # Convert channel index lists to tensors on the current device.
    frontal_idx = torch.as_tensor(
        BCIC_T3_FRONTAL_IDX,
        device=device,
        dtype=torch.long,
    )

    central_idx = torch.as_tensor(
        BCIC_T3_CENTRAL_IDX,
        device=device,
        dtype=torch.long,
    )

    left_temporal_idx = torch.as_tensor(
        BCIC_T3_LEFT_TEMPORAL_IDX,
        device=device,
        dtype=torch.long,
    )

    right_temporal_idx = torch.as_tensor(
        BCIC_T3_RIGHT_TEMPORAL_IDX,
        device=device,
        dtype=torch.long,
    )

    parietal_idx = torch.as_tensor(
        BCIC_T3_PARIETAL_IDX,
        device=device,
        dtype=torch.long,
    )

    # Global relative powers.
    global_delta_rel = delta_rel.mean(dim=1)
    global_theta_rel = theta_rel.mean(dim=1)
    global_alpha_rel = alpha_rel.mean(dim=1)
    global_beta_rel = beta_rel.mean(dim=1)
    global_gamma_rel = gamma_rel.mean(dim=1)

    # Regional summaries.
    frontal_theta_rel = theta_rel.index_select(
        dim=1,
        index=frontal_idx,
    ).mean(dim=1)

    central_beta_rel = beta_rel.index_select(
        dim=1,
        index=central_idx,
    ).mean(dim=1)

    parietal_alpha_rel = alpha_rel.index_select(
        dim=1,
        index=parietal_idx,
    ).mean(dim=1)

    # Combine beta and low-gamma as a compact temporal high-frequency
    # summary for speech-related activity.
    beta_gamma_rel = beta_rel + gamma_rel

    left_temporal_beta_gamma = beta_gamma_rel.index_select(
        dim=1,
        index=left_temporal_idx,
    ).mean(dim=1)

    right_temporal_beta_gamma = beta_gamma_rel.index_select(
        dim=1,
        index=right_temporal_idx,
    ).mean(dim=1)

    temporal_beta_gamma_rel = 0.5 * (
        left_temporal_beta_gamma
        + right_temporal_beta_gamma
    )

    temporal_beta_gamma_asymmetry = (
        left_temporal_beta_gamma
        - right_temporal_beta_gamma
    ) / (
        left_temporal_beta_gamma
        + right_temporal_beta_gamma
        + eps
    )

    # Normalized spectral entropy over 1-45 Hz.
    entropy_mask = (
        (freqs >= 1.0)
        & (freqs < 45.0)
    )

    entropy_psd = psd[..., entropy_mask]
    entropy_prob = entropy_psd / (
        entropy_psd.sum(dim=-1, keepdim=True)
        + eps
    )

    channel_entropy = -(
        entropy_prob
        * torch.log(entropy_prob + eps)
    ).sum(dim=-1)

    num_entropy_bins = int(entropy_mask.sum().item())

    channel_entropy = channel_entropy / math.log(
        max(num_entropy_bins, 2)
    )

    global_spectral_entropy = channel_entropy.mean(dim=1)

    features = torch.stack(
        [
            global_delta_rel,
            global_theta_rel,
            global_alpha_rel,
            global_beta_rel,
            global_gamma_rel,
            frontal_theta_rel,
            central_beta_rel,
            parietal_alpha_rel,
            temporal_beta_gamma_rel,
            temporal_beta_gamma_asymmetry,
            global_spectral_entropy,
        ],
        dim=1,
    )

    feature_names = [
        "global_delta_rel",
        "global_theta_rel",
        "global_alpha_rel",
        "global_beta_rel",
        "global_gamma_rel",
        "frontal_theta_rel",
        "central_beta_rel",
        "parietal_alpha_rel",
        "temporal_beta_gamma_rel",
        "temporal_beta_gamma_asymmetry",
        "global_spectral_entropy",
    ]

    return features, feature_names
