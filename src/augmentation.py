import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional


def add_gaussian_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add Gaussian white noise at specified SNR level.

    From TF-MDA and ECTN papers:
    SNR_dB = 10 * log10(P_signal / P_noise)

    Args:
        signal: Input signal (any shape, operates on last axis)
        snr_db: Signal-to-noise ratio in decibels
                Higher = less noise (10dB is mild, -4dB is severe)

    Returns:
        Noisy signal
    """
    signal_power = np.mean(signal ** 2)
    if signal_power < 1e-10:  # Avoid division by zero
        return signal

    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise


def amplitude_scaling(signal: np.ndarray, scale_range: tuple = (0.8, 1.2)) -> np.ndarray:
    """
    Random amplitude scaling.

    From WDCNN empirical studies - helps with amplitude variations
    between different fault severities.

    Args:
        signal: Input signal
        scale_range: (min_scale, max_scale) tuple

    Returns:
        Scaled signal
    """
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return signal * scale


def time_shift(signal: np.ndarray, max_shift_ratio: float = 0.1) -> np.ndarray:
    """
    Random circular time shift.

    Simulates phase variations in fault signals.

    Args:
        signal: Input signal (channels, time)
        max_shift_ratio: Maximum shift as fraction of signal length

    Returns:
        Shifted signal
    """
    if signal.ndim == 1:
        shift = int(np.random.uniform(-max_shift_ratio, max_shift_ratio) * len(signal))
        return np.roll(signal, shift)
    else:
        # For multi-channel, shift along time axis (last axis)
        shift = int(np.random.uniform(-max_shift_ratio, max_shift_ratio) * signal.shape[-1])
        return np.roll(signal, shift, axis=-1)


def jittering(signal: np.ndarray, sigma: float = 0.05) -> np.ndarray:
    """
    Add small random perturbations (jitter).

    From WDCNN empirical studies - different from Gaussian noise,
    this adds independent perturbations scaled to signal std.

    Args:
        signal: Input signal
        sigma: Standard deviation of jitter as fraction of signal std

    Returns:
        Jittered signal
    """
    noise = np.random.normal(0, sigma * np.std(signal), signal.shape)
    return signal + noise


def time_warp(signal: np.ndarray, sigma: float = 0.2, knots: int = 4) -> np.ndarray:
    """
    Time warping via smooth random distortion.

    Creates non-linear time distortion using cubic spline interpolation.
    This is important for bearing fault diagnosis because fault
    frequencies can vary with speed/load.

    Args:
        signal: Input signal (1D or 2D with shape (channels, time))
        sigma: Standard deviation of warp magnitude
        knots: Number of control points for warping curve

    Returns:
        Time-warped signal
    """
    from scipy.interpolate import CubicSpline

    orig_steps = signal.shape[-1]

    # Create random warp path
    random_warp = np.random.normal(loc=1.0, scale=sigma, size=(knots + 2,))
    warp_steps = np.linspace(0, orig_steps - 1, num=knots + 2)

    # Ensure monotonicity by using cumulative sum approach
    time_warp_fn = CubicSpline(warp_steps, np.cumsum(random_warp) * (orig_steps - 1) / np.sum(random_warp))

    # Generate warped time indices
    warped_indices = time_warp_fn(np.arange(orig_steps))
    warped_indices = np.clip(warped_indices, 0, orig_steps - 1)

    # Interpolate signal at warped positions
    if signal.ndim == 1:
        return np.interp(warped_indices, np.arange(orig_steps), signal).astype(signal.dtype)
    else:
        # Multi-channel: warp each channel
        result = np.zeros_like(signal)
        for c in range(signal.shape[0]):
            result[c] = np.interp(warped_indices, np.arange(orig_steps), signal[c])
        return result.astype(signal.dtype)


def window_crop(signal: np.ndarray, crop_ratio: float = 0.9) -> np.ndarray:
    """
    Random window cropping and resize back to original length.

    Simulates variations in window position relative to fault events.

    Args:
        signal: Input signal (channels, time) or (time,)
        crop_ratio: Fraction of signal to keep (0.8 = keep 80%)

    Returns:
        Cropped and resized signal
    """
    orig_len = signal.shape[-1]
    crop_len = int(orig_len * crop_ratio)

    # Random start position
    max_start = orig_len - crop_len
    start = np.random.randint(0, max_start + 1)

    if signal.ndim == 1:
        cropped = signal[start:start + crop_len]
        # Resize back using linear interpolation
        indices = np.linspace(0, crop_len - 1, orig_len)
        return np.interp(indices, np.arange(crop_len), cropped).astype(signal.dtype)
    else:
        cropped = signal[:, start:start + crop_len]
        result = np.zeros_like(signal)
        indices = np.linspace(0, crop_len - 1, orig_len)
        for c in range(signal.shape[0]):
            result[c] = np.interp(indices, np.arange(crop_len), cropped[c])
        return result.astype(signal.dtype)


def permutation(signal: np.ndarray, n_segments: int = 4) -> np.ndarray:
    """
    Segment permutation augmentation.

    Divides signal into segments and randomly permutes them.
    Can help with learning local features independent of global position.

    Args:
        signal: Input signal
        n_segments: Number of segments to create

    Returns:
        Permuted signal
    """
    orig_len = signal.shape[-1]
    segment_len = orig_len // n_segments

    if signal.ndim == 1:
        segments = [signal[i * segment_len:(i + 1) * segment_len] for i in range(n_segments)]
        # Handle remainder
        if orig_len % n_segments != 0:
            segments.append(signal[n_segments * segment_len:])
        np.random.shuffle(segments)
        return np.concatenate(segments)[:orig_len].astype(signal.dtype)
    else:
        segments = [signal[:, i * segment_len:(i + 1) * segment_len] for i in range(n_segments)]
        if orig_len % n_segments != 0:
            segments.append(signal[:, n_segments * segment_len:])
        np.random.shuffle(segments)
        return np.concatenate(segments, axis=1)[:, :orig_len].astype(signal.dtype)


# =============================================================================
# Augmentations (Combinations of transforms)
# =============================================================================

class Augmentations:
    """
    Defines which augmentations to apply and with what probability.

    Based on paper findings:
    - Noise injection is most critical for robustness
    - Time warping helps with frequency variations
    - Amplitude scaling helps with severity variations
    """

    def __init__(
            self,
            noise_prob: float = 0.5,
            noise_snr_range: tuple = (2, 10),  # 2-10 dB
            scale_prob: float = 0.3,
            scale_range: tuple = (0.8, 1.2),
            shift_prob: float = 0.3,
            shift_ratio: float = 0.1,
            warp_prob: float = 0.3,
            warp_sigma: float = 0.2,
            jitter_prob: float = 0.3,
            jitter_sigma: float = 0.03,
            crop_prob: float = 0.2,
            crop_ratio: float = 0.9,
    ):
        self.noise_prob = noise_prob
        self.noise_snr_range = noise_snr_range
        self.scale_prob = scale_prob
        self.scale_range = scale_range
        self.shift_prob = shift_prob
        self.shift_ratio = shift_ratio
        self.warp_prob = warp_prob
        self.warp_sigma = warp_sigma
        self.jitter_prob = jitter_prob
        self.jitter_sigma = jitter_sigma
        self.crop_prob = crop_prob
        self.crop_ratio = crop_ratio

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """Apply augmentations with configured probabilities."""

        # Noise injection (most likely has the largest impact?)
        if np.random.random() < self.noise_prob:
            snr = np.random.uniform(self.noise_snr_range[0], self.noise_snr_range[1])
            signal = add_gaussian_noise(signal, snr)

        # Amplitude scaling
        if np.random.random() < self.scale_prob:
            signal = amplitude_scaling(signal, self.scale_range)

        # Time shift
        if np.random.random() < self.shift_prob:
            signal = time_shift(signal, self.shift_ratio)

        # Time warp
        if np.random.random() < self.warp_prob:
            signal = time_warp(signal, self.warp_sigma)

        # Jittering
        if np.random.random() < self.jitter_prob:
            signal = jittering(signal, self.jitter_sigma)

        # Cropping
        if np.random.random() < self.crop_prob:
            signal = window_crop(signal, self.crop_ratio)

        return signal


# =============================================================================
# Predefined augmentations (these are pretty much inspired by papers)
# =============================================================================

def get_augmentations(name: str) -> Augmentations:
    """
    Get predefined augmentations.

    Augmentations based on findings from papers:
    - 'none': No augmentation (baseline)
    - 'light': Conservative augmentation
    - 'moderate': Balanced augmentation
    - 'heavy': Aggressive augmentation
    - 'noise_only': Only Gaussian noise (for test)
    - 'warp_only': Only time warping (for test)
    """

    augmentations = {
        'none': Augmentations(
            noise_prob=0, scale_prob=0, shift_prob=0,
            warp_prob=0, jitter_prob=0, crop_prob=0
        ),

        'light': Augmentations(
            noise_prob=0.3, noise_snr_range=(6, 12),
            scale_prob=0.2, scale_range=(0.9, 1.1),
            shift_prob=0.2, shift_ratio=0.05,
            warp_prob=0.1, warp_sigma=0.1,
            jitter_prob=0.2, jitter_sigma=0.02,
            crop_prob=0.1, crop_ratio=0.95,
        ),

        'moderate': Augmentations(
            noise_prob=0.5, noise_snr_range=(2, 10),
            scale_prob=0.3, scale_range=(0.8, 1.2),
            shift_prob=0.3, shift_ratio=0.1,
            warp_prob=0.3, warp_sigma=0.2,
            jitter_prob=0.3, jitter_sigma=0.03,
            crop_prob=0.2, crop_ratio=0.9,
        ),

        'heavy': Augmentations(
            noise_prob=0.7, noise_snr_range=(-2, 8),  # Can go to negative SNR
            scale_prob=0.5, scale_range=(0.7, 1.3),
            shift_prob=0.5, shift_ratio=0.15,
            warp_prob=0.5, warp_sigma=0.3,
            jitter_prob=0.5, jitter_sigma=0.05,
            crop_prob=0.3, crop_ratio=0.85,
        ),

        # Test augmentations
        'noise_only': Augmentations(
            noise_prob=0.5, noise_snr_range=(2, 10),
            scale_prob=0, shift_prob=0, warp_prob=0,
            jitter_prob=0, crop_prob=0
        ),

        'warp_only': Augmentations(
            noise_prob=0, scale_prob=0, shift_prob=0,
            warp_prob=0.5, warp_sigma=0.2,
            jitter_prob=0, crop_prob=0
        ),

        'scale_only': Augmentations(
            noise_prob=0, warp_prob=0, shift_prob=0,
            scale_prob=0.5, scale_range=(0.7, 1.3),
            jitter_prob=0, crop_prob=0
        ),
    }

    if name not in augmentations:
        raise ValueError(f"Unknown augmentation: {name}. Available: {list(augmentations.keys())}")

    return augmentations[name]


# =============================================================================
# Augmented wrapper for Dataset
# =============================================================================

class AugmentedSignalDataset(Dataset):
    """
    Dataset wrapper that applies augmentation during training.

    Usage:
        from .data import SignalDataset
        from .augmentation import AugmentedSignalDataset, get_augmentations

        base_dataset = SignalDataset(X_train, y_train, mode)
        augmentations = get_augmentations('moderate')
        train_dataset = AugmentedSignalDataset(base_dataset, augmentation)
    """

    def __init__(
            self,
            base_dataset: Dataset,
            augmentations: Optional[Augmentations] = None,
            augment_prob: float = 0.8,  # Probability of augmenting any sample
    ):
        """
        Args:
            base_dataset: Original dataset
            augmentations: Defining which augmentations to apply
            augment_prob: Overall probability of applying any augmentation
        """
        self.base_dataset = base_dataset
        self.augmentations = augmentations
        self.augment_prob = augment_prob

    def __len__(self):
        return len(self.base_dataset)  # Expected type 'Sized', got 'Dataset' instead

    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]

        # Apply augmentation with probability
        if self.augmentations is not None and np.random.random() < self.augment_prob:
            # Convert to numpy for augmentation
            if isinstance(x, torch.Tensor):
                x_np = x.numpy()
            else:
                x_np = x

            # Apply augmentation
            x_np = self.augmentations(x_np)

            # Convert back to tensor
            x = torch.from_numpy(x_np.copy()).float()

        return x, y


# =============================================================================
# Visualization of Augmentations
# =============================================================================

def visualize_augmentations(signal: np.ndarray, augmentation_name: str = 'moderate', n_samples: int = 5):
    """
    Visualize effect of augmentations on a sample signal.

    Args:
        signal: Original signal (1D or 2D)
        augmentation_name: Name of augmentation (duh)
        n_samples: Number of augmented versions to show

    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt

    augmentation = get_augmentations(augmentation_name)

    fig, axes = plt.subplots(n_samples + 1, 1, figsize=(12, 2 * (n_samples + 1)))

    # Original
    if signal.ndim == 2:
        plot_signal = signal[0]  # First channel
    else:
        plot_signal = signal

    axes[0].plot(plot_signal)
    axes[0].set_title('Original Signal')
    axes[0].set_ylabel('Amplitude')

    # Augmented versions
    for i in range(n_samples):
        aug_signal = augmentation(signal.copy())
        if aug_signal.ndim == 2:
            plot_aug = aug_signal[0]
        else:
            plot_aug = aug_signal

        axes[i + 1].plot(plot_aug)
        axes[i + 1].set_title(f'Augmented #{i + 1}')
        axes[i + 1].set_ylabel('Amplitude')

    axes[-1].set_xlabel('Time (samples)')
    plt.tight_layout()

    return fig


if __name__ == "__main__":
    # Quick test
    print("Testing augmentation functions...")

    # Create synthetic signal
    t = np.linspace(0, 1, 1024)
    signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 25 * t)
    signal = signal.astype(np.float32)

    # Test each augmentation
    print(f"Original signal: shape={signal.shape}, std={signal.std():.4f}")

    noisy = add_gaussian_noise(signal, snr_db=6)
    print(f"Noisy (6dB SNR): std={noisy.std():.4f}")

    scaled = amplitude_scaling(signal, (0.8, 1.2))
    print(f"Scaled: std={scaled.std():.4f}")

    shifted = time_shift(signal, 0.1)
    print(f"Shifted: std={shifted.std():.4f}")

    warped = time_warp(signal, sigma=0.2)
    print(f"Warped: shape={warped.shape}, std={warped.std():.4f}")

    jittered = jittering(signal, 0.05)
    print(f"Jittered: std={jittered.std():.4f}")

    cropped = window_crop(signal, 0.9)
    print(f"Cropped: shape={cropped.shape}")

    # Test augmentations
    augmentations = get_augmentations('moderate')
    augmented = augmentations(signal)
    print(f"Augmentations 'moderate': shape={augmented.shape}, std={augmented.std():.4f}")

    print("\nAll augmentation tests passed!")
