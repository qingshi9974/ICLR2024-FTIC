# ICLR2024-Frequency-aware-Transformer-for-Learned-Image-Compression
[ICLR2024] FLIC: Frequency-aware Transformer for Learned Image Compression

## Introduction
This repository is the offical [Pytorch](https://pytorch.org/) implementation of [FLIC: Frequency-aware Transformer for Learned Image Compression (ICLR2024)]https://openreview.net/forum?id=HKGQDDTuvZ). 

**Abstract.**
Learned image compression (LIC) has gained traction as an effective solution for image storage and transmission in recent years. However, existing LIC methods are redundant in latent representation due to limitations in capturing anisotropic frequency components and preserving directional details. To overcome these challenges, we propose a novel frequency-aware transformer (FAT) block that for the first time achieves multiscale directional ananlysis for LIC. The FAT block comprises frequency-decomposition window attention (FDWA) modules to capture multiscale and directional frequency components of natural images. Additionally, we introduce frequency-modulation feed-forward network (FMFFN) to adaptively modulate different frequency components, improving rate-distortion performance. Furthermore, we present a transformer-based channel-wise autoregressive (T-CA) model that effectively exploits channel dependencies. Experiments show that our method achieves state-of-the-art rate-distortion performance compared to existing LIC methods, and evidently outperforms latest standardized codec VTM-12.1 by 14.5%, 15.1%, 13.0% in BD-rate on the Kodak, Tecnick, and CLIC datasets.


