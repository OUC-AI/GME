GME: A Multi-Granularity Attention and Mamba-Driven Underwater Image Enhancer<p align="center"><img src="https://www.google.com/search?q=https://img.shields.io/badge/Task-Underwater_Image_Enhancement-blue" alt="Task"><img src="https://www.google.com/search?q=https://img.shields.io/badge/Framework-PyTorch-red" alt="PyTorch"><img src="https://www.google.com/search?q=https://img.shields.io/badge/License-MIT-green" alt="License"></p>OverviewGME (Group-mix attention and Mamba-driven Enhancer) is a novel deep learning framework designed to restore high-quality visual content from degraded underwater imagery. Underwater images often suffer from severe light absorption, scattering, low contrast, and color distortion.To address these challenges, GME introduces a multi-scale U-shaped encoder-decoder architecture that integrates the efficiency of the Mamba (Visual State Space) model with multi-granularity attention mechanisms. By effectively balancing global modeling capabilities with local detail refinement, GME achieves state-of-the-art performance in recovering color fidelity and contrast while maintaining computational efficiency.FeaturesMulti-Granularity Attention (GMA): Captures multi-scale spatial dependencies to refine fine-grained local details and suppress noise.Visual State Space (VSS) Core: Leverages the Mamba model's linear computational complexity to model long-range global dependencies effectively.Adaptive Enhancement Module (AEM): Utilizes dynamic convolution and learnable modulation factors to perform content-adaptive enhancement based on specific image characteristics.Dual-Domain Correction: Simultaneously addresses global color bias via Squeeze-and-Excitation (SE) and local structural degradation via GMA.State-of-the-Art Performance: Outperforms existing methods (e.g., SyreaNet, WaterNet, UGAN) on benchmarks like LSUI, UIEB, and EUVP.Method ArchitectureThe GME framework is built upon a multi-scale U-shaped architecture consisting of three elaborately designed core modules:Enhancement Layer (EL):Group-Mix Attention (GB/GMA): Splits features into groups and applies depth-wise separable convolutions with varying kernel sizes ($3\times3, 5\times5, 7\times7$) to capture region-level tokens. It employs a unified attention mechanism to model structural degradations.Enhancement Block (EB):VSS Module: Uses 2D Selective Scan (SS2D) operations for direction-aware, global feature enhancement.SE Module: Performs channel-wise recalibration to correct color imbalances (e.g., red channel attenuation).Adaptive Enhancement Module (AEM):Positioned at the bottleneck (end of the encoder).Generates input-dependent kernels via dynamic convolution.Computes a learnable modulation factor $\alpha$ to adaptively regulate enhancement intensity, preventing over-enhancement artifacts.Decoder & Reconstruction:Symmetric patch expansion layers restore spatial resolution.Final reconstruction via convolution and Sigmoid activation.PipelineThe training and inference pipeline follows these steps:Input Processing: Input underwater images ($I_u$) are resized (e.g., to $256 \times 256$) and divided into non-overlapping patches.Patch Embedding: Patches are projected into a $d$-dimensional latent space to form a token sequence.Hierarchical Encoding: The sequence passes through multiple stages of Enhancement Layers (EL). In each stage, the GMA block refines local details, followed by the EB block which corrects global color and contrast using VSS and SE.Adaptive Modulation: The deep features pass through the AEM, where dynamic weights and modulation factors adjust the features based on the specific scene content.Decoding: Features are upsampled via patch expansion and fused with encoder features via skip connections.Optimization: The model is trained end-to-end using a composite loss function: $\mathcal{L}_{total} = \lambda_{MAE}\mathcal{L}_{MAE} + \lambda_{SSIM}\mathcal{L}_{SSIM}$.InstallationPrerequisitesLinux (Ubuntu 20.04 recommended)Python 3.8+PyTorch 1.13+ (with CUDA support)CUDA 11.6+SetupCreate a conda environment and install dependencies. Note that mamba_ssm is required for the VSS module.conda create -n gme python=3.10 -y
conda activate gme

# Install PyTorch (adjust cuda version as needed)
# Example for CUDA 11.8:
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# Install dependencies
pip install -r requirements.txt
DatasetGME is trained primarily on the LSUI (Large Scale Underwater Image) dataset.Data StructureOrganize your dataset as follows:data/
├── LSUI/
│   ├── train/
│   │   ├── input/      # Underwater images
│   │   └── target/     # Ground truth reference images
│   ├── val/
│   │   ├── input/
│   │   └── target/
│   └── test/
│       ├── input/
│       └── target/
Supported DatasetsThe paper evaluates on the following datasets. You may need to download and arrange them similarly:LSUI (Training set: 2,995 pairs)UIEBEUVPOceanExU45 / RUIE / UPoor200 (No-reference benchmarks)UsageTrainingTo train the GME model on the LSUI dataset:python train.py \
  --data_dir ./data/LSUI \
  --batch_size 8 \
  --patch_size 256 \
  --lr 1e-3 \
  --epochs 200 \
  --save_dir ./checkpoints/GME_LSUI
Inference / TestingTo evaluate the model on a test folder:python test.py \
  --input_dir ./data/LSUI/test/input \
  --weights ./checkpoints/GME_LSUI/best_model.pth \
  --result_dir ./results/
ResultsGME demonstrates superior performance on both Full-Reference (FR) and No-Reference (NR) metrics.Quantitative Comparison (LSUI-Test)MethodPSNR ↑SSIM ↑UT-UIE24.4110.822SyreaNet20.9160.764WaterNet22.0100.857GME (Ours)28.5650.887Visual ComparisonQualitative results showing color correction and contrast enhancement:InputUT-UIESyreaNetGME (Ours)Ground Truth<img src="assets/sample_input.jpg" width="120" alt="Input"><img src="assets/sample_utuie.jpg" width="120" alt="UT-UIE"><img src="assets/sample_syreanet.jpg" width="120" alt="SyreaNet"><img src="assets/sample_gme.jpg" width="120" alt="GME"><img src="assets/sample_gt.jpg" width="120" alt="GT">Note: Please replace the image paths in assets/ with your actual result images.Ablation StudyKey components were validated through ablation experiments on the LSUI dataset:w/o VSS: Removing the Visual State Space module significantly drops global contrast performance.w/o GB: Removing Group-Mix Attention leads to loss of fine local details.w/o AEM: Removing the Adaptive Enhancement Module reduces robustness across diverse scenes.Full Model: Achieves the highest PSNR (28.565) and SSIM (0.887), proving the synergy of all modules.Model ZooDownload pretrained weights for different datasets:DatasetMetrics (PSNR/SSIM)DownloadLSUI28.565 / 0.887LSUI_ModelEUVP28.647 / 0.873EUVP_ModelUIEB23.419 / 0.880UIEB_ModelCitationIf you find this work useful in your research, please consider citing:@article{cao2025gme,
  title={GME: A Multi-Granularity Attention and Mamba-Driven Underwater Image Enhancer},
  author={Cao, Jingchao and Liu, Hongqing and Peng, Wangzhen and Liu, Yutao and Gu, Ke and Zhai, Guangtao and Dong, Junyu and Kwong, Sam},
  journal={arXiv preprint},
  year={2025}
}
LicenseThis project is released under the MIT License.
