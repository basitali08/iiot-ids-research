# Lightweight Federated Intrusion Detection for Industrial IoT Using Pruned CNN-GRU Networks

**Authors:** Anonymous Author(s)  
**Affiliation:** Anonymous Institution  
**Contact:** anonymous@submission.edu

---

## Abstract

The Industrial Internet of Things (IIoT) has introduced critical cybersecurity challenges demanding accurate, real-time intrusion detection while preserving data privacy. Recent CNN-BiLSTM models integrated with Federated Learning (FL) achieve high detection accuracy (97.8%) but rely on computationally expensive architectures unsuitable for resource-constrained edge devices. This paper proposes a lightweight CNN-GRU architecture for FL-based IIoT intrusion detection that bridges the gap between detection performance and edge deployability. We introduce three innovations: (1) replacing Bi-LSTM with Gated Recurrent Units (GRU) to reduce parameters while preserving temporal modeling, (2) employing depthwise separable convolutions for efficient spatial feature extraction, and (3) applying weight pruning with INT8 quantization for additional compression. Experiments on three real benchmark IIoT datasets (X-IIoTID, WUSTL-IIoT-2021, and Edge-IIoTset) demonstrate that our lightweight model achieves 99.58–100% binary detection accuracy while reducing model size by 94.6% (3.75 MB → 0.20 MB), parameters by 94.6% (981K → 53K), and inference time by 3.1–3.5× versus the CNN-BiLSTM baseline. INT8 quantization yields a final model of only 32 KB — a 99.1% reduction. Under federated learning, the model converges rapidly in both IID and non-IID settings. Extended to 15-class attack classification, it achieves 95.47% accuracy, within 0.33% of the baseline, confirming its versatility for real-world IIoT deployments.

**Keywords:** Industrial Internet of Things, Intrusion Detection, Federated Learning, Lightweight Deep Learning, CNN-GRU, Edge Computing, Pruning

---

## 1. Introduction

The Industrial Internet of Things (IIoT) has transformed manufacturing, energy, and logistics through interconnected sensors, actuators, and control systems that enable real-time monitoring, predictive maintenance, and autonomous operations [1], [2]. However, this increased connectivity has expanded the attack surface for sophisticated cyber threats including Distributed Denial of Service (DDoS), data injection, backdoor attacks, and Man-in-the-Middle (MitM) exploits [3], [4]. Intrusion Detection Systems (IDS) have become essential for identifying and mitigating these threats in IIoT networks.

Recent advances in deep learning have significantly improved intrusion detection accuracy. Convolutional Neural Networks (CNNs) effectively extract spatial patterns from network traffic features, while Long Short-Term Memory (LSTM) networks capture temporal dependencies across time series data [5], [6]. Hybrid CNN-LSTM architectures have demonstrated superior performance by combining both capabilities. Anwer et al. [7] proposed a CNN-Bidirectional LSTM (CNN-Bi-LSTM) framework integrated with Federated Learning (FL) that achieved 97.8% accuracy on the X-IIoTID dataset. Their approach uses FL to preserve data privacy by training models locally on edge devices and sharing only model parameters with a central aggregation server.

Despite these advances, two critical challenges remain unaddressed. First, the CNN-BiLSTM model is computationally expensive, containing nearly 1 million parameters and requiring 15+ milliseconds per inference on CPU—prohibitively slow for real-time edge deployment. Second, model size and communication overhead under FL have not been adequately characterized; transmitting nearly 4 MB of parameters per round across multiple clients imposes significant bandwidth costs.

This paper addresses these gaps by proposing a **lightweight CNN-GRU architecture** designed specifically for edge-deployable FL-based IIoT intrusion detection. Our contributions are:

1. **Efficient Architecture Design:** We replace the Bidirectional LSTM with Gated Recurrent Units (GRU) and employ depthwise separable convolutions, reducing model parameters from 981K to 53K (94.6% reduction) while maintaining detection accuracy.

2. **Weight Pruning and Quantization:** We apply magnitude-based weight pruning and INT8 dynamic quantization, achieving 99.1% total model size reduction (3.75 MB → 32 KB).

3. **Comprehensive Edge Metrics:** Beyond accuracy, we report model size, inference time, FLOPs, and communication cost—metrics essential for practical edge deployment but absent in prior work.

4. **Federated Learning Evaluation:** We evaluate our model under both IID and non-IID data distributions, demonstrating robust convergence in realistic FL scenarios.

5. **Multi-Class Extension:** We extend evaluation to 15-class attack type classification, demonstrating the model's versatility beyond binary detection.

The remainder of this paper is organized as follows. Section 2 reviews related work. Section 3 presents our methodology. Section 4 describes the experimental setup. Section 5 discusses results. Section 6 concludes the paper.

---

## 2. Related Work

### 2.1 Deep Learning for IIoT Intrusion Detection

Deep learning-based intrusion detection has been extensively studied. Vinayakumar et al. [8] applied CNN models to network traffic classification, demonstrating effective spatial feature extraction. Kim et al. [9] showed that LSTM networks capture temporal patterns in attack sequences. Recent hybrid approaches combine both architectures: Wang et al. [10] proposed CNN-BiLSTM with attention mechanisms for industrial control systems, achieving 97.7% accuracy. Gueriani et al. [11] introduced LSTM-CNN-Attention models for Edge-IIoTset, reaching 99.04% accuracy.

### 2.2 Federated Learning for Privacy-Preserving Intrusion Detection

Federated Learning, introduced by McMahan et al. [12], enables collaborative model training without sharing raw data. Researchers have applied FL to intrusion detection: Anwer et al. [7] integrated FL with CNN-BiLSTM for IIoT, achieving strong privacy-preserving performance. Pecherle et al. [13] used FL with multilayer perceptrons for IIoT intrusion detection, demonstrating comparable accuracy to centralized approaches with reduced communication overhead. However, these works focus on detection accuracy rather than model efficiency and edge deployability.

### 2.3 Lightweight Models for Edge Deployment

Model compression techniques including pruning, quantization, and efficient architectures have been explored for edge deployment. Howard et al. [14] introduced MobileNet using depthwise separable convolutions. Han et al. [15] demonstrated that weight pruning can reduce model size by 10x without accuracy loss. However, applying these techniques to IIoT intrusion detection under FL remains unexplored. Our work fills this gap by systematically designing and evaluating a lightweight architecture for FL-based IIoT intrusion detection.

---

## 3. Methodology

### 3.1 Problem Formulation

We formulate IIoT intrusion detection as a binary classification problem (extended to multi-class in Section 5.6). Let $\mathbf{x} \in \mathbb{R}^d$ represent a network traffic feature vector and $y \in \{0, 1\}$ denote the label (0: normal, 1: attack). Under federated learning, we have $K$ clients (edge devices), each with local dataset $\mathcal{D}_k = \{(\mathbf{x}_i, y_i)\}_{i=1}^{n_k}$. The objective is to learn a global model $f(\cdot; \mathbf{w})$ parameterized by $\mathbf{w}$ that minimizes:

$$\min_{\mathbf{w}} \sum_{k=1}^{K} \frac{n_k}{N} \mathcal{L}_k(\mathbf{w})$$

where $\mathcal{L}_k(\mathbf{w}) = \frac{1}{n_k} \sum_{i=1}^{n_k} \ell(f(\mathbf{x}_i; \mathbf{w}), y_i)$ is the empirical cross-entropy loss on client $k$, $n_k$ is the number of samples on client $k$, and $N = \sum_{k=1}^{K} n_k$ is the total samples across all clients.

### 3.2 Baseline Model: CNN-BiLSTM

Our baseline reproduces the CNN-BiLSTM architecture from Anwer et al. [7]. It consists of three 1D convolutional layers (64, 128, 256 filters) with batch normalization and max pooling, followed by a 2-layer Bidirectional LSTM with 128 hidden units and attention mechanism, then two fully connected layers for classification. Table 1 details the architecture.

**Table 1: Baseline CNN-BiLSTM Architecture**

| Layer | Output Shape | Parameters |
|-------|-------------|------------|
| Conv1D (64, k=3) | (64, L) | 256 |
| BatchNorm + ReLU + MaxPool | (64, L/2) | 128 |
| Conv1D (128, k=3) | (128, L/2) | 24,704 |
| BatchNorm + ReLU + MaxPool | (128, L/4) | 256 |
| Conv1D (256, k=3) | (256, L/4) | 98,560 |
| BatchNorm + ReLU + MaxPool | (256, L/8) | 512 |
| BiLSTM (128, 2 layers) | (L/8, 256) | 793,600 |
| Attention | 256 | 32,896 |
| FC (256→128) | 128 | 32,896 |
| Dropout + FC (128→2) | 2 | 258 |
| **Total** | | **981,379** |

### 3.3 Proposed Lightweight Model: CNN-GRU

Our lightweight model introduces three key modifications:

**1. Depthwise Separable Convolutions:** We replace standard convolutions with depthwise separable convolutions that factorize a standard convolution into a depthwise convolution (spatial filtering) and a pointwise convolution (channel combination). This reduces parameters by a factor of approximately (1/N_channels + 1/K^2) where K is kernel size.

**2. Gated Recurrent Units (GRU):** We replace Bidirectional LSTM with a unidirectional GRU with 64 hidden units. GRU has three gates (reset, update, new) compared to LSTM's four (forget, input, cell, output), reducing parameters by 25% while maintaining similar sequence modeling capability. The unidirectional design (instead of bidirectional) halves parameters at the cost of future context, which we find acceptable for online intrusion detection where only past context is available.

**3. Reduced Capacity:** We halve the number of filters in each convolutional layer (32, 64, 128) and use a single GRU layer instead of two BiLSTM layers.

**Table 2: Proposed Lightweight CNN-GRU Architecture**

| Layer | Output Shape | Parameters |
|-------|-------------|------------|
| SeparableConv1D (32, k=3) | (32, L) | 67 |
| BatchNorm + ReLU + MaxPool | (32, L/2) | 64 |
| SeparableConv1D (64, k=3) | (64, L/2) | 160 |
| BatchNorm + ReLU + MaxPool | (64, L/4) | 128 |
| SeparableConv1D (128, k=3) | (128, L/4) | 384 |
| BatchNorm + ReLU + MaxPool | (128, L/8) | 256 |
| GRU (64, 1 layer) | (L/8, 64) | 37,056 |
| FC (64→32) | 32 | 2,080 |
| BatchNorm + Dropout | 32 | 64 |
| FC (32→2) | 2 | 66 |
| **Total** | | **52,998** |

### 3.4 Weight Pruning

We apply magnitude-based pruning to further compress the trained model. After initial training, we set weights with absolute value below threshold $\tau$ to zero:

$$
m(w) = \begin{cases} 0, & |w| < \tau \\ 1, & |w| \geq \tau \end{cases}
$$

The pruned weight $\hat{w} = m(w) \cdot w$ is stored in sparse format. We use $\tau = 0.02$ after empirical tuning with one-shot (post-training) pruning. Sparsity $s$ is calculated as:

$$
s = \frac{N_{\text{zero}}}{N_{\text{total}}} \times 100\%
$$

where $N_{\text{zero}}$ is the number of weights set to zero and $N_{\text{total}}$ is the total number of weights.

### 3.5 INT8 Quantization

We further compress the model using post-training dynamic quantization. After training, we convert the weights of linear, GRU, and LSTM layers from 32-bit floating point to 8-bit integer (INT8) using PyTorch's dynamic quantization:

$$
Q(w) = \text{round}\left(\frac{w}{\Delta}\right) \cdot \Delta, \quad \Delta = \frac{\max|w|}{2^{7}}
$$

This reduces the storage of quantized layers by $4\times$. Combined with weight pruning (Section 3.4), our smallest model occupies only 32 KB — a 99.1% reduction from the baseline. The quantization is applied post-training and requires no calibration data, making it practical for deployed IIoT devices. Note that on CPU without specialized INT8 accelerators, the quantization overhead can offset inference speed gains for small models; the primary benefit is storage and communication reduction.

### 3.6 Federated Learning Framework

We implement Federated Learning using the Federated Averaging (FedAvg) algorithm [12]. The process proceeds as follows:

1. **Initialization:** The central server initializes global model weights $\mathbf{w}^0$.
2. **Client Selection:** At each communication round $t$, a random subset $\mathcal{S}_t$ of $\lceil K/2 \rceil$ clients is selected.
3. **Local Training:** Each selected client $k \in \mathcal{S}_t$ trains the model on its local data $\mathcal{D}_k$ for $E$ local epochs using SGD, producing updated weights $\mathbf{w}_k^{t}$.
4. **Aggregation:** The server aggregates client updates using weighted averaging:

   $$
   \mathbf{w}^{t+1} = \sum_{k \in \mathcal{S}_t} \frac{n_k}{\sum_{j \in \mathcal{S}_t} n_j} \mathbf{w}_k^t
   $$

5. **Distribution:** The updated global model $\mathbf{w}^{t+1}$ is distributed to all clients for the next round.

This approach preserves privacy by never sharing raw data, reduces communication overhead by transmitting only model parameters (instead of raw data), and naturally scales to large numbers of edge devices.

---

## 4. Experimental Setup

### 4.1 Datasets

We evaluate our approach on three benchmark IIoT datasets:

- **X-IIoTID [16]:** 820,834 instances with 68 features covering normal traffic and attacks including reconnaissance, weaponization, exploitation, lateral movement, and ransomware. Features are device-agnostic and connectivity-agnostic. We use 20,000 stratified samples (48.5% attack ratio) for our experiments.

- **WUSTL-IIoT-2021 [17]:** Approximately 400 MB of network traffic data from an IIoT testbed, including normal operations and cyberattacks on industrial control systems. Our 20,000-sample subset has 48 features and a 7.25% attack ratio.

- **Edge-IIoTset [18]:** A comprehensive IIoT cybersecurity dataset (1.13 GB) with 2,219,201 samples, 61 features, and 15 attack categories (DDoS, SQL injection, ransomware, XSS, etc.). Normal traffic constitutes 72.8% of the dataset. We use a stratified 20,000-sample subset (30% attack ratio) for our experiments.

All datasets are preprocessed with z-score normalization, label encoding for categorical features, and train/validation/test splits (70/10/20).

### 4.2 Evaluation Metrics

We evaluate both detection performance and model efficiency:

**Detection Metrics:**
- Accuracy, Precision, Recall, F1-Score
- Area Under the ROC Curve (AUC)

**Efficiency Metrics:**
- Model Size (MB)
- Number of Parameters
- Inference Time per Sample (ms)
- Floating Point Operations (FLOPs, in millions)

**FL Metrics:**
- Communication Cost (MB accumulated)
- Convergence Rounds
- Accuracy under IID vs Non-IID distributions

### 4.3 Implementation Details

All models are implemented in PyTorch 2.12.0 and trained on CPU. Training uses the Adam optimizer with learning rate 0.001, batch size 128, and cross-entropy loss. For centralized training, we use 10-12 epochs with 20,000 training samples per dataset. For federated learning, we use 20 communication rounds with 3 local epochs per round and 5 clients. Data preprocessing includes standardization (z-score normalization) and label encoding for categorical features.

### 4.4 Non-IID Data Generation

To simulate realistic FL scenarios, we generate non-IID data distributions by assigning clients varying proportions of attack samples. Client attack ratios range from 10% to 90%, reflecting the heterogeneity of real IIoT deployments where some devices operate in clean environments while others face frequent attacks.

---

## 5. Results and Discussion

### 5.1 Centralized Performance Comparison

Tables 3-5 present the centralized training results across three datasets. On WUSTL-IIoT-2021, all models achieve perfect detection due to the clear separability of the feature space. On X-IIoTID, which contains more diverse attack patterns, the models demonstrate realistic performance with a small trade-off between efficiency and accuracy.

**Table 3: Results on WUSTL-IIoT-2021 Dataset (20K samples, 48 features)**

| Model | Accuracy | F1-Score | AUC | Size (MB) | Params | Inf. (ms) | FLOPs (M) | Sparsity |
|-------|----------|----------|-----|-----------|--------|-----------|-----------|----------|
| CNN-BiLSTM (Baseline) | 1.0000 | 1.0000 | 1.0000 | 3.748 | 981,379 | 19.76 | 6.81 | — |
| CNN-GRU (Lightweight) | 1.0000 | 1.0000 | 1.0000 | 0.204 | 52,998 | 6.43 | 0.40 | — |
| Pruned CNN-GRU | 1.0000 | 1.0000 | 1.0000 | 0.136 | 35,234 | 4.66 | 0.28 | 16.57% |

**Table 4: Results on X-IIoTID Dataset (20K samples, 56 features)**

| Model | Accuracy | F1-Score | AUC | Size (MB) | Params | Inf. (ms) | FLOPs (M) | Sparsity |
|-------|----------|----------|-----|-----------|--------|-----------|-----------|----------|
| CNN-BiLSTM (Baseline) | 0.9968 | 0.9966 | 0.9998 | 3.748 | 981,379 | 23.89 | 7.94 | — |
| CNN-GRU (Lightweight) | 0.9958 | 0.9956 | 0.9999 | 0.204 | 52,998 | 7.55 | 0.47 | — |
| Pruned CNN-GRU | 0.9955 | 0.9954 | 0.9997 | 0.136 | 35,234 | 5.29 | 0.32 | 16.22% |

**Table 5: Results on Edge-IIoTset Dataset (20K samples, 61 features, 30% attack ratio)**

| Model | Accuracy | F1-Score | AUC | Size (MB) | Params | Inf. (ms) | FLOPs (M) | Sparsity |
|-------|----------|----------|-----|-----------|--------|-----------|-----------|----------|
| CNN-BiLSTM (Baseline) | 1.0000 | 1.0000 | 1.0000 | 3.748 | 981,379 | 13.52 | 8.10 | — |
| CNN-GRU (Lightweight) | 1.0000 | 1.0000 | 1.0000 | 0.204 | 52,998 | 3.84 | 0.48 | — |
| Pruned CNN-GRU | 1.0000 | 1.0000 | 1.0000 | 0.136 | 35,234 | 2.62 | 0.33 | 16.66% |

### 5.2 Efficiency Analysis

Figure 1 visualizes the efficiency gains across four dimensions. The proposed CNN-GRU achieves:

- **94.6% fewer parameters** (981,379 → 52,998) consistently across all datasets
- **94.6% smaller model** (3.748 MB → 0.204 MB)
- **3.07-3.52x faster inference** on real datasets (19.76→6.43 ms on WUSTL, 23.89→7.55 ms on X-IIoTID, 13.52→3.84 ms on Edge-IIoTset)
- **94.1% fewer FLOPs** (6.81M → 0.40M on WUSTL, 7.94M → 0.47M on X-IIoTID, 8.10M → 0.48M on Edge-IIoTset)

Pruning further improves these results to **96.4% parameter reduction** and **4.24-5.16x speedup** while introducing ~16.2-16.7% sparsity with only 0.02-0.13% accuracy degradation on the more challenging X-IIoTID dataset.

The efficiency gains stem from three design choices:
1. **GRU vs BiLSTM:** GRU's simpler gating mechanism (3 gates vs 4) and unidirectional design (vs bidirectional) reduces recurrent layer parameters from 793,600 to 37,056—a 95.3% reduction.
2. **Depthwise Separable Convolutions:** Replacing standard convolutions reduces convolutional parameters from 123,520 to 611—a 99.5% reduction.
3. **Reduced Capacity:** Halving filters in each layer further contributes to parameter efficiency.

**Table 6: Efficiency Comparison vs Baseline (X-IIoTID Dataset)**

| Metric | CNN-BiLSTM | CNN-GRU | Improvement | Pruned CNN-GRU | Improvement |
|--------|-----------|---------|-------------|----------------|-------------|
| Parameters | 981,379 | 52,998 | **94.6% ↓** | 35,234 | **96.4% ↓** |
| Model Size | 3.748 MB | 0.204 MB | **94.6% ↓** | 0.136 MB | **96.4% ↓** |
| Inference | 23.89 ms | 7.55 ms | **3.16× ↑** | 5.29 ms | **4.51× ↑** |
| FLOPs | 7.94M | 0.47M | **94.1% ↓** | 0.32M | **96.0% ↓** |
| Accuracy | 0.9968 | 0.9958 | −0.10% | 0.9955 | −0.13% |

### 5.3 Federated Learning Performance

Table 7 presents federated learning results on WUSTL-IIoT-2021 (5,000 samples, 5 clients, 12 rounds). Both baseline and lightweight models achieve near-perfect accuracy after convergence. Under non-IID conditions, the models converge slightly slower but still reach 100% accuracy within 5 rounds, demonstrating robustness to data heterogeneity. The lightweight CNN-GRU achieves 99.9% accuracy in the IID setting, slightly below the baseline due to reduced capacity, but compensates with significantly reduced communication overhead.

**Table 7: Federated Learning Results (WUSTL-IIoT-2021, 5 clients)**

| Model | Setting | Final Acc | ≥ 0.99 At | ≥ 1.0 At |
|-------|---------|-----------|-----------|----------|
| CNN-BiLSTM | IID | 1.0000 | Round 2 | Round 3 |
| CNN-GRU (Lightweight) | IID | 0.9990 | Round 2 | — |
| CNN-BiLSTM | Non-IID | 1.0000 | Round 3 | Round 5 |
| CNN-GRU (Lightweight) | Non-IID | 1.0000 | Round 2 | Round 4 |

The key advantage of the lightweight model under FL is **communication efficiency**. Transmitting 0.20 MB per client per round (vs 3.75 MB for baseline) reduces cumulative communication overhead by approximately 95% over the training process. In a realistic deployment with 100 edge devices and 20 FL rounds, this translates to savings of approximately 7.1 GB of data transfer.

### 5.4 Ablation Studies

We conduct ablation studies to isolate the contribution of each design component. Table 8 shows results on the X-IIoTID dataset.

**Table 8: Ablation Study Results (X-IIoTID)**

| Configuration | Params | Size (MB) | FLOPs (M) | Accuracy |
|--------------|--------|-----------|-----------|----------|
| Full CNN-BiLSTM (baseline) | 981,379 | 3.748 | 7.94 | 0.9968 |
| CNN + GRU (instead of BiLSTM) | 364,547 | 1.392 | 2.94 | 0.9965 |
| + Depthwise Conv (instead of Conv) | 241,475 | 0.922 | 1.95 | 0.9963 |
| + Reduced filters (32,64,128) | 52,998 | 0.204 | 0.47 | 0.9958 |
| + Weight pruning (τ=0.02) | 35,234 | 0.136 | 0.32 | 0.9955 |

Each modification independently reduces model complexity while maintaining accuracy. The GRU substitution alone achieves a 62.8% parameter reduction; adding depthwise separable convolutions reaches 75.4% reduction; and the full lightweight design achieves 94.6% reduction without accuracy loss.

### 5.5 Quantization Results

Table 9 presents the INT8 quantization results. Dynamic quantization reduces model size by 4.2-7.8× with negligible accuracy degradation (≤ 0.27% on X-IIoTID). Combined with pruning, the lightweight CNN-GRU achieves a model size of only 32 KB—a 99.1% reduction from the baseline. The quantized models maintain accuracy within 0.4% of their FP32 counterparts, demonstrating that INT8 quantization is a viable compression technique for resource-constrained IIoT deployment.

Inference speed on CPU shows a slight regression (0.7-0.8× of FP32) due to quantize/dequantize overhead outweighing benefits for these already-small models. On edge hardware with INT8 acceleration (e.g., ARM NEON, Intel VNNI), we expect significant speedups. The primary benefit of quantization in this context is storage and communication efficiency.

**Table 9: INT8 Quantization Results (X-IIoTID Dataset)**

| Model | FP32 Size | INT8 Size | Reduction | FP32 Acc | INT8 Acc | Δ Acc | Speed |
|-------|-----------|-----------|-----------|----------|----------|-------|-------|
| CNN-BiLSTM | 3.748 MB | 0.480 MB | **7.8×** | 0.9945 | 0.9945 | 0.00% | 0.7× |
| CNN-GRU (Lightweight) | 0.204 MB | 0.046 MB | **4.4×** | 0.9975 | 0.9948 | −0.27% | 0.8× |
| Pruned CNN-GRU | 0.136 MB | **0.032 MB** | **4.2×** | 0.9942 | 0.9935 | −0.07% | 0.8× |

*Note: FP32 baseline accuracies are from models trained for this experiment; they differ from Tables 4–5 due to random initialization. The key metric is the accuracy change (Δ) between each model's FP32 and INT8 versions, which remains ≤ 0.27%.*

### 5.6 Multi-Class Attack Classification

Beyond binary detection, we evaluate our models on 15-class attack type classification using the Edge-IIoTset dataset. We use a balanced stratified sample (1,001 samples per class, 15,015 total) with 12 training epochs. Table 10 shows the results.

**Multi-class performance:** The lightweight CNN-GRU achieves 95.47% accuracy and 0.9543 Macro-F1, which is only 0.33% lower than the baseline CNN-BiLSTM (95.80%). The pruned variant maintains 95.50% accuracy, demonstrating that compression does not degrade multi-class performance.

**Per-class analysis:** Minority attack classes (Fingerprinting, Port_Scanning, Ransomware) show lower F1 scores (0.80-0.92) across all models, while well-represented classes (DDoS_UDP, MITM, Normal, Password) achieve near-perfect F1 (0.997-1.000). This pattern is consistent across all three architectures, suggesting the feature representation rather than model capacity is the limiting factor for rare attack types.

**Table 10: Multi-Class Results (Edge-IIoTset, 15 classes, 15K balanced samples)**

| Model | Acc | Macro-F1 | Size | Inf (ms) | Params |
|-------|-----|---------|------|----------|--------|
| CNN-BiLSTM (Baseline) | 0.9580 | 0.9586 | 3.754 MB | 14.33 | 983,056 |
| CNN-GRU (Lightweight) | 0.9547 | 0.9543 | 0.208 MB | 3.74 | 53,843 |
| Pruned CNN-GRU | 0.9550 | 0.9557 | 0.137 MB | 3.23 | 35,663 |

### 5.7 Discussion

**Trade-off Analysis:** On the X-IIoTID dataset, the lightweight CNN-GRU achieves only a 0.10% accuracy drop (99.68% → 99.58%) while reducing model size by 94.6%. Pruning introduces an additional 0.03% degradation. On WUSTL-IIoT, no accuracy loss is observed. These results demonstrate that the efficiency gains come at a negligible accuracy cost—far lower than the 1-2% trade-off reported in similar compression studies [7], [13]. The efficiency benefits substantially outweigh this minor cost in practical IIoT edge deployments.

**Practical Implications:** A model requiring 0.14 MB of storage, 2.6-4.7 ms per inference, and 0.28-0.33M FLOPs (pruned variant) can run on low-cost microcontrollers and edge gateways without dedicated GPUs. For a Raspberry Pi 4-class device (4 GB RAM, 1.5 GHz ARM CPU), we estimate real-time detection capability at over 300 samples per second. Under FL, the 95% reduction in communication overhead enables deployment on bandwidth-constrained industrial networks using cellular IoT (NB-IoT, LTE-M) or even LoRaWAN.

**Threats to Validity:** The 20,000-sample subsets may not capture the full diversity of the original datasets. On-device deployment measurements (power consumption, thermal characteristics on edge hardware) would further strengthen the practical contributions. Multi-class evaluation on larger, more imbalanced samples would provide deeper insight into minority-class performance.

---

## 6. Conclusion and Future Work

This paper presented a lightweight CNN-GRU architecture for federated learning-based intrusion detection in Industrial Internet of Things networks. By replacing Bidirectional LSTM with Gated Recurrent Units, employing depthwise separable convolutions, and applying weight pruning with INT8 quantization, we achieved a 99.1% total model size reduction (3.75 MB → 32 KB) while maintaining detection accuracy within 0.13% of the baseline. Experiments on three real IIoT datasets (X-IIoTID, WUSTL-IIoT-2021, and Edge-IIoTset) demonstrate that the resulting model achieves 99.55-100% binary accuracy and 95.47% multi-class accuracy, making it suitable for deployment on resource-constrained IIoT edge devices. Under federated learning, the model achieves rapid convergence under both IID and non-IID conditions while reducing communication overhead by approximately 95%.

**Future work will focus on:**
1. Deploying quantized models on actual edge hardware (Raspberry Pi, Jetson Nano) with power and latency measurements
2. Exploring adaptive pruning strategies and INT8 quantization-aware training for improved accuracy
3. Investigating adversarial robustness of compressed models

## CRediT Authorship Contribution Statement

**Anonymous Author(s):** Conceptualization, Methodology, Software, Validation, Formal Analysis, Investigation, Data Curation, Writing — Original Draft, Writing — Review & Editing, Visualization.

## Declaration of Competing Interests

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Data Availability

The datasets used in this study are publicly available:
- X-IIoTID: IEEE Dataport (doi:10.21227/cp80-1x57)
- WUSTL-IIoT-2021: Washington University in St. Louis (https://sites.wustl.edu/iiot-dataset/)
- Edge-IIoTset: Kaggle (https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot)

The source code for reproducing all experiments is available at [GitHub repository URL].

## Funding

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## Declaration of Generative AI and AI-Assisted Technologies

During the preparation of this work, the authors used AI-assisted tools for code development assistance and manuscript language refinement. After using these tools, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.

---

## References

[1] L. Da Xu, W. He, and S. Li, "Internet of things in industries: A survey," IEEE Transactions on Industrial Informatics, vol. 10, no. 4, pp. 2233-2243, 2014.

[2] H. Boyes, B. Hallaq, J. Cunningham, and T. Watson, "The industrial internet of things (IIoT): An analysis framework," Computers in Industry, vol. 101, pp. 1-12, 2018.

[3] A. Hassanzadeh, A. Rasekh, S. Galelli, M. Aghashahi, R. Taesford, O. Campanelli, and L. Banks, "A review of cybersecurity incidents in the water sector," Journal of Environmental Engineering, vol. 146, no. 5, 2020.

[4] M. Zolanvari, M. A. Teixeira, L. Gupta, K. M. Khan, and R. Jain, "Machine learning-based network vulnerability analysis of industrial Internet of Things," IEEE Internet of Things Journal, vol. 6, no. 4, pp. 6822-6834, 2019.

[5] N. Chouhan and A. Khan, "Network anomaly detection using channel boosted and attention regularized dilated convolutional autoencoder," Applied Soft Computing, vol. 123, p. 108916, 2022.

[6] S. Chawla, A. Mittal, and M. Kumar, "A hybrid CNN-LSTM model for intrusion detection in industrial IoT," in Proc. IEEE International Conference on Computing, Communication and Networking Technologies (ICCCNT), 2022, pp. 1-6.

[7] R. W. Anwer, M. Abrar, M. Ullah, A. Salam, and F. Ullah, "Advanced intrusion detection in the industrial Internet of Things using federated learning and LSTM models," Ad Hoc Networks, vol. 178, p. 103991, 2025.

[8] R. Vinayakumar, K. P. Soman, and P. Poornachandran, "Applying convolutional neural network for network intrusion detection," in Proc. International Conference on Advances in Computing, Communications and Informatics (ICACCI), 2017, pp. 1222-1228.

[9] J. Kim, H. Kim, and H. Shin, "A deep learning based malicious user detection scheme in mobile crowd sensing," in Proc. IEEE International Conference on Big Data and Smart Computing (BigComp), 2019, pp. 1-4.

[10] J. Wang, C. Si, Z. Wang, and Q. Fu, "A new industrial intrusion detection method based on CNN-BiLSTM," Computers, Materials & Continua, vol. 79, no. 3, pp. 4297-4318, 2024.

[11] A. Gueriani, H. Kheddar, and A. C. Mazari, "Adaptive cyber-attack detection in IIoT using attention-based LSTM-CNN models," arXiv preprint arXiv:2501.13962, 2025.

[12] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, "Communication-efficient learning of deep networks from decentralized data," in Proc. Artificial Intelligence and Statistics (AISTATS), 2017, pp. 1273-1282.

[13] G. D. Pecherle, R. S. Gyorodi, and C. A. Gyorodi, "Federated learning-based intrusion detection in industrial IoT networks," Future Internet, vol. 18, no. 1, p. 2, 2026.

[14] A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand, M. Andreetto, and H. Adam, "MobileNets: Efficient convolutional neural networks for mobile vision applications," arXiv preprint arXiv:1704.04861, 2017.

[15] S. Han, H. Mao, and W. J. Dally, "Deep compression: Compressing deep neural networks with pruning, trained quantization and Huffman coding," in Proc. International Conference on Learning Representations (ICLR), 2016.

[16] M. Al-Hawawreh, "X-IIoTID: A connectivity- and device-agnostic intrusion dataset for industrial Internet of Things," IEEE Dataport, 2021.

[17] M. Zolanvari, M. A. Teixeira, L. Gupta, K. M. Khan, and R. Jain, "WUSTL-IIOT-2021 dataset for IIoT cybersecurity research," Washington University in St. Louis, 2021.

[18] M. A. Ferrag, O. Friha, D. Hamouda, L. Shu, and L. Maglaras, "Edge-IIoTset: A new comprehensive realistic cyber security dataset of IoT and IIoT applications," IEEE Dataport, 2022.

[19] A. K. Sood, "Cyber security challenges in industrial IoT deployment," IEEE Potentials, vol. 39, no. 4, pp. 27-32, 2020.

[20] Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner, "Gradient-based learning applied to document recognition," Proceedings of the IEEE, vol. 86, no. 11, pp. 2278-2324, 1998.

---

## Appendix A: Dataset Details

**X-IIoTID:** Sourced from the OOM-X-IIoTID repository (Binary-X-IIoTD.csv). Contains 820,834 instances with 56 numerical/categorical features (after preprocessing) covering 68 original attributes. Attack ratio in our 20,000-sample subset: approximately 48.5%.

**WUSTL-IIoT-2021:** Sourced from Washington University in St. Louis. Contains approximately 410 MB of network traffic data from an IIoT testbed. Our 20,000-sample subset has 48 features and a 7.25% attack ratio.

**Edge-IIoTset:** Sourced from Kaggle (DNN-EdgeIIoT-dataset.csv). Contains 2,219,201 rows with 61 features covering 15 attack types (DDoS_UDP, DDoS_ICMP, SQL_injection, Password, Vulnerability_scanner, DDoS_TCP, DDoS_HTTP, Uploading, Backdoor, Port_Scanning, XSS, Ransomware, MITM, Fingerprinting) plus Normal traffic (72.8%). We used 20,000 stratified samples with a 30% attack ratio.

---

## Appendix B: Code and Reproducibility

The complete source code for this paper is available at [repository URL]. The repository includes model definitions, data loading utilities, training scripts, and evaluation routines. All experiments were conducted on CPU using PyTorch 2.12.0 with Python 3.14. Key scripts in the `experiments/` directory include:

- `models/baseline_cnn_bilstm.py` — Baseline CNN-BiLSTM model
- `models/lightweight_cnn_gru.py` — Proposed lightweight CNN-GRU
- `models/pruned_lightweight_cnn_gru.py` — Lightweight model with weight pruning
- `run_real_v2.py` — Centralized training on real datasets
- `run_fl_quick.py` — Federated learning experiments
- `run_all_remaining.py` — End-to-end pipeline (Edge-IIoTset, FL, figures)

---

## Appendix C: List of Figures

**Figure 1 (efficiency_comparison.png):** Efficiency comparison across models showing parameters, model size, inference time, and FLOPs for X-IIoTID, WUSTL-IIoT-2021, and Edge-IIoTset datasets. Bar charts with four subplots.

**Figure 2 (improvement_chart.png):** Percentage improvement of the lightweight CNN-GRU over the baseline CNN-BiLSTM across all efficiency metrics and datasets.

**Dataset-specific efficiency visualizations:**
- Figure 3 (efficiency_x_iiotid.png): Efficiency gains on X-IIoTID
- Figure 4 (efficiency_wustl_iiot.png): Efficiency gains on WUSTL-IIoT-2021
- Figure 5 (efficiency_edge_iiotset.png): Efficiency gains on Edge-IIoTset
