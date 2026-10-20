## Model Architecture Analysis - Complete Calculations

### Assumptions:
- **Input**: Grayscale image with `NIC = 1` channel
- **Base filters**: `NF`
- **Output classes**: `NC = 24`
- **Kernel sizes**: `k` for conv layers, `kt` for transpose conv layers
- **Input spatial size**: `S × S`
- **Padding**: `k // 2` for conv layers, no padding for transpose conv
- **Stride**: 2 for pooling and transpose conv layers

### 1. Weight Calculations

**General Formula for Layer i**: `W_i = C_in × C_out × K_h × K_w`

| Layer | Type | Channels | Kernel | Weight Calculation |
|-------|------|----------|--------|-------------------|
| 1 | Conv2d | NIC → NF | k×k | `W₁ = NIC × NF × k × k = k² × NIC × NF` |
| 2 | Conv2d | NF → 2NF | k×k | `W₂ = NF × 2NF × k × k = 2k² × NF²` |
| 3 | Conv2d | 2NF → 4NF | k×k | `W₃ = 2NF × 4NF × k × k = 8k² × NF²` |
| 4 | ConvTranspose2d | 4NF → 2NF | kt×kt | `W₄ = 4NF × 2NF × kt × kt = 8kt² × NF²` |
| 5 | ConvTranspose2d | 2NF → NF | kt×kt | `W₅ = 2NF × NF × kt × kt = 2kt² × NF²` |
| 6 | ConvTranspose2d | NF → NC | kt×kt | `W₆ = NF × NC × kt × kt = kt² × NF × NC` |
| 7 | Conv2d 1×1 | NC → NC | 1×1 | `W₇ = NC × NC × 1 × 1 = NC²` |

#### Total Number of Weights

$$
W_{\text{total}} = W_1 + W_2 + W_3 + W_4 + W_5 + W_6 + W_7
$$

$$
W_{\text{total}} = k^2 \times NIC \times NF \;+\; 2k^2 \times NF^2 \;+\; 8k^2 \times NF^2 \;+\; 8k_t^2 \times NF^2 \;+\; 2k_t^2 \times NF^2 \;+\; k_t^2 \times NF \times NC \;+\; NC^2
$$

$$
W_{\text{total}} = k^2 \times NIC \times NF \;+\; 10k^2 \times NF^2 \;+\; 10k_t^2 \times NF^2 \;+\; k_t^2 \times NF \times NC \;+\; NC^2
$$


### 2. Output Calculations

**General Formula for Layer i**: `O_i = C_out × H_out × W_out`

| Layer | Output Dimensions | Output Calculation |
|-------|------------------|-------------------|
| 1 | [B, NF, S/2, S/2] | `O₁ = NF × (S/2) × (S/2) = NF × S²/4` |
| 2 | [B, 2NF, S/4, S/4] | `O₂ = 2NF × (S/4) × (S/4) = 2NF × S²/16` |
| 3 | [B, 4NF, S/8, S/8] | `O₃ = 4NF × (S/8) × (S/8) = 4NF × S²/64` |
| 4 | [B, 2NF, S/4, S/4] | `O₄ = 2NF × (S/4) × (S/4) = 2NF × S²/16` |
| 5 | [B, NF, S/2, S/2] | `O₅ = NF × (S/2) × (S/2) = NF × S²/4` |
| 6 | [B, NC, S, S] | `O₆ = NC × S × S = NC × S²` |
| 7 | [B, NC, S, S] | `O₇ = NC × S × S = NC × S²` |

#### Total Number of Outputs

$$
O_{\text{total}} = O_1 + O_2 + O_3 + O_4 + O_5 + O_6 + O_7
$$

$$
O_{\text{total}} = NF \times \frac{S^2}{4} + 2NF \times \frac{S^2}{16} + 4NF \times \frac{S^2}{64} + 2NF \times \frac{S^2}{16} + NF \times \frac{S^2}{4} + NC \times S^2 + NC \times S^2
$$

$$
O_{\text{total}} = NF \times \frac{S^2}{4} + NF \times \frac{S^2}{8} + NF \times \frac{S^2}{16} + NF \times \frac{S^2}{8} + NF \times \frac{S^2}{4} + 2NC \times S^2
$$

$$
O_{\text{total}} = \left(\frac{1}{4} + \frac{1}{8} + \frac{1}{16} + \frac{1}{8} + \frac{1}{4}\right) \times NF \times S^2 + 2NC \times S^2
$$

$$
O_{\text{total}} = \left(\frac{13}{16}\right) \times NF \times S^2 + 2NC \times S^2
$$

$$
O_{\text{total}} = 0.8125 \times NF \times S^2 + 2NC \times S^2
$$


### 3. Connection Calculations

**General Formula for Layer i**: `C_i = W_i × (H_in × W_in)`

| Layer | Input Dimensions | Connection Calculation |
|-------|------------------|----------------------|
| 1 | [B, NIC, S, S] | `C₁ = (k²×NIC×NF) × (S×S) = k²×NIC×NF×S²` |
| 2 | [B, NF, S/2, S/2] | `C₂ = (2k²×NF²) × (S/2×S/2) = 2k²×NF²×S²/4 = 0.5k²×NF²×S²` |
| 3 | [B, 2NF, S/4, S/4] | `C₃ = (8k²×NF²) × (S/4×S/4) = 8k²×NF²×S²/16 = 0.5k²×NF²×S²` |
| 4 | [B, 4NF, S/8, S/8] | `C₄ = (8kt²×NF²) × (S/8×S/8) = 8kt²×NF²×S²/64 = 0.125kt²×NF²×S²` |
| 5 | [B, 2NF, S/4, S/4] | `C₅ = (2kt²×NF²) × (S/4×S/4) = 2kt²×NF²×S²/16 = 0.125kt²×NF²×S²` |
| 6 | [B, NF, S/2, S/2] | `C₆ = (kt²×NF×NC) × (S/2×S/2) = kt²×NF×NC×S²/4 = 0.25kt²×NF×NC×S²` |
| 7 | [B, NC, S, S] | `C₇ = (NC²) × (S×S) = NC²×S²` |

#### Total Number of Connections

$$
C_{\text{total}} = C_1 + C_2 + C_3 + C_4 + C_5 + C_6 + C_7
$$

$$
C_{\text{total}} = k^2 \times NIC \times NF \times S^2 + 0.5k^2 \times NF^2 \times S^2 + 0.5k^2 \times NF^2 \times S^2 + 0.125k_t^2 \times NF^2 \times S^2 + 0.125k_t^2 \times NF^2 \times S^2 + 0.25k_t^2 \times NF \times NC \times S^2 + NC^2 \times S^2
$$

$$
C_{\text{total}} = k^2 \times NIC \times NF \times S^2 + k^2 \times NF^2 \times S^2 + 0.25k_t^2 \times NF^2 \times S^2 + 0.25k_t^2 \times NF \times NC \times S^2 + NC^2 \times S^2
$$



### Summary of General Formulas

| Quantity | General Formula (k, kt terms) |
|----------|-------------------------------|
| **Weights** | `W = k²×NIC×NF + 10k²×NF² + 10kt²×NF² + kt²×NF×NC + NC²` |
| **Outputs** | `O = 0.8125×NF×S² + 2NC×S²` |
| **Connections** | `C = k²×NIC×NF×S² + k²×NF²×S² + 0.25kt²×NF²×S² + 0.25kt²×NF×NC×S² + NC²×S²` |

### For Specific Parameters (k=3, kt=2)

| Quantity | Expression |
|----------|------------|
| **Weights** | `W = 9×NIC×NF + 90×NF² + 40×NF² + 4×NF×NC + NC² = 9×NIC×NF + 130×NF² + 4×NF×NC + NC²` |
| **Outputs** | `O = 0.8125×NF×S² + 2NC×S²` |
| **Connections** | `C = 9×NIC×NF×S² + 9×NF²×S² + 1×NF²×S² + 1×NF×NC×S² + NC²×S² = 9×NIC×NF×S² + 10×NF²×S² + NF×NC×S² + NC²×S²` |

### Results for Different Input Sizes

#### For 32×32 Input (S=32, NIC=1, NF=16, NC=24)

| Quantity | Calculation | Value |
|----------|-------------|-------|
| **Weights** | `W = 9×1×16 + 130×256 + 4×16×24 + 576`<br>`= 144 + 33280 + 1536 + 576` | `35,536` |
| **Outputs** | `O = 832×16 + 2048×24`<br>`= 13312 + 49152` | `62,464` |
| **Connections** | `C = 9216×1×16 + 10240×256 + 1024×16×24 + 1024×576`<br>`= 147456 + 2621440 + 393216 + 589824` | `3,751,936` |

#### For 64×64 Input (S=64, NIC=1, NF=16, NC=24)

| Quantity | Calculation | Value |
|----------|-------------|-------|
| **Weights** | `W = 9×1×16 + 130×256 + 4×16×24 + 576`<br>`= 144 + 33280 + 1536 + 576` | `35,536` |
| **Outputs** | `O = 3328×16 + 8192×24`<br>`= 53248 + 196608` | `249,856` |
| **Connections** | `C = 36864×1×16 + 40960×256 + 4096×16×24 + 4096×576`<br>`= 589824 + 10485760 + 1572864 + 2359296` | `15,007,744` |

### Final Summary

| Input Size | Weights | Outputs | Connections |
|------------|---------|---------|-------------|
| 32×32 | 35,536 | 62,464 | 3,751,936 |
| 64×64 | 35,536 | 249,856 | 15,007,744 |



---

*.

---

###  **Sweep Summary**

| **Sweep #** |  **Run ID**  | **Optimizer** |   **LR**  | **Batch Size** | **Kernel Size** | **NC** | **NF** | **NIC** | **Epochs** | **Best Epoch** | **Best Val Loss** | **Final Val Loss** | **Test Acc** | **Train Acc** |
| :---------: | :----------: | :-----------: | :-------: | :------------: | :-------------: | :----: | :----: | :-----: | :--------: | :------------: | :---------------: | :----------------: | :----------: | :-----------: |
|      1      |   hwvcaav0   |      SGD      |   0.003   |       32       |        5        |   24   |   16   |    1    |     25     |       23       |      1.71934      |       1.73714      |    0.38453   |    0.37633    |
|      2      |   oq6vjpr3   |      Adam     |   0.003   |       64       |        3        |   24   |   16   |    1    |     25     |       24       |      1.57339      |       1.61668      |    0.42651   |    0.40994    |
|      3      |   yglj5yqg   |      SGD      |   0.0001  |       64       |        3        |   24   |    8   |    1    |     25     |       21       |      1.70793      |       1.73693      |    0.38797   |    0.37694    |
|      4      |   ok96afry   |      SGD      |   0.003   |       64       |        5        |   24   |   32   |    1    |     25     |       25       |      2.63935      |       2.63935      |    0.16391   |    0.16022    |
|      5      |   hh582kwt   |      Adam     |   0.003   |       32       |        5        |   24   |   32   |    1    |     25     |       24       |      1.49865      |       1.57416      |    0.45104   |    0.43318    |
|      6      |   oefi3syn   |      Adam     |   0.003   |       32       |        5        |   24   |   32   |    1    |     25     |       23       |      1.38955      |       1.41194      |    0.47735   |    0.48434    |
|      7      |   mfylu36a   |      Adam     |   0.003   |       32       |        5        |   24   |   32   |    1    |     25     |       23       |      1.37660      |       1.39234      |    0.48022   |    0.48502    |
|   🏆 **8**  | **4s07yn98** |    **SGD**    | **0.003** |     **32**     |      **5**      | **24** | **32** |  **1**  |   **25**   |     **18**     |    **1.36773**    |     **1.41728**    |  **0.48625** |  **0.48451**  |
|      9      |   jttkcnii   |      Adam     |   0.003   |       32       |        3        |   24   |   32   |    1    |     25     |       25       |      1.45700      |       1.45700      |    0.46263   |    0.44066    |
|      10     |   sg0f59fk   |      Adam     |   0.001   |       32       |        5        |   24   |   32   |    1    |     25     |       22       |      1.57370      |       1.58575      |    0.42608   |    0.41995    |
|      11     |   2nmyn6x0   |      Adam     |   0.003   |       32       |        5        |   24   |   32   |    1    |     25     |       23       |      1.37792      |       1.41705      |    0.48386   |    0.47964    |
|      12     |   hdtkr486   |      SGD      |   0.003   |       32       |        3        |   24   |   32   |    1    |     25     |       22       |      1.39378      |       1.44628      |    0.47621   |    0.48371    |
|      13     |   6gt8f8dv   |      Adam     |   0.001   |       32       |        5        |   24   |   32   |    1    |     25     |       23       |      1.66964      |       1.67341      |    0.40122   |    0.38740    |
|      14     |   rb40cfvz   |      Adam     |   0.003   |       32       |        5        |   24   |   32   |    1    |     25     |       22       |      1.39322      |       1.42075      |    0.47667   |    0.47811    |
|      15     |   2452yw5d   |      SGD      |   0.003   |       32       |        5        |   24   |   32   |    1    |     25     |       22       |      1.39306      |       1.41053      |    0.47927   |    0.48281    |
|      16     |   o34seic7   |      Adam     |   0.003   |       32       |        5        |   24   |   32   |    1    |     25     |       22       |      1.48374      |       1.56294      |    0.45353   |    0.44216    |
|      17     |   zqx8o135   |      Adam     |   0.003   |       32       |        5        |   24   |   32   |    1    |     25     |       23       |      1.38200      |       1.39723      |    0.47955   |    0.48263    |
|      18     |   4q7phnvj   |      Adam     |   0.003   |       32       |        5        |   24   |   32   |    1    |     25     |       20       |      1.37633      |       1.38368      |    0.48259   |    0.48523    |
|      19     |   vtx5th2v   |      Adam     |   0.003   |       32       |        5        |   24   |   32   |    1    |     25     |       20       |      1.37539      |       1.39900      |    0.48256   |    0.48500    |
|      20     |   qiize13l   |      Adam     |   0.003   |       32       |        5        |   24   |   32   |    1    |     25     |       25       |      1.37551      |       1.37551      |    0.48256   |    0.48343    |

---


*  **Best validation loss:** `1.36773` → **Sweep 8 (4s07yn98)**
*  **Optimizer:** SGD  **LR:** 0.003  **Batch size:** 32  **Kernel size:** 5
*  **Best epoch:** 18  **Train acc:** 0.48451  **Test acc:** 0.48625
* Link to wandb sweep - https://wandb.ai/tempakash48-iiit-hyderabad/cifar10-colorization/runs/4q7phnvj?nw=nwusertempakash48
---

