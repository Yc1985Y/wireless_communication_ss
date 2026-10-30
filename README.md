# 无线通信系统仿真（Sionna + PyTorch）

课程《专业综合实践 · 无线系统仿真》的实验代码与结果。全部实验用 Python + [Sionna 2.2](https://github.com/NVlabs/sionna)（PyTorch 后端）实现，
**每个实验一个 Jupyter Notebook**，notebook 内保存了完整输出（图 + 数值），结果数据另存为 `data/*.csv`、图另存为 `figures/*.png`。

- 必做实验：基带调制、AWGN/瑞利信道 BER、QPSK-OFDM（含 CP 扫描与导频 LS 两个拓展）
- 补充实验：OFDM 的配对 CP 比较与导频 LS 的等效 SNR 理论
- 选做实验：空时编码、直扩/RAKE、接收分集、MIMO 检测、脉冲成形、BICM-LDPC、神经接收机、自编码器、实测水声信道

> 说明：课程讲义、教材 PDF、实测信道数据（`NOF1_00*.mat`）与实验报告均**未**包含在本仓库中。

---

## 目录结构

```
.
├── notebooks/                 14 个实验 notebook（已执行、含输出）
│   ├── 01_baseband_modulation.ipynb    实验一：BPSK/QPSK/8-PSK/16-QAM 基带调制
│   ├── 02_channel_ber.ipynb            实验二：AWGN 与瑞利衰落信道下的 BER
│   ├── 03A_ofdm.ipynb                  实验三：QPSK-OFDM（基本 + CP 扫描 + 导频 LS）
│   ├── 04_ofdm_supplement.ipynb        补充：CP 配对比较、LS 等效 SNR 理论
│   └── 03B…03J_*.ipynb                 9 个选做实验
├── figures/                   58 张结果图（PNG，200 dpi）
├── data/                      56 个结果数据表（CSV，报告中的每个数字都来自这里）
├── scripts/                   实验生成脚本与独立验证脚本
│   ├── make_notebooks.py              由课程模板生成 00/01/02/03A
│   ├── build_03B…03J.py               由课程模板生成 9 个选做实验
│   ├── supplement_ofdm.py             生成补充实验 04
│   ├── verify_algorithms.py           必做部分 18 项独立判据（自写实现，不复用 notebook 函数）
│   └── verify_optional.py             选做部分 19 项独立判据
└── docs/experiment_notes/     每个选做实验的方法与结果说明（Markdown）
```

## 环境

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| PyTorch | 2.11（CPU 即可，无需 GPU） |
| Sionna | 2.2.0 |
| NumPy / SciPy / Matplotlib / pandas | 2.2 / 1.17 / 3.10 / 3.0 |

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 运行

单个实验（推荐，一次一本）：

```bash
cd notebooks
jupyter notebook                 # 或 VS Code 打开 ipynb 并选择本仓库的解释器
```

命令行就地重跑（会刷新 `figures/` 与 `data/`）：

```bash
cd notebooks
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=2400 03A_ofdm.ipynb
```

`03E_watermark_mcbfsk.ipynb` 需要课程提供的实测水声信道数据 `NOF1_001.mat … NOF1_004.mat`：
把 `无线仿真资料2026/` 目录放在仓库同级（或用环境变量 `NOF1_DATA_DIR` 指定路径）即可运行。

## 可复现性

- 所有随机源固定：`torch.manual_seed` / `np.random.seed` / `sionna.phy.config.seed`；扫描的每个信噪比点重新设种子。
- 因此同一 notebook 重跑后，`data/*.csv` 逐字节不变（已用 sha256 校验 56 个 CSV）。
- 仿真点数固定（如 01 每个调制 2.1×10⁶ bit、03A 每点 1000 个信道块），报告中的"仿真 vs 理论"差异均可用二项统计涨落解释（`1/√N_e`）。

## 主要结果

**实验一 · 基带调制（Eb/N0 = 8 dB）**

| 调制 | bit/符号 | 仿真 BER | 理论 BER |
| --- | --- | --- | --- |
| BPSK | 1 | 2.21×10⁻⁴ | 1.91×10⁻⁴ |
| QPSK | 2 | 2.29×10⁻⁴ | 1.91×10⁻⁴ |
| 8-PSK | 3 | 6.13×10⁻³ | 6.18×10⁻³ |
| 16-QAM | 4 | 1.38×10⁻² | 1.38×10⁻² |

BER = 10⁻³ 所需 Eb/N0：BPSK/QPSK 6.7 dB、8-PSK 10.0 dB、16-QAM 10.8 dB。
注：16-QAM 采用教材给定的非 Gray 星座表，理论值由判决域积分求得（Gray 闭式会低估约 1.5 倍）。

**实验二 · AWGN vs 瑞利（QPSK，8 dB）**：AWGN 2.06×10⁻⁴，瑞利 3.55×10⁻²（约 190 倍差距）；
瑞利信道只按 1/SNR 缓慢改善，12 dB 仍有 1.52×10⁻²。

**实验三 · QPSK-OFDM（N=64，CP=16，三径 [0,1,4]）**

| Eb/N0 | 0 dB | 12 dB | 24 dB |
| --- | --- | --- | --- |
| 多径瑞利（仿真） | 1.45×10⁻¹ | 1.43×10⁻² | 1.13×10⁻³ |
| 单抽头瑞利（理论） | 1.46×10⁻¹ | 1.51×10⁻² | 9.92×10⁻⁴ |

- CP 扫描（18 dB）：CP=0 → 1.81×10⁻²，CP=4 → 4.02×10⁻³，CP ≥ τmax 后饱和；CP=0 约为 CP=16 的 4.8 倍。
- 导频 LS：相比理想 CSI 损失 2.0~3.1 dB，高 SNR 收敛到 3.01 dB（等效噪声方差加倍）。

**选做实验关键定量结论**

| 实验 | 结论（摘要） |
| --- | --- |
| 03B 空时编码 | @BER=10⁻² 相对 SISO 增益：2×1 Alamouti 5.38 dB、3×1 G3 6.91 dB、2×2 10.63 dB |
| 03C 直扩/RAKE | 三径 RAKE 达 10⁻² 需单指 18.61 dB → EGC 7.64 dB → MRC 6.87 dB；处理增益 10lg31 = 14.9 dB |
| 03D 接收分集 | 16-QAM SER=10⁻²：1×1 28.11 dB → 1×4 MRC 12.03 dB（省 16.07 dB） |
| 03H MIMO 检测 | 2×2 QPSK @10⁻²：ZF 13.93 / LMMSE 12.07 / ML 6.94 dB；1×1 SISO 参考 13.85 dB |
| 03G 脉冲成形 | β 0.15→0.8：最大旁瓣 −32.9→−68.1 dB，残余 ISI 3.6×10⁻²→6.8×10⁻⁴ |
| 03I BICM-LDPC | QPSK R=1/2 @1.5 dB 编码后 2.51×10⁻⁴（未编码 4.65×10⁻²，编码增益 186 倍） |
| 03F 神经接收机 | 10 dB 处等效 SNR 惩罚 0.147 dB（95% CI [−0.05, +0.32] dB） |
| 03J 自编码器 | 相对 QPSK 惩罚 0.389 dB；误差分解：非 Gray 标号 +41.6%、几何 +7.6%、网络自身 +35.5% |
| 03E 实测水声信道 | 四条实测信道无噪回放 0/1940 误码；选择性代价 6.8~9.9 dB |

## 算法正确性验证

`scripts/verify_algorithms.py` 与 `scripts/verify_optional.py` 用**独立实现**（闭式解、穷举最近邻、显式 DFT 矩阵、
`np.convolve`、解析判决域积分等，不调用被验证 notebook 的任何函数）逐项复核：

```bash
python scripts/verify_algorithms.py    # 必做部分 18 项判据
python scripts/verify_optional.py      # 选做部分 19 项判据
```

覆盖：星座能量与最小距离、比特→符号往返一致性、噪声方差口径、16-QAM 理论值、FFT/IFFT 互逆、CP 结构、
时域多径与频域乘性等效、ZF 残差、LS 无偏性与方差比、Alamouti 正交性、MRC 合并增益、m 序列周期与旁瓣、
RRC 零 ISI、LDPC 校验矩阵、MC-BFSK 帧结构等。当前结论：**18/18 与 19/19 全部通过**。

## 数据与图的来源

`figures/*.png` 与 `data/*.csv` 均由对应 notebook 直接生成（图用 Matplotlib `savefig(dpi=200)`，数据用 pandas `to_csv`），
不经过手工编辑；`data/*.csv` 是实验报告里所有数值的唯一来源，便于逐项核对。
