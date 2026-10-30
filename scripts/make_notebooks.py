# -*- coding: utf-8 -*-
'''根据学生版模板生成补全后的实验 Notebook（01 / 02 / 03A）。

做法：读取 无线仿真资料2026/student 下的原始 Notebook，保留其 Markdown 讲义，
仅替换其中的代码单元格为完整实现，并在末尾追加补充实验/过程可视化单元格。
'''
from __future__ import annotations

import sys

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # 仓库根目录
SRC = ROOT / "无线仿真资料2026" / "student"         # 课程提供的模板目录
OUT = ROOT / "notebooks"
OUT.mkdir(parents=True, exist_ok=True)


def load(name: str) -> dict:
    return json.loads((SRC / name).read_text(encoding="utf-8"))


def code_cell(src: str) -> dict:
    src = src.strip("\n") + "\n"
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def md_cell(src: str) -> dict:
    src = src.strip("\n") + "\n"
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


def save(nb: dict, name: str) -> None:
    # 统一 nbformat 版本
    nb["nbformat"] = 4
    nb["nbformat_minor"] = 5
    (OUT / name).write_text(
        json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print("written", OUT / name)


# ----------------------------------------------------------------------------
# 公共：01/02 共用的链路基础代码
# ----------------------------------------------------------------------------
COMMON_HEAD = r'''
# 导入依赖并准备调制、解调和 BER 统计的通用函数。
from pathlib import Path
import sys

# 从当前目录向上查找 src/sionna（若使用仓库源码则优先使用）。
for p in [Path.cwd(), *Path.cwd().parents]:
    if (p / "src" / "sionna").exists():
        sys.path.insert(0, str(p / "src"))
        break

import numpy as np
import torch
import matplotlib.pyplot as plt

# 设置 Matplotlib 中文字体，避免图标题和坐标轴中文乱码。
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

# Word 报告中的图宽约 14.4 cm，而画布按英寸设置，插入时会被缩放，
# 因此把 matplotlib 的字号整体放大，使打印后的字号与正文（10.5 pt）接近。
plt.rcParams.update(
    {
        "font.size": 13,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.dpi": 200,
        "savefig.dpi": 200,
    }
)

# 信道部分使用 Sionna 模块，星座映射与解映射由本实验自主实现（类似 16QAM 示例）。
from sionna.phy.channel import AWGN, RayleighBlockFading

# 固定使用 CPU，保证没有 GPU 的学生电脑也能运行。
DEVICE = "cpu"
PRECISION = "single"

# 固定随机种子，便于复现实验结果。
# 注意：Sionna 的 AWGN / 衰落信道使用自己的 torch.Generator（config.torch_rng），
# 因此除了 torch / numpy 之外还必须设置 sionna.phy.config.seed，否则噪声每次运行都不同。
from sionna.phy import config as sionna_config


def set_seed(seed=7):
    """固定 Python / NumPy / PyTorch / Sionna 四个随机源，使仿真结果可复现。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    sionna_config.seed = int(seed)


set_seed(7)

# 图、数据输出目录：便于把过程性与结果性材料归档到同一文件夹。
FIG_DIR = Path.cwd() / "figures"
DATA_DIR = Path.cwd() / "data"
FIG_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


# 将 Eb/N0 转成 AWGN 模块需要的噪声功率谱密度 no。
def ebno_db_to_no(ebno_db, nbits, coderate=1.0):
    ebno = 10 ** (torch.as_tensor(ebno_db, dtype=torch.float32, device=DEVICE) / 10.0)
    return 1.0 / (ebno * nbits * coderate)


# 统计误比特数，BER 会在此基础上除以总 bit 数。
def count_bit_errors(b, b_hat):
    # 统一转成 int32 后逐元素比较，统计不同 bit 的个数。
    return int(torch.sum(b.to(torch.int32) != b_hat.to(torch.int32)).item())


def ber(b, b_hat):
    return count_bit_errors(b, b_hat) / b.numel()


# 生成各调制方式的星座表，BPSK/QPSK/16-QAM 归一化至平均符号能量 Es = 1。
def get_constellation(name):
    if name == "BPSK":
        # 1 bit: 0 -> -1, 1 -> +1
        pts = torch.tensor([-1.0 + 0j, 1.0 + 0j], dtype=torch.complex64, device=DEVICE)
    elif name == "QPSK":
        # 2 bit: 00, 01, 10, 11
        pts = torch.tensor(
            [1.0 + 1.0j, -1.0 + 1.0j, 1.0 - 1.0j, -1.0 - 1.0j],
            dtype=torch.complex64,
            device=DEVICE,
        ) / np.sqrt(2.0)
    elif name == "8-PSK":
        # 3 bit Gray 码映射: 相邻星座点仅相差 1 bit
        gray_order = [0, 1, 3, 2, 6, 7, 5, 4]
        angles = torch.zeros(8, dtype=torch.float32, device=DEVICE)
        for k, bit_val in enumerate(gray_order):
            angles[bit_val] = k * (2 * np.pi / 8)
        pts = torch.exp(1j * angles).to(torch.complex64)
    elif name == "16-QAM":
        # 4 bit 16-QAM 星座点（参考 16QAM/QAM_awgn.m）
        c16 = [
            1 + 1j,
            1 + 3j,
            3 + 1j,
            3 + 3j,
            -1 + 1j,
            -1 + 3j,
            -3 + 1j,
            -3 + 3j,
            -3 - 3j,
            -3 - 1j,
            -1 - 3j,
            -1 - 1j,
            1 - 1j,
            1 - 3j,
            3 - 1j,
            3 - 3j,
        ]
        pts = torch.tensor(c16, dtype=torch.complex64, device=DEVICE)
        es = torch.mean(torch.abs(pts) ** 2)
        pts = pts / torch.sqrt(es)
    else:
        raise ValueError(f"未知调制方式: {name}")
    return pts


# 类似 16QAM/modulation.m：将比特流按 nbits 查表映射到星座点
def modulate(bits, constell, nbits):
    # 把每个符号的 nbits 个比特视为一个 nbits 位二进制数（高位在前）作为星座索引。
    bits = bits.to(torch.int64)
    idx = torch.zeros(bits.shape[:-1], dtype=torch.long, device=DEVICE)
    for i in range(nbits):
        idx = (idx << 1) | bits[..., i]
    return constell[idx]


# 类似 16QAM/demodulation.m：基于最小欧氏距离判决解映射还原比特
def demodulate(rx_symbols, constell, nbits):
    # (..., 1) 与 (M,) 广播，得到接收符号到每个星座点的距离平方 (..., M)。
    dist = torch.abs(rx_symbols.unsqueeze(-1) - constell) ** 2
    idx = torch.argmin(dist, dim=-1)
    bits = torch.zeros(*idx.shape, nbits, dtype=torch.int32, device=DEVICE)
    for i in range(nbits):
        shift = nbits - 1 - i
        bits[..., i] = ((idx >> shift) & 1).to(torch.int32)
    return bits
'''

# ----------------------------------------------------------------------------
# 01 基带调制链路
# ----------------------------------------------------------------------------
NB01_CELL2 = r'''
# 对四种调制方式：生成比特 -> 映射 -> AWGN 加噪 -> 判决解映射 -> 统计 BER，并绘制接收星座图。
MODES = {"BPSK": 1, "QPSK": 2, "8-PSK": 3, "16-QAM": 4}
EbN0_DB = 8.0
BATCH_SIZE = 512
NUM_SYMBOLS = 256

results = []

fig, axes = plt.subplots(2, 2, figsize=(9.4, 8.0))
axes = axes.ravel()

for idx, (ax, (name, nbits)) in enumerate(zip(axes, MODES.items())):
    constell = get_constellation(name)
    b = torch.randint(
        0, 2, (BATCH_SIZE, NUM_SYMBOLS, nbits), dtype=torch.int32, device=DEVICE
    )

    # 1) 发送端：比特查表映射为星座符号
    x = modulate(b, constell, nbits)
    # 2) 信道：按 Eb/N0 计算噪声功率谱密度并叠加 AWGN
    no = ebno_db_to_no(EbN0_DB, nbits)
    y = AWGN(precision=PRECISION, device=DEVICE)(x, no)
    # 3) 接收端：最小欧氏距离判决解映射
    b_hat = demodulate(y, constell, nbits)
    # 4) 统计误比特率
    this_ber = ber(b, b_hat)

    results.append(
        {"modulation": name, "nbits": nbits, "ebno_db": EbN0_DB, "ber": this_ber}
    )

    # 接收散点图：发散程度反映噪声大小，理想星座点用红色叉号标注
    ax.scatter(
        y.real.cpu().flatten(),
        y.imag.cpu().flatten(),
        s=5,
        alpha=0.25,
        c="#1f77b4",
        label="接收符号",
    )
    ax.scatter(
        constell.real.cpu(),
        constell.imag.cpu(),
        s=60,
        marker="x",
        c="red",
        linewidths=1.5,
        label="理想星座点",
    )
    ax.set_title(f"({'abcd'[idx]}) {name}（{nbits} bit/符号，BER={this_ber:.2e}）", fontsize=14)
    ax.set_xlabel("同相分量 I", fontsize=13)
    ax.set_ylabel("正交分量 Q", fontsize=13)
    ax.tick_params(labelsize=12)
    ax.grid(alpha=0.3)
    ax.axis("equal")
    ax.legend(fontsize=11, loc="best")

plt.tight_layout()
plt.savefig(FIG_DIR / "01_received_constellation.png", dpi=200, bbox_inches="tight")
plt.show()

import pandas as pd

df_results = pd.DataFrame(results)
print("表 1  四种调制方式在 Eb/N0 = 8 dB 下的误比特率")
print(df_results.to_string(index=False))
df_results.to_csv(DATA_DIR / "01_ber_at_8dB.csv", index=False, encoding="utf-8-sig")

assert len(results) == 4
'''

NB01_CELL3 = r'''
# 补充扩展：扫描 0~12 dB 的 Eb/N0，得到四种调制的 BER 曲线与理论对照（供报告分析使用）。
from scipy.special import erfc

EbN0_SWEEP = np.arange(0, 13, 2)
# 每个 Eb/N0 点统计 1024*512 个符号，兼顾统计精度与运行时间
BATCH_SIZE_SWEEP = 1024
NUM_SYMBOLS_SWEEP = 512

ber_sweep = {name: [] for name in MODES}
for name, nbits in MODES.items():
    constell = get_constellation(name)
    for ebno_db in EbN0_SWEEP:
        # 每个扫描点固定种子（含 Sionna 的生成器），保证结果可复现
        set_seed(7)
        b = torch.randint(
            0,
            2,
            (BATCH_SIZE_SWEEP, NUM_SYMBOLS_SWEEP, nbits),
            dtype=torch.int32,
            device=DEVICE,
        )
        x = modulate(b, constell, nbits)
        no = ebno_db_to_no(float(ebno_db), nbits)
        y = AWGN(precision=PRECISION, device=DEVICE)(x, no)
        b_hat = demodulate(y, constell, nbits)
        ber_sweep[name].append(ber(b, b_hat))


# ---------------------------------------------------------------------------
# 理论 BER 曲线：用于验证仿真链路的正确性
# ---------------------------------------------------------------------------
def qfunc(z):
    """高斯 Q 函数。"""
    return 0.5 * erfc(np.asarray(z, dtype=np.float64) / np.sqrt(2.0))


def theory_psk_mpsk(name, ebno_db):
    """BPSK / QPSK / 8-PSK 的 Gray 编码理论误比特率（Eb/N0 为单比特信噪比）。"""
    gb = 10 ** (np.asarray(ebno_db, dtype=np.float64) / 10.0)
    if name in ("BPSK", "QPSK"):
        # BPSK 与 QPSK 的比特错误率相同：Pb = 0.5*erfc(sqrt(gamma_b))
        return 0.5 * erfc(np.sqrt(gb))
    if name == "8-PSK":
        # 符号错误率 Ps ≈ 2*Q(sqrt(2*gamma_s)*sin(pi/8))，Gray 编码下 Pb ≈ Ps/3
        gs = 3.0 * gb
        ps = 2.0 * qfunc(np.sqrt(2.0 * gs) * np.sin(np.pi / 8.0))
        return ps / 3.0
    raise ValueError(name)


def theory_16qam_table(ebno_db):
    """按教材 16-QAM 星座表的实际比特标注，数值计算 AWGN 下的精确误比特率。

    16-QAM 的判决域是矩形，噪声在两个维度独立高斯，因此
    P(发送 i 被判为 j) 可用两个一维 Q 函数之差解析求出；
    再按 (i -> j) 的比特翻转个数加权求和，即为该星座标注下的精确 BER。
    """
    pts = get_constellation("16-QAM").cpu().numpy()
    levels = np.array([-3.0, -1.0, 1.0, 3.0]) / np.sqrt(10.0)
    # 每个电平对应的判决区间边界（相邻电平中点，两侧外延到无穷）
    bounds = [-np.inf, -2.0 / np.sqrt(10.0), 0.0, 2.0 / np.sqrt(10.0), np.inf]
    row = torch.arange(16, dtype=torch.int64)
    hamming = torch.zeros((16, 16), dtype=torch.float64)
    for i in range(16):
        for j in range(16):
            hamming[i, j] = (bin(i ^ j).count("1"))

    out = []
    for db in np.atleast_1d(ebno_db):
        gb = 10 ** (float(db) / 10.0)
        no = 1.0 / (gb * 4.0)          # 复噪声方差（Es = 1 归一化，Es/N0 = 4*Eb/N0）
        sigma = np.sqrt(no / 2.0)      # 每个实数维度的噪声标准差
        pe = np.zeros((16, 16), dtype=np.float64)
        for i in range(16):
            xi, yi = pts[i].real, pts[i].imag
            for j in range(16):
                xj, yj = pts[j].real, pts[j].imag
                kx = int(np.argmin(np.abs(levels - xj)))
                ky = int(np.argmin(np.abs(levels - yj)))
                px = qfunc((bounds[kx] - xi) / sigma) - qfunc((bounds[kx + 1] - xi) / sigma)
                py = qfunc((bounds[ky] - yi) / sigma) - qfunc((bounds[ky + 1] - yi) / sigma)
                pe[i, j] = max(px * py, 0.0)
        ber_th = float((pe * hamming.numpy()).sum() / 16.0 / 4.0)
        out.append(ber_th)
    return np.array(out)


theory_sweep = {}
for name in MODES:
    if name == "16-QAM":
        theory_sweep[name] = theory_16qam_table(EbN0_SWEEP)
    else:
        theory_sweep[name] = theory_psk_mpsk(name, EbN0_SWEEP)

fig, ax = plt.subplots(figsize=(7.2, 4.8))
markers = {"BPSK": "o", "QPSK": "s", "8-PSK": "^", "16-QAM": "d"}
for name, values in ber_sweep.items():
    # 样本有限时高 SNR 下可能统计不到错误（BER=0），log 轴上令曲线断开而不出现垂直断崖
    shown = [v if v > 0 else np.nan for v in values]
    line = ax.semilogy(EbN0_SWEEP, shown, marker=markers[name], ms=5, lw=1.4, label=f"{name} 仿真")
    ax.semilogy(
        EbN0_SWEEP,
        theory_sweep[name],
        ls="--",
        lw=1.2,
        color=line[0].get_color(),
        label=f"{name} 理论",
    )
ax.set_xlabel("Eb/N0 (dB)")
ax.set_ylabel("误比特率 BER")
ax.set_ylim(1e-6, 0.5)
ax.grid(True, which="both", alpha=0.3)
ax.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "01_ber_vs_ebno.png", dpi=200, bbox_inches="tight")
print("图4  四种调制方式的 BER-Eb/N0 曲线：仿真与理论对比（AWGN 信道）")
plt.show()

df_sweep = pd.DataFrame({"EbN0_dB": EbN0_SWEEP, **ber_sweep})
df_theory = pd.DataFrame({"EbN0_dB": EbN0_SWEEP, **{f"{k}_theory": v for k, v in theory_sweep.items()}})
print("表 2  四种调制方式在不同 Eb/N0 下的仿真 BER")
print(df_sweep.to_string(index=False))
print()
print("表 3  四种调制方式在不同 Eb/N0 下的理论 BER")
print(df_theory.to_string(index=False))
df_sweep.to_csv(DATA_DIR / "01_ber_sweep.csv", index=False, encoding="utf-8-sig")
df_theory.to_csv(DATA_DIR / "01_ber_theory.csv", index=False, encoding="utf-8-sig")
'''

# ----------------------------------------------------------------------------
# 02 信道与 BER
# ----------------------------------------------------------------------------
NB02_CELL2 = r'''
# 封装 AWGN 与 Rayleigh 链路的 BER 扫描。
# AWGN 链路：发送信号直接叠加高斯白噪声。
def transmit_awgn(bits, constell, nbits, ebno_db):
    x = modulate(bits, constell, nbits)
    no = ebno_db_to_no(ebno_db, nbits)

    y = AWGN(precision=PRECISION, device=DEVICE)(x, no)

    return demodulate(y, constell, nbits)


# Rayleigh 链路：衰落信道加单抽头零陷均衡。
def transmit_rayleigh(bits, constell, nbits, ebno_db):
    x = modulate(bits, constell, nbits)
    no = ebno_db_to_no(ebno_db, nbits)

    # RayleighBlockFading 生成服从 CN(0,1) 的单径平坦衰落系数（块衰落）。
    ch = RayleighBlockFading(1, 1, 1, 1, precision=PRECISION, device=DEVICE)
    h, _ = ch(batch_size=x.numel(), num_time_steps=1)
    h = h.reshape(x.shape)

    # 接收端已知信道（理想 CSI），做单抽头迫零均衡 y / h。
    y = AWGN(precision=PRECISION, device=DEVICE)(h * x, no)
    y_eq = y / h

    return demodulate(y_eq, constell, nbits)


# 对 Eb/N0 列表扫描，返回一组 BER 曲线数值。
def simulate_ber(
    name, nbits, channel_type, ebno_db_list, batch_size=256, num_symbols=256
):
    constell = get_constellation(name)
    ber_list = []
    for ebno_db in ebno_db_list:
        # 每个 Eb/N0 点重新生成独立比特流，避免同一组比特重复统计。
        b = torch.randint(
            0, 2, (batch_size, num_symbols, nbits), dtype=torch.int32, device=DEVICE
        )

        if channel_type == "awgn":
            b_hat = transmit_awgn(b, constell, nbits, float(ebno_db))
        elif channel_type == "rayleigh":
            b_hat = transmit_rayleigh(b, constell, nbits, float(ebno_db))
        else:
            raise ValueError(f"未知信道类型: {channel_type}")

        ber_list.append(ber(b, b_hat))
    return np.array(ber_list)
'''

NB02_CELL3 = r'''
# 扫描 Eb/N0，并把不同调制和信道的 BER 曲线画在同一张图中。
ebno_db = np.arange(0, 13, 2)

curves = {}

for name, nbits in {"QPSK": 2, "8-PSK": 3, "16-QAM": 4}.items():
    for ch in ["awgn", "rayleigh"]:
        curves[(name, ch)] = simulate_ber(name, nbits, ch, ebno_db)

# 理论基准：AWGN 信道下 QPSK/BPSK 的差错概率与单抽头瑞利信道下的理论 BER。
import scipy.special as sp

ebno_lin = 10.0 ** (ebno_db / 10.0)
theory_qpsk_awgn = 0.5 * sp.erfc(np.sqrt(ebno_lin))
theory_qpsk_rayleigh = 0.5 * (1.0 - np.sqrt(ebno_lin / (1.0 + ebno_lin)))

# 使用 semilogy 画出 6 条仿真 BER 曲线
fig, ax = plt.subplots(figsize=(7.6, 5.0))
styles = {
    ("QPSK", "awgn"): ("o", "-"),
    ("QPSK", "rayleigh"): ("o", "--"),
    ("8-PSK", "awgn"): ("s", "-"),
    ("8-PSK", "rayleigh"): ("s", "--"),
    ("16-QAM", "awgn"): ("^", "-"),
    ("16-QAM", "rayleigh"): ("^", "--"),
}
for key, values in curves.items():
    marker, ls = styles[key]
    # 高 SNR 下未统计到误码时 BER=0，绘制时用 NaN 断开曲线
    shown = [v if v > 0 else np.nan for v in values]
    ax.semilogy(
        ebno_db,
        shown,
        marker=marker,
        ls=ls,
        lw=1.3,
        ms=5,
        label=f"{key[0]} - {key[1].upper()} 仿真",
    )

ax.semilogy(ebno_db, theory_qpsk_awgn, color="k", ls=":", lw=1.6, label="QPSK AWGN 理论")
ax.semilogy(
    ebno_db, theory_qpsk_rayleigh, color="gray", ls=":", lw=1.6, label="QPSK Rayleigh 理论"
)

ax.set_xlabel("Eb/N0 (dB)")
ax.set_ylabel("误比特率 BER")
ax.set_ylim(1e-7, 0.5)
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(FIG_DIR / "02_ber_all_channels.png", dpi=200, bbox_inches="tight")
print("图5  AWGN 与平坦瑞利衰落信道下三种调制的 BER 曲线")
plt.show()

# 数值表（结果性数据）
import pandas as pd

df_curves = pd.DataFrame({"EbN0_dB": ebno_db})
for (name, ch), values in curves.items():
    df_curves[f"{name}_{ch}"] = values
print("表 3  AWGN / Rayleigh 信道下各调制的 BER 仿真数据")
print(df_curves.to_string(index=False))
df_curves.to_csv(DATA_DIR / "02_ber_curves.csv", index=False, encoding="utf-8-sig")

assert len(curves) == 6
'''

NB02_CELL4 = r'''
# 补充扩展：分信道绘制，并与理论曲线逐点对比（便于在报告中讨论吻合程度）。
def _q(x):
    """高斯 Q 函数（与 scipy 的 erfc 等价写法）。"""
    return 0.5 * sp.erfc(np.asarray(x, dtype=np.float64) / np.sqrt(2.0))


def theory_16qam_exact(ebno_db):
    """按教材 16 点星座表的实际比特标注，对 AWGN 判决域做数值积分得到精确 BER。

    该表不是 Gray 排序（相邻点之间平均比特差大于 1），因此不能直接套用 Gray 编码的
    近似公式 3/8·erfc(√(2γb/5))，否则会低估约 1.4~1.6 倍；这里按判决域精确求和。
    """
    pts = get_constellation("16-QAM").cpu().numpy()
    levels = np.array([-3.0, -1.0, 1.0, 3.0]) / np.sqrt(10.0)
    bounds = [-np.inf, -2.0 / np.sqrt(10.0), 0.0, 2.0 / np.sqrt(10.0), np.inf]
    hamming = np.array(
        [[bin(i ^ j).count("1") for j in range(16)] for i in range(16)], dtype=np.float64
    )
    out = []
    for db in np.atleast_1d(ebno_db):
        gb = 10 ** (float(db) / 10.0)
        sigma = np.sqrt(1.0 / (gb * 4.0) / 2.0)   # Es=1 ⇒ Es/N0=4γb，每维噪声方差 no/2
        pe = np.zeros((16, 16))
        for i in range(16):
            xi, yi = pts[i].real, pts[i].imag
            for j in range(16):
                kx = int(np.argmin(np.abs(levels - pts[j].real)))
                ky = int(np.argmin(np.abs(levels - pts[j].imag)))
                px = _q((bounds[kx] - xi) / sigma) - _q((bounds[kx + 1] - xi) / sigma)
                py = _q((bounds[ky] - yi) / sigma) - _q((bounds[ky + 1] - yi) / sigma)
                pe[i, j] = max(px * py, 0.0)
        out.append(float((pe * hamming).sum() / 16.0 / 4.0))
    return np.array(out)


def theory_awgn(name, ebno_db):
    g = 10.0 ** (np.asarray(ebno_db) / 10.0)
    if name in ("BPSK", "QPSK"):
        return 0.5 * sp.erfc(np.sqrt(g))
    if name == "8-PSK":
        return (1.0 / 3.0) * sp.erfc(np.sqrt(3.0 * g) * np.sin(np.pi / 8.0))
    if name == "16-QAM":
        return theory_16qam_exact(ebno_db)
    raise ValueError(name)


fig, axes = plt.subplots(2, 1, figsize=(8.0, 9.0))
for ax, ch in zip(axes, ["awgn", "rayleigh"]):
    for name in ["QPSK", "8-PSK", "16-QAM"]:
        shown = [v if v > 0 else np.nan for v in curves[(name, ch)]]
        ax.semilogy(ebno_db, shown, marker="o", ms=4, lw=1.3, label=f"{name} 仿真")
        if ch == "awgn":
            ax.semilogy(
                ebno_db,
                theory_awgn(name, ebno_db),
                ls="--",
                lw=1.2,
                label=f"{name} 理论",
            )
    if ch == "rayleigh":
        ax.semilogy(
            ebno_db,
            theory_qpsk_rayleigh,
            ls="--",
            lw=1.4,
            color="k",
            label="QPSK 瑞利理论",
        )
    ax.set_xlabel("Eb/N0 (dB)", fontsize=14)
    ax.set_ylabel("误比特率 BER", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_title(
        "(a) AWGN 信道" if ch == "awgn" else "(b) 平坦瑞利衰落信道", fontsize=13
    )
    # AWGN 面板需要覆盖到 1e-7；瑞利面板曲线只落在 1e-2~1e-1 这一段，单独收紧纵轴
    if ch == "awgn":
        ax.set_ylim(1e-7, 0.5)
    else:
        ax.set_ylim(5e-3, 4e-1)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(FIG_DIR / "02_ber_theory_compare.png", dpi=200, bbox_inches="tight")
print("图6  AWGN 与瑞利衰落信道下仿真结果与理论曲线对比")
plt.show()

# 计算仿真值与理论值的相对偏差（AWGN 信道，QPSK）
rel_err = np.abs(curves[("QPSK", "awgn")] - theory_awgn("QPSK", ebno_db)) / theory_awgn(
    "QPSK", ebno_db
)
print("表 4  QPSK/AWGN 仿真 BER 相对理论值的偏差")
for e, sim, th, r in zip(ebno_db, curves[("QPSK", "awgn")], theory_awgn("QPSK", ebno_db), rel_err):
    print(f"  Eb/N0={e:2d} dB: 仿真={sim:.4e}  理论={th:.4e}  相对偏差={r*100:5.2f}%")
'''

# ----------------------------------------------------------------------------
# 03A OFDM
# ----------------------------------------------------------------------------
NB03A_CELL1 = r'''
# 模块一：环境准备与 OFDM 链路核心函数
from pathlib import Path
import sys

# 从当前目录向上查找 src/sionna，优先使用本仓库源码。
for p in [Path.cwd(), *Path.cwd().parents]:
    if (p / "src" / "sionna").exists():
        sys.path.insert(0, str(p / "src"))
        break

import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.special as sp

# OFDM 链路只用到 AWGN 模块（多径衰落由自定义抽头 + 卷积实现）。
from sionna.phy.channel import AWGN

# 设置 Matplotlib 中文字体，避免图标题和坐标轴中文乱码。
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

# Word 报告中的图宽约 14.4 cm，而画布按英寸设置，插入时会被缩放，
# 因此把 matplotlib 的字号整体放大，使打印后的字号与正文（10.5 pt）接近。
plt.rcParams.update(
    {
        "font.size": 13,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.dpi": 200,
        "savefig.dpi": 200,
    }
)

# 固定使用 CPU，保证没有 GPU 的学生电脑也能运行。
DEVICE = "cpu"
PRECISION = "single"

# 固定随机种子，便于复现实验结果。
# Sionna 的 AWGN / 衰落信道使用自己的 torch.Generator（config.torch_rng），
# 因此除 torch / numpy 外还必须设置 sionna.phy.config.seed。
from sionna.phy import config as sionna_config


def set_seed(seed=42):
    """固定 Python / NumPy / PyTorch / Sionna 四个随机源，使仿真结果可复现。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    sionna_config.seed = int(seed)


set_seed(42)

# 图、数据输出目录：便于把过程性与结果性材料归档到同一文件夹。
FIG_DIR = Path.cwd() / "figures"
DATA_DIR = Path.cwd() / "data"
FIG_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# --- 系统基础仿真参数（完全对标教材第 23~24 页）---
NUM_SUBCARRIERS = 64  # FFT 长度 / 子载波数 N
DEFAULT_CP_LEN = 16   # 默认循环前缀 (CP) 长度
BITS_PER_SYMBOL = 2   # QPSK 调制每个符号包含的比特数

# 3 径瑞利衰落信道参数（教材参数：path_delays=[0, 1, 4], path_gains=[0.8, 0.8, 0.8]）
PATH_DELAYS = [0, 1, 4]
MAX_PATH_DELAY = max(PATH_DELAYS)  # 最大多径时延 tau_max = 4
RAW_GAINS = torch.tensor([0.8, 0.8, 0.8], dtype=torch.float32, device=DEVICE)
# 对径功率做归一化，使得平均信道增益 E[|h|^2] = 1.0，与理论 BER 曲线严格对齐
PATH_GAINS = RAW_GAINS / torch.sqrt(torch.sum(RAW_GAINS ** 2))


# 将 Eb/N0 (dB) 换算成 AWGN 模块所需的复噪声方差 no：no = 1/(Eb/N0 * log2(M))
def ebno_db_to_no(ebno_db, nbits, coderate=1.0):
    ebno = torch.as_tensor(ebno_db, dtype=torch.float32, device=DEVICE)
    return 1.0 / (10.0 ** (ebno / 10.0) * nbits * coderate)


def qpsk_modulate(bits):
    """自主实现 QPSK 星座映射（能量归一化 Es = 1）。

    输入 bits 形状: (..., 2)，每个元素取值为 0 或 1。
    映射规则: s = ((1 - 2*b0) + 1j * (1 - 2*b1)) / sqrt(2)
    """
    bits = bits.to(torch.float32)
    b0 = 1.0 - 2.0 * bits[..., 0]
    b1 = 1.0 - 2.0 * bits[..., 1]
    return (b0 + 1j * b1).to(torch.complex64) / np.sqrt(2.0)


def qpsk_demodulate(symbols):
    """自主实现 QPSK 硬判决解映射。

    实部 > 0 判决为 0，否则判为 1；虚部 > 0 判决为 0，否则判为 1。
    输出形状为 (..., 2) 的判决比特。
    """
    b0 = (symbols.real < 0).to(torch.int32)
    b1 = (symbols.imag < 0).to(torch.int32)
    return torch.stack([b0, b1], dim=-1)


def ofdm_modulate(symbols, cp_len):
    """OFDM 发射机：频域到时域 IFFT 变换并添加循环前缀 (CP)。

    完全对标教材公式: time_domain_ofdm = sqrt(N) * ifft(symbols, N)
    """
    # 频域 -> 时域：sqrt(N) * IDFT，使时域样点平均能量保持为 Es
    time_sym = torch.fft.ifft(symbols, dim=-1) * np.sqrt(NUM_SUBCARRIERS)
    if cp_len > 0:
        # 循环前缀：复制符号尾部 cp_len 个样点放到符号前面
        time_sym = torch.cat([time_sym[..., -cp_len:], time_sym], dim=-1)
    return time_sym


def ofdm_demodulate(rx_time, cp_len):
    """OFDM 接收机：丢弃循环前缀 (CP) 并进行 FFT 变换回到频域。

    完全对标教材公式: rx_freq = (1 / sqrt(N)) * fft(rx_no_cp, N)
    """
    if cp_len > 0:
        rx_no_cp = rx_time[..., cp_len:]
    else:
        rx_no_cp = rx_time
    # 时域 -> 频域：(1/sqrt(N)) * DFT，与发射端构成酉变换对
    return torch.fft.fft(rx_no_cp, dim=-1) / np.sqrt(NUM_SUBCARRIERS)


def zf_equalize(rx_freq, channel_freq):
    """频域单抽头迫零 (Zero-Forcing) 均衡器。"""
    # 每个子载波上做复数除法：X_hat[k] = Y[k] / H[k]
    # 极小值保护：避免深衰落子载波除零导致 inf
    return rx_freq / (channel_freq + 1e-12)


def count_bit_errors(b, b_hat):
    """统计误比特数。"""
    return int(torch.sum(b.to(torch.int32) != b_hat.to(torch.int32)).item())


def ber(b, b_hat):
    """计算误比特率 BER。"""
    return count_bit_errors(b, b_hat) / b.numel()


def theory_awgn_ber(ebno_db):
    """QPSK 在 AWGN 信道下的理论 BER: Pb = Q(sqrt(2 * Eb/N0)) = 0.5 * erfc(sqrt(Eb/N0))。"""
    ebno_linear = 10.0 ** (np.asarray(ebno_db) / 10.0)
    return 0.5 * sp.erfc(np.sqrt(ebno_linear))


def theory_rayleigh_ber(ebno_db):
    """未编码 QPSK 在单抽头瑞利衰落信道下的理论 BER: Pb = 0.5 * (1 - sqrt(gamma / (1 + gamma)))。"""
    gamma = 10.0 ** (np.asarray(ebno_db) / 10.0)
    return 0.5 * (1.0 - np.sqrt(gamma / (1.0 + gamma)))


# --- 多径信道与 OFDM 链路的公共辅助函数 ---
def sample_multipath_taps(batch_size):
    """生成 3 径瑞利衰落信道的复增益, 形状 (batch_size, P)。"""
    real = torch.randn(batch_size, len(PATH_DELAYS), dtype=torch.float32, device=DEVICE)
    imag = torch.randn(batch_size, len(PATH_DELAYS), dtype=torch.float32, device=DEVICE)
    taps = (real + 1j * imag) / np.sqrt(2.0)
    return taps.to(torch.complex64) * PATH_GAINS.to(torch.complex64)


def channel_frequency_response(taps):
    """由时域多径抽头计算各子载波频域响应 H[k]，形状 (batch_size, N)。"""
    k = torch.arange(NUM_SUBCARRIERS, dtype=torch.float32, device=DEVICE)
    delays = torch.tensor(PATH_DELAYS, dtype=torch.float32, device=DEVICE)
    # exp(-j 2*pi*k*tau/N): (N, P)
    delay_phase = torch.exp(-2j * np.pi * torch.outer(k, delays) / NUM_SUBCARRIERS)
    return taps @ delay_phase.T


def apply_multipath_time(stream, taps):
    """对流式时域信号做多径线性卷积（按各径时延移位后加权求和）。"""
    total_len = stream.shape[-1]
    out = torch.zeros_like(stream)
    for p, d in enumerate(PATH_DELAYS):
        coeff = taps[:, p].unsqueeze(-1)
        if d == 0:
            shifted = stream
        else:
            shifted = torch.cat(
                [torch.zeros_like(stream[..., :d]), stream[..., : total_len - d]], dim=-1
            )
        out = out + coeff * shifted
    return out


def extract_ofdm_symbols(stream, num_symbols, cp_len):
    """从连续时域流中逐符号去 CP 并做 FFT，返回频域符号 (batch, num_symbols, N)。"""
    block_len = NUM_SUBCARRIERS + cp_len
    windows = []
    for i in range(num_symbols):
        start = i * block_len + cp_len
        windows.append(stream[:, start : start + NUM_SUBCARRIERS])
    windows = torch.stack(windows, dim=1)
    return ofdm_demodulate(windows, 0)  # 已手动去 CP，此处直接 FFT
'''

NB03A_CELL2 = r'''
# 模块二（基本实验）：多径瑞利衰落信道 vs. AWGN 信道下的 QPSK-OFDM 系统性能仿真
# 完全对标教材第 23~27 页基本实验脚本逻辑与理论对比曲线

def simulate_basic_ofdm(ebno_db_range, num_blocks=1000, symbols_per_block=10):
    """仿真 QPSK-OFDM 在多径瑞利信道与 AWGN 信道下的误码率特性。"""
    ber_rayleigh_list = []
    ber_awgn_list = []

    N = NUM_SUBCARRIERS
    cp = DEFAULT_CP_LEN
    block_len = N + cp

    for ebno_db in ebno_db_range:
        no = ebno_db_to_no(float(ebno_db), BITS_PER_SYMBOL).item()

        # 1) 发送端：随机比特 -> QPSK 映射 -> IFFT + CP -> 连续时域符号流
        bits = torch.randint(
            0,
            2,
            (num_blocks, symbols_per_block, N, BITS_PER_SYMBOL),
            dtype=torch.int32,
            device=DEVICE,
        )
        freq_sym = qpsk_modulate(bits)                                  # (B, S, N)
        time_sym = ofdm_modulate(freq_sym, cp)                          # (B, S, L)
        stream = time_sym.reshape(num_blocks, symbols_per_block * block_len)

        # 2) 每个块（帧）抽取一次 3 径瑞利信道，并计算该帧的频域响应 H[k]
        taps = sample_multipath_taps(num_blocks)
        h_freq = channel_frequency_response(taps)                       # (B, N)

        # 3) 多径信道传输 + AWGN
        rx_multipath = apply_multipath_time(stream, taps)
        rx_multipath = AWGN(precision=PRECISION, device=DEVICE)(rx_multipath, no)
        # 用于对比的纯 AWGN 信道（等效 H[k] = 1）
        rx_awgn = AWGN(precision=PRECISION, device=DEVICE)(stream, no)

        # 4) 接收端：去 CP + FFT + 频域迫零均衡 + QPSK 解映射
        rx_freq_mp = extract_ofdm_symbols(rx_multipath, symbols_per_block, cp)
        rx_eq_mp = zf_equalize(rx_freq_mp, h_freq.unsqueeze(1))
        bits_hat_mp = qpsk_demodulate(rx_eq_mp)

        rx_freq_aw = extract_ofdm_symbols(rx_awgn, symbols_per_block, cp)
        bits_hat_aw = qpsk_demodulate(rx_freq_aw)  # H[k] = 1，无需均衡

        ber_rayleigh_list.append(ber(bits, bits_hat_mp))
        ber_awgn_list.append(ber(bits, bits_hat_aw))

    return np.array(ber_rayleigh_list), np.array(ber_awgn_list)


# 设定信噪比扫描范围 (dB)
ebno_range_basic = np.arange(0, 26, 3)

print("正在执行基本实验：QPSK-OFDM 在多径瑞利衰落信道与 AWGN 信道下的性能仿真...")
ber_sim_rayleigh, ber_sim_awgn = simulate_basic_ofdm(ebno_range_basic, num_blocks=1000, symbols_per_block=10)

# 计算对应的理论误码率
ber_theory_ray = theory_rayleigh_ber(ebno_range_basic)
ber_theory_awg = theory_awgn_ber(ebno_range_basic)

# 绘制误码率性能对比曲线 (包含仿真瑞利、仿真AWGN、理论瑞利、理论AWGN四条曲线)
fig, ax = plt.subplots(figsize=(7.6, 5.0))
ber_sim_rayleigh_plot = [v if v > 0 else np.nan for v in ber_sim_rayleigh]
ber_sim_awgn_plot = [v if v > 0 else np.nan for v in ber_sim_awgn]
ax.semilogy(ebno_range_basic, ber_sim_rayleigh_plot, "o-", lw=1.5, ms=6, label="OFDM 多径瑞利衰落（仿真）")
ax.semilogy(ebno_range_basic, ber_theory_ray, "k--", lw=1.2, label="单抽头瑞利衰落（理论）")
ax.semilogy(ebno_range_basic, ber_sim_awgn_plot, "s-", lw=1.5, ms=6, label="OFDM AWGN（仿真）")
ax.semilogy(ebno_range_basic, ber_theory_awg, "b--", lw=1.2, label="AWGN（理论）")
ax.set_xlabel("Eb/N0 (dB)")
ax.set_ylabel("误比特率 BER")
ax.set_ylim(5e-4, 0.3)
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "03A_basic_ber.png", dpi=200, bbox_inches="tight")
print("图10  QPSK-OFDM 在 AWGN 与多径瑞利衰落信道下的误码性能")
plt.show()

# 数值表（结果性数据）
import pandas as pd

df_basic = pd.DataFrame(
    {
        "EbN0_dB": ebno_range_basic,
        "BER_多径瑞利_仿真": ber_sim_rayleigh,
        "BER_单抽头瑞利_理论": ber_theory_ray,
        "BER_AWGN_仿真": ber_sim_awgn,
        "BER_AWGN_理论": ber_theory_awg,
    }
)
print("表 5  基本实验：QPSK-OFDM 的 BER 仿真与理论值")
print(df_basic.to_string(index=False))
df_basic.to_csv(DATA_DIR / "03A_basic_ber.csv", index=False, encoding="utf-8-sig")

# 验证断言
assert len(ber_sim_rayleigh) == len(ebno_range_basic), "仿真结果长度与 Eb/N0 范围不符"
assert ber_sim_rayleigh[0] > ber_sim_rayleigh[-1], "瑞利信道 BER 应随 Eb/N0 上升而单调下降"
assert np.all(ber_sim_rayleigh >= ber_sim_awgn), "多径衰落信道误码率应普遍高于理想高斯信道"
print("基本实验仿真完成并通过断言检验！")
'''

NB03A_CELL3 = r'''
# 模块三（拓展实验一）：保护间隔 (CP) 长度与最大多径时延关系研究
# 完全对标教材第 28 页拓展实验 1 要求：
# “研究最大多径时延扩展 tau_m 与保护间隔 G_i (N_cp) 长度的关系。
#  分别观察当 N_cp < tau_m 和 N_cp >= tau_m 时系统 BER 性能的变化，并分析原因。”

def simulate_cp_sweep(cp_candidates, ebno_db=18.0, num_blocks=1000, symbols_per_block=8):
    """测试不同循环前缀长度下的连续 OFDM 帧传输误码率。

    采用连续符号流传输模式：前一符号的多径响应会泄漏到后一符号的前端。
    当 N_cp < tau_max 时，残留多径时延破坏正交性，引发 ISI 与 ICI，产生不可消除的误码平台。
    当 N_cp >= tau_max 时，多径时延被循环前缀完全吸收，ISI/ICI 消除。
    """
    ber_cp_list = []
    N = NUM_SUBCARRIERS
    no = ebno_db_to_no(float(ebno_db), BITS_PER_SYMBOL).item()

    for cp_len in cp_candidates:
        block_len = N + cp_len

        # 1) 发送端：QPSK -> IFFT + CP -> 连续符号流
        bits = torch.randint(
            0,
            2,
            (num_blocks, symbols_per_block, N, BITS_PER_SYMBOL),
            dtype=torch.int32,
            device=DEVICE,
        )
        freq_sym = qpsk_modulate(bits)
        time_sym = ofdm_modulate(freq_sym, cp_len)
        stream = time_sym.reshape(num_blocks, symbols_per_block * block_len)

        # 2) 3 径瑞利信道（每帧一次实现）+ AWGN
        taps = sample_multipath_taps(num_blocks)
        h_freq = channel_frequency_response(taps)
        rx_stream = apply_multipath_time(stream, taps)
        rx_stream = AWGN(precision=PRECISION, device=DEVICE)(rx_stream, no)

        # 3) 逐符号去 CP、FFT、迫零均衡、解映射与误码统计
        rx_freq = extract_ofdm_symbols(rx_stream, symbols_per_block, cp_len)
        rx_eq = zf_equalize(rx_freq, h_freq.unsqueeze(1))
        bits_hat = qpsk_demodulate(rx_eq)
        ber_cp_list.append(ber(bits, bits_hat))

    return np.array(ber_cp_list)


cp_candidates = [0, 1, 2, 4, 8, 16]
test_ebno = 18.0

print(f"正在执行拓展实验一：固定 Eb/N0 = {test_ebno} dB 下扫描循环前缀 CP 长度...")
ber_cp_results = simulate_cp_sweep(cp_candidates, ebno_db=test_ebno, num_blocks=1000, symbols_per_block=8)

# 打印数值统计
for cp, b in zip(cp_candidates, ber_cp_results):
    print(f"  CP 长度 = {cp:2d} (最大时延 tau_max={MAX_PATH_DELAY}): BER = {b:.5e}")

# 绘制 CP 长度与 BER 关系曲线，并在 x = MAX_PATH_DELAY 处绘制垂直虚线辅助线
fig, ax = plt.subplots(figsize=(7.0, 4.8))
ax.semilogy(cp_candidates, ber_cp_results, "o-", lw=1.6, ms=7, color="#d62728")
ax.axvline(MAX_PATH_DELAY, color="gray", ls="--", lw=1.2, label=f"最大多径时延 τmax = {MAX_PATH_DELAY}")
ax.set_xlabel("循环前缀长度 N_cp（样点）")
ax.set_ylabel("误比特率 BER")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "03A_cp_sweep.png", dpi=200, bbox_inches="tight")
print("图11  BER 随循环前缀长度的变化（Eb/N0 = 18 dB）")
plt.show()

# 数值表（结果性数据）
import pandas as pd

df_cp = pd.DataFrame({"N_cp": cp_candidates, "BER": ber_cp_results})
print("表 6  不同循环前缀长度下的 BER")
print(df_cp.to_string(index=False))
df_cp.to_csv(DATA_DIR / "03A_cp_sweep.csv", index=False, encoding="utf-8-sig")

# 验证断言：无 CP 或 CP 过短时的误码率显著高于满足条件的 CP
assert ber_cp_results[0] > ber_cp_results[-1], "无 CP 时的 BER 应显著高于充分 CP"
assert ber_cp_results[3] < ber_cp_results[0] * 0.5, "当 CP >= tau_max 时误码率应大幅降低"
print("拓展实验一仿真完成并通过断言检验！")
'''

NB03A_CELL4 = r'''
# 模块四（拓展实验二）：导频插入与 LS 信道估计性能对比
# 完全对标教材第 28 页拓展实验 2 要求：
# “以上实验假定信道信息在接收端是理想已知的。请修改仿真程序，通过插入导频 (Pilot)
#  的方式进行信道估计，并使用估计出的信道信息进行均衡。比较此时的 BER 性能与理想信道已知情况下的性能差异。”

def simulate_ls_channel_estimation(ebno_db_range, num_blocks=1000, data_symbols_per_block=4):
    """块状导频传输结构下的 LS 信道估计与理想已知 CSI 对比实验。

    每个传输块包含: 1 个已知块状导频符号 X_p + data_symbols_per_block 个数据符号 X_d。
    在信道相干时间内，导频与数据经历相同的信道响应 H[k]。
    接收端利用导频通过最小二乘估计算法: H_hat_ls[k] = Y_p[k] / X_p[k] 进行信道估计，
    然后使用 H_hat_ls[k] 均衡数据符号。
    """
    ber_ideal_list = []
    ber_ls_list = []

    N = NUM_SUBCARRIERS
    cp = DEFAULT_CP_LEN
    block_len = N + cp
    num_symbols = 1 + data_symbols_per_block

    for ebno_db in ebno_db_range:
        no = ebno_db_to_no(float(ebno_db), BITS_PER_SYMBOL).item()

        # 1) 构造已知导频符号与数据符号并拼接成一个发送块
        pilot_bits = torch.randint(0, 2, (num_blocks, N, BITS_PER_SYMBOL), dtype=torch.int32, device=DEVICE)
        pilot_sym = qpsk_modulate(pilot_bits)                                    # (B, N)
        data_bits = torch.randint(
            0,
            2,
            (num_blocks, data_symbols_per_block, N, BITS_PER_SYMBOL),
            dtype=torch.int32,
            device=DEVICE,
        )
        data_sym = qpsk_modulate(data_bits)                                      # (B, D, N)
        block_sym = torch.cat([pilot_sym.unsqueeze(1), data_sym], dim=1)         # (B, S, N)

        # 2) OFDM 调制 + 多径信道 + AWGN
        time_sym = ofdm_modulate(block_sym, cp)
        stream = time_sym.reshape(num_blocks, num_symbols * block_len)
        taps = sample_multipath_taps(num_blocks)
        h_freq = channel_frequency_response(taps)                                # (B, N)
        rx_stream = apply_multipath_time(stream, taps)
        rx_stream = AWGN(precision=PRECISION, device=DEVICE)(rx_stream, no)
        rx_freq = extract_ofdm_symbols(rx_stream, num_symbols, cp)               # (B, S, N)

        # 3) 导频位置最小二乘信道估计: H_hat = Y_p / X_p
        y_pilot = rx_freq[:, 0, :]
        h_ls = y_pilot / (pilot_sym + 1e-12)

        # 4) 分别用理想 CSI 与 LS 估计 CSI 均衡数据符号
        y_data = rx_freq[:, 1:, :]
        rx_eq_ideal = zf_equalize(y_data, h_freq.unsqueeze(1))
        rx_eq_ls = zf_equalize(y_data, h_ls.unsqueeze(1))

        ber_ideal_list.append(ber(data_bits, qpsk_demodulate(rx_eq_ideal)))
        ber_ls_list.append(ber(data_bits, qpsk_demodulate(rx_eq_ls)))

    return np.array(ber_ideal_list), np.array(ber_ls_list)


ebno_range_ls = np.arange(0, 25, 3)

print("正在执行拓展实验二：导频插入与 LS 信道估计性能仿真...")
ber_ideal_results, ber_ls_results = simulate_ls_channel_estimation(ebno_range_ls, num_blocks=1000, data_symbols_per_block=4)

# 绘制 BER 对比图 (包含理想已知 CSI、LS 估计 CSI 与理论瑞利参考线)
fig, ax = plt.subplots(figsize=(7.4, 4.9))
ax.semilogy(ebno_range_ls, ber_ideal_results, "o-", lw=1.5, ms=6, label="理想已知 CSI（仿真）")
ax.semilogy(ebno_range_ls, ber_ls_results, "s--", lw=1.5, ms=6, label="LS 信道估计（仿真）")
ax.semilogy(ebno_range_ls, theory_rayleigh_ber(ebno_range_ls), "k:", lw=1.4, label="单抽头瑞利衰落（理论）")
ax.set_xlabel("Eb/N0 (dB)")
ax.set_ylabel("误比特率 BER")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "03A_ls_estimation.png", dpi=200, bbox_inches="tight")
print("图12  理想 CSI 与基于导频 LS 估计的 OFDM 误码性能对比")
plt.show()

# 数值表（结果性数据）
import pandas as pd

df_ls = pd.DataFrame(
    {
        "EbN0_dB": ebno_range_ls,
        "BER_理想CSI": ber_ideal_results,
        "BER_LS估计": ber_ls_results,
        "BER_单抽头瑞利理论": theory_rayleigh_ber(ebno_range_ls),
    }
)
print("表 7  理想 CSI 与 LS 信道估计的 BER 对比")
print(df_ls.to_string(index=False))
df_ls.to_csv(DATA_DIR / "03A_ls_estimation.csv", index=False, encoding="utf-8-sig")

# 验证断言：LS 估计性能略差于理想已知信道（噪声引入估计误差），但在高信噪比下整体趋势一致
assert len(ber_ls_results) == len(ebno_range_ls), "结果长度不匹配"
assert ber_ls_results[0] > ber_ls_results[-1], "LS 估计 BER 应随 Eb/N0 提升而下降"
assert np.all(ber_ls_results >= ber_ideal_results * 0.9), "LS 估计引入信道估计噪声，BER 应普遍高于或接近理想 CSI"
print("拓展实验二仿真完成并通过断言检验！")
'''

# 03A 补充过程可视化
NB03A_CELL5 = r'''
# 补充（过程性材料）：OFDM 时域符号结构与循环前缀示意、频域子载波与多径信道频响。
set_seed(2026)

N = NUM_SUBCARRIERS
cp = DEFAULT_CP_LEN

# --- 图 9(a)：两个连续 OFDM 符号的时域波形与 CP 区间 ---
bits2 = torch.randint(0, 2, (1, 2, N, BITS_PER_SYMBOL), dtype=torch.int32, device=DEVICE)
sym2 = qpsk_modulate(bits2)
x2 = ofdm_modulate(sym2, cp).reshape(-1)      # 2 * (N + cp) 个时域样点

fig, ax = plt.subplots(figsize=(8.6, 3.0))
ax.plot(np.arange(x2.numel()), x2.real.cpu().numpy(), lw=1.0, label="同相分量 I")
ax.plot(np.arange(x2.numel()), x2.imag.cpu().numpy(), lw=1.0, alpha=0.8, label="正交分量 Q")
for s in range(2):
    start = s * (N + cp)
    ax.axvspan(start, start + cp, color="orange", alpha=0.25)
    ax.axvline(start + cp, color="k", ls="--", lw=0.8)
ax.set_xlabel("时域样点序号 n", fontsize=14)
ax.set_ylabel("幅度", fontsize=14)
ax.tick_params(labelsize=12)
ax.legend(fontsize=10, loc="upper right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "03A_time_cp_structure.png", dpi=200, bbox_inches="tight")
print("图7  OFDM 符号的时域波形与循环前缀结构")
plt.show()

# --- 图 9(b)：各子载波幅度与 3 径信道频域响应 ---
taps = sample_multipath_taps(1)
h_freq = channel_frequency_response(taps)[0].cpu().numpy()
k = np.arange(N)

fig, ax = plt.subplots(figsize=(8.2, 3.2))
ax.stem(k, np.abs(sym2[0, 0].cpu().numpy()), basefmt=" ", label="|X[k]|（发送）")
ax.plot(k, np.abs(h_freq), "r.-", lw=1.0, label="|H[k]|（信道频响）")
ax.set_xlabel("子载波序号 k", fontsize=14)
ax.set_ylabel("幅度", fontsize=14)
ax.tick_params(labelsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "03A_freq_response.png", dpi=200, bbox_inches="tight")
print("图8  三径信道的频率响应与子载波位置")
plt.show()

# --- 图 9(c)：单抽头均衡前后接收星座图 ---
# 取较高的 Eb/N0（20 dB）：低信噪比下 ZF 会把深衰落子载波的噪声放大，
# 四个簇彼此重叠，不利于展示均衡恢复相位结构的效果。
ebb = 20.0
no8 = ebno_db_to_no(ebb, BITS_PER_SYMBOL).item()
bits3 = torch.randint(0, 2, (2000, NUM_SUBCARRIERS, BITS_PER_SYMBOL), dtype=torch.int32, device=DEVICE)
sym3 = qpsk_modulate(bits3)
ts = ofdm_modulate(sym3, cp)
st = ts.reshape(2000, N + cp)
tp = sample_multipath_taps(2000)
hf = channel_frequency_response(tp)
rx = AWGN(precision=PRECISION, device=DEVICE)(apply_multipath_time(st, tp), no8)
rf = extract_ofdm_symbols(rx, 1, cp)
rx_eq = zf_equalize(rf, hf.unsqueeze(1))

# 深衰落子载波 |H[k]| 很小，ZF 均衡会把噪声放大成离群点；
# 这里不截断数据，而是用坐标轴范围把离群点裁掉（比截断更干净，也不改变 BER 统计）。
eq_re = rx_eq[:, 0, :].real.cpu().flatten()
eq_im = rx_eq[:, 0, :].imag.cpu().flatten()

fig, axes = plt.subplots(2, 1, figsize=(7.6, 8.2))
axes[0].scatter(rf[:, 0, :].real.cpu().flatten(), rf[:, 0, :].imag.cpu().flatten(), s=3, alpha=0.25)
axes[0].set_title("(a) 均衡前（多径造成幅度/相位失真）", fontsize=13)
axes[1].scatter(eq_re, eq_im, s=3, alpha=0.25)
axes[1].set_title("(b) 频域 ZF 均衡后（恢复 QPSK 四象限结构）", fontsize=13)
for ax in axes:
    ax.set_xlabel("I", fontsize=13)
    ax.set_ylabel("Q", fontsize=13)
    ax.tick_params(labelsize=12)
    ax.grid(alpha=0.3)
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
axes[1].set_aspect("equal")
plt.tight_layout()
plt.savefig(FIG_DIR / "03A_constellation_eq.png", dpi=200, bbox_inches="tight")
print("图9  OFDM 接收端 ZF 均衡前后的星座图（Eb/N0 = 20 dB）")
plt.show()
'''


def build_01():
    nb = load("01_baseband_modulation.ipynb")
    nb["cells"][1] = code_cell(COMMON_HEAD)
    nb["cells"][2] = code_cell(NB01_CELL2)
    nb["cells"].append(
        md_cell(
            """## 补充：BER 随 Eb/N0 变化曲线

在完成模板要求的 8 dB 单点统计之后，进一步扫描 Eb/N0 = 0~12 dB，
得到四种调制方式的 BER 曲线，作为报告第 3.1 节定量分析的依据。"""
        )
    )
    nb["cells"].append(code_cell(NB01_CELL3))
    save(nb, "01_baseband_modulation.ipynb")


def build_02():
    nb = load("02_channel_ber.ipynb")
    nb["cells"][1] = code_cell(COMMON_HEAD)
    nb["cells"][2] = code_cell(NB02_CELL2)
    nb["cells"][3] = code_cell(NB02_CELL3)
    nb["cells"].append(
        md_cell(
            """## 补充：分信道对比与理论曲线

将 AWGN 与瑞利衰落两种信道分开展示，并叠加理论曲线，用于定量评估仿真与理论的吻合程度。"""
        )
    )
    nb["cells"].append(code_cell(NB02_CELL4))
    save(nb, "02_channel_ber.ipynb")


def build_03a():
    nb = load("03A_ofdm.ipynb")
    nb["cells"][1] = code_cell(NB03A_CELL1)
    nb["cells"][2] = code_cell(NB03A_CELL2)
    nb["cells"][3] = code_cell(NB03A_CELL3)
    nb["cells"][4] = code_cell(NB03A_CELL4)
    nb["cells"].append(
        md_cell(
            """## 补充：OFDM 关键过程可视化

为便于在报告中说明 OFDM 的时域结构、频域子载波与均衡效果，这里补充输出时域波形（含 CP 区间）、
子载波与信道频响，以及迫零均衡前后的接收星座图。"""
        )
    )
    nb["cells"].append(code_cell(NB03A_CELL5))
    save(nb, "03A_ofdm.ipynb")


def build_00():
    nb = load("00_env_check.ipynb")
    save(nb, "00_env_check.ipynb")


def build_04():
    import nbformat
    work = Path(__file__).resolve().parent.parent
    nb = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell("# OFDM 补充实验\n配对 CP 多信噪比实验和 LS 精确理论。"),
        nbformat.v4.new_code_cell((work / "scripts/supplement_ofdm.py").read_text(encoding="utf-8"))])
    nb.metadata.kernelspec = dict(display_name="Python 3", language="python", name="python3")
    nbformat.write(nb, work / "notebooks/04_ofdm_supplement.ipynb")


if __name__ == "__main__":
    # 支持只重建部分 notebook：python make_notebooks.py 03a
    wanted = set(sys.argv[1:])
    todo = {
        "00": build_00,
        "01": build_01,
        "02": build_02,
        "03a": build_03a,
        "04": build_04,
    }
    for key, func in todo.items():
        if not wanted or key in wanted:
            func()
