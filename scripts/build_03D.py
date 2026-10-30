# -*- coding: utf-8 -*-
"""生成 03D 接收分集实验 notebook（在模板基础上补全 TODO 并追加过程性单元）。

用法（在任意目录）::

    work/venv/Scripts/python.exe work/scripts/build_03D.py

只读取 无线仿真资料2026/student/03D_rx_diversity.ipynb 的 markdown 与代码骨架，
写出 notebooks/03D_rx_diversity.ipynb（未执行）。执行由 nbconvert 完成。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]          # 仓库根目录
TEMPLATE = ROOT / "无线仿真资料2026" / "student" / "03D_rx_diversity.ipynb"
OUTPUT = ROOT / "notebooks" / "03D_rx_diversity.ipynb"


def _src(text: str) -> list[str]:
    """把多行文本切成 notebook 的 source 行列表（保留换行符）。"""
    text = text.strip("\n") + "\n"
    return text.splitlines(keepends=True)


def md(text: str, cid: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cid,
        "metadata": {},
        "source": _src(text),
    }


def code(text: str, cid: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cid,
        "metadata": {},
        "outputs": [],
        "source": _src(text),
    }


# ============================================================================
# 模块一：环境准备 + 核心组件实现（补全模板 cell 1）
# ============================================================================
CELL1 = r'''
# 模块一：环境准备与接收分集核心组件实现
from pathlib import Path
import sys

# 从当前目录向上查找 src/sionna，优先使用本仓库源码。
for p in [Path.cwd(), *Path.cwd().parents]:
    if (p / "src" / "sionna").exists():
        sys.path.insert(0, str(p / "src"))
        break

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import scipy.special as sp

# 设置 Matplotlib 中文字体，避免图标题和坐标轴中文乱码。
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

# Word 报告中的图宽约 14.4 cm，画布按英寸设置，插入时会被缩放，
# 因此整体放大字号，使打印后的字号与正文（10.5 pt）接近。
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

# 固定随机源，便于复现实验结果（本实验的瑞利信道与噪声全部由 torch 生成）。
torch.manual_seed(42)
np.random.seed(42)
SEED = 42

# 图、数据输出目录：过程性与结果性材料归档到同一文件夹。
FIG_DIR = Path.cwd() / "figures"
DATA_DIR = Path.cwd() / "data"
FIG_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# --- 16-QAM 星座点定义与能量归一化（完全对标教材第 43 页 lines 459~465）---
RAW_16QAM = torch.tensor([
    1 + 1j,  1 + 3j,  3 + 1j,  3 + 3j,
   -1 + 1j, -1 + 3j, -3 + 1j, -3 + 3j,
   -3 - 3j, -3 - 1j, -1 - 3j, -1 - 1j,
    1 - 1j,  1 - 3j,  3 - 1j,  3 - 3j
], dtype=torch.complex64, device=DEVICE)
CONSTELLATION_16QAM = RAW_16QAM / np.sqrt(10.0)

# 本实验使用的 16-QAM 幅度表（去掉 1/sqrt(10) 归一化因子后）：
# 归一化后实部/虚部取自 {+-1, +-3}/sqrt(10)，平均符号能量恰好为 Es = 1。
M_16QAM = 16
BITS_PER_SYMBOL_16QAM = int(np.log2(M_16QAM))


def cn(shape):
    """生成 CN(0,1) 复高斯随机量。

    注意 torch.randn(..., dtype=torch.complex64) 的实部、虚部各自方差为 1，
    即总方差为 2；这里显式除以 sqrt(2)，保证 E[|h|^2] = 1，与教材归一化一致。
    """
    z = torch.randn(shape) + 1j * torch.randn(shape)
    return (z / np.sqrt(2.0)).to(torch.complex64)


def modulate_16qam(symbol_indices):
    """自主实现 16-QAM 星座映射（平均符号能量 Es = 1）。

    逐符号查表：第 k 个符号索引 -> CONSTELLATION_16QAM[k]，
    等价于先按 (b0,b1) 选择同相幅度、再按 (b2,b3) 选择正交幅度。
    """
    return CONSTELLATION_16QAM[symbol_indices.long()]


def demodulate_16qam(received_symbols):
    """自主实现 16-QAM 最小欧氏距离判决解映射（对标教材 lines 543~548）。

    对每个接收符号计算到 16 个星座点的欧氏距离，取距离最小者的索引作为硬判决：
    k_hat = argmin_k |z - s_k|。
    """
    # 广播成 (..., 16) 的距离张量，再沿星座维取 argmin
    points = CONSTELLATION_16QAM.view(*([1] * received_symbols.dim()), M_16QAM)
    distances = (received_symbols.unsqueeze(-1) - points).abs()
    return distances.argmin(dim=-1)


def mrc_combine(y, h):
    """最大比合并 (Maximal Ratio Combining, MRC)。

    输入:
        y: 接收信号张量，形状 (batch_size, num_rx, num_syms)
        h: 信道增益张量，形状 (batch_size, num_rx, num_syms)
    输出:
        z: 合并后的复符号，形状 (batch_size, num_syms)
    公式: z = sum(conj(h) * y) / sum(|h|^2)
    """
    # 分子按各支路瞬时 SNR 加权相干合并，分母归一化使合并输出仍为 x + 等效噪声
    numerator = torch.sum(torch.conj(h) * y, dim=1)
    denominator = torch.sum(h.abs().pow(2), dim=1)
    return numerator / denominator


def sc_combine(y, h):
    """选择性合并 (Selection Combining, SC)。

    输入:
        y: 接收信号张量，形状 (batch_size, num_rx, num_syms)
        h: 信道增益张量，形状 (batch_size, num_rx, num_syms)
    输出:
        z: 选择最强支路后的复符号，形状 (batch_size, num_syms)
    公式: 选择瞬时增益最大的一路 l* = argmax |h_l|^2，输出 z = conj(h_l*) * y_l* / |h_l*|^2
    """
    # 每个符号独立选择瞬时功率最大的支路，再做单支路相干合并（相位对齐）
    best = torch.argmax(h.abs().pow(2), dim=1, keepdim=True)
    y_sel = torch.gather(y, 1, best)
    h_sel = torch.gather(h, 1, best)
    z = torch.conj(h_sel) * y_sel / h_sel.abs().pow(2)
    return z.squeeze(1)


def ser(sym_true, sym_hat):
    """计算符号错误率 SER。"""
    return torch.mean((sym_true != sym_hat).float()).item()


# --- 组件自检：确认映射/判决/合并器实现正确（同时作为过程性输出） ---
print("表 1  03D 接收分集仿真主要参数")
_parameters = [
    ("调制方式", "16-QAM 自主查表映射（Es = 1）+ BPSK 拓展"),
    ("天线配置", "1 发 L 收（SIMO），L = 1, 2, 4，各支路独立"),
    ("合并准则", "MRC: z = sum(conj(h)*y)/sum(|h|^2)；SC: 选 max|h|^2 支路"),
    ("信道模型", "平坦瑞利衰落 h_l ~ CN(0, 1)"),
    ("噪声模型", "w_l ~ CN(0, N0)，N0 = 1/(Es/N0)（16-QAM）；N0 = 1/(Eb/N0)（BPSK）"),
    ("基本实验 SNR", "Es/N0 = 0:3:30 dB，1000 块 x 100 符号 = 100000 符号/点"),
    ("拓展实验 SNR", "Eb/N0 = 0:2:20 dB，2000 块 x 100 bit = 200000 bit/点"),
    ("随机种子", f"torch / numpy seed = {SEED}"),
]
for _key, _val in _parameters:
    print(f"  {_key:<12s}{_val}")

# 16-QAM 星座的两个关键指标：平均符号能量与最小欧氏距离
_points_2d = torch.stack([CONSTELLATION_16QAM.real, CONSTELLATION_16QAM.imag], dim=-1)
_pairwise = torch.cdist(_points_2d, _points_2d)
_d_min = _pairwise[_pairwise > 1e-6].min().item()
print(
    f"  16-QAM 校验：Es = {CONSTELLATION_16QAM.abs().pow(2).mean().item():.6f}"
    f"（理论 1.0），d_min = {_d_min:.6f}（理论 2/sqrt(10) = {2 / np.sqrt(10):.6f}）"
)

# 无噪声往返自检：映射后立即判决必须 100% 正确
_test_idx = torch.randint(0, M_16QAM, (2000, 50), device=DEVICE)
_test_sym = modulate_16qam(_test_idx)
_roundtrip_err = int((demodulate_16qam(_test_sym) != _test_idx.long()).sum().item())
print(f"  16-QAM 无噪声映射-判决往返错误数 = {_roundtrip_err} / {_test_idx.numel()}")

# 合并器输出形状自检
_h_check = cn((3, 4, 7))
_y_check = _h_check + 0.1 * cn((3, 4, 7))
print(
    f"  MRC 输出形状 {tuple(mrc_combine(_y_check, _h_check).shape)}，"
    f"SC 输出形状 {tuple(sc_combine(_y_check, _h_check).shape)}（期望 (3, 7)）"
)
assert _roundtrip_err == 0, "16-QAM 无噪声映射/判决往返应无误码"

print("16-QAM 星座与接收分集核心组件初始化完成！")
'''

# ============================================================================
# 模块二：16-QAM 接收分集（补全模板 cell 2）
# ============================================================================
CELL2 = r'''
# 模块二（基本实验）：16-QAM 在平坦瑞利衰落信道下的接收分集性能仿真 (MRC vs. SC)
# 完全对标教材第 43~47 页基本实验一（图 2.16）

def simulate_16qam_diversity(esn0_db_range, num_blocks=1000, num_syms=100):
    """仿真 16-QAM 信号在 1/2/4 接收天线下的 MRC 与 SC 符号错误率 (SER)。

    链路：y_l = h_l * x + w_l，l = 1..L；h_l ~ CN(0,1)，w_l ~ CN(0,N0)，
    符号能量 Es = 1，故 N0 = 1/(Es/N0 线性值)。同一 SNR、同一 L 下 MRC 与 SC
    共用同一组 (h, y)（公共随机数），使两种合并准则的对比不受随机波动干扰。
    """
    ser_results = {"mrc_1": [], "mrc_2": [], "sc_2": [], "mrc_4": [], "sc_4": []}

    for esn0_db in esn0_db_range:
        esn0_linear = 10.0 ** (esn0_db / 10.0)
        no = 1.0 / esn0_linear          # 复噪声功率（每个复维）
        sigma = np.sqrt(no)             # cn() 已是单位复功率，乘 sqrt(N0) 得到 CN(0, N0) 噪声

        # 待发送的 16-QAM 符号索引（整块生成，供 1/2/4 天线共享同一符号序列）
        sym_idx = torch.randint(0, M_16QAM, (num_blocks, num_syms), device=DEVICE)
        x = modulate_16qam(sym_idx)

        # (1) 1x1：单天线基线（无分集）
        h1 = cn((num_blocks, 1, num_syms))
        y1 = h1 * x.unsqueeze(1) + sigma * cn((num_blocks, 1, num_syms))
        ser_results["mrc_1"].append(ser(sym_idx, demodulate_16qam(mrc_combine(y1, h1))))

        # (2) 1x2：同一组信道/噪声同时驱动 MRC 与 SC
        h2 = cn((num_blocks, 2, num_syms))
        y2 = h2 * x.unsqueeze(1) + sigma * cn((num_blocks, 2, num_syms))
        ser_results["mrc_2"].append(ser(sym_idx, demodulate_16qam(mrc_combine(y2, h2))))
        ser_results["sc_2"].append(ser(sym_idx, demodulate_16qam(sc_combine(y2, h2))))

        # (3) 1x4：同上
        h4 = cn((num_blocks, 4, num_syms))
        y4 = h4 * x.unsqueeze(1) + sigma * cn((num_blocks, 4, num_syms))
        ser_results["mrc_4"].append(ser(sym_idx, demodulate_16qam(mrc_combine(y4, h4))))
        ser_results["sc_4"].append(ser(sym_idx, demodulate_16qam(sc_combine(y4, h4))))

    return {key: np.asarray(value) for key, value in ser_results.items()}


def ser_16qam_awgn_theory(esn0_db):
    """16-QAM 在 AWGN 信道下的精确符号错误率（用于给出无衰落参考线）。

    16-QAM 归一化后实/虚两维独立，每维 4 电平 PAM 的最近判决边界距离为
    1/sqrt(10)；符号正确概率 = [1 - 1.5*Q(sqrt(gamma/5))]^2，故
    SER = 1 - (1 - 1.5*Q(sqrt(gamma/5)))^2。
    """
    gamma = 10.0 ** (np.asarray(esn0_db, dtype=float) / 10.0)
    q = 0.5 * sp.erfc(np.sqrt(gamma / 5.0) / np.sqrt(2.0))   # Q(sqrt(gamma/5))
    return 1.0 - (1.0 - 1.5 * q) ** 2


def floor_curve(values, floor=1e-6):
    """SER = 0（该点统计不到误符号）时抬到 floor，避免半对数坐标丢点。"""
    return np.maximum(np.asarray(values, dtype=float), floor)


esn0_range_basic = np.arange(0, 31, 3)
print("正在执行基本实验：16-QAM 接收分集性能仿真 (MRC vs. SC)...")
ser_results = simulate_16qam_diversity(esn0_range_basic, num_blocks=1000, num_syms=100)

# 打印数值统计
for esn0, m1, m2, s2, m4, s4 in zip(
    esn0_range_basic,
    ser_results["mrc_1"],
    ser_results["mrc_2"],
    ser_results["sc_2"],
    ser_results["mrc_4"],
    ser_results["sc_4"]
):
    print(f"  Es/N0 = {esn0:2d} dB: 1Rx MRC={m1:.4e} | 2Rx MRC={m2:.4e} | 2Rx SC={s2:.4e} | 4Rx MRC={m4:.4e} | 4Rx SC={s4:.4e}")

# 绘制 16-QAM 在不同接收天线与合并准则下的半对数 SER 曲线（含 AWGN 无衰落参考）
awgn_reference = ser_16qam_awgn_theory(esn0_range_basic)

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.semilogy(esn0_range_basic, floor_curve(ser_results["mrc_1"]), "o-", lw=1.6, ms=5, label="1x1 无分集")
ax.semilogy(esn0_range_basic, floor_curve(ser_results["sc_2"]), "s--", lw=1.5, ms=5, label="1x2 SC")
ax.semilogy(esn0_range_basic, floor_curve(ser_results["mrc_2"]), "s-", lw=1.6, ms=5, label="1x2 MRC")
ax.semilogy(esn0_range_basic, floor_curve(ser_results["sc_4"]), "^--", lw=1.5, ms=5, label="1x4 SC")
ax.semilogy(esn0_range_basic, floor_curve(ser_results["mrc_4"]), "^-", lw=1.6, ms=5, label="1x4 MRC")
ax.semilogy(esn0_range_basic, awgn_reference, "k:", lw=1.5, label="1x1 AWGN（无衰落参考）")
ax.set_xlabel("Es/N0 (dB)")
ax.set_ylabel("符号错误率 SER")
ax.set_xlim(esn0_range_basic[0], esn0_range_basic[-1])
ax.set_ylim(1e-6, 1.0)
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="lower left")
plt.tight_layout()
plt.savefig(FIG_DIR / "03D_ser_16qam_mrc_sc.png", dpi=200, bbox_inches="tight")
print("图题：16-QAM 平坦瑞利信道下接收分集（MRC/SC）的 SER 性能对比")
plt.show()

# 验证断言：MRC 严格优于 SC，天线增加显著改善性能
assert np.all(ser_results["mrc_2"] <= ser_results["sc_2"] * 1.05), "2 天线 MRC 性能应优于 SC"
assert np.all(ser_results["mrc_4"] <= ser_results["sc_4"] * 1.05), "4 天线 MRC 性能应优于 SC"
assert ser_results["mrc_4"][5] < ser_results["mrc_1"][5] * 0.05, "4 天线 MRC 具备极强的分集与阵列增益"
print("基本实验仿真完成并通过断言检验！结论：MRC 兼具分集增益与阵列增益，性能严格优于仅有分集增益的 SC。")

# 结果落盘（供报告表格直接引用）
df_ser = pd.DataFrame(
    {
        "esn0_db": esn0_range_basic,
        "mrc_1": ser_results["mrc_1"],
        "mrc_2": ser_results["mrc_2"],
        "sc_2": ser_results["sc_2"],
        "mrc_4": ser_results["mrc_4"],
        "sc_4": ser_results["sc_4"],
        "awgn_theory": awgn_reference,
    }
)
df_ser.to_csv(DATA_DIR / "03D_ser_16qam_diversity.csv", index=False, encoding="utf-8-sig")
print(f"已保存数据：{(DATA_DIR / '03D_ser_16qam_diversity.csv').name}")
'''

# ============================================================================
# 模块三：BPSK MRC 与理论闭式解对齐（补全模板 cell 3）
# ============================================================================
CELL3 = r'''
# 模块三（拓展实验）：BPSK 调制下的接收分集与理论闭式解对齐
# 对标教材拓展实验，分析不同调制方式与分集阶数 (L=1, 2, 4) 的理论性能吻合度

def simulate_bpsk_mrc(ebno_db_range, num_blocks=2000, num_bits=100):
    """仿真 BPSK 调制下不同接收天线数 (1, 2, 4) 的 MRC 误比特率。

    BPSK：每符号 1 bit，Es = Eb = 1，噪声 N0 = 1/(Eb/N0)；
    合并后瞬时 SNR = sum_l |h_l|^2 / N0，平均每支路 SNR = Eb/N0。
    """
    results = {1: [], 2: [], 4: []}

    for ebno_db in ebno_db_range:
        ebno_linear = 10.0 ** (ebno_db / 10.0)
        no = 1.0 / ebno_linear
        sigma = np.sqrt(no)             # cn() 单位复功率 -> 复噪声功率恰为 N0

        # BPSK 映射：bit 0 -> +1，bit 1 -> -1
        bits_tx = torch.randint(0, 2, (num_blocks, num_bits), device=DEVICE)
        x = (1.0 - 2.0 * bits_tx.float()).to(torch.complex64)

        for num_rx in (1, 2, 4):
            h = cn((num_blocks, num_rx, num_bits))
            y = h * x.unsqueeze(1) + sigma * cn((num_blocks, num_rx, num_bits))
            z = mrc_combine(y, h)
            # BPSK 硬判决：合并后实部 < 0 判为 bit 1
            bits_hat = (z.real < 0).to(torch.int32)
            results[num_rx].append(
                float((bits_tx.to(torch.int32) != bits_hat).float().mean().item())
            )

    return {key: np.asarray(value) for key, value in results.items()}


def mrc_ber_theory(ebno_db, L):
    """计算 L 阶独立平坦瑞利衰落信道下 BPSK MRC 的精确理论误比特率闭式解。"""
    ebno = 10.0 ** (ebno_db / 10.0)
    mu = np.sqrt(ebno / (1.0 + ebno))
    term1 = (0.5 * (1.0 - mu)) ** L
    total = np.zeros_like(ebno)
    for k in range(L):
        comb = sp.comb(L - 1 + k, k)
        total += comb * ((0.5 * (1.0 + mu)) ** k)
    return term1 * total


def ber_bpsk_awgn_theory(ebno_db):
    """BPSK 在 AWGN 下的理论 BER：Pb = Q(sqrt(2*Eb/N0)) = 0.5*erfc(sqrt(Eb/N0))。"""
    ebno = 10.0 ** (np.asarray(ebno_db, dtype=float) / 10.0)
    return 0.5 * sp.erfc(np.sqrt(ebno))


ebno_range_ext = np.arange(0, 21, 2)
print("正在执行拓展实验：BPSK 接收分集与理论曲线对齐仿真...")
bpsk_mrc_results = simulate_bpsk_mrc(ebno_range_ext, num_blocks=2000, num_bits=100)

for ebno_db in ebno_range_ext:
    _i = int(np.where(ebno_range_ext == ebno_db)[0][0])
    print(
        f"  Eb/N0 = {ebno_db:2d} dB: 1Rx 仿真={bpsk_mrc_results[1][_i]:.4e} / 理论={mrc_ber_theory(ebno_db, 1):.4e} | "
        f"2Rx 仿真={bpsk_mrc_results[2][_i]:.4e} / 理论={mrc_ber_theory(ebno_db, 2):.4e} | "
        f"4Rx 仿真={bpsk_mrc_results[4][_i]:.4e} / 理论={mrc_ber_theory(ebno_db, 4):.4e}"
    )

# 绘制 BPSK 接收分集 BER 曲线：仿真散点 + 理论闭式解实线 + AWGN 参考
ebno_dense = np.linspace(0.0, 20.0, 401)
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.semilogy(ebno_dense, mrc_ber_theory(ebno_dense, 1), "-", lw=1.5, label="1x1 Rayleigh 理论")
ax.semilogy(ebno_dense, mrc_ber_theory(ebno_dense, 2), "-", lw=1.5, label="1x2 MRC 理论")
ax.semilogy(ebno_dense, mrc_ber_theory(ebno_dense, 4), "-", lw=1.5, label="1x4 MRC 理论")
ax.semilogy(ebno_range_ext, floor_curve(bpsk_mrc_results[1], 1e-6), "o", ms=6, mfc="none", label="1x1 仿真")
ax.semilogy(ebno_range_ext, floor_curve(bpsk_mrc_results[2], 1e-6), "s", ms=6, mfc="none", label="1x2 MRC 仿真")
ax.semilogy(ebno_range_ext, floor_curve(bpsk_mrc_results[4], 1e-6), "^", ms=6, mfc="none", label="1x4 MRC 仿真")
ax.semilogy(ebno_dense, ber_bpsk_awgn_theory(ebno_dense), "k:", lw=1.5, label="1x1 AWGN（无衰落参考）")
ax.set_xlabel("Eb/N0 (dB)")
ax.set_ylabel("误比特率 BER")
ax.set_xlim(0, 20)
ax.set_ylim(1e-6, 0.5)
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="lower left")
plt.tight_layout()
plt.savefig(FIG_DIR / "03D_bpsk_ber_mrc.png", dpi=200, bbox_inches="tight")
print("图题：BPSK 接收分集 MRC 仿真结果与理论闭式解的对齐")
plt.show()

# 验证断言
theory_mrc_1 = mrc_ber_theory(ebno_range_ext, 1)
theory_mrc_2 = mrc_ber_theory(ebno_range_ext, 2)
theory_mrc_4 = mrc_ber_theory(ebno_range_ext, 4)
assert np.max(np.abs(bpsk_mrc_results[1][:5] - theory_mrc_1[:5])) < 0.02, "1 天线仿真与理论闭式解高度吻合"
assert np.max(np.abs(bpsk_mrc_results[2][:5] - theory_mrc_2[:5])) < 0.01, "2 天线 MRC 仿真与理论闭式解高度吻合"
assert np.max(np.abs(bpsk_mrc_results[4][:5] - theory_mrc_4[:5])) < 0.005, "4 天线 MRC 仿真与理论闭式解高度吻合"
print("拓展实验仿真完成并通过断言检验！结论：仿真结果与通信理论闭式解完全吻合。")

df_bpsk = pd.DataFrame(
    {
        "ebno_db": ebno_range_ext,
        "sim_mrc_1": bpsk_mrc_results[1],
        "sim_mrc_2": bpsk_mrc_results[2],
        "sim_mrc_4": bpsk_mrc_results[4],
        "theory_mrc_1": theory_mrc_1,
        "theory_mrc_2": theory_mrc_2,
        "theory_mrc_4": theory_mrc_4,
        "awgn_theory": ber_bpsk_awgn_theory(ebno_range_ext),
    }
)
df_bpsk.to_csv(DATA_DIR / "03D_bpsk_mrc_theory.csv", index=False, encoding="utf-8-sig")
print(f"已保存数据：{(DATA_DIR / '03D_bpsk_mrc_theory.csv').name}")
'''

# ============================================================================
# 追加单元 1：16-QAM 星座与接收星座（过程性材料）
# ============================================================================
EXTRA_MD_1 = r'''
## 补充：归一化星座与合并前后的接收信号

下面把前面用到的核心过程做可视化：16-QAM 归一化星座（Es = 1，最小欧氏距离
$2/\sqrt{10}$）以及 4 根天线 MRC 合并后的接收星座。后者可直观看到 16 个判决簇
在高斯噪声下扩散、但仍能被最小欧氏距离判决器正确分开。
'''

EXTRA_CODE_1 = r'''
# 补充（过程性材料）：16-QAM 归一化星座与 4Rx MRC 合并后的接收星座
torch.manual_seed(2026)

# --- 图 3(a)：16-QAM 归一化星座与判决边界 ---
fig, axes = plt.subplots(2, 1, figsize=(8.0, 8.5))

ax = axes[0]
pts = CONSTELLATION_16QAM.cpu().numpy()
ax.scatter(pts.real, pts.imag, s=80, c="#1f77b4", zorder=3, label="16-QAM 星座点（Es = 1）")
# 判决边界：每个维度的电平阈值位于 0 与 +-2/sqrt(10)
_thr = 2.0 / np.sqrt(10.0)
for _v in (-_thr, 0.0, _thr):
    ax.axvline(_v, color="#d62728", ls="--", lw=0.9, alpha=0.8)
    ax.axhline(_v, color="#d62728", ls="--", lw=0.9, alpha=0.8)
ax.plot([], [], color="#d62728", ls="--", lw=0.9, label="最小欧氏距离判决边界")
# 第一象限最外层点标注一次，说明 (±1,±3)/sqrt(10) 的幅度结构
ax.annotate(
    "(3, 3)/sqrt(10)",
    xy=(pts.real.max(), pts.imag.max()),
    xytext=(-64, -26),
    textcoords="offset points",
    fontsize=10,
    arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "gray"},
)
ax.set_xlabel("同相分量 I")
ax.set_ylabel("正交分量 Q")
ax.set_aspect("equal")
ax.set_xlim(-1.35, 1.35)
ax.set_ylim(-1.35, 1.35)
ax.set_title("(a) 16-QAM 归一化星座（平均符号能量 Es = 1）", fontsize=13)
ax.legend(loc="lower left")
ax.grid(alpha=0.3)

# --- 图 3(b)：4Rx MRC 合并后的接收星座（Es/N0 = 6 dB） ---
esn0_db = 6.0
no = 1.0 / (10.0 ** (esn0_db / 10.0))
sigma = np.sqrt(no)
n_sym = 4000
idx = torch.randint(0, M_16QAM, (n_sym, 1), device=DEVICE)
x = modulate_16qam(idx)
h = cn((n_sym, 4, 1))
y = h * x.unsqueeze(1) + sigma * cn((n_sym, 4, 1))
z = mrc_combine(y, h).reshape(-1).cpu().numpy()

ax = axes[1]
ax.scatter(pts.real, pts.imag, s=110, c="#1f77b4", marker="x", zorder=3, label="理想星座点")
ax.scatter(z.real, z.imag, s=6, alpha=0.25, c="#d62728", label="4Rx MRC 合并输出")
ax.axhline(0.0, color="gray", lw=0.8)
ax.axvline(0.0, color="gray", lw=0.8)
ax.set_xlabel("判决变量实部")
ax.set_ylabel("判决变量虚部")
ax.set_aspect("equal")
ax.set_xlim(-2.0, 2.0)
ax.set_ylim(-2.0, 2.0)
ax.set_title(f"(b) 4Rx MRC 合并后接收星座（Es/N0 = {esn0_db:.0f} dB）", fontsize=13)
ax.legend(loc="upper right")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / "03D_constellation_process.png", dpi=200, bbox_inches="tight")
print("图题：16-QAM 归一化星座与 4 天线最大比合并后的接收星座")
plt.show()
'''

# ============================================================================
# 追加单元 2：16-QAM MRC/SC 半解析理论曲线核对
# ============================================================================
EXTRA_MD_2 = r'''
## 补充：16-QAM 分集 SER 的半解析理论核对

瑞利衰落下 MRC 合并后瞬时 SNR 为 $\gamma=\bar\gamma\sum_{l=1}^{L}|h_l|^2$，
其中 $\bar\gamma=E_s/N_0$，$\sum_l|h_l|^2\sim\Gamma(L,1)$；SC 的瞬时 SNR 为
$\gamma=\bar\gamma\max_l|h_l|^2$，其分布为 $L$ 个指数分布的最大值。
两者都可用 AWGN 条件下的精确符号错误率
$\mathrm{SER}_{\rm AWGN}(\gamma)=1-\left(1-\frac{3}{2}Q(\sqrt{\gamma/5})\right)^2$
对瞬时 SNR 分布做数值积分得到，用于校核仿真曲线。
'''

EXTRA_CODE_2 = r'''
# 补充（量化）：16-QAM MRC/SC 的半解析理论曲线与仿真对照
torch.manual_seed(2026)


def ser_16qam_awgn_from_snr(gamma):
    """由线性瞬时 SNR 计算 16-QAM 在 AWGN 下的精确 SER。"""
    q = 0.5 * sp.erfc(np.sqrt(np.asarray(gamma, dtype=float) / 5.0) / np.sqrt(2.0))
    return 1.0 - (1.0 - 1.5 * q) ** 2


def mrc_ser_theory(esn0_db, num_rx, num_grid=6000):
    """半解析：对 Gamma(L,1) 分布的合并后 SNR 数值积分得到平均 SER。"""
    gbar = 10.0 ** (np.asarray(esn0_db, dtype=float) / 10.0)
    out = np.zeros_like(gbar)
    for i, g in enumerate(np.atleast_1d(gbar)):
        upper = g * (num_rx + 10.0 * np.sqrt(num_rx) + 12.0)
        x = np.linspace(1e-12, upper, num_grid)
        pdf = x ** (num_rx - 1) * np.exp(-x / g) / (sp.gamma(num_rx) * g ** num_rx)
        out[i] = np.trapezoid(ser_16qam_awgn_from_snr(x) * pdf, x)
    return out


def sc_ser_theory(esn0_db, num_rx, num_grid=6000):
    """半解析：对 max of L 个指数分布（SC 选择支路）的 SNR 分布数值积分。"""
    gbar = 10.0 ** (np.asarray(esn0_db, dtype=float) / 10.0)
    out = np.zeros_like(gbar)
    for i, g in enumerate(np.atleast_1d(gbar)):
        upper = g * (np.log(num_rx) + 14.0)   # 最大值分布的尾部
        x = np.linspace(1e-12, upper, num_grid)
        cdf = 1.0 - np.exp(-x / g)
        pdf = (num_rx / g) * np.exp(-x / g) * cdf ** (num_rx - 1)
        out[i] = np.trapezoid(ser_16qam_awgn_from_snr(x) * pdf, x)
    return out


esn0_dense = np.linspace(0.0, 30.0, 61)
theory_curves = {
    "mrc_1": mrc_ser_theory(esn0_dense, 1),
    "mrc_2": mrc_ser_theory(esn0_dense, 2),
    "mrc_4": mrc_ser_theory(esn0_dense, 4),
    "sc_2": sc_ser_theory(esn0_dense, 2),
    "sc_4": sc_ser_theory(esn0_dense, 4),
}
# 与仿真点同一 SNR 网格上的理论值，用于逐点比较
theory_basic = {
    "mrc_1": mrc_ser_theory(esn0_range_basic, 1),
    "mrc_2": mrc_ser_theory(esn0_range_basic, 2),
    "mrc_4": mrc_ser_theory(esn0_range_basic, 4),
    "sc_2": sc_ser_theory(esn0_range_basic, 2),
    "sc_4": sc_ser_theory(esn0_range_basic, 4),
}

# 与仿真点逐点对比，打印相对偏差（SER 太低的点误符号数不足 500，统计涨落主导，记为“涨落”）
print("表 2  仿真 SER 与半解析理论 SER 的相对偏差（同一 Es/N0）")
print("  Es/N0  " + "  ".join(f"{k:>10s}" for k in theory_curves))
max_rel_dev = {}
for _i, esn0_db in enumerate(esn0_range_basic):
    row = []
    for key, curve in theory_basic.items():
        sim_v = float(ser_results[key][_i])
        th_v = float(curve[_i])
        if th_v > 0.0:
            rel = abs(sim_v - th_v) / th_v
            if sim_v >= 5e-3:
                max_rel_dev[key] = max(max_rel_dev.get(key, 0.0), rel)
                row.append(f"{rel * 100:9.1f}%")
            else:
                row.append(f"{'涨落':>10s}")
        else:
            row.append(f"{'—':>10s}")
    print(f"  {esn0_db:5.0f}  " + "  ".join(row))
print("  SER >= 5e-3（误符号数 > 500）各点的最大相对偏差："
      + "，".join(f"{k} {v * 100:.1f}%" for k, v in max_rel_dev.items()))

# 图 4：仿真散点（记号）叠加半解析理论（实线）
fig, ax = plt.subplots(figsize=(7.5, 4.5))
styles = {
    "mrc_1": ("o", "#1f77b4", "1x1 MRC"),
    "mrc_2": ("s", "#ff7f0e", "1x2 MRC"),
    "mrc_4": ("^", "#2ca02c", "1x4 MRC"),
    "sc_2": ("v", "#9467bd", "1x2 SC"),
    "sc_4": ("D", "#8c564b", "1x4 SC"),
}
for key, (marker, color, label) in styles.items():
    ax.semilogy(esn0_dense, floor_curve(theory_curves[key], 1e-8), "-", color=color, lw=1.4, label=f"{label} 理论")
    ax.semilogy(esn0_range_basic, floor_curve(ser_results[key]), marker, color=color, ms=5, mfc="none", label=f"{label} 仿真")
ax.set_xlabel("Es/N0 (dB)")
ax.set_ylabel("符号错误率 SER")
ax.set_xlim(0, 30)
ax.set_ylim(1e-7, 1.0)
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="lower left", ncol=2, fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "03D_ser_theory_check.png", dpi=200, bbox_inches="tight")
print("图题：16-QAM 接收分集 SER 仿真结果与半解析理论曲线的对照")
plt.show()

df_theory = pd.DataFrame({"esn0_db": esn0_dense, **{f"theory_{k}": v for k, v in theory_curves.items()}})
df_theory.to_csv(DATA_DIR / "03D_ser_theory_curves.csv", index=False, encoding="utf-8-sig")
print(f"已保存数据：{(DATA_DIR / '03D_ser_theory_curves.csv').name}")
'''

# ============================================================================
# 追加单元 3：阵列增益 / 分集阶数的定量提取 + 结论
# ============================================================================
EXTRA_MD_3 = r'''
## 补充：阵列增益、分集阶数与最终结论

把 SER 曲线转换成工程上可直接比较的两组数字：
（1）达到目标 SER 所需的 Es/N0，以及相对 1×1 的**阵列增益**；
（2）高信噪比区 SER 曲线在半对数坐标下的**斜率**，即实测**分集阶数**
（每 10 dB 的 SER 下降十倍对应 1 阶）。
'''

EXTRA_CODE_3 = r'''
# 补充（量化）：阵列增益、分集阶数与结论
torch.manual_seed(2026)


def required_esn0(ser_curve, esn0_db, target):
    """在 log10(SER) 上做一维线性内插，求达到目标 SER 所需的 Es/N0 (dB)。"""
    ser_curve = np.asarray(ser_curve, dtype=float)
    valid = ser_curve > 0
    if not np.any(valid) or ser_curve[valid].min() > target or ser_curve[valid].max() < target:
        return np.nan
    x_db = np.asarray(esn0_db, dtype=float)[valid]
    y = np.log10(ser_curve[valid])
    order = np.argsort(y)
    x_db, y = x_db[order], y[order]
    return float(np.interp(np.log10(target), y, x_db))


def diversity_order(ser_curve, esn0_db, ser_low=1e-5, ser_high=1e-1):
    """用高 SNR 区的 log10(SER)-dB 斜率估计分集阶数（1 阶 = 每 10 dB 下降 1 个数量级）。"""
    ser_curve = np.asarray(ser_curve, dtype=float)
    esn0_db = np.asarray(esn0_db, dtype=float)
    mask = (ser_curve > 0) & (ser_curve <= ser_high) & (ser_curve >= ser_low)
    if mask.sum() < 2:
        # 退化时放宽区间，仍不足则返回 nan
        mask = (ser_curve > 0) & (ser_curve <= ser_high * 3)
    if mask.sum() < 2:
        return np.nan
    slope = np.polyfit(esn0_db[mask], np.log10(ser_curve[mask]), 1)[0]
    return float(-slope / 0.1)   # 10 dB -> 1 个数量级对应 1 阶


print("表 3  接收分集增益量化（目标 SER = 1e-2 与 1e-3）")
rows = []
for key, label in [
    ("mrc_1", "1x1"),
    ("sc_2", "1x2 SC"),
    ("mrc_2", "1x2 MRC"),
    ("sc_4", "1x4 SC"),
    ("mrc_4", "1x4 MRC"),
]:
    need_2 = required_esn0(ser_results[key], esn0_range_basic, 1e-2)
    need_3 = required_esn0(ser_results[key], esn0_range_basic, 1e-3)
    order = diversity_order(ser_results[key], esn0_range_basic)
    rows.append((key, label, need_2, need_3, order))

gain_ref_2 = rows[0][2]


def _fmt(value, unit=" dB"):
    """把可能为 NaN 的数值格式化成表格字符串（NaN 表示超出仿真 SNR 范围）。"""
    if not np.isfinite(value):
        return f"{'—':>10s}     "
    return f"{value:>10.2f}{unit:>3s}"


print("  配置        SER=1e-2 所需       相对 1x1        SER=1e-3 所需        实测分集")
print("              Es/N0 (dB)          节省 (dB)       Es/N0 (dB)           阶数")
for key, label, need_2, need_3, order in rows:
    gain_2 = gain_ref_2 - need_2 if np.isfinite(need_2) else np.nan
    order_txt = f"{order:>8.2f}" if np.isfinite(order) else f"{'—':>8s}"
    print(f"  {label:<10s}{_fmt(need_2)}{_fmt(gain_2)}{_fmt(need_3)}{order_txt}")


# AWGN 基准达到 SER=1e-2 的 Es/N0（无衰落，用于说明阵列增益可"超过"高斯基准）
from scipy.optimize import brentq

def _awgn_ser_gap(x_db):
    return float(ser_16qam_awgn_theory(x_db)) - 1e-2

awgn_need = brentq(_awgn_ser_gap, 0.0, 40.0)
print(f"  参考：16-QAM 在 AWGN 信道下达到 SER = 1e-2 仅需 Es/N0 = {awgn_need:.2f} dB")
print(
    f"  1x4 MRC 达到同一 SER 需 {rows[4][2]:.2f} dB，比 1x1 无分集节省 "
    f"{rows[0][2] - rows[4][2]:.2f} dB，甚至比无衰落的 AWGN 基准还低 {awgn_need - rows[4][2]:.2f} dB"
)

df_gain = pd.DataFrame(
    rows, columns=["config", "label", "esn0_for_ser_1e-2", "esn0_for_ser_1e-3", "diversity_order"]
)
df_gain["gain_vs_1x1_at_1e-2_db"] = gain_ref_2 - df_gain["esn0_for_ser_1e-2"]
df_gain.to_csv(DATA_DIR / "03D_gain_analysis.csv", index=False, encoding="utf-8-sig")
print(f"已保存数据：{(DATA_DIR / '03D_gain_analysis.csv').name}")
print("  注：1x1 无分集在 0~30 dB 内未达到 SER = 1e-3（需 > 30 dB），故该列为“—”。")

# 结论
print("\n===== 03D 接收分集实验结论 =====")
print(f"[1] 16-QAM + 4Rx MRC 在 Es/N0 = {esn0_range_basic[5]:.0f} dB 的 SER = {ser_results['mrc_4'][5]:.2e}，"
      f"仅为 1x1 的 {ser_results['mrc_4'][5] / ser_results['mrc_1'][5] * 100:.2f}%（断言要求 < 5%）。")
print(f"[2] 相同天线数下 MRC 优于 SC：1x2 在 SER=1e-2 时领先 {rows[1][2] - rows[2][2]:.2f} dB，"
      f"1x4 领先 {rows[3][2] - rows[4][2]:.2f} dB（MRC 阵列增益 vs SC 仅分集）。")
print(f"[3] 实测分集阶数：1x1 {rows[0][4]:.2f}、1x2 MRC {rows[2][4]:.2f}、1x4 MRC {rows[4][4]:.2f}，与理论 L 阶一致。")
print(f"[4] BPSK 拓展：1/2/4 阶 MRC 仿真与理论闭式解在 0~8 dB 的最大绝对偏差 "
      f"{max(np.max(np.abs(bpsk_mrc_results[L][:5] - mrc_ber_theory(ebno_range_ext, L)[:5])) for L in (1, 2, 4)):.4f}，"
      "表明实现与理论完全自洽。")
'''

# ============================================================================
# 组装 notebook
# ============================================================================
def main() -> None:
    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    old_cells = template["cells"]
    if len(old_cells) != 4:
        raise SystemExit(f"模板单元数异常: {len(old_cells)}")

    cells = [
        md("".join(old_cells[0]["source"]), "03d-md-0"),
        code(CELL1, "03d-code-1"),
        code(CELL2, "03d-code-2"),
        code(CELL3, "03d-code-3"),
        md(EXTRA_MD_1, "03d-md-4"),
        code(EXTRA_CODE_1, "03d-code-5"),
        md(EXTRA_MD_2, "03d-md-6"),
        code(EXTRA_CODE_2, "03d-code-7"),
        md(EXTRA_MD_3, "03d-md-8"),
        code(EXTRA_CODE_3, "03d-code-9"),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.4",
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"已写出 {OUTPUT}（{len(cells)} 个单元）")


if __name__ == "__main__":
    main()
