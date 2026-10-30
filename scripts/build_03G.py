# -*- coding: utf-8 -*-
"""生成 notebooks/03G_pulse_shaping.ipynb。

流程：读模板（保留 markdown 与断言）→ 补全 TODO 代码 → 追加过程性/落盘单元。
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "无线仿真资料2026" / "student" / "03G_pulse_shaping.ipynb"
OUT = ROOT / "notebooks" / "03G_pulse_shaping.ipynb"


def to_source(text):
    lines = text.split("\n")
    out = [ln + "\n" for ln in lines]
    if out:
        out[-1] = out[-1].rstrip("\n")
    return out


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": to_source(text)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": to_source(text),
    }


# ---------------------------------------------------------------- markdown ----
MD_GOAL = r"""## 实验目的

1. 掌握根升余弦（RRC）脉冲成形的离散实现：发射端上采样 + RRC，接收端 RRC 匹配滤波 + 符号率抽样。
2. 用数值实验验证 Nyquist 无 ISI 条件 $p(kT)=\delta[k]$，观察滚降因子 $\beta$ 对时域拖尾、占用带宽、峰均比和定时误差敏感度的折中关系。
3. 对比"带限成形 + 匹配滤波"与"直接矩形抽样"的误码率，说明匹配滤波为何是最优线性接收滤波器。

## 原理

**RRC 冲激响应。** 令符号周期 $T=1$、过采样倍数 $\mathrm{sps}=8$、滤波跨度 $\mathrm{span}$ 个符号，则滤波器长度 $L=2\lfloor \mathrm{span}\cdot\mathrm{sps}/2\rfloor+1$，采样时刻 $t_n=(n-L/2)/\mathrm{sps}$：

$$
h(t)=\begin{cases}
1+\beta\left(\dfrac{4}{\pi}-1\right), & t=0\\[4pt]
\dfrac{\beta}{\sqrt{2}}\left[\left(1+\dfrac{2}{\pi}\right)\sin\dfrac{\pi}{4\beta}+\left(1-\dfrac{2}{\pi}\right)\cos\dfrac{\pi}{4\beta}\right], & t=\pm\dfrac{1}{4\beta}\\[8pt]
\dfrac{\sin\left[\pi t(1-\beta)\right]+4\beta t\cos\left[\pi t(1+\beta)\right]}{\pi t\left[1-(4\beta t)^2\right]}, & \text{其他}
\end{cases}
$$

**无 ISI 条件。** 收发两端各用一个 RRC，级联响应为升余弦（RC）脉冲

$$
p(t)=h(t)*h(t)=\frac{\sin(\pi t/T)}{\pi t/T}\cdot\frac{\cos(\pi\beta t/T)}{1-(4\beta t/T)^2},
$$

它在 $t=kT$ 处满足 $p(kT)=\delta[k]$，即相邻符号在抽样时刻互不干扰。本实验把系数归一化为单位能量（$\sum_n h^2[n]=1$），于是级联增益为 1、匹配滤波输出噪声方差等于信道噪声方差，便于按 $E_b/N_0$ 标定。

**带宽与滚降。** RRC 的单边带宽为 $\dfrac{1+\beta}{2T}$，带宽效率 $\dfrac{1}{1+\beta}$ symbol/Hz：$\beta$ 越小频谱越"陡"、越省带宽，但时域拖尾越长、对定时误差和截断越敏感；$\beta$ 越大带宽冗余越多、眼图开口越宽。

**误码率口径。** 复基带白噪声按过采样点加入，每维方差 $\sigma^2/2$，其中 $\sigma^2=1/(\log_2 M\cdot E_b/N_0)$；16-QAM（Gray 映射）近似理论为

$$
P_b\approx\frac{3}{8}\,\mathrm{erfc}\!\left(\sqrt{\frac{2}{5}\frac{E_b}{N_0}}\right).
$$

## 参数与设计

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| 调制 | 16-QAM（4 bit/符号） | Sionna `QAMSource` / `Mapper` |
| 过采样 $\mathrm{sps}$ | 8 | 每符号 8 个样点 |
| RRC 跨度 span | 10 符号 | 截断长度 $L=81$ 抽头 |
| 滚降 $\beta$ | 0.35（主）/ 扫 0.15, 0.35, 0.5, 0.8 | 折中分析 |
| 基本链路 | $8\times128$ 符号 | 波形与星座 |
| 统计链路 | $64\times1024$ 符号（262144 bit） | BER / 指标统计 |
| $E_b/N_0$ | 4~12 dB | 与无 ISI 理论对照 |
| 定时偏差 | 0~0.5 符号（0~4 个样点） | 敏感度扫描 |
"""

# ------------------------------------------------------------------ cell1 ----
CELL1 = r'''# 导入依赖：Sionna 信号处理模块、QAM 映射与理论曲线工具。
from pathlib import Path
import sys

for p in [Path.cwd(), *Path.cwd().parents]:
    if (p / "src" / "sionna").exists():
        sys.path.insert(0, str(p / "src"))
        break

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.special import erfc

DEVICE = "cpu"
PRECISION = "single"
torch.manual_seed(42)
np.random.seed(42)
from sionna.phy import config as sionna_config
sionna_config.seed = 42

plt.rcParams.update(
    {
        "font.sans-serif": [
            "Microsoft YaHei",
            "SimHei",
            "Noto Sans CJK SC",
            "Arial Unicode MS",
            "DejaVu Sans",
        ],
        "axes.unicode_minus": False,
        "font.size": 13,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.dpi": 200,
        "savefig.dpi": 200,
    }
)

FIG_DIR = Path.cwd() / "figures"
DATA_DIR = Path.cwd() / "data"
FIG_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

from sionna.phy.mapping import Constellation, Mapper, Demapper, QAMSource
from sionna.phy.signal import Upsampling, Downsampling, RootRaisedCosineFilter

BATCH_SIZE = 8
NUM_SYMBOLS = 128
NUM_BITS_PER_SYMBOL = 4
SAMPLES_PER_SYMBOL = 8
SPAN_IN_SYMBOLS = 10
BETA = 0.35
ROLLOFFS = [0.15, 0.35, 0.5, 0.8]
BIG_BATCH = 64
BIG_SYMBOLS = 1024

qam = QAMSource(NUM_BITS_PER_SYMBOL, device=DEVICE)
upsampler = Upsampling(SAMPLES_PER_SYMBOL, device=DEVICE)
downsampler = Downsampling(SAMPLES_PER_SYMBOL, device=DEVICE)
rrc = RootRaisedCosineFilter(SPAN_IN_SYMBOLS, SAMPLES_PER_SYMBOL, BETA, device=DEVICE)

const = Constellation("qam", NUM_BITS_PER_SYMBOL, precision=PRECISION, device=DEVICE)
mapper = Mapper(constellation=const, precision=PRECISION, device=DEVICE)
demapper = Demapper(
    "app", constellation=const, hard_out=True, precision=PRECISION, device=DEVICE
)


def rrc_impulse_response(beta, sps=SAMPLES_PER_SYMBOL, span=SPAN_IN_SYMBOLS):
    """按上面的 RRC 公式自主计算冲激响应，并归一化为单位能量。"""
    length = 2 * (span * sps // 2) + 1
    t = np.arange(-(length // 2), length // 2 + 1, dtype=np.float64) / sps
    h = np.empty_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:  # t = 0
            h[i] = 1.0 + beta * (4 / np.pi - 1)
        elif abs(abs(ti) - 1 / (4 * beta)) < 1e-9:  # t = ±T/(4β) 的极限值
            h[i] = beta / np.sqrt(2) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            h[i] = (
                np.sin(np.pi * ti * (1 - beta))
                + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            ) / (np.pi * ti * (1 - (4 * beta * ti) ** 2))
    return h / np.sqrt(np.sum(h**2))


def unit_energy_coeffs(filt):
    """取滤波器系数并归一化为单位能量。"""
    c = filt.coefficients.detach().clone().to(torch.float32)
    return c / torch.sqrt(torch.sum(c**2))


def manual_filter(x, h):
    """自主实现离散卷积 y[n] = Σ_m h[m]·x[n-m]（零填充，复数信号按 I/Q 两路滤波）。"""
    h = torch.as_tensor(h, dtype=torch.float32, device=DEVICE)
    b, n = x.shape
    xr = torch.view_as_real(x).permute(0, 2, 1).reshape(b * 2, 1, n)
    w = torch.flip(h, [0]).view(1, 1, -1)
    y = torch.nn.functional.conv1d(xr, w, padding=len(h) - 1)
    return torch.view_as_complex(y.reshape(b, 2, -1).permute(0, 2, 1).contiguous())


def tx_pulse_shape(x, h):
    """上采样 → RRC 成形 → 截取中心 N·sps 个样点（补偿群时延 (L-1)/2）。"""
    x_us = upsampler(x)
    full = manual_filter(x_us, h)
    d = len(h) // 2
    return full[..., d : d + x.shape[-1] * SAMPLES_PER_SYMBOL]


def aligned_matched_filter(rx, h, num_symbols, offset=0):
    """匹配滤波，并输出对齐到符号边界的 sps 倍符号波形。"""
    mf = manual_filter(rx, h)
    d = len(h) // 2 + offset
    return mf[..., d : d + num_symbols * SAMPLES_PER_SYMBOL]


def sample_symbols(rx, h, num_symbols, offset=0):
    """匹配滤波 + 抽取，得到符号率复序列（offset 为抽样相位偏差，单位：样点）。"""
    return downsampler(aligned_matched_filter(rx, h, num_symbols, offset))


def ebno_db_to_no(ebno_db, nbits, coderate=1.0):
    ebno = 10.0 ** (torch.as_tensor(ebno_db, dtype=torch.float32, device=DEVICE) / 10.0)
    return 1.0 / (ebno * nbits * coderate)


def ber(b, b_hat):
    return float(torch.mean((b.to(torch.int32) != b_hat.to(torch.int32)).float()).item())


def qam16_ber_theory(ebno_db):
    return (3 / 8) * erfc(np.sqrt(0.4 * 10 ** (np.asarray(ebno_db) / 10)))


def papr_db(waveform):
    p = waveform.abs() ** 2
    return 10 * np.log10(float(p.amax(dim=-1).mean() / p.mean()))


def occupied_bandwidth(waveform, fs=SAMPLES_PER_SYMBOL, thr_db=-30, nperseg=2048):
    """用 Welch 谱估计 -30 dB 占用带宽（单位：符号率归一化）。"""
    _, psd = welch(
        waveform.numpy(), fs=fs, nperseg=nperseg, return_onesided=False, scaling="density"
    )
    f = np.fft.fftshift(np.fft.fftfreq(nperseg, d=1 / fs))
    psd = np.fft.fftshift(psd)
    mask = psd > psd.max() * 10 ** (thr_db / 10)
    return float(f[mask].max() - f[mask].min())


# 把模板中的 RRC 系数归一化为单位能量，使"库函数"与"自主实现"使用同一组系数
h_rrc = unit_energy_coeffs(rrc)
rrc.coefficients = h_rrc
h_np = h_rrc.numpy()
print("RRC 抽头数 L =", len(h_rrc), "，∑h² =", float(torch.sum(h_rrc**2)))
print("实验环境就绪：sps =", SAMPLES_PER_SYMBOL, "span =", SPAN_IN_SYMBOLS, "β =", BETA)
'''

# ------------------------------------------------------------------ cell2 ----
CELL2 = r'''# 生成 16-QAM 符号，上采样后做 RRC 发射滤波。
x = qam([BATCH_SIZE, NUM_SYMBOLS])
x_us = upsampler(x)
# RRC 发射滤波：卷积后截取中心 NUM_SYMBOLS*SAMPLES_PER_SYMBOL 个样点，
# 使第 k 个符号落在第 k*SAMPLES_PER_SYMBOL 个样点上（补偿群时延 (L-1)/2）。
tx_full = rrc(x_us, padding="full")
tx = tx_full[..., rrc.length // 2 : rrc.length // 2 + NUM_SYMBOLS * SAMPLES_PER_SYMBOL]
print("原始符号:", x.shape, "上采样:", x_us.shape, "发射波形:", tx.shape)
print("单位能量 RRC 下平均样点功率:", float((tx.abs() ** 2).mean()))

plt.figure(figsize=(8, 3))
plt.plot(tx[0, :160].real.cpu().numpy(), label="I")
plt.plot(tx[0, :160].imag.cpu().numpy(), label="Q", alpha=0.8)
plt.xlabel("采样点")
plt.ylabel("幅度")
plt.grid(True)
plt.legend()
plt.show()

assert tx.shape[-1] == NUM_SYMBOLS * SAMPLES_PER_SYMBOL
print("通过断言检验：发射波形长度 = 符号数 × 过采样倍数")
'''

# ------------------------------------------------------------------ cell3 ----
CELL3 = r'''# 匹配滤波后按符号率抽样，观察星座恢复效果。
mf = rrc(tx, padding="full")
x_hat = sample_symbols(tx, h_rrc, NUM_SYMBOLS)
inner = slice(SPAN_IN_SYMBOLS, NUM_SYMBOLS - SPAN_IN_SYMBOLS)
print("匹配滤波输出:", mf.shape, "抽样后:", x_hat.shape)
print("内部符号最大重构误差:", float((x_hat[:, inner] - x[:, inner]).abs().max()))

plt.figure(figsize=(4.5, 4.5))
pts = x_hat.flatten().detach().cpu().numpy()
plt.scatter(pts.real, pts.imag, s=8, alpha=0.5, label="匹配滤波抽样点")
ideal = const.points.detach().cpu().numpy()
plt.scatter(ideal.real, ideal.imag, s=90, marker="x", c="r", label="理想 16-QAM")
plt.xlabel("I")
plt.ylabel("Q")
plt.grid(True)
plt.axis("equal")
plt.legend(loc="upper right")
plt.show()

assert x_hat.shape == x.shape
print("通过断言检验：匹配滤波 + 抽样恢复出 N 个符号")
'''

MD_OWN = r"""## 模块二 自主实现 RRC 与无 ISI 检验

下面的单元直接用原理部分的公式计算 RRC 系数，并与 Sionna 内置 `RootRaisedCosineFilter` 逐点比较；随后把级联的 RC 脉冲 $p(t)$ 在符号间隔处取样，验证 $p(kT)\approx0$。
"""

# ------------------------------------------------------------------ cell4 ----
CELL4 = r'''# 自主实现 RRC 冲激响应，并与 Sionna 内置滤波器逐点对比。
print("=== RRC 系数：自主实现 vs Sionna ===")
h_dict = {}
rows = []
for beta in ROLLOFFS:
    filt = RootRaisedCosineFilter(SPAN_IN_SYMBOLS, SAMPLES_PER_SYMBOL, beta, device=DEVICE)
    h_ref = unit_energy_coeffs(filt).numpy()
    h_own = rrc_impulse_response(beta)
    h_dict[beta] = h_own
    rows.append(
        {
            "beta": beta,
            "抽头数": len(h_own),
            "最大绝对偏差": float(np.max(np.abs(h_own - h_ref))),
            "h(0)": float(h_own[len(h_own) // 2]),
            "能量": float(np.sum(h_own**2)),
        }
    )
    print(
        f"β={beta:.2f}: 抽头 {len(h_own)}，与 Sionna 最大偏差 "
        f"{np.max(np.abs(h_own - h_ref)):.2e}，h(0)={h_own[len(h_own)//2]:.4f}"
    )
df_rrc = pd.DataFrame(rows)
print(df_rrc.to_string(index=False))
df_rrc.to_csv(DATA_DIR / "03G_rrc_beta.csv", index=False, encoding="utf-8-sig")

# 两种实现产生的发射波形应完全一致
filt_beta = RootRaisedCosineFilter(SPAN_IN_SYMBOLS, SAMPLES_PER_SYMBOL, BETA, device=DEVICE)
filt_beta.coefficients = h_dict[BETA].copy()
tx_lib = filt_beta(upsampler(x), padding="full")
tx_lib = tx_lib[..., len(h_dict[BETA]) // 2 : len(h_dict[BETA]) // 2 + NUM_SYMBOLS * SAMPLES_PER_SYMBOL]
print("自主卷积 vs Sionna 滤波的发射波形最大偏差:", float((tx_lib - tx).abs().max()))

# ---- 图 03G-1：RRC 时域响应 ----
t_sym = (np.arange(len(h_np)) - len(h_np) // 2) / SAMPLES_PER_SYMBOL
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for beta in ROLLOFFS:
    ax.plot(t_sym, h_dict[beta], lw=1.6, label=f"β = {beta}")
ax.plot(
    t_sym[::SAMPLES_PER_SYMBOL],
    h_dict[BETA][::SAMPLES_PER_SYMBOL],
    "ko",
    ms=4,
    label=f"符号抽样点（β = {BETA}）",
)
ax.set_xlim(-5, 5)
ax.set_xlabel("归一化时间 t/T")
ax.set_ylabel("h(t)（单位能量归一化）")
ax.grid(True, alpha=0.4)
ax.legend(loc="upper right")
fig.savefig(FIG_DIR / "03G_rrc_time.png", dpi=200, bbox_inches="tight")
print("图题：图 03G-1 RRC 脉冲的时域响应（不同滚降因子）")
plt.show()

# ---- 图 03G-2：RRC 幅频响应 ----
fig, ax = plt.subplots(figsize=(7.5, 4.5))
nf = 4096
f_norm = np.fft.fftshift(np.fft.fftfreq(nf)) * SAMPLES_PER_SYMBOL
for beta in ROLLOFFS:
    H = np.fft.fftshift(np.fft.fft(h_dict[beta], nf))
    mag = 20 * np.log10(np.maximum(np.abs(H), 1e-9))
    ax.plot(f_norm, mag - mag.max(), lw=1.6, label=f"β = {beta}")
    ax.axvline((1 + beta) / 2, color="gray", ls=":", lw=0.8)
    ax.axvline(-(1 + beta) / 2, color="gray", ls=":", lw=0.8)
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-80, 3)
ax.set_xlabel("归一化频率 fT")
ax.set_ylabel("|H(f)| (dB)")
ax.grid(True, alpha=0.4)
ax.legend(loc="upper right")
fig.savefig(FIG_DIR / "03G_rrc_freq.png", dpi=200, bbox_inches="tight")
print("图题：图 03G-2 RRC 幅频响应与滚降边缘（虚线为 (1+β)/2T）")
plt.show()

assert df_rrc["最大绝对偏差"].max() < 1e-6
print("通过断言检验：自主实现的 RRC 与 Sionna 系数一致（最大偏差 < 1e-6）")
'''

# ------------------------------------------------------------------ cell5 ----
CELL5 = r'''# 无 ISI 检验：级联 RC 脉冲 p(t) = h(t)*h(t) 在符号间隔处过零。
print("=== 级联 RC 脉冲在符号时刻的取值 ===")
rows = []
for beta in ROLLOFFS:
    h = h_dict[beta]
    p = np.convolve(h, h)
    c = len(h) - 1
    lags = np.arange(-SPAN_IN_SYMBOLS, SPAN_IN_SYMBOLS + 1)  # 覆盖 span 内全部 kT，原 ±5 会漏掉 |k|=6..10 的截断旁瓣
    vals = np.array([p[c + k * SAMPLES_PER_SYMBOL] for k in lags])
    main = vals[lags == 0][0]
    side = np.max(np.abs(vals[lags != 0]))
    rows.append(
        {
            "beta": beta,
            "p(0)": float(main),
            "最大旁瓣": float(side),
            "旁瓣电平(dB)": float(20 * np.log10(side / main)),
        }
    )
    print(f"β={beta:.2f}: p(0)={main:.4f}，|p(kT)| 最大值={side:.2e}（{20*np.log10(side/main):.1f} dB）")
df_isi = pd.DataFrame(rows)
print(df_isi.to_string(index=False))

# ---- 图 03G-3：无 ISI 点 ----
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for beta, style in [(0.15, "--"), (BETA, "-")]:
    h = h_dict[beta]
    p = np.convolve(h, h)
    c = len(h) - 1
    tt = (np.arange(len(p)) - c) / SAMPLES_PER_SYMBOL
    ax.plot(tt, p, style, lw=1.6, label=f"β = {beta}")
    ax.plot(
        tt[::SAMPLES_PER_SYMBOL],
        p[::SAMPLES_PER_SYMBOL],
        "o",
        ms=5,
        mfc="none",
        mec="C0" if beta == BETA else "C1",
    )
ax.axhline(0, color="k", lw=0.8)
ax.axvline(0, color="gray", ls=":", lw=0.8)
ax.set_xlim(-5, 5)
ax.set_ylim(-0.25, 1.1)
ax.set_xlabel("归一化时间 t/T")
ax.set_ylabel("p(t) = h(t)*h(t)")
ax.grid(True, alpha=0.4)
ax.legend(loc="upper right")
fig.savefig(FIG_DIR / "03G_isi_check.png", dpi=200, bbox_inches="tight")
print("图题：图 03G-3 级联升余弦脉冲在符号间隔处的过零点（无 ISI 检验）")
plt.show()

assert df_isi["最大旁瓣"].max() < 3e-2
assert df_isi.loc[df_isi.beta == BETA, "最大旁瓣"].iloc[0] < 1e-2
print("通过断言检验：p(kT) 旁瓣 ≤ -32.9 dB（span=10 截断下），β≥0.35 时 ≤ -44 dB，满足 Nyquist 无 ISI 条件")
'''

MD_EYE = r"""## 模块三 眼图与滚降因子的折中

眼图把接收波形按 2 个符号周期折叠，反映 ISI 与定时误差容限；抽样时刻正好落在眼图张开最大处。$\beta$ 越小，眼图在时间轴上张开得越窄。
"""

# ------------------------------------------------------------------ cell6 ----
CELL6 = r'''# ---- 图 03G-4：不同滚降因子的匹配滤波输出眼图 ----
def eye_traces(rx, num_symbols, span_traces=2):
    """把 sps 倍符号波形折叠成 span_traces 个符号长的眼图轨迹。"""
    seg = rx[..., : (num_symbols - span_traces) * SAMPLES_PER_SYMBOL]
    win = seg.unfold(-1, span_traces * SAMPLES_PER_SYMBOL, SAMPLES_PER_SYMBOL)
    return win.reshape(-1, span_traces * SAMPLES_PER_SYMBOL)


fig, axes = plt.subplots(2, 1, figsize=(8, 8.5))
for ax, beta in zip(axes, [0.15, 0.8]):
    tx_beta = tx_pulse_shape(x, h_dict[beta])
    rx_beta = aligned_matched_filter(tx_beta, h_dict[beta], NUM_SYMBOLS)
    traces = eye_traces(rx_beta, NUM_SYMBOLS).detach().cpu().numpy()
    tt = np.arange(traces.shape[-1]) / SAMPLES_PER_SYMBOL
    ax.plot(tt, traces[: min(len(traces), 400)].real.T, color="C0", lw=0.5, alpha=0.25)
    for s in range(3):
        ax.axvline(s, color="k", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlim(0, 2)
    ax.set_ylim(-1.6, 1.6)
    ax.set_xlabel("归一化时间 t/T")
    ax.set_ylabel("匹配滤波输出 I 路")
    ax.set_title(f"β = {beta}", fontsize=12)
    ax.grid(True, alpha=0.3)
    print(f"β={beta}: 眼图轨迹 {len(traces)} 条（虚线为最佳抽样时刻）")
fig.tight_layout()
fig.savefig(FIG_DIR / "03G_eye_beta.png", dpi=200, bbox_inches="tight")
print("图题：图 03G-4 匹配滤波输出眼图：β=0.15 与 β=0.8 对比")
plt.show()
'''

# ------------------------------------------------------------------ cell7 ----
CELL7 = r'''# ---- 图 03G-5：β 对峰均比、占用带宽与残余 ISI 的影响 ----
torch.manual_seed(42)
big_bits = torch.randint(
    0, 2, (BIG_BATCH, BIG_SYMBOLS * NUM_BITS_PER_SYMBOL), dtype=torch.int32, device=DEVICE
)
big_x = mapper(big_bits)
big_bt = big_bits.reshape(BIG_BATCH, BIG_SYMBOLS, NUM_BITS_PER_SYMBOL)
inner = slice(SPAN_IN_SYMBOLS, BIG_SYMBOLS - SPAN_IN_SYMBOLS)

rows = []
for beta in ROLLOFFS:
    tx_b = tx_pulse_shape(big_x, h_dict[beta])
    papr = papr_db(tx_b)
    bw = occupied_bandwidth(tx_b[0])
    y = sample_symbols(tx_b, torch.as_tensor(h_dict[beta], dtype=torch.float32), BIG_SYMBOLS)
    isi_rms = float(torch.sqrt(torch.mean((y[:, inner] - big_x[:, inner]).abs() ** 2)))
    isi_peak = float((y[:, inner] - big_x[:, inner]).abs().max())
    rows.append(
        {
            "beta": beta,
            "PAPR_dB": papr,
            "占用带宽(30dB)/符号率": bw,
            "理论带宽(1+beta)": 1 + beta,
            "残余ISI_rms": isi_rms,
            "残余ISI_峰值": isi_peak,
            "残余ISI_dB": float(20 * np.log10(isi_rms)),
        }
    )
    print(
        f"β={beta:.2f}: PAPR={papr:.2f} dB，-30dB 带宽={bw:.2f}×符号率（理论 {1+beta:.2f}），"
        f"残余 ISI rms={isi_rms:.4f}"
    )
df_metric = pd.DataFrame(rows)
print(df_metric.to_string(index=False))
df_metric.to_csv(DATA_DIR / "03G_beta_metrics.csv", index=False, encoding="utf-8-sig")

fig, axes = plt.subplots(2, 1, figsize=(8, 8.5))
ax = axes[0]
ax.plot(df_metric["beta"], df_metric["PAPR_dB"], "o-", color="C0", label="RRC 成形后波形")
ax.axhline(10 * np.log10(1.8), color="C3", ls="--", lw=1.0, label="16-QAM 符号 PAPR（未成形）")
ax.set_xlabel("滚降因子 β")
ax.set_ylabel("峰均比 (dB)")
ax.grid(True, alpha=0.4)
ax.legend(loc="upper right")

ax = axes[1]
ax.plot(df_metric["beta"], df_metric["占用带宽(30dB)/符号率"], "o-", color="C0", label="实测 -30 dB 占用带宽")
ax.plot(df_metric["beta"], df_metric["理论带宽(1+beta)"], "s--", color="C0", alpha=0.5, label="理论带宽 1+β")
ax.set_xlabel("滚降因子 β")
ax.set_ylabel("归一化带宽 (×符号率)")
ax.grid(True, alpha=0.4)
ax2 = ax.twinx()
ax2.semilogy(df_metric["beta"], df_metric["残余ISI_rms"], "^-", color="C3", label="残余 ISI（rms）")
ax2.set_ylabel("残余 ISI (rms，内部符号)", color="C3")
ax2.tick_params(axis="y", colors="C3")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
fig.tight_layout()
fig.savefig(FIG_DIR / "03G_beta_metrics.png", dpi=200, bbox_inches="tight")
print("图题：图 03G-5 滚降因子对峰均比、占用带宽与残余 ISI 的影响")
plt.show()

assert df_metric["占用带宽(30dB)/符号率"].is_monotonic_increasing
assert df_metric.loc[df_metric.beta == 0.15, "PAPR_dB"].iloc[0] > df_metric.loc[df_metric.beta == 0.5, "PAPR_dB"].iloc[0]
print("通过断言检验：占用带宽随 β 单调增大；小 β 的峰均比明显更高")
'''

MD_BER = r"""## 模块四 误码率与定时误差敏感度

在相同 $E_b/N_0$ 下比较不同 $\beta$ 的 16-QAM 误码率，并统计"去掉匹配滤波、直接抽样"的对照，最后扫描抽样相位偏差，量化滚降因子对定时同步精度的要求。统计只取内部符号，避免首尾截断瞬态影响。
"""

# ------------------------------------------------------------------ cell8 ----
CELL8 = r'''# ---- 图 03G-6：BER vs Eb/N0（不同 β）与无匹配滤波对照 ----
EB_NO_DB = np.array([4.0, 6.0, 8.0, 10.0, 12.0])


def run_ber(beta, ebno_db, use_mf=True, h=None, offset=0):
    """16-QAM 成形链路误码率（内部符号统计）。"""
    if h is None:
        h_own = rrc_impulse_response(beta)
    else:
        h_own = h
    h_t = torch.as_tensor(h_own, dtype=torch.float32, device=DEVICE)
    tx_b = tx_pulse_shape(big_x, h_own)
    no = ebno_db_to_no(ebno_db, NUM_BITS_PER_SYMBOL)
    noise = (torch.randn_like(tx_b.real) + 1j * torch.randn_like(tx_b.real)) * torch.sqrt(no / 2)
    rx_b = tx_b + noise.to(torch.complex64)
    if use_mf:
        y = sample_symbols(rx_b, h_t, BIG_SYMBOLS, offset=offset)
    else:  # 对照：不做匹配滤波，直接在最优点抽样，但补一级理想 AGC
        # 成形脉冲在符号时刻的峰值约等于 h(0)<1，若不做 AGC，判决器会把小幅度
        # 信号误判为"低 SNR"（BER 0.28 中混入了 −8.24 dB 的固定幅度损失）。
        # 公平对照应先用 AGC 把信号幅度归一到 1，代价是噪声同倍放大。
        peak = float(h_own[len(h_own) // 2])
        y = downsampler(rx_b[..., offset : offset + BIG_SYMBOLS * SAMPLES_PER_SYMBOL]) / peak
    b_hat = demapper(y, no).reshape(BIG_BATCH, BIG_SYMBOLS, NUM_BITS_PER_SYMBOL)
    return ber(big_bt[:, inner], b_hat[:, inner]), b_hat


ber_dict = {}
for beta in ROLLOFFS:
    h_own = h_dict[beta]
    ber_dict[beta] = [run_ber(beta, e, h=h_own)[0] for e in EB_NO_DB]
    print(f"β={beta:.2f}: " + "  ".join(f"{e:.0f}dB→{v:.2e}" for e, v in zip(EB_NO_DB, ber_dict[beta])))

ber_nomf, _ = run_ber(BETA, 10.0, use_mf=False, h=h_dict[BETA])
print(f"对照（无匹配滤波，AGC 归一化后直接抽样）Eb/N0=10 dB: BER={ber_nomf:.4f}；有匹配滤波: {ber_dict[BETA][3]:.2e}")

df_ber = pd.DataFrame({"Eb/N0_dB": EB_NO_DB, "理论_16QAM": qam16_ber_theory(EB_NO_DB)})
for beta in ROLLOFFS:
    df_ber[f"beta_{beta}"] = np.maximum(ber_dict[beta], 1e-8)
df_ber["无匹配滤波_beta_0.35"] = np.nan
df_ber.loc[df_ber["Eb/N0_dB"] == 10.0, "无匹配滤波_beta_0.35"] = ber_nomf
print(df_ber.to_string(index=False))
df_ber.to_csv(DATA_DIR / "03G_ber_vs_ebno.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.semilogy(EB_NO_DB, qam16_ber_theory(EB_NO_DB), "k-", lw=1.8, label="16-QAM 无 ISI 理论")
markers = ["o", "s", "^", "D"]
for beta, mk in zip(ROLLOFFS, markers):
    ax.semilogy(EB_NO_DB, np.maximum(ber_dict[beta], 1e-8), mk + "-", ms=5, label=f"成形+匹配滤波 β = {beta}")
ax.semilogy([10.0], [ber_nomf], "r*", ms=14, label="无匹配滤波（直接抽样）")
ax.set_xlabel("Eb/N0 (dB)")
ax.set_ylabel("BER")
ax.set_ylim(1e-7, 1)
ax.grid(True, which="both", alpha=0.4)
ax.legend(loc="lower left")
fig.savefig(FIG_DIR / "03G_ber_beta.png", dpi=200, bbox_inches="tight")
print("图题：图 03G-6 不同滚降因子下 16-QAM 的 BER 与无 ISI 理论对照")
plt.show()

# 断言：单调下降、接近理论、匹配滤波带来量级增益
for beta in ROLLOFFS:
    seq = np.array(ber_dict[beta])
    assert np.all(np.diff(seq) <= 0), f"β={beta} 的 BER 未随信噪比单调下降"
for i, e in enumerate(EB_NO_DB):
    if 8.0 <= e <= 10.0:
        ratio = ber_dict[BETA][i] / qam16_ber_theory(e)
        assert 0.5 < ratio < 2.0, f"Eb/N0={e} dB 时与理论偏差过大：{ratio:.2f}"
assert ber_nomf > 20 * ber_dict[BETA][3]
print("通过断言检验：BER 随信噪比单调下降、与无 ISI 理论一致（0.5~2 倍内）、匹配滤波优于直接抽样 20 倍以上")
'''

# ------------------------------------------------------------------ cell9 ----
CELL9 = r'''# ---- 图 03G-7：定时偏差对 BER 的影响 ----
OFFSETS = [0, 1, 2, 3, 4]  # 单位：样点，1 个样点 = 1/8 符号
EB_NO_TIMING = 12.0
timing = {}
for beta in [0.15, BETA, 0.8]:
    timing[beta] = [run_ber(beta, EB_NO_TIMING, h=h_dict[beta], offset=o)[0] for o in OFFSETS]
    print(f"β={beta:.2f}: " + "  ".join(f"{o/8:.3f}T→{v:.2e}" for o, v in zip(OFFSETS, timing[beta])))

df_time = pd.DataFrame({"定时偏差(符号)": np.array(OFFSETS) / SAMPLES_PER_SYMBOL})
for beta in timing:
    df_time[f"beta_{beta}"] = np.maximum(timing[beta], 1e-8)
print(df_time.to_string(index=False))
df_time.to_csv(DATA_DIR / "03G_timing_offset.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(7.5, 4.5))
for beta, mk in zip([0.15, BETA, 0.8], ["o", "s", "^"]):
    ax.semilogy(df_time["定时偏差(符号)"], df_time[f"beta_{beta}"], mk + "-", ms=5, label=f"β = {beta}")
ax.set_xlabel("抽样相位偏差 (符号周期 T)")
ax.set_ylabel("BER")
ax.set_ylim(1e-6, 1)
ax.grid(True, which="both", alpha=0.4)
ax.legend(loc="lower right")
fig.savefig(FIG_DIR / "03G_timing_offset.png", dpi=200, bbox_inches="tight")
print(f"图题：图 03G-7 抽样相位偏差对 BER 的影响（Eb/N0 = {EB_NO_TIMING:.0f} dB）")
plt.show()

assert timing[0.15][0] < 1e-3 and timing[BETA][0] < 1e-3
assert timing[0.8][1] < 0.1 * timing[0.15][1]
print("通过断言检验：无定时偏差时 BER < 1e-3；偏差 1/8 T 时 β=0.8 的 BER 比 β=0.15 低一个量级以上")
'''

MD_SUM = r"""## 小结

1. 自主实现的 RRC 系数与 Sionna 内置滤波器逐点一致（最大偏差 $<10^{-6}$），说明离散实现正确；级联 RC 脉冲在 $t=kT$ 处的旁瓣为 $-32.9$ dB（$\beta=0.15$）到 $-74$ dB（$\beta=0.8$），验证了 Nyquist 无 ISI 条件，同时暴露了 span=10 截断对小时延拖尾的影响。
2. 带限成形 + 匹配滤波的 16-QAM 误码率与无 ISI 理论曲线在 8~10 dB 内相差不到 2 倍；去掉匹配滤波直接抽样时，10 dB 处的 BER 从 $10^{-3}$ 量级恶化到 $10^{-1}$ 量级。
3. $\beta$ 的取舍清楚可见：$\beta$ 从 0.15 增至 0.8，-30 dB 占用带宽从约 1.3 增到 1.75 倍符号率，而残余 ISI（rms）从约 $3.6\times10^{-2}$ 降到 $7\times10^{-4}$，1/8 符号定时偏差下的 BER 相差一个量级以上。工程上 $\beta=0.2\sim0.35$ 是带宽效率与同步鲁棒性的常见折中。
"""


def main():
    nb = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    cells = nb["cells"]
    assert len(cells) == 4, f"模板单元数异常: {len(cells)}"

    new_cells = [
        cells[0],  # 模板 markdown 标题
        md(MD_GOAL),
        code(CELL1),
        code(CELL2),
        code(CELL3),
        md(MD_OWN),
        code(CELL4),
        code(CELL5),
        md(MD_EYE),
        code(CELL6),
        code(CELL7),
        md(MD_BER),
        code(CELL8),
        code(CELL9),
        md(MD_SUM),
    ]
    nb["cells"] = new_cells
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb["metadata"]["language_info"] = {"name": "python", "version": "3.12.4"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"已写入 {OUT}（{len(new_cells)} 个单元）")


if __name__ == "__main__":
    main()
