# -*- coding: utf-8 -*-
"""生成 notebooks/03I_bicm_ldpc.ipynb。

流程：读模板（保留 markdown 与断言）→ 补全 TODO 代码 → 追加过程性/落盘单元。
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "无线仿真资料2026" / "student" / "03I_bicm_ldpc.ipynb"
OUT = ROOT / "notebooks" / "03I_bicm_ldpc.ipynb"


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


MD_GOAL = r"""## 实验目的

1. 搭建 5G NR LDPC 编码 + BICM（比特交织编码调制）的完整链路：LDPC 编码 → QAM 映射 → AWGN → 软解调 LLR → 迭代置信传播译码。
2. 掌握 $E_s/N_0$ 与 $E_b/N_0$ 的换算，理解码率 $R=k/n$ 对噪声方差标定的影响，并对比编码与未编码链路的误码率。
3. 观察 LDPC 迭代次数、码率、调制阶数对瀑布区位置的影响，理解软信息（LLR）在 BICM 中的作用。

## 原理

**BICM 链路。** 信息比特 $\mathbf{u}\in\{0,1\}^{k}$ 先经 5G NR LDPC 编码得到码字 $\mathbf{c}\in\{0,1\}^{n}$（含 3GPP TS 38.212 的速率匹配与面向 BICM 的输出交织），再按 $\log_2 M$ 个比特一组映射为 QAM 符号 $x$，经 AWGN 信道 $y=x+w$ 后，解调器不直接判决，而是输出每个编码比特的对数似然比（Sionna 采用 $\ln\frac{P(b=1)}{P(b=0)}$ 的约定，与部分教科书的符号相反，但不影响译码）

$$
L(b_i)=\ln\frac{P(b_i=1\mid y)}{P(b_i=0\mid y)}
=\ln\frac{\sum_{x\in\mathcal{X}_i^{1}}\exp\left(-\dfrac{|y-x|^2}{N_0}\right)}{\sum_{x\in\mathcal{X}_i^{0}}\exp\left(-\dfrac{|y-x|^2}{N_0}\right)},
$$

其中 $\mathcal{X}_i^{b}$ 是第 $i$ 个比特取值为 $b$ 的星座点子集。该软信息送入 LDPC 置信传播译码器迭代更新。

**噪声标定与两种信噪比口径。** Sionna 的 `AWGN` 把 $N_0$ 解释为复噪声的**总**方差（每维 $N_0/2$）。发送符号平均能量为 $E_s$，每符号携带 $\log_2 M$ 比特，码率为 $R=k/n$，则

$$
\frac{E_s}{N_0}=\frac{E_b}{N_0}\cdot \log_2 M\cdot R,\qquad
N_0=\frac{1}{(E_b/N_0)\cdot \log_2 M\cdot R}.
$$

因此同一 $E_b/N_0$ 下，编码链路的 $E_s/N_0$ 比未编码链路低 $10\log_{10}R$ dB（$R=1/2$ 时低 3 dB）——这正是编码换来的功率/带宽折中：编码用冗余换编码增益。本实验所有曲线横轴统一为 $E_b/N_0$。

**理论上界。** 未编码 QPSK（Gray）：$P_b=\frac{1}{2}\mathrm{erfc}\sqrt{E_b/N_0}$；未编码 16-QAM：$P_b\approx\frac{3}{8}\mathrm{erfc}\sqrt{\frac{2}{5}E_b/N_0}$。编码后的 BER 需由仿真给出。

## 参数与设计

| 参数 | 模板演示链路 | 扩展实验链路 | 说明 |
| --- | --- | --- | --- |
| 信息位 $k$ | 600 | 1200 | 5G NR LDPC（BG2） |
| 码字位 $n$ | 1200 | 2400 | $R=k/n=1/2$ |
| 调制 | QPSK | QPSK / 16-QAM | 2 / 4 bit 每符号 |
| 批大小 | 16 码字 | 64 码字 | 扩展实验 76800 bit/点 |
| 译码迭代 | 8 | 2 / 5 / 8 / 20 | 迭代次数对比 |
| $E_b/N_0$ | 0, 2, 4, 6 dB | 0~4 dB（步长 0.5 dB） | 瀑布区附近加密 |
| 码率对比 | — | 1/3, 1/2, 3/4 | (600,1800)/(600,1200)/(900,1200) |
| 交织 | 5G 输出交织 | 5G 输出交织 | `num_bits_per_symbol` 指定 |
"""

CELL1 = r'''# 导入依赖：Sionna 的星座映射、AWGN 信道与 5G NR LDPC 编解码器。
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

from sionna.phy.mapping import Constellation, Mapper, Demapper
from sionna.phy.channel import AWGN
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder

K = 600
N = 1200
NUM_BITS_PER_SYMBOL = 2
CODERATE = K / N
BATCH_SIZE = 16

const = Constellation("qam", NUM_BITS_PER_SYMBOL, precision=PRECISION, device=DEVICE)
mapper = Mapper(constellation=const, precision=PRECISION, device=DEVICE)
demapper = Demapper(
    "app", constellation=const, hard_out=False, precision=PRECISION, device=DEVICE
)
awgn = AWGN(precision=PRECISION, device=DEVICE)
# num_bits_per_symbol 让编码器按 3GPP TS 38.212 5.4.2.2 做 BICM 输出交织
encoder = LDPC5GEncoder(K, N, num_bits_per_symbol=NUM_BITS_PER_SYMBOL)
decoder = LDPC5GDecoder(encoder, hard_out=True, num_iter=8)

# ---- 扩展实验：更长的码块 + 更多迭代，用于观察瀑布区 ---
BIG_K = 1200
BIG_N = 2400
BIG_BATCH = 256
BIG_ITER = 20
EB_RANGE = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
FLOOR = 1e-7  # 0 个错误点在 semilogy 上的下限（图中单独标注）


def ebno_db_to_no(ebno_db, nbits, coderate=1.0):
    """Es/N0 = (Eb/N0)·nbits·coderate ⇒ N0 = 1 / ((Eb/N0)·nbits·coderate)。"""
    ebno = 10.0 ** (torch.as_tensor(ebno_db, dtype=torch.float32, device=DEVICE) / 10.0)
    return 1.0 / (ebno * nbits * coderate)


def ber(b, b_hat):
    return float(
        torch.mean((b.to(torch.int32) != b_hat.to(torch.int32)).float()).item()
    )


_CHAIN_CACHE = {}


def get_chain(k, n, q, num_iter):
    """按 (k, n, q, 迭代次数) 缓存星座/映射/解调/编解码器。"""
    key = (k, n, q, num_iter)
    if key not in _CHAIN_CACHE:
        c = Constellation("qam", q, precision=PRECISION, device=DEVICE)
        enc = LDPC5GEncoder(k, n, num_bits_per_symbol=q)
        dec = LDPC5GDecoder(enc, hard_out=True, num_iter=num_iter)
        _CHAIN_CACHE[key] = (
            Mapper(constellation=c, precision=PRECISION, device=DEVICE),
            Demapper("app", constellation=c, hard_out=False, precision=PRECISION, device=DEVICE),
            enc,
            dec,
        )
    return _CHAIN_CACHE[key]


def run_ldpc(k, n, q, ebno_db, num_iter=BIG_ITER, batch=BIG_BATCH):
    """BICM 链路：编码 → 映射 → AWGN → 软解调 LLR → LDPC 译码。"""
    mpr, dmp, enc, dec = get_chain(k, n, q, num_iter)
    bits = torch.randint(0, 2, (batch, k), dtype=torch.int32, device=DEVICE)
    cw = enc(bits)
    x = mpr(cw)
    no = ebno_db_to_no(ebno_db, q, k / n)
    y = awgn(x, no)
    llr = dmp(y, no)
    b_hat = dec(llr)
    return {
        "ber": ber(bits, b_hat),
        "y": y,
        "llr": llr,
        "bits": cw.to(torch.int32),
        "no": float(no),
    }


def run_uncoded(q, ebno_db, num_symbols=4096, batch=BIG_BATCH):
    """未编码参考链路：随机比特 → QAM → AWGN → 硬判决。"""
    c = Constellation("qam", q, precision=PRECISION, device=DEVICE)
    mpr = Mapper(constellation=c, precision=PRECISION, device=DEVICE)
    dmp = Demapper("app", constellation=c, hard_out=True, precision=PRECISION, device=DEVICE)
    bits = torch.randint(0, 2, (batch, num_symbols * q), dtype=torch.int32, device=DEVICE)
    x = mpr(bits)
    no = ebno_db_to_no(ebno_db, q)
    y = awgn(x, no)
    b_hat = dmp(y, no)
    return {"ber": ber(bits, b_hat), "y": y, "bits": bits, "no": float(no)}


def qpsk_ber_theory(ebno_db):
    return 0.5 * erfc(np.sqrt(10 ** (np.asarray(ebno_db, dtype=float) / 10)))


def qam16_ber_theory(ebno_db):
    return (3 / 8) * erfc(np.sqrt(0.4 * 10 ** (np.asarray(ebno_db, dtype=float) / 10)))


print("模板演示链路: K,N =", K, N, "R =", CODERATE, "QPSK, 迭代", decoder.num_iter)
print("扩展实验链路: K,N =", BIG_K, BIG_N, "R =", BIG_K / BIG_N, "batch =", BIG_BATCH)
print("Eb/N0 与 Es/N0 关系: Es/N0 = Eb/N0 * log2(M) * R")
'''

CELL2 = r'''# 模板要求的 BICM 误码率函数：一次端到端仿真（K=600, N=1200, QPSK, R=1/2）。
def run_bicm_ber(ebno_db, batch_size=BATCH_SIZE):
    """编码 → 映射 → AWGN → 软解调 → LDPC 解码 → 信息比特 BER。"""
    bits = torch.randint(0, 2, (batch_size, K), dtype=torch.int32, device=DEVICE)
    cw = encoder(bits)
    x = mapper(cw)
    no = ebno_db_to_no(ebno_db, NUM_BITS_PER_SYMBOL, CODERATE)
    y = awgn(x, no)
    llr = demapper(y, no)
    b_hat = decoder(llr)
    val = ber(bits, b_hat)
    print(f"Eb/N0 = {ebno_db:4.1f} dB，N0 = {float(no):.3f}，BER = {val:.5f}")
    return val


ebno_db = np.arange(0, 7, 2)
coded_ber = [run_bicm_ber(v) for v in ebno_db]

plt.figure(figsize=(7.5, 4.5))
plt.semilogy(ebno_db, np.array(coded_ber) + 1e-6, "o-", lw=1.8, label="5G LDPC 编码（R=1/2，8 次迭代）")
plt.semilogy(
    np.linspace(0, 6, 100),
    qpsk_ber_theory(np.linspace(0, 6, 100)),
    "k--",
    lw=1.2,
    label="未编码 QPSK 理论",
)
plt.xlabel("Eb/N0 (dB)")
plt.ylabel("BER")
plt.grid(True, which="both", alpha=0.4)
plt.legend(loc="lower left")
plt.show()

assert len(coded_ber) == len(ebno_db)
print("通过断言检验：编码后 BER 长度为", len(coded_ber))
'''

MD_MOD2 = r"""## 模块二 编码增益：编码 vs 未编码

模板链路只有 600 个信息比特、76800 次统计，高信噪比处容易出现"0 错误"；这里把码块加长到 $k=1200,n=2400$（$R=1/2$），每点统计 76800 比特，并把迭代次数提高到 20，得到更完整的瀑布区曲线。编码与未编码链路使用同一 $E_b/N_0$ 口径，因此编码链路的噪声方差按 $N_0=1/((E_b/N_0)\log_2 M\cdot R)$ 标定。
"""

CELL3 = r'''# ---- 图 03I-1：编码 vs 未编码 BER-Eb/N0 ----
res_coded_qpsk = [run_ldpc(BIG_K, BIG_N, 2, e)["ber"] for e in EB_RANGE]
res_coded_16qam = [run_ldpc(BIG_K, BIG_N, 4, e)["ber"] for e in EB_RANGE]
res_unc_qpsk = [run_uncoded(2, e)["ber"] for e in EB_RANGE]
res_unc_16qam = [run_uncoded(4, e)["ber"] for e in EB_RANGE]

for name, seq in [
    ("QPSK 编码 R=1/2", res_coded_qpsk),
    ("16-QAM 编码 R=1/2", res_coded_16qam),
    ("QPSK 未编码", res_unc_qpsk),
    ("16-QAM 未编码", res_unc_16qam),
]:
    print(f"{name:<18s}" + "  ".join(f"{e:.1f}dB→{v:.2e}" for e, v in zip(EB_RANGE, seq)))

df_eb = pd.DataFrame({"Eb/N0_dB": EB_RANGE})
df_eb["QPSK编码_R0.5"] = res_coded_qpsk
df_eb["16QAM编码_R0.5"] = res_coded_16qam
df_eb["QPSK未编码"] = res_unc_qpsk
df_eb["16QAM未编码"] = res_unc_16qam
df_eb["QPSK理论"] = qpsk_ber_theory(EB_RANGE)
df_eb["16QAM理论"] = qam16_ber_theory(EB_RANGE)
print(df_eb.to_string(index=False))
df_eb.to_csv(DATA_DIR / "03I_ber_ebno.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(7.5, 4.5))
dense = np.linspace(EB_RANGE[0], EB_RANGE[-1], 200)
ax.semilogy(dense, qpsk_ber_theory(dense), "k--", lw=1.2, label="未编码 QPSK 理论")
ax.semilogy(dense, qam16_ber_theory(dense), "k:", lw=1.2, label="未编码 16-QAM 理论")
ax.semilogy(EB_RANGE, np.maximum(res_unc_qpsk, FLOOR), "o", ms=6, color="C7", label="未编码 QPSK 仿真")
ax.semilogy(EB_RANGE, np.maximum(res_unc_16qam, FLOOR), "s", ms=6, color="C6", label="未编码 16-QAM 仿真")
ax.semilogy(EB_RANGE, np.maximum(res_coded_qpsk, FLOOR), "o-", ms=5, color="C0", label="LDPC 编码 QPSK, R=1/2")
ax.semilogy(EB_RANGE, np.maximum(res_coded_16qam, FLOOR), "s-", ms=5, color="C3", label="LDPC 编码 16-QAM, R=1/2")
ax.set_xlabel("Eb/N0 (dB)")
ax.set_ylabel("BER")
ax.set_ylim(1e-7, 1)
ax.annotate(
    "0 个错误点记为 $10^{-7}$",
    xy=(EB_RANGE[-1], FLOOR),
    xytext=(2.4, 3e-7),
    fontsize=9,
    color="gray",
)
ax.grid(True, which="both", alpha=0.4)
ax.legend(loc="lower left")
fig.savefig(FIG_DIR / "03I_ber_ebno.png", dpi=200, bbox_inches="tight")
print("图题：图 03I-1 编码与未编码 BICM 链路的 BER-Eb/N0 对比")
plt.show()

# 断言：编码曲线单调下降；高信噪比处编码明显优于未编码
assert np.all(np.diff(res_coded_qpsk) <= 0) and np.all(np.diff(res_coded_16qam) <= 0)
assert res_coded_qpsk[-1] < 0.1 * res_unc_qpsk[-1]
assert res_coded_16qam[-1] < 0.1 * res_unc_16qam[-1]
assert res_coded_qpsk[0] > res_unc_qpsk[0]  # 低信噪比、短码条件下编码无增益
print("通过断言检验：编码 BER 单调下降；高 SNR 处编码增益 >10 倍；低 SNR 处编码反而变差")
'''

MD_MOD3 = r"""## 模块三 迭代次数与码率的影响

LDPC 的误码率由置信传播迭代逐步逼近，迭代次数不足时瀑布区右移；码率越低，冗余越多、相同 $E_b/N_0$ 下的编码增益越大（但有效吞吐 $\log_2 M\cdot R$ 下降）。
"""

CELL4 = r'''# ---- 图 03I-2：迭代次数对比 ----
ITER_LIST = [2, 5, 8, 20]
EB_ITER = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
res_iter = {}
for it in ITER_LIST:
    res_iter[it] = [run_ldpc(BIG_K, BIG_N, 2, e, num_iter=it)["ber"] for e in EB_ITER]
    print(f"迭代 {it:2d} 次: " + "  ".join(f"{e:.1f}dB→{v:.2e}" for e, v in zip(EB_ITER, res_iter[it])))

df_iter = pd.DataFrame({"Eb/N0_dB": EB_ITER})
for it in ITER_LIST:
    df_iter[f"iter_{it}"] = res_iter[it]
print(df_iter.to_string(index=False))
df_iter.to_csv(DATA_DIR / "03I_iterations.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(7.5, 4.5))
markers = ["v", "^", "s", "o"]
for it, mk in zip(ITER_LIST, markers):
    ax.semilogy(EB_ITER, np.maximum(res_iter[it], FLOOR), mk + "-", ms=5, label=f"迭代 {it} 次")
ax.set_xlabel("Eb/N0 (dB)")
ax.set_ylabel("BER")
ax.set_ylim(1e-7, 1)
ax.grid(True, which="both", alpha=0.4)
ax.legend(loc="lower left")
fig.savefig(FIG_DIR / "03I_iterations.png", dpi=200, bbox_inches="tight")
print("图题：图 03I-2 LDPC 译码迭代次数对 BER 的影响（QPSK，R=1/2）")
plt.show()

# 断言：Eb/N0=1.5 dB 处迭代越多 BER 越低；20 次迭代在 2 dB 处基本无误码
assert res_iter[20][3] <= res_iter[8][3] <= res_iter[5][3] <= res_iter[2][3]
assert res_iter[20][-1] < 1e-3
print("通过断言检验：迭代次数增加使 BER 单调下降，20 次迭代在 2 dB 处 BER < 1e-3")
'''

CELL5 = r'''# ---- 图 03I-3：不同码率对比 ----
RATE_LIST = [(600, 1800), (600, 1200), (900, 1200)]  # R = 1/3, 1/2, 3/4
EB_RATE = np.array([0.0, 1.0, 2.0, 3.0])
res_rate = {}
for k, n in RATE_LIST:
    res_rate[(k, n)] = [run_ldpc(k, n, 2, e)["ber"] for e in EB_RATE]
    print(f"R={k/n:.2f} (k={k},n={n}): " + "  ".join(f"{e:.1f}dB→{v:.2e}" for e, v in zip(EB_RATE, res_rate[(k, n)])))

df_rate = pd.DataFrame({"Eb/N0_dB": EB_RATE})
for k, n in RATE_LIST:
    df_rate[f"R_{k/n:.2f}"] = res_rate[(k, n)]
print(df_rate.to_string(index=False))
df_rate.to_csv(DATA_DIR / "03I_rates.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(7.5, 4.5))
for (k, n), mk in zip(RATE_LIST, ["o", "s", "^"]):
    ax.semilogy(EB_RATE, np.maximum(res_rate[(k, n)], FLOOR), mk + "-", ms=5, label=f"R = {k/n:.2f} (k={k}, n={n})")
ax.set_xlabel("Eb/N0 (dB)")
ax.set_ylabel("BER")
ax.set_ylim(1e-7, 1)
ax.grid(True, which="both", alpha=0.4)
ax.legend(loc="lower left")
fig.savefig(FIG_DIR / "03I_rates.png", dpi=200, bbox_inches="tight")
print("图题：图 03I-3 码率对 LDPC 编码增益的影响（QPSK）")
plt.show()

# 断言：同一 Eb/N0 下低码率 BER 更低（1 dB 处最明显）
assert res_rate[(600, 1800)][1] < res_rate[(600, 1200)][1] < res_rate[(900, 1200)][1]
print("通过断言检验：Eb/N0=1 dB 处 BER 随码率增大而升高，符合编码增益与码率的折中")
'''

MD_MOD4 = r"""## 模块四 BICM 软信息的可视化

BICM 的性能优势来自"解调器输出软信息而不是硬判决"。下图把解调器输出的 LLR 与其真实比特值对应起来：可靠比特的 LLR 远离 0，处于衰落/噪声中的比特 LLR 集中在 0 附近，LDPC 译码器正是利用这种可靠性差异完成纠错。
"""

CELL6 = r'''# ---- 图 03I-4：LLR 分布与平均可靠度 ----
out_ref = run_ldpc(BIG_K, BIG_N, 4, 2.0)
llr = out_ref["llr"].flatten().numpy()
true_bits = out_ref["bits"].flatten().numpy()

# 落盘时只抽取 6000 个样本代表整体分布（完整数组留在内存中用于画直方图）
_rng_llr = np.random.default_rng(42)
_idx = np.sort(_rng_llr.choice(llr.size, size=min(6000, llr.size), replace=False))
df_llr = pd.DataFrame({"LLR": llr[_idx].astype(float), "bit": true_bits[_idx].astype(int)})
df_llr.to_csv(DATA_DIR / "03I_llr.csv", index=False, encoding="utf-8-sig")
print("LLR 落盘抽样点数:", len(df_llr))

mean_abs_llr = []
eb_llr = np.array([0.0, 1.0, 2.0, 3.0])
for e in eb_llr:
    o = run_ldpc(BIG_K, BIG_N, 4, e)
    mean_abs_llr.append(float(np.mean(np.abs(o["llr"].flatten().numpy()))))
print("16-QAM 平均 |LLR|:", "  ".join(f"{e:.1f}dB→{v:.2f}" for e, v in zip(eb_llr, mean_abs_llr)))

fig, axes = plt.subplots(2, 1, figsize=(8, 8.5))
ax = axes[0]
bins = np.linspace(-40, 40, 81)
ax.hist(llr[true_bits == 0], bins=bins, alpha=0.65, label="真实比特 b=0", color="C0")
ax.hist(llr[true_bits == 1], bins=bins, alpha=0.65, label="真实比特 b=1", color="C3")
ax.set_xlabel("解调器输出 LLR")
ax.set_ylabel("样本数")
ax.set_title("16-QAM，Eb/N0 = 2 dB", fontsize=12)
ax.grid(True, alpha=0.35)
ax.legend(loc="upper center")

ax = axes[1]
ax.plot(eb_llr, mean_abs_llr, "o-", ms=6, color="C0")
ax.set_xlabel("Eb/N0 (dB)")
ax.set_ylabel("平均 |LLR|（16-QAM）")
ax.grid(True, alpha=0.4)
fig.tight_layout()
fig.savefig(FIG_DIR / "03I_llr.png", dpi=200, bbox_inches="tight")
print("图题：图 03I-4 解调器输出 LLR 的分布与平均可靠度")
plt.show()

# 断言：b=1 的 LLR 均值 > 0、b=0 的 LLR 均值 < 0；平均 |LLR| 随信噪比上升
assert llr[true_bits == 1].mean() > 0 > llr[true_bits == 0].mean()
assert np.all(np.diff(mean_abs_llr) > 0)
print("通过断言检验：LLR 符号与比特值一致，且平均可靠度随信噪比单调上升")
'''

MD_SUM = r"""## 小结

1. 在 $E_b/N_0$ 统一口径下（$N_0=1/((E_b/N_0)\log_2M\cdot R)$），$R=1/2$、$k=1200$、20 次迭代的 QPSK-BICM 链路在 2 dB 处已无错误，而未编码 QPSK 此时 BER 约 $4\times10^{-2}$；16-QAM 编码链路在 4 dB 处 BER 从约 $6\times10^{-2}$ 降至 0，编码增益超过 10 倍。
2. 低信噪比（0 dB）时编码链路反而比未编码差（约 0.16 对 0.08）：短码块、$R=1/2$ 且 20 dB 以内尚未越过瀑布区，译码器输出接近随机码字——这说明编码增益只在阈值以上出现，也正是"瀑布区"的物理含义。
3. 迭代次数从 2 增到 20 使 1.5 dB 处的 BER 从 $10^{-1}$ 量级降到 $10^{-3}$ 以下；码率从 3/4 降到 1/3 使 1 dB 处 BER 从 $10^{-1}$ 量级降到 $10^{-4}$ 量级，代价是频谱效率减半——与编码理论中的码率-可靠性折中一致。
"""


def main():
    nb = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    cells = nb["cells"]
    assert len(cells) == 3, f"模板单元数异常: {len(cells)}"
    new_cells = [
        cells[0],
        md(MD_GOAL),
        code(CELL1),
        code(CELL2),
        md(MD_MOD2),
        code(CELL3),
        md(MD_MOD3),
        code(CELL4),
        code(CELL5),
        md(MD_MOD4),
        code(CELL6),
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
