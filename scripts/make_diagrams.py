# -*- coding: utf-8 -*-
"""生成实验报告"实验原理"一节所需的系统框图（纯 matplotlib 绘制）。"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.stdout.reconfigure(encoding="utf-8")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BOX = dict(boxstyle="round,pad=0.02,rounding_size=0.10", fc="#EAF2FB", ec="#2F6DB5", lw=1.3)
BOXC = dict(boxstyle="round,pad=0.02,rounding_size=0.10", fc="#FDF1E3", ec="#C86A16", lw=1.3)


def box(ax, cx, cy, w, h, label, style, fs=11):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, **style))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fs, linespacing=1.5)


def arrow(ax, p1, p2, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            p1, p2, arrowstyle="-|>", mutation_scale=15, lw=1.4,
            color="#444444", shrinkA=2, shrinkB=2,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


# ------------------------------------------------- 图 1：基带调制链路总体模型
fig, ax = plt.subplots(figsize=(11.0, 2.3))
ax.set_xlim(0, 13.9)
ax.set_ylim(0, 2.0)
ax.axis("off")
W, H, Y = 1.85, 0.95, 1.0
centers = [1.10, 3.45, 5.80, 8.15, 10.50, 12.85]
labels = [
    "随机比特源\n$\\mathbf{b}$",
    "星座映射\n$s=\\mu(\\mathbf{b})$",
    "无线信道\nAWGN / 瑞利",
    "最小距离判决\n$\\hat{s}=\\arg\\min|y-s|^2$",
    "解映射\n$\\hat{\\mathbf{b}}$",
    "误码统计\nBER",
]
styles = [BOX, BOX, BOXC, BOX, BOX, BOXC]
for c, lab, st in zip(centers, labels, styles):
    box(ax, c, Y, W, H, lab, st, fs=10.5)
for i in range(len(centers) - 1):
    arrow(ax, (centers[i] + W / 2, Y), (centers[i + 1] - W / 2, Y))
ax.text(5.80, 0.10, "$y = h\\,s + n$", ha="center", fontsize=10.5, color="#C86A16")
plt.tight_layout()
fig.savefig(OUT / "fig01_system_block.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------- 图 2：OFDM 收发机结构
fig, ax = plt.subplots(figsize=(11.0, 4.2))
ax.set_xlim(0, 14.4)
ax.set_ylim(0, 5.0)
ax.axis("off")
W, H = 2.00, 0.95
Y_TX, Y_RX = 3.9, 1.1
tx = [
    (1.20, "比特源"),
    (3.75, "QPSK 映射\n$s_k$"),
    (6.30, "串/并 + IFFT\nN = 64"),
    (8.85, "加循环前缀\n$N_{cp}$"),
]
rx = [
    (8.85, "去循环前缀"),
    (6.30, "FFT\n$Y_k$"),
    (3.75, "单抽头 ZF 均衡\n$\\hat{H}_k=Y_k/H_k$"),
    (1.20, "QPSK 解映射\n+ BER 统计"),
]
for c, lab in tx:
    box(ax, c, Y_TX, W, H, lab, BOX, fs=10)
for c, lab in rx:
    box(ax, c, Y_RX, W, H, lab, BOX, fs=10)
for i in range(len(tx) - 1):
    arrow(ax, (tx[i][0] + W / 2, Y_TX), (tx[i + 1][0] - W / 2, Y_TX))
for i in range(len(rx) - 1):
    arrow(ax, (rx[i][0] - W / 2, Y_RX), (rx[i + 1][0] + W / 2, Y_RX))

# 信道框（右侧，跨两行）
box(ax, 12.10, 2.5, 1.85, 2.4, "多径瑞利\n衰落信道\n3 径 [0,1,4]\n+ AWGN", BOXC, fs=9.5)
arrow(ax, (8.85 + W / 2, Y_TX), (11.15, 3.25))
arrow(ax, (11.15, 1.75), (8.85 + W / 2, Y_RX))

ax.text(0.55, 4.75, "发送端（基带等效）", fontsize=10.5, color="#2F6DB5", ha="left")
ax.text(0.55, 0.30, "接收端", fontsize=10.5, color="#2F6DB5", ha="left")
plt.tight_layout()
fig.savefig(OUT / "fig02_ofdm_block.png", dpi=220, bbox_inches="tight")
plt.close(fig)

print("saved:", sorted(p.name for p in OUT.glob("*.png")))
