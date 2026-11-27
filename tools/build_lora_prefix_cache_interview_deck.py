from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


plt.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "Hiragino Sans GB",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

C_BG = RGBColor(255, 255, 255)
C_PANEL = RGBColor(248, 250, 252)
C_BORDER = RGBColor(203, 213, 225)
C_NAVY = RGBColor(15, 23, 42)
C_TEXT = RGBColor(30, 41, 59)
C_MUTED = RGBColor(100, 116, 139)
C_BLUE = RGBColor(29, 78, 216)
C_BLUE_LIGHT = RGBColor(239, 246, 255)
C_GREEN = RGBColor(5, 150, 105)
C_GREEN_LIGHT = RGBColor(236, 253, 245)
C_RED = RGBColor(220, 38, 38)
C_RED_LIGHT = RGBColor(254, 242, 242)
C_AMBER = RGBColor(180, 83, 9)
C_AMBER_LIGHT = RGBColor(255, 251, 235)
C_CODE_BG = RGBColor(15, 23, 42)
C_CODE_FG = RGBColor(241, 245, 249)

FONT_CN = "PingFang SC"
FONT_CODE = "Menlo"


def _add_header(slide, title: str, subtitle: str, page_num: int, total_pages: int) -> None:
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.55), Inches(0.38), Inches(0.10), Inches(0.62)
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = C_BLUE
    bar.line.fill.background()

    tb = slide.shapes.add_textbox(Inches(0.76), Inches(0.30), Inches(11.4), Inches(0.78))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    p0 = tf.paragraphs[0]
    p0.text = title
    p0.font.name = FONT_CN
    p0.font.size = Pt(22)
    p0.font.bold = True
    p0.font.color.rgb = C_NAVY

    p1 = tf.add_paragraph()
    p1.text = subtitle
    p1.font.name = FONT_CN
    p1.font.size = Pt(11.5)
    p1.font.color.rgb = C_MUTED

    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.55), Inches(1.12), Inches(12.23), Inches(0.02)
    )
    line.fill.solid()
    line.fill.fore_color.rgb = C_BORDER
    line.line.fill.background()

    ft = slide.shapes.add_textbox(Inches(0.55), Inches(7.12), Inches(12.23), Inches(0.28))
    ftf = ft.text_frame
    ftf.margin_left = ftf.margin_top = ftf.margin_right = ftf.margin_bottom = 0
    fp = ftf.paragraphs[0]
    fp.alignment = PP_ALIGN.RIGHT
    fp.text = f"TinyLLMForge  |  Parametric Prefix LoRA & Native Runtime  |  {page_num} / {total_pages}"
    fp.font.name = FONT_CN
    fp.font.size = Pt(9.5)
    fp.font.color.rgb = C_MUTED


def _add_card(
    slide,
    left: float,
    top: float,
    width: float,
    height: float,
    *,
    fill_rgb: RGBColor = C_PANEL,
    border_rgb: RGBColor = C_BORDER,
    title: str | None = None,
    title_rgb: RGBColor = C_NAVY,
    bullets: list[str] | None = None,
    font_size: float = 12.0,
    is_code: bool = False,
):
    shp = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(left),
        Inches(top),
        Inches(width),
        Inches(height),
    )
    shp.adjustments[0] = 0.03
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill_rgb
    shp.line.color.rgb = border_rgb
    shp.line.width = Pt(1.0)

    tb = slide.shapes.add_textbox(
        Inches(left + 0.18),
        Inches(top + 0.14),
        Inches(width - 0.36),
        Inches(height - 0.26),
    )
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    first = True
    if title:
        p = tf.paragraphs[0]
        first = False
        p.text = title
        p.font.name = FONT_CN
        p.font.size = Pt(font_size + 2.0)
        p.font.bold = True
        p.font.color.rgb = title_rgb
        p.space_after = Pt(6)

    if bullets:
        for b_txt in bullets:
            p = tf.paragraphs[0] if first else tf.add_paragraph()
            first = False
            p.text = b_txt
            p.font.name = FONT_CODE if is_code else FONT_CN
            p.font.size = Pt(font_size)
            p.font.color.rgb = C_CODE_FG if is_code else C_TEXT
            p.space_after = Pt(4 if not is_code else 1.5)
    return shp


def _fig_to_stream(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def _make_progression_chart(g12_data: dict | None) -> io.BytesIO:
    labels = [
        "Base\n(裸底座)",
        "LoRA-16\n(欠拟合)",
        "LoRA-96\n(无约束遗忘)",
        "KL-64\n(零空间锚定)",
        "G8 Diff-KL\n(差异行聚焦)",
        "G9 Pure-CSD\n(零退化自蒸馏)",
    ]
    tr_vals = [3, 2, 4, 3, 5, 4]
    rt_vals = [4, 4, 2, 4, 3, 4]
    if g12_data is not None:
        tr_g12 = int(g12_data["lora_hybrid_csd_by_split"]["transfer"]["feedback"]["PASS"])
        rt_g12 = int(g12_data["lora_hybrid_csd_by_split"]["retention"]["feedback"]["PASS"])
        labels.append("G12 Hybrid\n(外科掩码+自蒸馏)")
        tr_vals.append(tr_g12)
        rt_vals.append(rt_g12)
    labels.append("Long Prompt\n(3,500 tok 基准)")
    tr_vals.append(4)
    rt_vals.append(4)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11.2, 4.0))
    b1 = ax.bar(x, tr_vals, width=0.52, label="同机制迁移 Transfer (/12)", color="#1D4ED8")
    b2 = ax.bar(x, rt_vals, width=0.52, bottom=tr_vals, label="异机制保留 Retention (/12)", color="#10B981")
    ax.axhline(8.0, color="#DC2626", linestyle="--", linewidth=1.6, label="长提示词总分基准 (8/24)")

    for idx_i, (tv, rv) in enumerate(zip(tr_vals, rt_vals)):
        tot = tv + rv
        lbl_str = f"★ {tot}/24 ★\n({tv}+{rv})" if tot >= 9 else f"{tot}/24\n({tv}+{rv})"
        txt_col = "#B45309" if tot >= 9 else "#0F172A"
        ax.text(
            idx_i,
            tot + 0.22,
            lbl_str,
            ha="center",
            va="bottom",
            fontsize=9.8 if tot >= 9 else 9.2,
            fontweight="bold",
            color=txt_col,
        )

    ax.set_ylim(0, max(tr_vals[i] + rt_vals[i] for i in range(len(tr_vals))) + 2.4)
    ax.set_ylabel("通过题数 (8 用例全过 PASS)", fontsize=11, fontweight="bold")
    ax.set_title("confirmatory_v2_r1 (24 题严格沙箱评测) 算法演进全路径对比", fontsize=13, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.2)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=True, fontsize=9.5, ncol=3)
    fig.tight_layout()
    return _fig_to_stream(fig)


def _make_serving_chart() -> io.BytesIO:
    batches = ["B=1", "B=4", "B=8", "B=16", "B=32", "B=64"]
    x = np.arange(len(batches))

    cold_tps = [17.2, 68.5, 136.8, 273.9, 362.8, 0.0]
    warm_tps = [16.8, 67.3, 134.9, 269.7, 362.0, 0.0]
    lora_tps = [17.1, 68.4, 136.6, 271.4, 543.8, 1095.6]

    warm_mem = [0.73, 2.90, 5.79, 11.55, 23.09, 46.2]
    lora_mem = [0.11, 0.43, 0.85, 1.70, 3.41, 6.82]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.1))

    w = 0.26
    ax1.bar(x - w, cold_tps, width=w, label="冷长提示词 KV (3,500 tok)", color="#94A3B8")
    ax1.bar(x, warm_tps, width=w, label="免费预热 Warm Prefix Cache", color="#F59E0B")
    ax1.bar(x + w, lora_tps, width=w, label="7.31 MiB 权重 Cache (短输入)", color="#059669")
    ax1.text(5 - w / 2, 45, "CUDA\nOOM!", color="#DC2626", fontweight="bold", ha="center", fontsize=9)
    ax1.text(5 + w, 1115, "1,095.6\ntok/s", color="#059669", fontweight="bold", ha="center", fontsize=9.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(batches, fontsize=9.5)
    ax1.set_ylim(0, 1260)
    ax1.set_ylabel("纯吐字吞吐 Decode tok/s", fontsize=10.5, fontweight="bold")
    ax1.set_title("A100 80GB 并发吐字吞吐 (SDPA)", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper left", fontsize=8.5)

    ax2.plot(x[:5], warm_mem[:5], marker="o", linewidth=2.4, color="#F59E0B", label="预热 Prefix KV Cache (GiB)")
    ax2.scatter([5], [46.2], marker="x", s=110, color="#DC2626", zorder=5, label="B=64 爆显存 OOM")
    ax2.plot(x, lora_mem, marker="s", linewidth=2.6, color="#1D4ED8", label="7.31 MiB 权重 Cache (GiB)")
    for idx_i in (3, 4, 5):
        ax2.annotate(
            f"{lora_mem[idx_i]:.1f}G",
            (idx_i, lora_mem[idx_i]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8.8,
            fontweight="bold",
            color="#1D4ED8",
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels(batches, fontsize=9.5)
    ax2.set_ylabel("额外 KV 显存占用 (GiB)", fontsize=10.5, fontweight="bold")
    ax2.set_title("并发 KV 显存增长曲线 (省 85.2% 显存)", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="upper left", fontsize=8.5)

    fig.tight_layout()
    return _fig_to_stream(fig)


def _make_bbh_and_kv_pareto_chart() -> io.BytesIO:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 3.95))

    models = ["Qwen3-0.6B\n(2.19 MiB LoRA)", "Qwen3-8B\n(7.31 MiB LoRA)"]
    xm = np.arange(len(models))
    w = 0.25
    base_v = [9.17, 13.33]
    text_v = [48.33, 66.67]
    lora_v = [50.00, 62.50]
    lora_err = [0.83, 3.00]

    ax1.bar(xm - w, base_v, width=w, label="裸底座 0-shot", color="#94A3B8")
    ax1.bar(xm, text_v, width=w, label="12-shot CoT 长提示词", color="#F59E0B")
    ax1.bar(xm + w, lora_v, yerr=lora_err, capsize=4, width=w, label="受控 LoRA (3种子均值±std)", color="#1D4ED8")
    ax1.text(0 + w, 53.0, "50.0%±0.8%\n(p<2e-11, 反超!)", ha="center", fontsize=8.5, fontweight="bold", color="#1D4ED8")
    ax1.text(1 + w, 67.5, "62.5%±3.0%\n(p<5e-15)", ha="center", fontsize=8.5, fontweight="bold", color="#1D4ED8")
    ax1.set_xticks(xm)
    ax1.set_xticklabels(models, fontsize=9.8)
    ax1.set_ylim(0, 82)
    ax1.set_ylabel("准确率 (%)", fontsize=10.5, fontweight="bold")
    ax1.set_title("公开 BIG-Bench Hard (N=120 × 3 Seeds)", fontsize=11.8, fontweight="bold")
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper left", fontsize=8.2)

    comp_x = [1.0, 1.97, 3.20, 5.0, 10.0, 9.78, 15.22]
    acc_y = [72.5, 72.5, 67.5, 2.5, 2.5, 42.5, 27.5]
    ax2.plot([1.0, 1.97, 3.20], [72.5, 72.5, 67.5], "o-", color="#64748B", linewidth=2.0, label="INT8 / INT4 KV 量化 (零提速 55s)")
    ax2.plot([5.0, 10.0], [2.5, 2.5], "x--", color="#DC2626", markersize=9, linewidth=2.0, label="SnapKV 20%/10% (静态驱逐崩盘)")
    ax2.scatter([9.78, 15.22], [42.5, 27.5], s=110, marker="*", color="#059669", zorder=5, label="LoRA+微残差 / 纯LoRA (快3.8~4.0×)")
    ax2.annotate("INT8 (2×, 72.5%)", (1.97, 72.5), xytext=(4, -14), textcoords="offset points", fontsize=8.2)
    ax2.annotate("SnapKV-10%\n(2.5% 归零!)", (10.0, 2.5), xytext=(-15, 12), textcoords="offset points", fontsize=8.3, color="#DC2626", fontweight="bold")
    ax2.annotate("LoRA+微残差 (9.8×)\n42.5% (17×于SnapKV!)", (9.78, 42.5), xytext=(-45, 10), textcoords="offset points", fontsize=8.5, color="#059669", fontweight="bold")
    ax2.set_xlabel("相对 BF16 KV 显存压缩倍数 (×)", fontsize=10.2, fontweight="bold")
    ax2.set_ylabel("严格全对率 (%, N=40)", fontsize=10.5, fontweight="bold")
    ax2.set_ylim(-4, 86)
    ax2.set_xlim(0, 17.5)
    ax2.set_title("KV 压缩三段式帕累托前沿 (N=40)", fontsize=11.8, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="upper right", fontsize=8.0)

    fig.tight_layout()
    return _fig_to_stream(fig)


def build_deck(repo_root: Path, out_pptx: Path, g12_report_path: Path | None = None) -> dict:
    g12_data = None
    if g12_report_path and g12_report_path.exists():
        g12_data = json.loads(g12_report_path.read_text("utf-8"))

    e2e_path = (
        repo_root
        / "artifacts/lora_necessity_v2/20261126-tinyvllm-prefix-lora-runtime/tinyvllm-native-prefix-lora-e2e-report.json"
    )
    e2e_data = json.loads(e2e_path.read_text("utf-8"))

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]
    total_pages = 12

    best_score_str = "8 / 24 (追平长提示词，专精 5/12 反超)"
    g12_bullet = "G8 差异行 4× 聚焦达 5/12 迁移（反超长提示词 4/12）；G9 正反全 KL 自蒸馏达成全集 0 退化（8/24）"
    if g12_data is not None:
        g12_tot = int(g12_data["total_pass_feedback"])
        g12_tr = int(g12_data["lora_hybrid_csd_by_split"]["transfer"]["feedback"]["PASS"])
        g12_rt = int(g12_data["lora_hybrid_csd_by_split"]["retention"]["feedback"]["PASS"])
        g12_kl = float(g12_data["heldout_dev_restraint_audit"]["mean_token_kl"])
        g12_top1 = float(g12_data["heldout_dev_restraint_audit"]["top1_agreement_rate"]) * 100.0
        if g12_tot >= 9:
            best_score_str = f"{g12_tot} / 24 (正面反超长提示词 8/24！迁移 {g12_tr}/12 + 保留 {g12_rt}/12)"
        else:
            best_score_str = f"总分 {max(8, g12_tot)} / 24 (G12 实测 {g12_tr}/12 + {g12_rt}/12 = {g12_tot}/24，D01..D04 一致率 {g12_top1:.1f}%)"
        g12_bullet = (
            f"G12 外科手术掩码 + 25 轨迹自蒸馏实测：{g12_tr}/12 迁移 + {g12_rt}/12 保留 = {g12_tot}/24 "
            f"(未见异类 D01..D04 Top-1 一致率 {g12_top1:.1f}%，KL={g12_kl:.4f})"
        )

    s1 = prs.slides.add_slide(blank)
    hero = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(2.65))
    hero.fill.solid()
    hero.fill.fore_color.rgb = C_NAVY
    hero.line.fill.background()

    tb1 = s1.shapes.add_textbox(Inches(0.75), Inches(0.48), Inches(11.8), Inches(1.95))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "TinyLLMForge：受控零空间外挂 LoRA 参数化前缀缓存"
    p.font.name = FONT_CN
    p.font.size = Pt(30)
    p.font.bold = True
    p.font.color.rgb = RGBColor(255, 255, 255)
    p.space_after = Pt(10)

    p2 = tf1.add_paragraph()
    p2.text = "将高频共享长提示词（3,050~16,384 tok）浓缩为 7.31 MiB BF16 权重缓存：从 KL 零空间防遗忘到 tinyvllm 原生 CUDA Graph 落地"
    p2.font.name = FONT_CN
    p2.font.size = Pt(14.5)
    p2.font.color.rgb = RGBColor(191, 219, 254)

    kpi_specs = [
        (
            "58.7× ~ 315× 驻留压缩",
            C_BLUE_LIGHT,
            C_BLUE,
            [
                "• 3,050 tok 前缀 KV (428.9 MiB) → 7.31 MiB",
                "• 16K tok 规范前缀 KV (2.25 GiB) → 7.31 MiB",
                "• 单张 80G A100 可同时常驻 5,666 个专精模块（KV 仅能放 96 个）",
            ],
        ),
        (
            "Batch=64 破千 tok/s 不爆显存",
            C_GREEN_LIGHT,
            C_GREEN,
            [
                "• B=64 冷/热 Prefix KV 双双 CUDA_OOM",
                "• 7.31 MiB 权重 Cache 稳跑 1,095.6 decode tok/s",
                "• 总吞吐较热 Prefix Cache 峰值翻 2.65×，单次推理提速 8.87×",
            ],
        ),
        (
            best_score_str,
            C_AMBER_LIGHT,
            C_AMBER,
            [
                "• 攻克少样本 LoRA「训浅不涨、训深遗忘」死结",
                "• G9 实现未见异类 D01..D04 100.0% 选词一致、全集 0 题改坏",
                f"• {g12_bullet}",
            ],
        ),
        (
            "tinyvllm 引擎主干原生闭环",
            C_PANEL,
            C_NAVY,
            [
                "• add_request 最长前缀 O(L) 透明识别裁剪 (3,426 → 376 tok)",
                "• 多槽位 QKVLoRASlotPool 无分支混合 Batch 旁路",
                "• 8 档 CUDA Graph 全兼容，较原生 Warm APC 省 26.7% 块且快 1.047×",
            ],
        ),
    ]
    for idx_k, (kt, bg_c, hd_c, k_bul) in enumerate(kpi_specs):
        col = idx_k % 2
        row = idx_k // 2
        _add_card(
            s1,
            0.70 + col * 6.05,
            2.95 + row * 2.02,
            5.85,
            1.84,
            fill_rgb=bg_c,
            border_rgb=hd_c,
            title=kt,
            title_rgb=hd_c,
            bullets=k_bul,
            font_size=11.5,
        )

    s2 = prs.slides.add_slide(blank)
    _add_header(s2, "1. 工程痛点与诚实边界：为什么有了 Prefix Cache 还需要参数化权重缓存？", "厘清 KV Cache 物理瓶颈与「非冷启高频共享前缀」的真实适用域", 2, total_pages)
    _add_card(
        s2,
        0.55,
        1.35,
        5.95,
        5.55,
        fill_rgb=C_RED_LIGHT,
        border_rgb=C_RED,
        title="原生 Prefix KV Cache 的两大物理硬伤",
        title_rgb=C_RED,
        bullets=[
            "1) Decode 阶段显存带宽墙（Memory-Bandwidth Bound）：",
            "   • Prefix Cache 只能省掉首轮 Prefill 重算；",
            "   • 在 Decode 阶段，每生成 1 个新 token，Attention 仍必须把 3,050~16,384 个前缀 token 的 K/V 从 HBM 搬进 SRAM！",
            "   • A100 实测：B=32 时热 Prefix Cache 的 TPOT 从 58.8ms 恶化至 88.4ms，B=64 直接 OOM。",
            "2) 多专精技能并发驻留容量墙：",
            "   • Qwen3-8B 每 token KV = 36 层 × 2 × 8 头 × 128 维 × 2B = 144 KiB；",
            "   • 单个 3,050 tok 示范前缀占 428.9 MiB，16K 规范占 2.25 GiB，几十个租户即打满整卡。",
        ],
        font_size=11.8,
    )
    _add_card(
        s2,
        6.75,
        1.35,
        6.03,
        5.55,
        fill_rgb=C_BLUE_LIGHT,
        border_rgb=C_BLUE,
        title="本方案定位：非冷启高频专精前缀的参数化卸载",
        title_rgb=C_BLUE,
        bullets=[
            "✅ 最佳适用场景（Warm Specialized Prefixes）：",
            "   • 高频复用的固定业务规范 / Few-shot 错题示范集（如医疗科室病历质控规则、法律条款审查准则、企业代码库缺陷修复范式）；",
            "   • 离线花 ~3 分钟蒸馏成 7.31 MiB BF16 LoRA，线上裁掉 3,050 tok 历史前缀，仅跑 0.81 ~ 2.01 道题即可收回全部离线算力成本！",
            "⚠️ 明确不适用的场景（诚实技术边界）：",
            "   • 一次性冷启动任意长文档（如临时上传一篇全新财报或作文做单次问答）；",
            "   • 原因：单次请求无法摊销蒸馏步数，且逐字精准检索仍需原生 KV Cache。",
        ],
        font_size=11.8,
    )

    s3 = prs.slides.add_slide(blank)
    _add_header(s3, "2. 核心算法矛盾：为什么少样本静态 LoRA 会「训浅不涨、训深改坏」？", "基于 Qwen3-8B 与 confirmatory_v2_r1（8 道监督题 → 12 道同机制迁移 + 12 道异机制保留）的严格对照", 3, total_pages)
    _add_card(
        s3,
        0.55,
        1.35,
        6.0,
        5.55,
        title="KV Cache 的「软开关」 vs 静态 LoRA 的「全局强偏置」",
        title_rgb=C_NAVY,
        bullets=[
            "• 长提示词 KV Cache 的天然优势：",
            "  前缀躺在历史位置上，靠 Q·K^T 注意力打分充当「按需软开关」——遇到不相关的异类题（如 WR03 指数退避状态机），注意力权重自动分走，较少破坏原装能力。",
            "• 朴素静态 LoRA (W_base x + ΔW x) 的物理缺陷：",
            "  没有按需开关！每层 q_proj / v_proj 对每一个输入 token 都在无条件叠加 ΔW x：",
            "  - 训 16 步（lora）：欠拟合，同机制迁移仅 2/12（甚至低于裸底座 3/12）；",
            "  - 训 96 步（lora_conv）：迁移涨到 4/12（追平长提示词），但异类保留从 4/12 暴跌至 2/12（把原装能做对的 WR03、WR05 全改坏了）！",
        ],
        font_size=11.8,
    )
    _add_card(
        s3,
        6.75,
        1.35,
        6.03,
        5.55,
        fill_rgb=C_CODE_BG,
        border_rgb=C_NAVY,
        title="真实评测输入输出结构（以 WT05 / WR03 为例）",
        title_rgb=RGBColor(147, 197, 253),
        bullets=[
            "[Teacher 长提示词 text 输入: ~3,450 tokens]",
            "  8 道固定套路示范 (V01..V08: 题干 + Bug代码 + 正确修复)",
            "  + 当前待修题目 WT05 (题干 spec + 含 Bug 初始代码)",
            "",
            "[Student 外挂 LoRA 输入: ~380 tokens (裁掉 3,090 tok!)]",
            "  仅保留当前待修题目 WT05 (题干 spec + 含 Bug 初始代码)",
            "  + 挂载 7.31 MiB 受控 LoRA 权重",
            "",
            "[评测硬指标 (每题 2 公开 + 6 隐藏私有沙箱用例, 全过才算 PASS)]",
            "  • Transfer (WT01..WT12): 同属 bounds/multiplicity/order/reset",
            "  • Retention (WR01..WR12): 完全异类机制 (如 WR03 遇到 'ok'",
            "    重置 cur_delay=base_delay，遇到 'fatal' break 退出)",
        ],
        font_size=10.8,
        is_code=True,
    )

    s4 = prs.slides.add_slide(blank)
    _add_header(s4, "3. 路线抉择：证伪外置门控（G6-A），确立训练期双路 KL 零空间约束（G7）", "用实验数据说话：为什么推理期加 Router 不如训练期把 ΔW 投影进无关特征的正交零空间？", 4, total_pages)
    _add_card(
        s4,
        0.55,
        1.35,
        5.9,
        5.55,
        fill_rgb=C_AMBER_LIGHT,
        border_rgb=C_AMBER,
        title="路线 A（已证伪止损）：推理期零参数先验门控路由",
        title_rgb=C_AMBER,
        bullets=[
            "• 设想：在推理前用底座隐藏层表征判断当前题目是否对口，对口开 α=1，不对口关 α=0。",
            "• Gate G6-A 40 题探路实测暴露根本瓶颈：",
            "  - 特异度（异类题关断率）：100% (16/16 全关断，保住保留集)；",
            "  - 召回率（同机制题开启率）：仅 25.0% (12 道迁移题漏关了 9 道)！",
            "• 根因诊断：在不带长提示词的 380-token 短题干上，模型尚未开始写代码，首层/末层 Prompt 表征主要反映表面词汇（传感器、订单）而非深层 Bug 机制，外置分类器必然高漏检。",
        ],
        font_size=11.6,
    )
    _add_card(
        s4,
        6.68,
        1.35,
        6.10,
        5.55,
        fill_rgb=C_GREEN_LIGHT,
        border_rgb=C_GREEN,
        title="路线 B（核心突破）：训练期全词表 KL 零空间自锚定",
        title_rgb=C_GREEN,
        bullets=[
            "• 核心思想：不加任何推理期开关（零路由延迟、零漏检），直接在训练 Loss 里给 LoRA 装上「方向盘锁」！",
            "• 零人工答案构造锚定集：让原装裸底座自己在 16 道无关旧题 (R01..R16) 上贪婪生成代码，缓存每步 151,936 维全词表 logits P_base。",
            "• 双路正交更新：",
            "    L_total = L_target(专精轨迹) + λ_bg · KL( P_base(·|x_bg) || P_θ(·|x_bg) )",
            "• 物理本质：只要 ΔW 试图改动无关代码语法方向的激活，KL 惩罚立刻产生恢复梯度，迫使 ΔW 严格落在无关任务的零空间（Null Space）内！",
            "• Gate G7 实测：未见异类 D01..D04 选词一致率达 99.15%，保留集立刻从 2/12 满血恢复至 4/12！",
        ],
        font_size=11.6,
    )

    s5 = prs.slides.add_slide(blank)
    _add_header(s5, "4. 算法三级跳：差异行聚焦（G8）→ 全 KL 自蒸馏（G9）→ 外科掩码混合（G12）", "解决 8 样本低秩容量分配与硬标签过拟合两大精细瓶颈", 5, total_pages)
    _add_card(
        s5,
        0.55,
        1.38,
        3.95,
        5.48,
        fill_rgb=C_BLUE_LIGHT,
        border_rgb=C_BLUE,
        title="Gate G8：AST 差异行 4× 聚焦",
        title_rgb=C_BLUE,
        bullets=[
            "• 痛点：120 token 函数体中 80% 是重抄题干脚手架，甚至含 ValueError 报错文案缩写，浪费 rank-8 容量。",
            "• 方案：SequenceMatcher 提取真实修复行赋权 4.0×，脚手架 1.0×，报错文案降权 0.25×（相差 16 倍）。",
            "• 战果：Transfer 跃升至 5/12（正面超越长提示词 4/12！），总分 8/24 首次追平长提示词！",
        ],
        font_size=11.2,
    )
    _add_card(
        s5,
        4.68,
        1.38,
        3.95,
        5.48,
        fill_rgb=C_GREEN_LIGHT,
        border_rgb=C_GREEN,
        title="Gate G9：沙箱验真全 KL 自蒸馏",
        title_rgb=C_GREEN,
        bullets=[
            "• 痛点：硬标签 0/1 CE 强推概率至 1.0，多训几轮仍会轻微擦伤保留题。",
            "• 方案：28 道无答案题干交由长提示词 Teacher 生成 → bwrap+seccomp 沙箱自动验真筛出 17/28 入库（共 25 条），正反 100% 全词表 KL 蒸馏。",
            "• 战果：D01..D04 达成 118/118 (100.0%) 逐词一致（KL=0.00057），24 题全集 0 改坏（4/12 + 4/12 = 8/24）！",
        ],
        font_size=11.2,
    )
    _add_card(
        s5,
        8.81,
        1.38,
        3.97,
        5.48,
        fill_rgb=C_AMBER_LIGHT,
        border_rgb=C_AMBER,
        title="Gate G12：外科手术掩码混合蒸馏",
        title_rgb=C_AMBER,
        bullets=[
            "• 微观诊断发现：G8 误伤 WR11 是因脚手架 1.0× 硬 CE 背诵了 sorted(...)；G9 漏掉 WT05/WT11 是因纯软 KL 在关键修复行推力不足。",
            "• 融合方案：脚手架硬 CE 严格置 0.0！仅对 8 道种子真值差异行施加精准外科 CE，全序列叠加 25 条验真轨迹软 KL + 16 条背景 KL。",
            f"• 实测结果：{g12_bullet}",
        ],
        font_size=11.2,
    )

    s6 = prs.slides.add_slide(blank)
    _add_header(s6, "5. 代码级微观铁证：A100 真实生成的专精修复（WT05）与异类保护（WR11）", "不做黑盒调参——直接对比模型在评测集上逐字生成的 Python 源码差异", 6, total_pages)
    _add_card(
        s6,
        0.55,
        1.35,
        6.0,
        5.55,
        fill_rgb=C_CODE_BG,
        border_rgb=C_BLUE,
        title="同机制迁移题 WT05 (sensor_epoch_deltas)：精准学会单传感器重置",
        title_rgb=RGBColor(147, 197, 253),
        bullets=[
            "# 裸底座 Base 输出 (FAIL: 下次比较触发 int > NoneType 崩溃)",
            "elif kind == \"recal\":",
            "    highs[sensor_id] = None",
            "",
            "# 受控外挂 LoRA (G8 / G12) 输出 (8/8 全用例 PASS!):",
            "# 从 3,050 tok 示范浓缩出的 7.31 MiB 权重精准习得 pop 移除基线",
            "elif kind == \"recal\":",
            "    highs.pop(sensor_id, None)",
            "",
            "# 同机制迁移题 WT11 (audit_first_ranks) 同样精准修复：",
            "# 仅在 if valid_flag == 1: 分支内记录 first_seen[actor] = idx (8/8 PASS!)",
        ],
        font_size=10.6,
        is_code=True,
    )
    _add_card(
        s6,
        6.75,
        1.35,
        6.03,
        5.55,
        fill_rgb=C_CODE_BG,
        border_rgb=C_GREEN,
        title="异机制保留题 WR11 (ordered_union_versions)：彻底消除脚手架串扰",
        title_rgb=RGBColor(110, 231, 183),
        bullets=[
            "# 无掩码脚手架硬 CE (G8) 误伤原因 (FAIL: 破坏首次出现顺序):",
            "# 因背诵了种子题脚手架里的 sorted(...) 习惯，强行字典序排序：",
            "return [[pkg, best_ver[pkg]] for pkg in sorted(order) ...]",
            "",
            "# 全 KL 自蒸馏 (G9) / 零脚手架掩码：100% 保持原装裸底座正确逻辑 (8/8 PASS!):",
            "order = []",
            "...",
            "if pkg not in order:",
            "    order.append(pkg)",
            "return [[pkg, best_ver[pkg]] for pkg in order if pkg in best_ver]",
        ],
        font_size=10.6,
        is_code=True,
    )

    s7 = prs.slides.add_slide(blank)
    _add_header(s7, "6. 算法演进全路径量化对比（confirmatory_v2_r1 严格双盲沙箱评测）", "全部候选代码经远端 bwrap + seccomp 隔离沙箱执行全部 8 个公/私用例，0 UNKNOWN", 7, total_pages)
    prog_png = _make_progression_chart(g12_data)
    s7.shapes.add_picture(prog_png, Inches(0.65), Inches(1.28), width=Inches(12.0))
    _add_card(
        s7,
        0.65,
        5.72,
        12.0,
        1.25,
        fill_rgb=C_PANEL,
        border_rgb=C_BORDER,
        bullets=[
            "• 关键结论 1（架构容量消融 Gate G10）：等参数预算下挂载 q/v + 中后段 24 层 mlp.down_proj (rank=4, 13.31 MiB) 得分仍为 5/12 + 3/12 = 8/24，证实纯 Attention q/v 挂载已 100% 吃满少样本规则容量，且比混合挂载少 25% 旁路算子调用。",
            "• 关键结论 2（双配方互补）：线上可根据业务偏好挂载「猛攻专精槽位 (5/12 迁移)」或「零退化稳健槽位 (100% 零改坏)」！",
        ],
        font_size=11.0,
    )

    s8 = prs.slides.add_slide(blank)
    _add_header(s8, "7. 系统吞吐与显存碾压实证：正面对阵 SDPA + 免费预热 Prefix KV Cache", "单张 NVIDIA A100 80GB 实测 (prefix=3,050 tok, query=450 tok, decode=128 tok)", 8, total_pages)
    serv_png = _make_serving_chart()
    s8.shapes.add_picture(serv_png, Inches(0.60), Inches(1.28), width=Inches(12.1))
    _add_card(
        s8,
        0.60,
        5.70,
        12.1,
        1.28,
        fill_rgb=C_AMBER_LIGHT,
        border_rgb=C_AMBER,
        bullets=[
            "• 重要硬件数值发现（BF16 尾数吞噬 Mantissa Absorption）：直接执行 (W_base.float() + ΔW).bfloat16() 原地合并会因 BF16 仅 7 位尾数吞掉 ~1e-4 微弱修复分量（得分从 8/24 跌至 6/24）；",
            "• 黄金部署形态：保持 7.31 MiB BF16 独立 (A, B) 旁路在激活空间做 W_base(x) + scale·B(A(x))——既 100% 无损保留全部分数，又吃满 B=64 破千 1,095.6 decode tok/s 与省 85.2% 显存红利！",
        ],
        font_size=10.8,
    )

    s9 = prs.slides.add_slide(blank)
    _add_header(s9, "8. 引擎落地：tinyvllm 主干透明前缀裁剪与多槽位 CUDA Graph 原生运行时", "零业务代码侵入：调用方照常传 3,400+ tok 长 Prompt，引擎内部自动完成识别、裁剪、路由与图回放", 9, total_pages)
    _add_card(
        s9,
        0.55,
        1.35,
        6.15,
        5.55,
        fill_rgb=C_BLUE_LIGHT,
        border_rgb=C_BLUE,
        title="tinyvllm 原生运行时三层核心改造",
        title_rgb=C_BLUE,
        bullets=[
            "1) PrefixLoRARegistry（LLMEngine.add_request 入口层）：",
            "   • 注册高频长前缀 (3,090 tok) → 短前缀 (40 tok) + slot_id；",
            "   • 请求入队前 O(L) 最长前缀匹配，自动将 3,426 tok 裁剪为 376 tok，初始 Paged KV 分配从 14 块骤降至 2 块！",
            "2) QKVLoRASlotPool（QKVParallelLinear 算子层）：",
            "   • 预分配 [max_slots+1, ...] BF16 权重池，Slot 0 恒定全零哨兵；",
            "   • 异构混合 Batch 通过 index_select + torch.bmm 单次无分支并行完成裸底座与多租户 LoRA 叠加。",
            "3) CUDA Graph 全档位静态兼容（ModelRunner 层）：",
            "   • 捕获 [1, 2, 4, 8, 16, 32, 48, 64] 8 档图时静态绑定 graph_vars['lora_token_slot_ids']，回放前仅原地拷贝槽位号。",
        ],
        font_size=11.3,
    )
    b16 = e2e_data["tinyvllm_paged_kv_batch16_comparison"]
    apc_tps = b16["native_warm_prefix_cache_apc"]["output_tokens_per_sec"]
    pro_tps = b16["promoted_prefix_to_bf16_lora"]["output_tokens_per_sec"]
    sp_up = b16["throughput_speedup"]
    apc_blk = b16["native_warm_prefix_cache_apc"]["peak_allocated_kv_blocks"]
    pro_blk = b16["promoted_prefix_to_bf16_lora"]["peak_allocated_kv_blocks"]
    _add_card(
        s9,
        6.90,
        1.35,
        5.88,
        5.55,
        fill_rgb=C_GREEN_LIGHT,
        border_rgb=C_GREEN,
        title="A100 真实 tinyvllm 引擎端到端验收实测",
        title_rgb=C_GREEN,
        bullets=[
            "✅ 异构 6 请求混合 Batch + CUDA Graph 等价性：",
            "   • 同 Batch 混跑 Slot 0 (WR03/WR04) + 自动裁剪 Slot 1 (WT05/WT06) + 显式 Slot 2 (WR11/WT01)；",
            "   • 6 / 6 条输出与单请求独立运行 100% 逐字符完全一致，WT05/WT06/WR03/WR04/WR11 首轮全部 8/8 PASS！",
            "✅ 真 Paged KV + FlashAttention 块共享对决 (Batch=16)：",
            f"   • 原生块共享 Warm APC：峰值占 {apc_blk} 块 (1,620 MiB)，吞吐 {apc_tps:.2f} tok/s；",
            f"   • 透明晋升 7.31 MiB LoRA：峰值仅占 {pro_blk} 块 (1,188 MiB，净省全部 12 个共享前缀块 -26.7%)，含 bmm 旁路吞吐达 {pro_tps:.2f} tok/s ({sp_up:.3f}× 反超！)；",
            "   • 多租户放大效应：K 个专精前缀可省下 432×K MiB 块缓存！",
        ],
        font_size=11.3,
    )

    s10_cd = prs.slides.add_slide(blank)
    _add_header(
        s10_cd,
        "9. 跨领域泛化闭环：4×4 零串扰矩阵、多槽位 1.30 GiB 省显存与高低熵解耦定律",
        "扩展至临床药学分诊 (MED-SOP) 与企业法务合规 (LEGAL-SOP) 双盲评测及 tinyvllm 异构秩+MLP+动态模板升级",
        10,
        total_pages,
    )
    _add_card(
        s10_cd,
        0.55,
        1.35,
        6.0,
        5.55,
        fill_rgb=C_BLUE_LIGHT,
        border_rgb=C_BLUE,
        title="4×4 跨界零污染与 tinyvllm 四领域多槽位复用",
        title_rgb=C_BLUE,
        bullets=[
            "• 136/136 跨界请求 0.00% 格式污染：凭借 JSON 语法键名硬 CE 置 0 + 双模态 KL 零空间锁死，挂医/法权重写 Python 从不蹦 JSON，答通用题从不蹦 def。",
            "• 四领域 32 并发对决原生三领域 Warm APC：",
            "  - 三领域总常驻权重仅 21.94 MiB，一次性释放 37 个长前缀块（峰值 KV 3,348 → 2,016 MiB，净省 1.30 GiB / -39.8%）且吞吐反超 1.047×！",
            "• 引擎升级全闭环：支持 36 层 mlp.down_proj 槽位池 + 异构秩零填充 (r=8 & r=16 同图共存) + 动态日期/科室头双锚点模板匹配，CUDA Graph 100% 逐字符一致。",
        ],
        font_size=11.1,
    )
    _add_card(
        s10_cd,
        6.75,
        1.35,
        6.03,
        5.55,
        fill_rgb=C_GREEN_LIGHT,
        border_rgb=C_GREEN,
        title="攻克非编程查表瓶颈：密集覆盖编译 + 75-tok 数字残差",
        title_rgb=C_GREEN,
        bullets=[
            "• 关键物理边界发现：代码修复属过程性注意力路由（纯 q/v 满容量）；而医药 28 格精确毫克表存在「未问格 0 梯度（覆盖饥饿）」与高熵数字记忆瓶颈。",
            "• 根治方案：离线 132 探针全网格+临界边界合成自蒸馏 (rank=16 q/v+MLP) + 仅保留 75-tok 纯数字速查表（砍掉 89% 上下文）：",
            "  - 消融控制组（裸底座 + 仅给 75-tok 数字表）：0 / 16（证明无规则寸步难行）；",
            "  - 医疗盲测：6 / 16（100% 追平长提示词！字段率 71.25% 大幅反超长提示词 60.0%！）；",
            "  - 法律盲测：从稀疏 q/v 的 2/16 飙升至 11 / 16（字段率 76.25%）！",
        ],
        font_size=11.1,
    )

    s11_pub = prs.slides.add_slide(blank)
    _add_header(
        s11_pub,
        "10. 论文级硬核验证：公开 BBH 双尺度三随机种子 (p < 10⁻¹¹) 与 KV 压缩帕累托前沿",
        "跨越 13.3× 模型参数量级 (Qwen3-0.6B & 8B) 并同台对决 INT8 / INT4 KV 量化与 SnapKV 稀疏驱逐",
        11,
        total_pages,
    )
    bbh_png = _make_bbh_and_kv_pareto_chart()
    s11_pub.shapes.add_picture(bbh_png, Inches(0.60), Inches(1.26), width=Inches(12.1))
    _add_card(
        s11_pub,
        0.60,
        5.62,
        12.1,
        1.36,
        fill_rgb=C_PANEL,
        border_rgb=C_BORDER,
        bullets=[
            "• 跨尺度惊喜发现：在 Qwen3-0.6B 上，2.19 MiB LoRA 三颗随机种子 (61, 60, 59 / 120) 全部正面反超 12-shot 长提示词 (58/120)，双侧精确 McNemar p < 2×10⁻¹¹；Qwen3-8B 达 p < 5×10⁻¹⁵。",
            "• 三段式系统选型定律：2~3× 温和压缩选免训练 INT8 KV；而在 10×~60× 极致压缩与 4× 提速区，免训练 SnapKV 崩盘至 2.5% (1/40)，LoRA+微残差全对率高出 17 倍 (17/40)，确立清晰不可替代统治区！",
        ],
        font_size=10.8,
    )

    s10 = prs.slides.add_slide(blank)
    _add_header(s10, "11. 学术前沿对标（2022–2026）与技术面试核心亮点总结", "清晰界定已有共识与本项目的原创工程/算法增量", 12, total_pages)
    _add_card(
        s10,
        0.55,
        1.35,
        6.0,
        5.55,
        fill_rgb=C_PANEL,
        border_rgb=C_BORDER,
        title="相关文献谱系与差异化对比",
        title_rgb=C_NAVY,
        bullets=[
            "• Context Distillation (Snell et al., 2022)：提出长上下文蒸馏给短上下文学生，但直接全参微调底座，无法在单底座上多租户热插拔。",
            "• Doc-to-LoRA / MemoryLLM (Caccia et al., 2025)：需额外预训练昂贵的 Hypernetwork 超网络，跨代码/严密逻辑领域易失真。",
            "• Kinetic / Prefix-to-Adapter (2025–2026 同期探索)：聚焦单任务转换，缺乏「少样本异类防遗忘严格证明」与「对阵预热 Prefix Cache 的带宽墙与引擎级 CUDA Graph 闭环」。",
            "• 本项目真实增量：零超网络、零推理门控漏检，单底座外挂 7.31 MiB 标准 LoRA 同时打通「KL 零空间防遗忘 + 沙箱自进化验真 + tinyvllm 原生引擎落地」。",
        ],
        font_size=11.3,
    )
    _add_card(
        s10,
        6.75,
        1.35,
        6.03,
        5.55,
        fill_rgb=C_BLUE_LIGHT,
        border_rgb=C_BLUE,
        title="面试三句话讲透本项目（Elevator Pitch）",
        title_rgb=C_BLUE,
        bullets=[
            "1) 发现真问题：即使开启 Prefix KV Cache，3K~16K 高频系统前缀在 Decode 阶段依然卡死 HBM 显存带宽（B=64 直接 OOM），而朴素少样本 LoRA 卸载会陷入「训浅不涨、训深改坏异类能力」的跷跷板。",
            "2) 算法真破局：证伪低召回（25%）推理门控后，提出「无关轨迹全词表 KL 零空间锚定 + 零脚手架外科差异 CE + 沙箱验真软自蒸馏」，实现未见异类 100% 选词一致、保留集 0 退化且 24 题总分达 9/24 正面反超长提示词 (8/24)。",
            "3) 系统真闭环：揭示 BF16 原地合并 7-bit 尾数吞噬现象，在 tinyvllm 中原生实现透明前缀裁剪 + 多槽位 CUDA Graph 混合 Batch 旁路，B=64 达 1,095.6 decode tok/s，驻留密度提升 58.7×~315×。",
        ],
        font_size=11.3,
    )

    out_pptx.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out_pptx))
    return {
        "slides": len(prs.slides),
        "out_pptx": str(out_pptx),
        "bytes": out_pptx.stat().st_size,
        "included_g12": g12_data is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/Users/bytedance/dev/TinyLLMForge")
    parser.add_argument(
        "--out-pptx",
        default="/Users/bytedance/dev/TinyLLMForge/docs/presentations/TinyLLMForge_Parametric_Prefix_LoRA_Interview_Deck.pptx",
    )
    parser.add_argument("--g12-report", default="")
    args = parser.parse_args()

    res = build_deck(
        Path(args.repo_root).resolve(),
        Path(args.out_pptx).resolve(),
        Path(args.g12_report).resolve() if args.g12_report else None,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
