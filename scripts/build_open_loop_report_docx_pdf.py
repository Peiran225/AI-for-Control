"""Build high-resolution DOCX and PDF reports for the Transformer u(t) experiment."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIG = ROOT / "paper_runs"
PY = os.environ.get("PYTHON", sys.executable)
RENDER = os.environ.get("RENDER_DOCX")


BLUE = RGBColor(46, 116, 181)
DARK = RGBColor(31, 41, 55)
MUTED = RGBColor(82, 96, 109)
HEADER_FILL = "F2F4F7"


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_width(cell, width_inches: float) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(int(width_inches * 1440)))
    tc_w.set(qn("w:type"), "dxa")


def set_run_font(run, size: float | None = None, bold: bool = False, color: RGBColor | None = None, name: str = "Calibri") -> None:
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "PingFang SC")
    if size is not None:
        run.font.size = Pt(size)
    run.bold = bold
    if color is not None:
        run.font.color.rgb = color


def add_para(doc: Document, text: str = "", style: str | None = None, size: float = 10.5, bold: bool = False) -> None:
    p = doc.add_paragraph(style=style)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.1
    r = p.add_run(text)
    set_run_font(r, size=size, bold=bold, color=DARK)


def add_note(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.05
    r = p.add_run(text)
    set_run_font(r, size=8.3, color=MUTED)


def add_heading(doc: Document, text: str, level: int = 1) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14 if level == 1 else 10)
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run(text)
    set_run_font(r, size=15 if level == 1 else 12.5, bold=True, color=BLUE)


def add_equation(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run(text)
    set_run_font(r, size=10.5, name="Cambria Math")
    r.font.color.rgb = RGBColor(17, 24, 39)


def add_table(doc: Document, headers: list[str], rows: list[list[str]], widths: list[float], font_size: float = 8.5) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    for i, text in enumerate(headers):
        set_cell_width(hdr[i], widths[i])
        set_cell_shading(hdr[i], HEADER_FILL)
        hdr[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(text)
        set_run_font(r, size=font_size, bold=True, color=DARK)
    for row in rows:
        cells = table.add_row().cells
        for i, text in enumerate(row):
            set_cell_width(cells[i], widths[i])
            cells[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cells[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i >= 2 and i <= 5 else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(text)
            set_run_font(r, size=font_size, color=DARK)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def add_figure(doc: Document, rel_path: str, caption: str | None = None, width: float = 6.3) -> None:
    path = ROOT / rel_path
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(3)
    p.add_run().add_picture(str(path), width=Inches(width))
    if caption:
        cp = doc.add_paragraph()
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cp.paragraph_format.space_after = Pt(8)
        r = cp.add_run(caption)
        set_run_font(r, size=9, color=MUTED)


def setup_doc() -> Document:
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(0.75)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)
    styles = doc.styles
    styles["Normal"].font.name = "Calibri"
    styles["Normal"]._element.rPr.rFonts.set(qn("w:eastAsia"), "PingFang SC")
    styles["Normal"].font.size = Pt(10.5)
    return doc


def add_title(doc: Document, title: str, subtitle: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(title)
    set_run_font(r, size=23, bold=True, color=DARK)
    sp = doc.add_paragraph()
    sp.paragraph_format.space_after = Pt(12)
    sr = sp.add_run(subtitle)
    set_run_font(sr, size=10.5, color=MUTED)


def build_en() -> Path:
    doc = setup_doc()
    add_title(doc, "Transformer u(t) Experiment Report", "Manuscript time-dependent control trained by PMP/KKT optimality gaps")
    add_para(doc, "This report reproduces the manuscript's time-dependent Transformer control strategy u_theta(t): [0,T] -> [0,u_max].")

    add_heading(doc, "1. Model and Training Objective")
    add_para(doc, "We use the population dynamics and cost from the manuscript:")
    add_equation(doc, "dN_i/dt = (r_i - phi_i u(t) - M_i G(N(t))) N_i(t)")
    add_equation(doc, "G(N) = log(1 + (1/m) sum_k N_k)")
    add_equation(doc, "J(u) = alpha^T N(T) + int_0^T [ beta^T N(t) + gamma u(t) ] dt")
    add_para(doc, "Reported parameters: T=10, m=21, u_max=3, alpha=1, beta=0.1, gamma=20, and N_i(0)=10. The control is represented by a small Transformer encoder over normalized time: u_theta(t_k)=u_max sigma(g_theta(t_k)).")
    add_para(doc, "Given u_theta(t), we roll out N_theta(t), solve the costate equation backward, and compute the switching function")
    add_equation(doc, "psi(t) = H_u(N, lambda, u) = gamma - sum_i phi_i lambda_i(t) N_i(t).")
    add_table(
        doc,
        ["component", "role"],
        [
            ["non-singular KKT loss", "enforces u=0 when psi>0 and u=u_max when psi<0"],
            ["singular loss", "near psi=0, matches u_theta(t) to the singular candidate u_sing(N(t))"],
        ],
        [1.7, 4.6],
        font_size=9.2,
    )
    add_para(doc, "Here q(t) is a smooth weight that switches between the singular condition near psi(t)=0 and the boundary KKT condition away from psi(t)=0.")

    add_heading(doc, "2. Learned u(t), N(t), and N(u)")
    add_para(doc, "The learned control starts high, transitions to a lower interior/singular-like region, and increases again near the terminal portion.")
    add_figure(doc, "paper_runs/open_loop_ut_report/ut_nt_trajectory_clean.png", width=6.35)
    add_para(doc, "The N(u) phase plot shows population directly against the applied control value; color indicates time.")
    add_figure(doc, "paper_runs/open_loop_ut_report/nu_phase_plot_clean.png", width=6.35)

    add_heading(doc, "3. Training Loss Trajectories for PMP/KKT Conditions")
    add_para(doc, "The table reports the smallest recorded training loss. In this experiment, the training loss is the manuscript's PMP/KKT optimality gap, composed of the singular condition and the non-singular Hamiltonian minimization condition.")
    add_table(
        doc,
        ["metric", "value"],
        [
            ["total training loss / PMP-KKT optimality gap", "0.02637"],
            ["singular-condition training loss", "0.00167"],
            ["non-singular Hamiltonian-minimization training loss", "0.02471"],
            ["objective J on the training grid", "384.76"],
            ["range and mean of u_theta(t)", "min 1.038, max 2.758, mean 1.372"],
            ["terminal mean state", "1.207"],
        ],
        [3.2, 2.0],
        font_size=9.2,
    )
    add_figure(doc, "paper_runs/open_loop_ut_report/training_loss_trajectory_clean.png", width=6.35)
    add_para(doc, "The final pointwise diagnostic below uses a smooth weight q(t) to separate the two regimes: near psi(t)=0 it emphasizes the singular-condition error; away from psi(t)=0 it emphasizes the boundary KKT error.")
    add_figure(doc, "paper_runs/open_loop_ut_report/pmp_condition_components_clean.png", width=6.35)

    add_heading(doc, "4. Comparison With Related Work [3]")
    add_para(doc, "Related work [3], Neural-PMP / PMP-gradient, also follows a Pontryagin-style procedure: forward state integration, backward costate recursion, and Hamiltonian-gradient updates of a discrete control sequence. For comparison on the same model and parameters, we implemented the corresponding control-update step from [3].")
    add_table(
        doc,
        ["method", "control update / training criterion", "PMP/KKT gap", "objective J*", "note"],
        [
            ["Transformer u(t)", "paper PMP/KKT optimality-gap loss", "0.0523", "386.70", "main u(t) reproduction"],
            ["Neural-PMP [3]", "Hamiltonian-gradient update of control sequence", "7.47", "387.02", "related-work comparison"],
            ["direct grid cost", "minimize discretized J", "0.805", "386.47", "reference only"],
            ["constant u=1.5", "no training", "76.89", "400.40", "scale check"],
        ],
        [1.35, 2.3, 0.85, 0.85, 1.15],
        font_size=8.0,
    )
    add_note(doc, "* The J values in this table are computed after fixing u(t), reintegrating N(t) with a finer-step fourth-order Runge-Kutta method, and then applying the manuscript objective definition. This is only to use the same numerical integration accuracy across methods.")
    add_para(doc, "Under the same PMP/KKT gap calculation, the Transformer u_theta(t) has a much smaller gap than the Neural-PMP implementation of [3]. When J is recomputed with the same numerical evaluator, the Transformer also has a slightly lower objective value in this run.")
    add_figure(doc, "paper_runs/neural_pmp_baseline_beta01/neural_pmp_ut_nt.png", width=6.35)
    add_figure(doc, "paper_runs/neural_pmp_baseline_beta01/neural_pmp_training_curve.png", width=6.35)
    add_figure(doc, "paper_runs/neural_pmp_baseline_beta01/neural_pmp_reference_gap_closeup.png", width=6.35)

    add_heading(doc, "5. Conclusion")
    add_para(doc, "The requested u(t) reproduction is complete. The Transformer control trajectory trained with the manuscript's PMP/KKT optimality-gap loss is smooth and satisfies the control bounds, reduces the training optimality gap from 76.26 to 0.02637, and gives J approximately 386.70 under the common numerical evaluation. The state-dependent extension u(t,N) is not included in this report because it requires different optimality conditions.")
    out = REPORTS / "ut_reproduction_report.docx"
    doc.save(out)
    return out


def build_zh() -> Path:
    doc = setup_doc()
    add_title(doc, "开环 u(t) 实验报告", "论文设定下的 Transformer 控制策略，使用 PMP/KKT optimality gap 训练")
    add_para(doc, "本报告只关注时间相关开环控制策略 u_theta(t): [0,T] -> [0,u_max]。不包含反馈控制 u(t,N)，因为该情形的最优性条件不同。")

    add_heading(doc, "1. 模型和训练目标")
    add_para(doc, "我们使用论文中的种群动力学模型和目标函数：")
    add_equation(doc, "dN_i/dt = (r_i - phi_i u(t) - M_i G(N(t))) N_i(t)")
    add_equation(doc, "G(N) = log(1 + (1/m) sum_k N_k)")
    add_equation(doc, "J(u) = alpha^T N(T) + int_0^T [ beta^T N(t) + gamma u(t) ] dt")
    add_para(doc, "本次参数为 T=10，m=21，u_max=3，alpha=1，beta=0.1，gamma=20，初始条件为 N_i(0)=10。控制函数由小型 Transformer encoder 表示：u_theta(t_k)=u_max sigma(g_theta(t_k))。")
    add_para(doc, "给定 u_theta(t) 后，先正向求解得到 N_theta(t)，再反向求解 costate，并计算 switching function：")
    add_equation(doc, "psi(t) = H_u(N, lambda, u) = gamma - sum_i phi_i lambda_i(t) N_i(t).")
    add_table(
        doc,
        ["component", "作用"],
        [
            ["non-singular KKT loss", "当 psi>0 时推动 u=0；当 psi<0 时推动 u=u_max"],
            ["singular loss", "当 psi 接近 0 时，使 u_theta(t) 接近 singular candidate u_sing(N(t))"],
        ],
        [1.85, 4.45],
        font_size=9.0,
    )
    add_para(doc, "其中 q(t) 是一个平滑权重，用来在 psi(t) 接近 0 的 singular condition 和 psi(t) 远离 0 的边界 KKT condition 之间切换。")

    add_heading(doc, "2. 学到的 u(t)、N(t) 和 N(u)")
    add_para(doc, "学到的开环控制在开始阶段较高，中间阶段下降到接近 singular control 的内部取值，末端附近再次升高。")
    add_figure(doc, "paper_runs/open_loop_ut_report/ut_nt_trajectory_clean.png", width=6.35)
    add_para(doc, "下面的 N(u) 图将同一次状态轨迹中的种群水平直接画在控制值 u(t) 上，颜色表示时间。")
    add_figure(doc, "paper_runs/open_loop_ut_report/nu_phase_plot_clean.png", width=6.35)

    add_heading(doc, "3. 各 PMP/KKT 最优性条件的 training loss trajectory")
    add_para(doc, "下表汇总训练过程中记录到的最小 training loss。这里的 training loss 正是论文中定义的 PMP/KKT optimality gap，由 singular condition 和 non-singular Hamiltonian minimization condition 两部分组成。")
    add_table(
        doc,
        ["metric", "value"],
        [
            ["total training loss / PMP-KKT optimality gap", "0.02637"],
            ["singular-condition training loss", "0.00167"],
            ["non-singular Hamiltonian-minimization training loss", "0.02471"],
            ["objective J on the training grid", "384.76"],
            ["range and mean of u_theta(t)", "min 1.038, max 2.758, mean 1.372"],
            ["terminal mean state", "1.207"],
        ],
        [3.2, 2.0],
        font_size=9.2,
    )
    add_para(doc, "训练曲线将论文中的两类最优性条件分开展示：当 psi(t) 接近 0 时，对应 singular condition；当 psi(t) 不为 0 时，对应 Hamiltonian 在控制边界 0 或 u_max 上取最小的 non-singular condition。结果显示，singular-condition gap 较小，主要剩余误差来自 non-singular Hamiltonian-minimization condition，尤其是控制切换附近。")
    add_figure(doc, "paper_runs/open_loop_ut_report/training_loss_trajectory_clean.png", width=6.35)
    add_para(doc, "下面的逐点诊断图使用平滑权重 q(t) 区分两类区域：当 psi(t) 接近 0 时，主要看 singular-condition error；当 psi(t) 远离 0 时，主要看边界 KKT error。")
    add_figure(doc, "paper_runs/open_loop_ut_report/pmp_condition_components_clean.png", width=6.35)

    add_heading(doc, "4. 与 related work [3] 的比较")
    add_para(doc, "Related work [3] 的 Neural-PMP / PMP-gradient 方法也基于 Pontryagin 思路：先正向求解状态，再反向求解 costate，并用 Hamiltonian gradient 更新离散控制序列。为了在同一个模型和参数下比较，我们实现了 [3] 中对应的控制更新步骤。")
    add_table(
        doc,
        ["方法", "训练/更新准则", "PMP/KKT gap", "目标函数 J*", "说明"],
        [
            ["Transformer u(t)", "论文中的 PMP/KKT optimality-gap loss", "0.0523", "386.70", "本报告主实验"],
            ["Neural-PMP [3]", "Hamiltonian-gradient 更新离散控制序列", "7.47", "387.02", "related work 对比"],
            ["direct grid cost", "直接最小化离散 J", "0.805", "386.47", "仅作参考"],
            ["constant u=1.5", "无训练", "76.89", "400.40", "尺度参考"],
        ],
        [1.35, 2.3, 0.85, 0.85, 1.15],
        font_size=8.0,
    )
    add_note(doc, "* 表中的 J 都是在控制 u(t) 固定后，用更细时间步长的四阶 Runge-Kutta 方法重新求解状态 N(t)，再按论文目标函数定义计算得到。这样做只是为了让不同方法的数值比较使用同一个积分精度。")
    add_para(doc, "在同一个 PMP/KKT gap 计算方式下，Transformer u_theta(t) 的 gap 明显小于 [3] 的 Neural-PMP 实现；按同一数值评估方式重新计算目标函数 J 时，本次实验中 Transformer 的 J 也略低。")
    add_figure(doc, "paper_runs/neural_pmp_baseline_beta01/neural_pmp_ut_nt.png", width=6.35)
    add_figure(doc, "paper_runs/neural_pmp_baseline_beta01/neural_pmp_training_curve.png", width=6.35)
    add_figure(doc, "paper_runs/neural_pmp_baseline_beta01/neural_pmp_reference_gap_closeup.png", width=6.35)

    add_heading(doc, "5. 结论")
    add_para(doc, "老师要求的 u(t) 复现实验已经完成。使用论文中的 PMP/KKT optimality-gap loss 训练得到的 Transformer 控制轨迹是一个平滑且满足控制约束的 u(t)，训练 optimality gap 从 76.26 降到 0.02637；按同一数值评估方式计算，目标函数值为 J 约 386.70。本报告不包含状态相关扩展 u(t,N)。")
    out = REPORTS / "open_loop_ut_reproduction_report_zh.docx"
    doc.save(out)
    return out


def convert_to_pdf(docx_path: Path) -> Path:
    env = os.environ.copy()
    env["TMPDIR"] = "/private/tmp"
    env["TEMP"] = "/private/tmp"
    env["TMP"] = "/private/tmp"
    subprocess.run(
        ["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(REPORTS), str(docx_path)],
        check=True,
        cwd=str(ROOT),
        env=env,
    )
    return docx_path.with_suffix(".pdf")


def render(docx_path: Path) -> None:
    if not RENDER:
        print(f"Skipping page render for {docx_path.name}; set RENDER_DOCX to enable it.")
        return
    out_dir = REPORTS / f"{docx_path.stem}_render"
    subprocess.run(
        [PY, RENDER, str(docx_path), "--output_dir", str(out_dir), "--emit_pdf"],
        check=True,
        cwd=str(ROOT),
    )


def main() -> None:
    for builder in (build_en, build_zh):
        docx_path = builder()
        pdf_path = convert_to_pdf(docx_path)
        render(docx_path)
        print(docx_path)
        print(pdf_path)


if __name__ == "__main__":
    main()
