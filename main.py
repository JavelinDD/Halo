"""
图片文字对比工具（Tkinter + PaddleOCR）

=======================
依赖安装（建议 Python 3.9+）
=======================
1) 安装 PaddlePaddle（根据你的系统选择 CPU/GPU 版本，以下示例为 CPU）
   pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple

2) 安装其余依赖
   pip install paddleocr opencv-python pillow numpy

运行方式：
   python main.py
"""

from __future__ import annotations

import os
import re
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from difflib import SequenceMatcher

import cv2
import numpy as np
from paddleocr import PaddleOCR


class OCRCompareApp:
    """主界面与业务逻辑封装类。"""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("图片文字一致性对比（PaddleOCR）")
        self.root.geometry("1100x760")

        # 保存用户选择的两张图片路径
        self.image_path_1 = tk.StringVar()
        self.image_path_2 = tk.StringVar()

        # 懒加载 OCR：首次对比时初始化，避免启动时卡顿
        self.ocr: PaddleOCR | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        """创建界面组件。"""
        top_frame = ttk.Frame(self.root, padding=12)
        top_frame.pack(fill="x")

        ttk.Label(top_frame, text="图片1：").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=6)
        ttk.Entry(top_frame, textvariable=self.image_path_1, width=90).grid(row=0, column=1, sticky="ew", pady=6)
        ttk.Button(top_frame, text="选择图片1", command=lambda: self.select_image(self.image_path_1)).grid(
            row=0, column=2, padx=(8, 0), pady=6
        )

        ttk.Label(top_frame, text="图片2：").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=6)
        ttk.Entry(top_frame, textvariable=self.image_path_2, width=90).grid(row=1, column=1, sticky="ew", pady=6)
        ttk.Button(top_frame, text="选择图片2", command=lambda: self.select_image(self.image_path_2)).grid(
            row=1, column=2, padx=(8, 0), pady=6
        )

        top_frame.columnconfigure(1, weight=1)

        action_frame = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        action_frame.pack(fill="x")

        self.compare_btn = ttk.Button(action_frame, text="开始识别并对比", command=self.run_compare_async)
        self.compare_btn.pack(side="left")

        self.status_var = tk.StringVar(value="请选择两张图片后开始。")
        ttk.Label(action_frame, textvariable=self.status_var, foreground="#1f4b99").pack(side="left", padx=12)

        result_frame = ttk.Frame(self.root, padding=12)
        result_frame.pack(fill="both", expand=True)

        # 左右两个文本框用于显示 OCR 文本及差异高亮
        left_panel = ttk.LabelFrame(result_frame, text="图片1识别结果", padding=10)
        right_panel = ttk.LabelFrame(result_frame, text="图片2识别结果", padding=10)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 6))
        right_panel.pack(side="left", fill="both", expand=True, padx=(6, 0))

        self.text_left = tk.Text(left_panel, wrap="word", font=("Consolas", 11))
        self.text_right = tk.Text(right_panel, wrap="word", font=("Consolas", 11))
        self.text_left.pack(fill="both", expand=True)
        self.text_right.pack(fill="both", expand=True)

        # 差异高亮样式：黄色背景 + 红色文字
        self.text_left.tag_configure("diff", background="#fff59d", foreground="#b71c1c")
        self.text_right.tag_configure("diff", background="#fff59d", foreground="#b71c1c")

        bottom_frame = ttk.LabelFrame(self.root, text="对比结论", padding=12)
        bottom_frame.pack(fill="x", padx=12, pady=(0, 12))

        self.summary_label = ttk.Label(bottom_frame, text="暂无结果", font=("Microsoft YaHei", 11, "bold"))
        self.summary_label.pack(anchor="w")

        self.detail_var = tk.StringVar(value="")
        ttk.Label(bottom_frame, textvariable=self.detail_var).pack(anchor="w", pady=(6, 0))

    def select_image(self, target_var: tk.StringVar) -> None:
        """弹出文件选择框并写入路径。"""
        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片文件", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"), ("所有文件", "*.*")],
        )
        if file_path:
            target_var.set(file_path)

    def run_compare_async(self) -> None:
        """异步执行 OCR 与比对，避免阻塞 GUI。"""
        path1 = self.image_path_1.get().strip()
        path2 = self.image_path_2.get().strip()

        if not path1 or not path2:
            messagebox.showwarning("提示", "请先选择两张图片。")
            return

        if not os.path.exists(path1) or not os.path.exists(path2):
            messagebox.showerror("错误", "图片路径无效，请重新选择。")
            return

        self.compare_btn.config(state="disabled")
        self.status_var.set("正在识别，请稍候...")

        thread = threading.Thread(target=self._compare_images, args=(path1, path2), daemon=True)
        thread.start()

    def _init_ocr_if_needed(self) -> None:
        """按需初始化 PaddleOCR。"""
        if self.ocr is None:
            # lang='ch' 模型可覆盖中文、英文与数字场景
            self.ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)

    @staticmethod
    def preprocess_image(image_path: str) -> np.ndarray:
        """图片预处理：灰度、去噪、二值化，提升 OCR 稳定性。"""
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取图片：{image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.medianBlur(gray, 3)
        binary = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            10,
        )

        # 轻量形态学操作：连接断裂笔画，降低噪点干扰
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return processed

    @staticmethod
    def clean_text(text: str) -> str:
        """文本清理：统一空白符并去除首尾空格。"""
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def extract_text(self, image_path: str) -> str:
        """执行 OCR 并提取全部识别文本。"""
        self._init_ocr_if_needed()
        assert self.ocr is not None

        processed_img = self.preprocess_image(image_path)
        ocr_result = self.ocr.ocr(processed_img, cls=True)

        lines: list[str] = []
        if ocr_result and ocr_result[0]:
            for line in ocr_result[0]:
                if line and len(line) >= 2:
                    text = line[1][0]
                    if text:
                        lines.append(text)

        raw_text = "\n".join(lines)
        return self.clean_text(raw_text)

    @staticmethod
    def similarity_ratio(text1: str, text2: str) -> float:
        """计算相似度（0~1）。"""
        return SequenceMatcher(None, text1, text2).ratio()

    def _set_text_with_diff(self, text_widget: tk.Text, content: str, diff_ranges: list[tuple[int, int]]) -> None:
        """写入文本并按字符范围高亮差异。"""
        text_widget.config(state="normal")
        text_widget.delete("1.0", "end")
        text_widget.insert("1.0", content)
        text_widget.tag_remove("diff", "1.0", "end")

        for start, end in diff_ranges:
            if end > start:
                text_widget.tag_add("diff", f"1.0+{start}c", f"1.0+{end}c")

        text_widget.config(state="disabled")

    @staticmethod
    def get_diff_ranges(text1: str, text2: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """使用序列匹配算法，找出两段文本各自差异区间。"""
        matcher = SequenceMatcher(None, text1, text2)
        ranges1: list[tuple[int, int]] = []
        ranges2: list[tuple[int, int]] = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "equal":
                ranges1.append((i1, i2))
                ranges2.append((j1, j2))

        return ranges1, ranges2

    def _compare_images(self, path1: str, path2: str) -> None:
        """后台线程执行识别+对比。"""
        try:
            text1 = self.extract_text(path1)
            text2 = self.extract_text(path2)

            is_equal = text1 == text2
            ratio = self.similarity_ratio(text1, text2)
            diff1, diff2 = self.get_diff_ranges(text1, text2)

            self.root.after(0, self._render_result, text1, text2, is_equal, ratio, diff1, diff2)
        except Exception as exc:
            self.root.after(0, self._on_error, str(exc))

    def _render_result(
        self,
        text1: str,
        text2: str,
        is_equal: bool,
        ratio: float,
        diff1: list[tuple[int, int]],
        diff2: list[tuple[int, int]],
    ) -> None:
        """将识别和对比结果渲染到界面。"""
        self._set_text_with_diff(self.text_left, text1, diff1)
        self._set_text_with_diff(self.text_right, text2, diff2)

        if is_equal:
            self.summary_label.config(text="✅ 两张图片识别出的文字完全一致", foreground="#1b5e20")
            self.detail_var.set(f"相似度：{ratio:.2%}")
        else:
            self.summary_label.config(text="❌ 两张图片文字不一致（已高亮差异）", foreground="#b71c1c")
            self.detail_var.set(f"相似度：{ratio:.2%}")

        self.status_var.set("识别完成。")
        self.compare_btn.config(state="normal")

    def _on_error(self, error_message: str) -> None:
        """处理错误并恢复按钮状态。"""
        self.status_var.set("识别失败。")
        self.compare_btn.config(state="normal")
        messagebox.showerror("识别失败", error_message)


def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
    app = OCRCompareApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
