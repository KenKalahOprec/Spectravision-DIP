"""
=============================================================================
SPECTRAVISION DIP LABORATORY — DIGITAL IMAGE PROCESSING & MATRIX COMPUTING
Laboratorium Pengolahan Citra Digital, Analisis Matriks & Filter Spasial
Peneliti / Developer: Jade Aurestha
Theme: Dark Slate & Peach Studio (Template Desain Matriks Spektral)
=============================================================================
Fitur & Kapabilitas Ilmiah:
1. 🧮 4 OPERASI ARITMATIKA MATRIKS CITRA LENGKAP:
   - [ + ] Penjumlahan: Penjumlahan Citra (A + B), Alpha Blending (αA + (1-α)B), Skalar Kecerahan (+c)
   - [ - ] Pengurangan: Pengurangan Terarah (A - B), Selisih Mutlak Spasial |A - B|, Skalar Reduksi (-c)
   - [ × ] Perkalian: Perkalian Masking Ter-normalisasi (A * B / 255), Penskalaan Kontras (*c)
   - [ ÷ ] Pembagian: Rasio Spektral (A / (B + 1) * 255), Penskalaan Reduksi (/c)
2. 🔬 SPATIAL FILTERING & MORPHOLOGY:
   - Grayscale, Canny Edge Detection (Slider Ambang T1/T2 Dinamis), Sobel Gradient,
     Laplacian Derivative, Gaussian Smoothing (Kernel Dinamis), Inversi Citra (Negatif),
     Thermal Jet False-Color, Otsu Adaptive Thresholding, Emboss 3D Texture, Deteksi Kontur.
   - Penyetelan Spasial Interaktif: Brightness, Contrast, Gamma, Blur Radius.
3. 🔪 INTENSITY & BIT-PLANE SLICING:
   - Gray-Level Intensity Slicing [A, B] dengan opsi Pertahankan Latar (Preserve BG).
   - Bit-Plane Decomposition (Bits 0 sampai 7 individual & Rekonstruksi Multi-Bit MSB).
4. 📊 SPECTRAL HISTOGRAM & HARALICK GLCM TEXTURE ANALYSIS:
   - Dual-Wave Density Plot (Peach #E8A87C untuk Asli & Teal #4ECDC4 untuk Hasil).
   - Entropi Informasi Shannon (bits/pixel), Mean Spasial (μ), Standar Deviasi (σ).
   - Gray-Level Co-occurrence Matrix (GLCM): Contrast, Dissimilarity, Homogeneity, Energy, Correlation.
5. 🔍 TELEMETRI SPASIAL & DUAL/3-WAY VIEWPORT INTERAKTIF:
   - Pelacak koordinat kursor live (X, Y) dan intensitas tensor piksel (R, G, B) real-time.
   - Citra Kalibrasi Spektral Sintetis Otomatis saat peluncuran.
   - Pratinjau Citra B dan kontrol pemilihan operan kedua yang transparan.
=============================================================================
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from skimage.feature import graycomatrix, graycoprops
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class SpectraVisionStudio(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("SpectraVision DIP Lab • Jade Aurestha")
        self.geometry("1400x920")
        self.minsize(1180, 780)

        # ---------------------------------------------------------------------
        # PALET WARNA TEMPLATE STUDIO ILMIAH
        # ---------------------------------------------------------------------
        self.C_BG = "#1E1E24"           # Deep Matte Charcoal
        self.C_CANVAS = "#141418"       # Main Workspace Canvas
        self.C_PANEL = "#2B2D42"        # Toolbar & Control Panels
        self.C_SUBPANEL = "#212332"     # Inset Header & Status Bar
        self.C_BORDER = "#3A3D52"       # Separator Lines
        self.C_PEACH = "#E8A87C"        # Primary Accent (Peach Gold)
        self.C_TEAL = "#4ECDC4"         # Secondary Accent (Teal Aqua)
        self.C_TEXT = "#F8F9FA"         # Clean Off-White
        self.C_MUTED = "#ADB5BD"        # Secondary Text Muted

        self.configure(bg=self.C_BG)

        # ---------------------------------------------------------------------
        # DATA CITRA & STATE PEMROSESAN
        # ---------------------------------------------------------------------
        self.filepath_a = None
        self.filename_a = "Citra Kalibrasi Spektral"
        self.img_a_orig = None          # Citra A Resolusi Penuh (RGB, uint8)
        self.img_a_initial = None       # Salinan murni citra asli (tidak terpengaruh pipeline untuk reset)

        self.filepath_b = None
        self.filename_b = "Pola Sintetis Lingkaran Konsentris"
        self.img_b_orig = None          # Citra B Resolusi Penuh (RGB, uint8)

        self.img_processed = None       # Hasil Akhir Pemrosesan (RGB, uint8)

        # Viewport Mode: "DUAL" (Split Komparasi) atau "3WAY" (Citra A, B, dan Hasil)
        self.view_mode = "DUAL"
        self.active_right_tab = "arithmetic"
        self.tk_cache = {}

        # Filter State
        self.active_filter_mode = "sliders"  # "sliders" atau id preset: "canny", "sobel", dll.

        # ---------------------------------------------------------------------
        # VARIABEL 4 OPERASI ARITMATIKA
        # ---------------------------------------------------------------------
        self.var_arith_op = tk.StringVar(value="ADD")
        self.var_arith_target = tk.StringVar(value="Antar Citra (Citra A & B)")  # Default operan antar dua citra A & B

        # Penjumlahan
        self.var_add_mode = tk.StringVar(value="Penjumlahan Biasa (A + B)")
        self.var_add_alpha = tk.DoubleVar(value=0.5)
        self.var_add_scalar = tk.IntVar(value=40)

        # Pengurangan
        self.var_sub_mode = tk.StringVar(value="Pengurangan Terarah (A - B)")
        self.var_sub_scalar = tk.IntVar(value=40)

        # Perkalian
        self.var_mul_mode = tk.StringVar(value="Perkalian Masking (A * B / 255)")
        self.var_mul_scalar = tk.DoubleVar(value=1.5)

        # Pembagian
        self.var_div_mode = tk.StringVar(value="Rasio Spektral (A / (B + 1) * 255)")
        self.var_div_scalar = tk.DoubleVar(value=2.0)

        # ---------------------------------------------------------------------
        # VARIABEL FILTER SPASIAL
        # ---------------------------------------------------------------------
        self.var_brightness = tk.IntVar(value=0)
        self.var_contrast = tk.IntVar(value=0)
        self.var_gamma = tk.DoubleVar(value=1.0)
        self.var_blur = tk.IntVar(value=1)
        self.var_canny1 = tk.IntVar(value=100)
        self.var_canny2 = tk.IntVar(value=200)

        # ---------------------------------------------------------------------
        # VARIABEL SLICING INTENSITAS
        # ---------------------------------------------------------------------
        self.var_slicing_type = tk.StringVar(value="Gray-Level Slicing")
        self.var_slice_low = tk.IntVar(value=80)
        self.var_slice_high = tk.IntVar(value=180)
        self.var_slice_highlight = tk.IntVar(value=255)
        self.var_slice_preserve_bg = tk.BooleanVar(value=False)
        self.var_bit_plane = tk.IntVar(value=7)

        # ---------------------------------------------------------------------
        # VARIABEL TABEL RGB & INSPEKSI TENSOR
        # ---------------------------------------------------------------------
        self.var_rgb_source = tk.StringVar(value="Citra Asli")
        self.var_rgb_x = tk.IntVar(value=0)
        self.var_rgb_y = tk.IntVar(value=0)

        # ---------------------------------------------------------------------
        # PEMBANGUNAN ANTARMUKA STUDIO
        # ---------------------------------------------------------------------
        self._setup_ttk_styles()
        self._build_top_menu_bar()
        self._build_main_workspace()
        self._build_bottom_status_bar()

        # Inisialisasi Citra B Default & Muat/Bangkitkan Citra Awal
        self._generate_default_image_b("circle")
        self._try_load_default_image()

        # Binding resize window agar viewport fleksibel dan adaptif
        self.bind("<Configure>", self._on_window_configure)
        self._resize_job = None

        # Pintasan Keyboard Cepat: Ctrl+R untuk Reset ke Citra Asli
        self.bind("<Control-r>", lambda e: self.reset_to_original_image())
        self.bind("<Control-R>", lambda e: self.reset_to_original_image())

    # =========================================================================
    # STYLING SISTEM TTK
    # =========================================================================
    def _setup_ttk_styles(self):
        self.style = ttk.Style(self)
        try:
            self.style.theme_use('clam')
        except Exception:
            pass

        self.style.configure(".", background=self.C_PANEL, foreground=self.C_TEXT, font=("Segoe UI", 9))
        self.style.configure("TLabel", background=self.C_PANEL, foreground=self.C_TEXT)
        self.style.configure("Header.TLabel", font=("Segoe UI", 10, "bold"), foreground=self.C_PEACH)
        self.style.configure("Muted.TLabel", foreground=self.C_MUTED, font=("Segoe UI", 8))

        self.style.configure("TLabelframe", background=self.C_PANEL, bordercolor=self.C_BORDER)
        self.style.configure("TLabelframe.Label", background=self.C_PANEL, foreground=self.C_PEACH, font=("Segoe UI", 9, "bold"))

        self.style.configure("TCombobox", fieldbackground=self.C_SUBPANEL, background=self.C_BORDER, foreground=self.C_TEXT)
        self.style.map("TCombobox", fieldbackground=[('readonly', self.C_SUBPANEL)])

        self.style.configure("Horizontal.TScale", background=self.C_PANEL, troughcolor=self.C_SUBPANEL)

        self.style.configure("Treeview", background=self.C_CANVAS, foreground=self.C_TEXT, fieldbackground=self.C_CANVAS, font=("Consolas", 8))
        self.style.configure("Treeview.Heading", background=self.C_SUBPANEL, foreground=self.C_PEACH, font=("Segoe UI", 8, "bold"))
        self.style.map("Treeview", background=[('selected', self.C_PEACH)], foreground=[('selected', self.C_BG)])

    # =========================================================================
    # 1. TOP MENU BAR (STUDIO HEADER & ACTIONS)
    # =========================================================================
    def _build_top_menu_bar(self):
        menu_bar = tk.Frame(self, bg=self.C_PANEL, height=42, bd=0)
        menu_bar.pack(side=tk.TOP, fill=tk.X)

        # Brand / Identitas Ilmiah Laboratorium
        lbl_logo = tk.Label(menu_bar, text="SPECTRAVISION DIP LAB", bg=self.C_PANEL, fg=self.C_TEXT,
                            font=("Segoe UI", 11, "bold"))
        lbl_logo.pack(side=tk.LEFT, padx=(16, 6), pady=8)

        lbl_by = tk.Label(menu_bar, text="•  Jade Aurestha", bg=self.C_PANEL, fg=self.C_PEACH,
                          font=("Segoe UI", 9, "bold"))
        lbl_by.pack(side=tk.LEFT, padx=(0, 20), pady=8)

        # Menu Action Items
        menu_actions = [
            ("File", self._show_file_menu),
            ("Aritmatika Matriks", lambda: self.switch_right_tab("arithmetic")),
            ("Filter Spasial", lambda: self.switch_right_tab("filters")),
            ("Analisis Spektral", lambda: self.switch_right_tab("histogram")),
            ("Slicing Intensitas", lambda: self.switch_right_tab("slicing")),
            ("Matriks Transformasi", lambda: self.switch_right_tab("gallery")),
            ("Tabel RGB", lambda: self.switch_right_tab("rgbtable")),
            ("Tentang Lab", self._show_about_dialog)
        ]

        for title, cmd in menu_actions:
            btn = tk.Button(menu_bar, text=title, font=("Segoe UI", 9), bg=self.C_PANEL, fg=self.C_MUTED,
                            activebackground=self.C_SUBPANEL, activeforeground=self.C_TEXT,
                            bd=0, relief=tk.FLAT, padx=8, pady=4, cursor="hand2", command=cmd)
            btn.pack(side=tk.LEFT, padx=1)
            btn.bind("<Enter>", lambda e, b=btn: b.configure(fg=self.C_TEXT))
            btn.bind("<Leave>", lambda e, b=btn: b.configure(fg=self.C_MUTED))

        # Viewport Switcher & Quick Actions di Kanan
        view_box = tk.Frame(menu_bar, bg=self.C_PANEL)
        view_box.pack(side=tk.RIGHT, padx=16)

        btn_reset_top = tk.Button(view_box, text="🔄 Reset Original", font=("Segoe UI", 8, "bold"),
                                  bg=self.C_SUBPANEL, fg=self.C_PEACH, activebackground=self.C_BORDER,
                                  bd=1, relief=tk.SOLID, padx=9, pady=3, cursor="hand2",
                                  command=self.reset_to_original_image)
        btn_reset_top.pack(side=tk.RIGHT, padx=6)

        btn_rgb_top = tk.Button(view_box, text="📊 Tabel RGB", font=("Segoe UI", 8, "bold"),
                                bg=self.C_SUBPANEL, fg=self.C_TEAL, activebackground=self.C_BORDER,
                                bd=1, relief=tk.SOLID, padx=8, pady=3, cursor="hand2",
                                command=self.show_rgb_table_dialog)
        btn_rgb_top.pack(side=tk.RIGHT, padx=4)

        self.btn_view_dual = tk.Button(view_box, text="Split View", font=("Segoe UI", 8, "bold"),
                                       bg=self.C_PEACH, fg=self.C_BG, bd=0, relief=tk.FLAT, padx=10, pady=3,
                                       cursor="hand2", command=lambda: self.set_view_mode("DUAL"))
        self.btn_view_dual.pack(side=tk.RIGHT, padx=4)

        self.btn_view_3way = tk.Button(view_box, text="3-Way View", font=("Segoe UI", 8),
                                       bg=self.C_SUBPANEL, fg=self.C_TEXT, bd=0, relief=tk.FLAT, padx=10, pady=3,
                                       cursor="hand2", command=lambda: self.set_view_mode("3WAY"))
        self.btn_view_3way.pack(side=tk.RIGHT, padx=4)

        # Garis batas bawah menu
        sep = tk.Frame(self, bg=self.C_BORDER, height=1)
        sep.pack(side=tk.TOP, fill=tk.X)

    def _show_file_menu(self):
        m = tk.Menu(self, tearoff=0, bg=self.C_SUBPANEL, fg=self.C_TEXT, activebackground=self.C_PEACH, activeforeground=self.C_BG)
        m.add_command(label="📂 Buka Citra A (Operan Utama)...", command=self.open_image_a_dialog)
        m.add_command(label="🖼️ Buka Citra B (Operan Sekunder)...", command=self.open_image_b_dialog)
        m.add_command(label="🎯 Bangkitkan Citra Kalibrasi Sintetis", command=self._generate_synthetic_test_chart)
        m.add_separator()
        m.add_command(label="🔄 Reset ke Citra Asli (Original)\tCtrl+R", command=self.reset_to_original_image)
        m.add_separator()
        m.add_command(label="💾 Simpan Hasil Citra Terproses...", command=self.save_processed_image)
        m.add_command(label="🔁 Jadikan Hasil Sebagai Citra A Baru", command=self.use_result_as_input_a)
        m.add_separator()
        m.add_command(label="🚪 Keluar", command=self.quit)
        m.post(self.winfo_rootx() + 180, self.winfo_rooty() + 42)

    def _show_about_dialog(self):
        info = (
            "SPECTRAVISION DIP LABORATORY\n"
            "Digital Image Processing, Spatial Analysis & Matrix Computing Suite\n\n"
            "Peneliti & Pengembang : Jade Aurestha\n"
            "Versi                   : 3.0 (Scientific Studio Release)\n"
            "Komputasi Numerik      : Python 3.14, NumPy, OpenCV, Scikit-Image, Matplotlib\n\n"
            "Kapabilitas Utama:\n"
            "• 4 Operasi Aritmatika Citra Lengkap (Penjumlahan, Pengurangan, Perkalian, Pembagian)\n"
            "• Analisis Spektral Histogram & Fitur Tekstur GLCM Haralick\n"
            "• Entropi Informasi Shannon, Standar Deviasi, & Mean Spasial\n"
            "• Intensity Slicing & Bit-Plane Decomposition (Bits 0-7)\n"
            "• Live Mouse Telemetry & Inspeksi Intensitas Tensor Piksel"
        )
        messagebox.showinfo("Tentang SpectraVision DIP Lab", info)

    def set_view_mode(self, mode):
        self.view_mode = mode
        if mode == "DUAL":
            self.btn_view_dual.configure(bg=self.C_PEACH, fg=self.C_BG, font=("Segoe UI", 8, "bold"))
            self.btn_view_3way.configure(bg=self.C_SUBPANEL, fg=self.C_TEXT, font=("Segoe UI", 8))
            self.frame_3way.pack_forget()
            self.frame_dual.pack(fill=tk.BOTH, expand=True)
        else:
            self.btn_view_3way.configure(bg=self.C_PEACH, fg=self.C_BG, font=("Segoe UI", 8, "bold"))
            self.btn_view_dual.configure(bg=self.C_SUBPANEL, fg=self.C_TEXT, font=("Segoe UI", 8))
            self.frame_dual.pack_forget()
            self.frame_3way.pack(fill=tk.BOTH, expand=True)
        self.refresh_active_view()

    # =========================================================================
    # 2. MAIN WORKSPACE (LEFT DOCK + CENTER VIEWPORT + RIGHT PANEL)
    # =========================================================================
    def _build_main_workspace(self):
        workspace = tk.Frame(self, bg=self.C_BG)
        workspace.pack(fill=tk.BOTH, expand=True)

        # A. LEFT VERTICAL TOOL DOCK (50px)
        left_dock = tk.Frame(workspace, bg=self.C_PANEL, width=50)
        left_dock.pack(side=tk.LEFT, fill=tk.Y)
        left_dock.pack_propagate(False)

        sep_dock = tk.Frame(workspace, bg=self.C_BORDER, width=1)
        sep_dock.pack(side=tk.LEFT, fill=tk.Y)

        dock_tools = [
            ("📂", "Buka Citra A", self.open_image_a_dialog),
            ("🖼️", "Buka Citra B", self.open_image_b_dialog),
            ("📊", "Tabel Matriks RGB", lambda: self.switch_right_tab("rgbtable")),
            ("🔄", "Reset ke Citra Asli (Original)", self.reset_to_original_image),
            ("💾", "Simpan Hasil", self.save_processed_image),
            ("🔁", "Gunakan Sebagai Citra A", self.use_result_as_input_a),
            ("🎯", "Sampel Matriks 3x3", self.sample_roi_matrix)
        ]

        for icon, tooltip, cmd in dock_tools:
            btn = tk.Button(left_dock, text=icon, font=("Segoe UI", 12), bg=self.C_PANEL, fg=self.C_TEXT,
                            activebackground=self.C_BORDER, activeforeground=self.C_PEACH,
                            bd=0, relief=tk.FLAT, width=3, height=2, cursor="hand2", command=cmd)
            btn.pack(pady=4)
            btn.bind("<Enter>", lambda e, b=btn: b.configure(bg=self.C_BORDER))
            btn.bind("<Leave>", lambda e, b=btn: b.configure(bg=self.C_PANEL))

        # B. CENTER MAIN CANVAS (#141418)
        self.center_canvas = tk.Frame(workspace, bg=self.C_CANVAS)
        self.center_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas_inner = tk.Frame(self.center_canvas, bg=self.C_CANVAS)
        self.canvas_inner.pack(fill=tk.BOTH, expand=True, padx=14, pady=10)

        # 1. Tampilan DUAL / SPLIT VIEW
        self.frame_dual = tk.Frame(self.canvas_inner, bg=self.C_CANVAS)
        self.frame_dual.pack(fill=tk.BOTH, expand=True)

        # Frame Citra Asli (Kiri)
        box_orig = tk.Frame(self.frame_dual, bg="#050505", bd=2, relief=tk.SOLID, highlightbackground=self.C_BORDER)
        box_orig.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        tag_orig = tk.Label(box_orig, text="Citra Asli (RGB Input)", bg=self.C_BG, fg=self.C_TEXT,
                            font=("Segoe UI", 9, "bold"), padx=10, pady=3)
        tag_orig.place(x=10, y=10)

        self.lbl_view_orig = tk.Label(box_orig, bg="#050505", text="Memuat Citra...", fg=self.C_MUTED)
        self.lbl_view_orig.pack(fill=tk.BOTH, expand=True)
        self.lbl_view_orig.bind("<Motion>", lambda e: self._on_canvas_mouse_move(e, is_proc=False))
        self.lbl_view_orig.bind("<Button-1>", lambda e: self._on_canvas_click(e, is_proc=False))

        # Frame Citra Hasil (Kanan)
        box_proc = tk.Frame(self.frame_dual, bg="#050505", bd=2, relief=tk.SOLID, highlightbackground=self.C_BORDER)
        box_proc.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(6, 0))

        self.tag_proc = tk.Label(box_proc, text="Hasil Transformasi Matriks", bg=self.C_BG, fg=self.C_PEACH,
                                 font=("Segoe UI", 9, "bold"), padx=10, pady=3)
        self.tag_proc.place(x=10, y=10)

        btn_quick_reset = tk.Button(box_proc, text="🔄 Reset Original", font=("Segoe UI", 8, "bold"),
                                    bg=self.C_BG, fg=self.C_PEACH, activebackground=self.C_BORDER,
                                    bd=1, relief=tk.SOLID, padx=8, pady=2, cursor="hand2",
                                    command=self.reset_to_original_image)
        btn_quick_reset.place(relx=1.0, y=10, x=-10, anchor="ne")

        self.lbl_view_proc = tk.Label(box_proc, bg="#050505", text="Memuat Hasil...", fg=self.C_MUTED)
        self.lbl_view_proc.pack(fill=tk.BOTH, expand=True)
        self.lbl_view_proc.bind("<Motion>", lambda e: self._on_canvas_mouse_move(e, is_proc=True))
        self.lbl_view_proc.bind("<Button-1>", lambda e: self._on_canvas_click(e, is_proc=True))

        # 2. Tampilan 3-WAY VIEW (Citra A, Citra B, Citra Hasil)
        self.frame_3way = tk.Frame(self.canvas_inner, bg=self.C_CANVAS)

        top_3way = tk.Frame(self.frame_3way, bg=self.C_CANVAS)
        top_3way.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 6))

        box_3a = tk.Frame(top_3way, bg="#050505", bd=1, relief=tk.SOLID, highlightbackground=self.C_BORDER)
        box_3a.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        tk.Label(box_3a, text="Citra A (Matriks 1)", bg=self.C_BG, fg=self.C_TEXT, font=("Segoe UI", 8, "bold")).place(x=8, y=8)
        self.lbl_3way_a = tk.Label(box_3a, bg="#050505", text="Citra A", fg=self.C_MUTED)
        self.lbl_3way_a.pack(fill=tk.BOTH, expand=True)

        box_3b = tk.Frame(top_3way, bg="#050505", bd=1, relief=tk.SOLID, highlightbackground=self.C_BORDER)
        box_3b.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4, 0))
        tk.Label(box_3b, text="Citra B (Matriks 2)", bg=self.C_BG, fg=self.C_PEACH, font=("Segoe UI", 8, "bold")).place(x=8, y=8)
        self.lbl_3way_b = tk.Label(box_3b, bg="#050505", text="Citra B", fg=self.C_MUTED)
        self.lbl_3way_b.pack(fill=tk.BOTH, expand=True)

        box_3res = tk.Frame(self.frame_3way, bg="#050505", bd=1, relief=tk.SOLID, highlightbackground=self.C_BORDER)
        box_3res.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, pady=(4, 0))
        tk.Label(box_3res, text="Hasil Aritmatika Spasial", bg=self.C_BG, fg=self.C_TEAL, font=("Segoe UI", 8, "bold")).place(x=8, y=8)

        btn_quick_reset_3w = tk.Button(box_3res, text="🔄 Reset Original", font=("Segoe UI", 8, "bold"),
                                       bg=self.C_BG, fg=self.C_PEACH, activebackground=self.C_BORDER,
                                       bd=1, relief=tk.SOLID, padx=8, pady=2, cursor="hand2",
                                       command=self.reset_to_original_image)
        btn_quick_reset_3w.place(relx=1.0, y=8, x=-8, anchor="ne")

        self.lbl_3way_res = tk.Label(box_3res, bg="#050505", text="Hasil Aritmatika", fg=self.C_MUTED)
        self.lbl_3way_res.pack(fill=tk.BOTH, expand=True)

        # C. RIGHT CONTROL PANEL (335px)
        sep_right = tk.Frame(workspace, bg=self.C_BORDER, width=1)
        sep_right.pack(side=tk.LEFT, fill=tk.Y)

        self.right_panel = tk.Frame(workspace, bg=self.C_PANEL, width=340, bd=0)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        self.right_panel.pack_propagate(False)

        self._build_right_control_panel(self.right_panel)

    # =========================================================================
    # 3. RIGHT CONTROL PANEL (NAVIGASI TAB & AKORDEON PARAMETER)
    # =========================================================================
    def _build_right_control_panel(self, parent):
        tab_header = tk.Frame(parent, bg=self.C_SUBPANEL, height=36)
        tab_header.pack(side=tk.TOP, fill=tk.X)

        self.tab_defs = [
            ("arithmetic", "Aritmatika"),
            ("histogram", "Histogram"),
            ("filters", "Filters"),
            ("slicing", "Slicing"),
            ("gallery", "Galeri"),
            ("rgbtable", "Tabel RGB")
        ]

        self.right_tab_btns = {}
        self.right_tab_frames = {}

        for tab_id, label in self.tab_defs:
            btn = tk.Button(tab_header, text=label, font=("Segoe UI", 9),
                            bg=self.C_SUBPANEL, fg=self.C_MUTED, bd=0, relief=tk.FLAT,
                            padx=8, pady=7, cursor="hand2", command=lambda t=tab_id: self.switch_right_tab(t))
            btn.pack(side=tk.LEFT, padx=1)
            self.right_tab_btns[tab_id] = btn

            f = tk.Frame(parent, bg=self.C_PANEL)
            self.right_tab_frames[tab_id] = f

        # Bangun konten setiap tab kontrol
        self._build_section_arithmetic(self.right_tab_frames["arithmetic"])
        self._build_section_histogram(self.right_tab_frames["histogram"])
        self._build_section_filters(self.right_tab_frames["filters"])
        self._build_section_slicing(self.right_tab_frames["slicing"])
        self._build_section_gallery(self.right_tab_frames["gallery"])
        self._build_section_rgb_table(self.right_tab_frames["rgbtable"])

        # Default buka tab Aritmatika
        self.switch_right_tab("arithmetic")

    def switch_right_tab(self, active_id):
        self.active_right_tab = active_id
        for tab_id, btn in self.right_tab_btns.items():
            if tab_id == active_id:
                btn.configure(fg=self.C_PEACH, font=("Segoe UI", 9, "bold"), bg=self.C_PANEL)
            else:
                btn.configure(fg=self.C_MUTED, font=("Segoe UI", 9), bg=self.C_SUBPANEL)

        for tab_id, frame in self.right_tab_frames.items():
            if tab_id == active_id:
                frame.pack(fill=tk.BOTH, expand=True)
            else:
                frame.pack_forget()

        if active_id == "arithmetic":
            self.on_arithmetic_change()
        elif active_id == "histogram":
            self._update_histogram_chart()
        elif active_id == "filters":
            self.on_filter_param_change()
        elif active_id == "slicing":
            self.on_slicing_change()
        elif active_id == "gallery":
            self.refresh_active_view()
        elif active_id == "rgbtable":
            self._update_rgb_table()

    # =========================================================================
    # SECTION 1: 4 OPERASI ARITMATIKA MATRIKS CITRA
    # =========================================================================
    def _build_section_arithmetic(self, parent):
        scroll_c = tk.Canvas(parent, bg=self.C_PANEL, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=scroll_c.yview)
        sf = tk.Frame(scroll_c, bg=self.C_PANEL)
        sf.bind("<Configure>", lambda e: scroll_c.configure(scrollregion=scroll_c.bbox("all")))
        scroll_c.create_window((0, 0), window=sf, anchor="nw", width=320)
        scroll_c.configure(yscrollcommand=scrollbar.set)
        scroll_c.pack(side="left", fill="both", expand=True, padx=(2, 0))
        scrollbar.pack(side="right", fill="y")

        lbl_head = tk.Label(sf, text="4 Operasi Aritmatika Citra", bg=self.C_PANEL, fg=self.C_TEXT,
                            font=("Segoe UI", 11, "bold"))
        lbl_head.pack(anchor=tk.W, padx=12, pady=(12, 6))

        # 4 TOMBOL OPERATOR
        btn_grid = tk.Frame(sf, bg=self.C_PANEL)
        btn_grid.pack(fill=tk.X, padx=10, pady=4)

        self.arith_buttons = {}
        ops_info = [
            ("ADD", "[ + ] Penjumlahan", 0, 0),
            ("SUB", "[ - ] Pengurangan", 0, 1),
            ("MUL", "[ × ] Perkalian",   1, 0),
            ("DIV", "[ ÷ ] Pembagian",   1, 1)
        ]

        for code, title, r, c in ops_info:
            b = tk.Button(btn_grid, text=title, font=("Segoe UI", 9, "bold"),
                          bd=1, relief=tk.SOLID, padx=6, pady=8, cursor="hand2",
                          command=lambda op=code: self.set_arithmetic_operator(op))
            b.grid(row=r, column=c, padx=3, pady=3, sticky="ew")
            btn_grid.grid_columnconfigure(c, weight=1)
            self.arith_buttons[code] = b

        # Mode Target Operasi
        box_target = ttk.LabelFrame(sf, text="Domain Operasi", padding=(10, 6))
        box_target.pack(fill=tk.X, padx=10, pady=6)

        r1 = tk.Radiobutton(box_target, text="Citra A & Skalar (Konstanta c)",
                            variable=self.var_arith_target, value="Citra A & Skalar",
                            bg=self.C_PANEL, fg=self.C_TEXT, selectcolor=self.C_SUBPANEL,
                            activebackground=self.C_PANEL, activeforeground=self.C_PEACH,
                            command=self._on_arith_target_changed)
        r1.pack(anchor=tk.W, pady=2)

        r2 = tk.Radiobutton(box_target, text="Antar Dua Citra (Citra A & B)",
                            variable=self.var_arith_target, value="Antar Citra (Citra A & B)",
                            bg=self.C_PANEL, fg=self.C_TEXT, selectcolor=self.C_SUBPANEL,
                            activebackground=self.C_PANEL, activeforeground=self.C_PEACH,
                            command=self._on_arith_target_changed)
        r2.pack(anchor=tk.W, pady=2)

        # Kontainer Parameter Dinamis Tiap Operator
        self.box_arith_dynamic = ttk.LabelFrame(sf, text="Parameter Transformasi", padding=(10, 6))
        self.box_arith_dynamic.pack(fill=tk.X, padx=10, pady=6)

        self.arith_ctrl_panels = {}
        self._build_arithmetic_dynamic_controls(self.box_arith_dynamic)

        # Panel Info Citra B (Operan Sekunder)
        self.box_citra_b_info = ttk.LabelFrame(sf, text="Status Citra B (Operan 2)", padding=(10, 6))
        self.box_citra_b_info.pack(fill=tk.X, padx=10, pady=6)

        self.lbl_citra_b_status = tk.Label(self.box_citra_b_info, text="Citra B: Lingkaran Konsentris",
                                           bg=self.C_PANEL, fg=self.C_PEACH, font=("Segoe UI", 8),
                                           wraplength=280, justify=tk.LEFT)
        self.lbl_citra_b_status.pack(anchor=tk.W, pady=(0, 4))

        f_b_btns = tk.Frame(self.box_citra_b_info, bg=self.C_PANEL)
        f_b_btns.pack(fill=tk.X)

        btn_open_b = tk.Button(f_b_btns, text="📂 Buka Citra B...", font=("Segoe UI", 8),
                               bg=self.C_SUBPANEL, fg=self.C_TEXT, bd=1, relief=tk.SOLID,
                               cursor="hand2", command=self.open_image_b_dialog)
        btn_open_b.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        btn_cycle_b = tk.Button(f_b_btns, text="🔄 Pola Sintetis", font=("Segoe UI", 8),
                                bg=self.C_SUBPANEL, fg=self.C_TEAL, bd=1, relief=tk.SOLID,
                                cursor="hand2", command=self._cycle_synthetic_b)
        btn_cycle_b.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))

        # Tombol Aksi Eksekusi
        btn_apply = tk.Button(sf, text="Apply Matrix Operation", font=("Segoe UI", 9, "bold"),
                              bg=self.C_SUBPANEL, fg=self.C_PEACH, activebackground=self.C_BORDER,
                              bd=1, relief=tk.SOLID, padx=10, pady=6, cursor="hand2",
                              command=self.on_arithmetic_change)
        btn_apply.pack(fill=tk.X, padx=10, pady=(8, 3))

        btn_reset_arith = tk.Button(sf, text="🔄 Reset ke Citra Asli (Original)", font=("Segoe UI", 9),
                                    bg=self.C_PANEL, fg=self.C_PEACH, activebackground=self.C_BORDER,
                                    bd=1, relief=tk.SOLID, padx=10, pady=5, cursor="hand2",
                                    command=self.reset_to_original_image)
        btn_reset_arith.pack(fill=tk.X, padx=10, pady=3)

        btn_save = tk.Button(sf, text="💾 Simpan Hasil Citra", font=("Segoe UI", 9),
                             bg=self.C_PANEL, fg=self.C_TEXT, activebackground=self.C_BORDER,
                             bd=1, relief=tk.SOLID, padx=10, pady=5, cursor="hand2",
                             command=self.save_processed_image)
        btn_save.pack(fill=tk.X, padx=10, pady=(2, 6))

        # Terminal Log Spesifikasi Ilmiah Matriks
        f_specs = tk.LabelFrame(sf, text="Spesifikasi & Metrik Matriks Spasial", bg=self.C_SUBPANEL, fg=self.C_PEACH,
                                font=("Segoe UI", 8, "bold"), bd=1, relief=tk.SOLID, padx=8, pady=6)
        f_specs.pack(fill=tk.X, padx=10, pady=6)

        self.lbl_specs_log = tk.Label(f_specs, text="Memuat telemetri spasial...",
                                      bg=self.C_SUBPANEL, fg=self.C_MUTED, font=("Consolas", 8),
                                      justify=tk.LEFT, anchor=tk.W)
        self.lbl_specs_log.pack(fill=tk.X)

        self.set_arithmetic_operator("ADD")

    def _on_arith_target_changed(self):
        target = self.var_arith_target.get()
        op = self.var_arith_op.get()
        if "Antar Citra" in target:
            self.set_view_mode("3WAY")
        else:
            self.set_view_mode("DUAL")
        self.set_arithmetic_operator(op)

    def _build_arithmetic_dynamic_controls(self, parent):
        # 1. Kontrol Penjumlahan
        f_add = tk.Frame(parent, bg=self.C_PANEL)
        self.panel_add_inter = tk.Frame(f_add, bg=self.C_PANEL)
        tk.Label(self.panel_add_inter, text="Metode Penjumlahan Citra:", bg=self.C_PANEL, fg=self.C_MUTED, font=("Segoe UI", 8)).pack(anchor=tk.W)
        combo_add = ttk.Combobox(self.panel_add_inter, textvariable=self.var_add_mode, state="readonly",
                                 values=["Penjumlahan Biasa (A + B)", "Rata-rata Citra / Averaging ((A + B) / 2)", "Alpha Blending (αA + (1-α)B)"])
        combo_add.pack(fill=tk.X, padx=2, pady=(0, 4))
        combo_add.bind("<<ComboboxSelected>>", lambda e: self.on_arithmetic_change())

        def on_alpha_slider():
            if "Alpha Blending" not in self.var_add_mode.get():
                self.var_add_mode.set("Alpha Blending (αA + (1-α)B)")
            self.on_arithmetic_change()

        self._create_slider_row(self.panel_add_inter, "Bobot Alpha (α):", self.var_add_alpha, 0.0, 1.0, on_alpha_slider, is_float=True)

        self.panel_add_scalar = tk.Frame(f_add, bg=self.C_PANEL)
        self._create_slider_row(self.panel_add_scalar, "Skalar Tambah (+c):", self.var_add_scalar, 0, 200, self.on_arithmetic_change)
        self.arith_ctrl_panels["ADD"] = f_add

        # 2. Kontrol Pengurangan
        f_sub = tk.Frame(parent, bg=self.C_PANEL)
        self.panel_sub_inter = tk.Frame(f_sub, bg=self.C_PANEL)
        tk.Label(self.panel_sub_inter, text="Metode Pengurangan Citra:", bg=self.C_PANEL, fg=self.C_MUTED, font=("Segoe UI", 8)).pack(anchor=tk.W)
        combo_sub = ttk.Combobox(self.panel_sub_inter, textvariable=self.var_sub_mode, state="readonly",
                                 values=["Pengurangan Terarah (A - B)", "Selisih Mutlak |A - B|", "Pengurangan Terbalik (B - A)"])
        combo_sub.pack(fill=tk.X, padx=2, pady=(0, 4))
        combo_sub.bind("<<ComboboxSelected>>", lambda e: self.on_arithmetic_change())

        self.panel_sub_scalar = tk.Frame(f_sub, bg=self.C_PANEL)
        self._create_slider_row(self.panel_sub_scalar, "Skalar Kurang (-c):", self.var_sub_scalar, 0, 200, self.on_arithmetic_change)
        self.arith_ctrl_panels["SUB"] = f_sub

        # 3. Kontrol Perkalian
        f_mul = tk.Frame(parent, bg=self.C_PANEL)
        self.panel_mul_inter = tk.Frame(f_mul, bg=self.C_PANEL)
        tk.Label(self.panel_mul_inter, text="Metode Perkalian Citra:", bg=self.C_PANEL, fg=self.C_MUTED, font=("Segoe UI", 8)).pack(anchor=tk.W)
        combo_mul = ttk.Combobox(self.panel_mul_inter, textvariable=self.var_mul_mode, state="readonly",
                                 values=["Perkalian Masking (A * B / 255)", "Perkalian Matriks (A * B / 128)"])
        combo_mul.pack(fill=tk.X, padx=2, pady=(0, 4))
        combo_mul.bind("<<ComboboxSelected>>", lambda e: self.on_arithmetic_change())

        self.panel_mul_scalar = tk.Frame(f_mul, bg=self.C_PANEL)
        self._create_slider_row(self.panel_mul_scalar, "Skalar Kontras (*c):", self.var_mul_scalar, 0.1, 4.0, self.on_arithmetic_change, is_float=True)
        self.arith_ctrl_panels["MUL"] = f_mul

        # 4. Kontrol Pembagian
        f_div = tk.Frame(parent, bg=self.C_PANEL)
        self.panel_div_inter = tk.Frame(f_div, bg=self.C_PANEL)
        tk.Label(self.panel_div_inter, text="Metode Pembagian Citra:", bg=self.C_PANEL, fg=self.C_MUTED, font=("Segoe UI", 8)).pack(anchor=tk.W)
        combo_div = ttk.Combobox(self.panel_div_inter, textvariable=self.var_div_mode, state="readonly",
                                 values=["Rasio Spektral ((A + 1)/(B + 1) * 128)", "Pembagian Citra (A / (B + 1) * 255)"])
        combo_div.pack(fill=tk.X, padx=2, pady=(0, 4))
        combo_div.bind("<<ComboboxSelected>>", lambda e: self.on_arithmetic_change())

        self.panel_div_scalar = tk.Frame(f_div, bg=self.C_PANEL)
        self._create_slider_row(self.panel_div_scalar, "Skalar Pembagi (/c):", self.var_div_scalar, 1.0, 10.0, self.on_arithmetic_change, is_float=True)
        self.arith_ctrl_panels["DIV"] = f_div

    def set_arithmetic_operator(self, op_code):
        self.var_arith_op.set(op_code)

        for code, btn in self.arith_buttons.items():
            if code == op_code:
                btn.configure(bg=self.C_PEACH, fg=self.C_BG, font=("Segoe UI", 9, "bold"))
            else:
                btn.configure(bg=self.C_SUBPANEL, fg=self.C_TEXT, font=("Segoe UI", 9))

        is_scalar = ("Skalar" in self.var_arith_target.get())

        for code, panel in self.arith_ctrl_panels.items():
            if code == op_code:
                panel.pack(fill=tk.X, padx=4, pady=4)
                if code == "ADD":
                    if is_scalar:
                        self.panel_add_inter.pack_forget()
                        self.panel_add_scalar.pack(fill=tk.X)
                    else:
                        self.panel_add_scalar.pack_forget()
                        self.panel_add_inter.pack(fill=tk.X)
                elif code == "SUB":
                    if is_scalar:
                        self.panel_sub_inter.pack_forget()
                        self.panel_sub_scalar.pack(fill=tk.X)
                    else:
                        self.panel_sub_scalar.pack_forget()
                        self.panel_sub_inter.pack(fill=tk.X)
                elif code == "MUL":
                    if is_scalar:
                        self.panel_mul_inter.pack_forget()
                        self.panel_mul_scalar.pack(fill=tk.X)
                    else:
                        self.panel_mul_scalar.pack_forget()
                        self.panel_mul_inter.pack(fill=tk.X)
                elif code == "DIV":
                    if is_scalar:
                        self.panel_div_inter.pack_forget()
                        self.panel_div_scalar.pack(fill=tk.X)
                    else:
                        self.panel_div_scalar.pack_forget()
                        self.panel_div_inter.pack(fill=tk.X)
            else:
                panel.pack_forget()

        self.on_arithmetic_change()

    def on_arithmetic_change(self):
        if self.img_a_orig is None:
            return

        # Pastikan format 3 kanal RGB yang konsisten
        if len(self.img_a_orig.shape) == 2:
            a_rgb = cv2.cvtColor(self.img_a_orig, cv2.COLOR_GRAY2RGB)
        else:
            a_rgb = self.img_a_orig

        if self.img_b_orig is None:
            self._generate_default_image_b("circle")

        if len(self.img_b_orig.shape) == 2:
            b_rgb = cv2.cvtColor(self.img_b_orig, cv2.COLOR_GRAY2RGB)
        else:
            b_rgb = self.img_b_orig

        h, w = a_rgb.shape[:2]
        img_b_aligned = cv2.resize(b_rgb, (w, h), interpolation=cv2.INTER_AREA)

        op = self.var_arith_op.get()
        target = self.var_arith_target.get()

        a_f = a_rgb.astype(np.float32)
        b_f = img_b_aligned.astype(np.float32)

        if "Skalar" in target:
            if op == "ADD":
                c = float(self.var_add_scalar.get())
                res = np.clip(a_f + c, 0, 255).astype(np.uint8)
                desc = f"Penjumlahan Skalar (+{int(c)})"
            elif op == "SUB":
                c = float(self.var_sub_scalar.get())
                res = np.clip(a_f - c, 0, 255).astype(np.uint8)
                desc = f"Pengurangan Skalar (-{int(c)})"
            elif op == "MUL":
                c = float(self.var_mul_scalar.get())
                res = np.clip(a_f * c, 0, 255).astype(np.uint8)
                desc = f"Penskalaan Kontras (*{c:.1f})"
            elif op == "DIV":
                c = max(0.01, float(self.var_div_scalar.get()))
                res = np.clip(a_f / c, 0, 255).astype(np.uint8)
                desc = f"Pembagian Skalar (/{c:.1f})"
            else:
                res = a_rgb.copy()
                desc = "Citra Asli"
        else:
            if op == "ADD":
                mode = self.var_add_mode.get()
                if "Alpha Blending" in mode:
                    a = float(self.var_add_alpha.get())
                    res = cv2.addWeighted(a_rgb, a, img_b_aligned, 1.0 - a, 0)
                    desc = f"Alpha Blending (α={a:.2f})"
                elif "Rata-rata" in mode or "Averaging" in mode:
                    res = cv2.addWeighted(a_rgb, 0.5, img_b_aligned, 0.5, 0)
                    desc = "Image Averaging ((A + B) / 2)"
                else:
                    res = cv2.add(a_rgb, img_b_aligned)
                    desc = "Penjumlahan Citra (A + B)"

            elif op == "SUB":
                mode = self.var_sub_mode.get()
                if "Selisih Mutlak" in mode:
                    res = cv2.absdiff(a_rgb, img_b_aligned)
                    desc = "Selisih Mutlak |A - B|"
                elif "Terbalik" in mode:
                    res = cv2.subtract(img_b_aligned, a_rgb)
                    desc = "Pengurangan Terbalik (B - A)"
                else:
                    res = cv2.subtract(a_rgb, img_b_aligned)
                    desc = "Pengurangan Terarah (A - B)"

            elif op == "MUL":
                mode = self.var_mul_mode.get()
                if "Masking" in mode:
                    res = np.clip((a_f * b_f) / 255.0, 0, 255).astype(np.uint8)
                    desc = "Perkalian Masking (A*B/255)"
                else:
                    res = np.clip(a_f * (b_f / 128.0), 0, 255).astype(np.uint8)
                    desc = "Perkalian Matriks (A*B/128)"

            elif op == "DIV":
                mode = self.var_div_mode.get()
                if "Rasio" in mode or "Spektral" in mode:
                    res = np.clip(((a_f + 1.0) / (b_f + 1.0)) * 128.0, 0, 255).astype(np.uint8)
                    desc = "Rasio Spektral ((A+1)/(B+1)*128)"
                else:
                    res = np.clip((a_f / (b_f + 1.0)) * 255.0, 0, 255).astype(np.uint8)
                    desc = "Pembagian Citra Ter-normalisasi"
            else:
                res = a_rgb.copy()
                desc = "Citra Asli"

        self.img_processed = res
        self.tag_proc.config(text=f"Aritmatika: {desc}")
        self.refresh_active_view()
        self._update_specs_log(desc)

    # =========================================================================
    # SECTION 2: LIVE HISTOGRAM SPEKTRAL & FITUR TEKSTUR HARALICK (GLCM)
    # =========================================================================
    def _build_section_histogram(self, parent):
        lbl_head = tk.Label(parent, text="Analisis Spektral & Histogram", bg=self.C_PANEL, fg=self.C_TEXT,
                            font=("Segoe UI", 11, "bold"))
        lbl_head.pack(anchor=tk.W, padx=12, pady=(12, 6))

        # Canvas Matplotlib Histogram (Gelombang Peach #E8A87C & Teal #4ECDC4)
        hist_container = tk.Frame(parent, bg=self.C_BG, bd=1, relief=tk.SOLID)
        hist_container.pack(fill=tk.X, padx=12, pady=6)

        self.fig_hist, self.ax_hist = plt.subplots(figsize=(3.4, 2.0), dpi=90)
        self.fig_hist.patch.set_facecolor("#1E1E24")
        self.ax_hist.set_facecolor("#1E1E24")
        self.ax_hist.tick_params(colors=self.C_MUTED, labelsize=7)
        self.ax_hist.grid(True, color="#2B2D42", linestyle='-', linewidth=0.8)
        self.ax_hist.set_xlim([0, 256])
        self.ax_hist.set_yticks([])
        self.fig_hist.tight_layout()

        self.canvas_hist_tk = FigureCanvasTkAgg(self.fig_hist, master=hist_container)
        self.canvas_hist_tk.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # GLCM & Metrik Tekstur Haralick
        lbl_m = tk.Label(parent, text="Fitur Tekstur Matriks GLCM Haralick", bg=self.C_PANEL, fg=self.C_PEACH,
                         font=("Segoe UI", 9, "bold"))
        lbl_m.pack(anchor=tk.W, padx=12, pady=(10, 4))

        self.lbl_glcm_info = tk.Label(parent, text="Memuat tekstur...", bg=self.C_SUBPANEL, fg=self.C_MUTED,
                                      font=("Consolas", 8), justify=tk.LEFT, bd=1, relief=tk.SOLID, padx=8, pady=6)
        self.lbl_glcm_info.pack(fill=tk.X, padx=12, pady=4)

        btn_calc_glcm = tk.Button(parent, text="🔬 Refresh Analisis GLCM", font=("Segoe UI", 8, "bold"),
                                  bg=self.C_SUBPANEL, fg=self.C_TEAL, activebackground=self.C_BORDER,
                                  bd=1, relief=tk.SOLID, padx=8, pady=4, cursor="hand2",
                                  command=self._update_histogram_chart)
        btn_calc_glcm.pack(fill=tk.X, padx=12, pady=4)

    def _update_histogram_chart(self):
        if self.img_processed is None or self.img_a_orig is None:
            return

        self.ax_hist.clear()
        self.ax_hist.set_facecolor("#1E1E24")
        self.ax_hist.tick_params(colors=self.C_MUTED, labelsize=7)
        self.ax_hist.grid(True, color="#2B2D42", linestyle='-', linewidth=0.8)

        # Citra Asli (Gelombang Peach #E8A87C)
        gray_a = cv2.cvtColor(self.img_a_orig, cv2.COLOR_RGB2GRAY) if len(self.img_a_orig.shape) == 3 else self.img_a_orig
        h_a, _ = np.histogram(gray_a.ravel(), bins=256, range=[0, 256])
        bins = np.arange(256)
        self.ax_hist.plot(bins, h_a, color=self.C_PEACH, alpha=0.9, linewidth=1.2, label='Asli')
        self.ax_hist.fill_between(bins, h_a, color=self.C_PEACH, alpha=0.25)

        # Citra Hasil (Gelombang Teal #4ECDC4)
        gray_res = cv2.cvtColor(self.img_processed, cv2.COLOR_RGB2GRAY) if len(self.img_processed.shape) == 3 else self.img_processed
        h_res, _ = np.histogram(gray_res.ravel(), bins=256, range=[0, 256])
        self.ax_hist.plot(bins, h_res, color=self.C_TEAL, alpha=0.9, linewidth=1.2, label='Hasil')
        self.ax_hist.fill_between(bins, h_res, color=self.C_TEAL, alpha=0.25)

        self.ax_hist.set_xlim([0, 255])
        self.ax_hist.set_yticks([])
        self.fig_hist.tight_layout()
        self.canvas_hist_tk.draw()

        # Ekstraksi Fitur Tekstur Matriks GLCM Haralick
        try:
            quantized = (gray_res // 16).astype(np.uint8)
            glcm = graycomatrix(quantized, distances=[1], angles=[0], levels=16, symmetric=True, normed=True)
            c = graycoprops(glcm, 'contrast')[0, 0]
            d = graycoprops(glcm, 'dissimilarity')[0, 0]
            h = graycoprops(glcm, 'homogeneity')[0, 0]
            e = graycoprops(glcm, 'energy')[0, 0]
            cor = graycoprops(glcm, 'correlation')[0, 0]

            self.lbl_glcm_info.config(text=(
                f"• Kontras Spasial    : {c:8.3f}\n"
                f"• Disimilaritas      : {d:8.3f}\n"
                f"• Homogenitas Matriks: {h:8.3f}\n"
                f"• Energi Haralick    : {e:8.3f}\n"
                f"• Korelasi Spektral  : {cor:8.3f}"
            ))
        except Exception as err:
            self.lbl_glcm_info.config(text=f"GLCM Error: {err}")

    # =========================================================================
    # SECTION 3: SPATIAL FILTERS & CONVOLUTION KERNELS
    # =========================================================================
    def _build_section_filters(self, parent):
        sf_c = tk.Canvas(parent, bg=self.C_PANEL, highlightthickness=0)
        sb = ttk.Scrollbar(parent, orient="vertical", command=sf_c.yview)
        f_inner = tk.Frame(sf_c, bg=self.C_PANEL)
        f_inner.bind("<Configure>", lambda e: sf_c.configure(scrollregion=sf_c.bbox("all")))
        sf_c.create_window((0, 0), window=f_inner, anchor="nw", width=320)
        sf_c.configure(yscrollcommand=sb.set)
        sf_c.pack(side="left", fill="both", expand=True, padx=(2, 0))
        sb.pack(side="right", fill="y")

        lbl_head = tk.Label(f_inner, text="Filter Spasial & Konvolusi", bg=self.C_PANEL, fg=self.C_TEXT,
                            font=("Segoe UI", 11, "bold"))
        lbl_head.pack(anchor=tk.W, padx=12, pady=(12, 6))

        # Preset Filter Populer
        f_presets = ttk.LabelFrame(f_inner, text="Preset Konvolusi Kernel", padding=(8, 6))
        f_presets.pack(fill=tk.X, padx=10, pady=4)

        presets = [
            ("Grayscale", "gray"),
            ("Canny Edge", "canny"),
            ("Sobel X/Y", "sobel"),
            ("Laplacian", "laplacian"),
            ("Gaussian Blur", "blur"),
            ("Inversi Negatif", "invert"),
            ("Thermal Jet", "thermal"),
            ("Otsu Binary", "otsu"),
            ("Emboss 3D", "emboss"),
            ("Deteksi Kontur", "contour")
        ]

        grid_p = tk.Frame(f_presets, bg=self.C_PANEL)
        grid_p.pack(fill=tk.X)

        self.filter_buttons = {}
        for i, (name, p_id) in enumerate(presets):
            b = tk.Button(grid_p, text=name, font=("Segoe UI", 8), bg=self.C_SUBPANEL, fg=self.C_TEXT,
                          activebackground=self.C_PEACH, activeforeground=self.C_BG, bd=1, relief=tk.SOLID,
                          padx=4, pady=4, cursor="hand2", command=lambda pid=p_id: self.apply_filter_preset(pid))
            b.grid(row=i//2, column=i%2, padx=3, pady=3, sticky="ew")
            grid_p.grid_columnconfigure(i%2, weight=1)
            self.filter_buttons[p_id] = b

        # Slider Parameter Manual
        f_manual = ttk.LabelFrame(f_inner, text="Penyetelan Spasial Interaktif", padding=(8, 6))
        f_manual.pack(fill=tk.X, padx=10, pady=6)

        self._create_slider_row(f_manual, "Brightness (Kecerahan):", self.var_brightness, -100, 100, self.on_filter_param_change)
        self._create_slider_row(f_manual, "Contrast (Kontras):", self.var_contrast, -100, 100, self.on_filter_param_change)
        self._create_slider_row(f_manual, "Gamma Spasial:", self.var_gamma, 0.2, 3.0, self.on_filter_param_change, is_float=True)
        self.scale_blur, self.lbl_val_blur = self._create_slider_row(f_manual, "Kernel Blur Radius:", self.var_blur, 1, 31, self.on_filter_param_change, step=2, odd_only=True)
        self._create_slider_row(f_manual, "Canny Threshold 1:", self.var_canny1, 10, 250, self.on_filter_param_change)
        self._create_slider_row(f_manual, "Canny Threshold 2:", self.var_canny2, 20, 300, self.on_filter_param_change)

        btn_reset_f = tk.Button(f_inner, text="🔄 Reset ke Citra Asli (Original)", font=("Segoe UI", 9),
                                bg=self.C_PANEL, fg=self.C_PEACH, activebackground=self.C_BORDER,
                                bd=1, relief=tk.SOLID, padx=8, pady=5, cursor="hand2", command=self.reset_to_original_image)
        btn_reset_f.pack(fill=tk.X, padx=10, pady=6)

    def _reset_filter_sliders(self):
        self.reset_to_original_image()

    def apply_filter_preset(self, preset_id):
        if self.img_a_orig is None:
            return

        self.active_filter_mode = preset_id

        # Update button highlights
        for pid, b in self.filter_buttons.items():
            if pid == preset_id:
                b.configure(bg=self.C_PEACH, fg=self.C_BG, font=("Segoe UI", 8, "bold"))
            else:
                b.configure(bg=self.C_SUBPANEL, fg=self.C_TEXT, font=("Segoe UI", 8))

        gray = cv2.cvtColor(self.img_a_orig, cv2.COLOR_RGB2GRAY) if len(self.img_a_orig.shape) == 3 else self.img_a_orig

        if preset_id == "gray":
            self.img_processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            desc = "Konversi Grayscale"
        elif preset_id == "canny":
            t1 = self.var_canny1.get()
            t2 = self.var_canny2.get()
            edges = cv2.Canny(gray, t1, t2)
            self.img_processed = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            desc = f"Canny Edge (T1={t1}, T2={t2})"
        elif preset_id == "sobel":
            sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sob = np.clip(np.sqrt(sx**2 + sy**2), 0, 255).astype(np.uint8)
            self.img_processed = cv2.cvtColor(sob, cv2.COLOR_GRAY2RGB)
            desc = "Sobel Gradient Filter"
        elif preset_id == "laplacian":
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            lap = np.clip(np.abs(lap), 0, 255).astype(np.uint8)
            self.img_processed = cv2.cvtColor(lap, cv2.COLOR_GRAY2RGB)
            desc = "Laplacian 2nd Derivative"
        elif preset_id == "blur":
            # Preset Gaussian Blur: pilih kernel jelas terlihat (9x9 jika masih bernilai netral 1)
            k = self.var_blur.get()
            if k <= 1:
                k = 9
                self.var_blur.set(k)
            elif k % 2 == 0:
                k += 1
                self.var_blur.set(k)

            if hasattr(self, 'scale_blur') and hasattr(self, 'lbl_val_blur'):
                try:
                    self.scale_blur.set(k)
                    self.lbl_val_blur.config(text=str(k))
                except Exception:
                    pass

            b = cv2.GaussianBlur(self.img_a_orig, (k, k), 0)
            self.img_processed = b
            desc = f"Gaussian Smoothing (Kernel {k}x{k})"
        elif preset_id == "invert":
            self.img_processed = cv2.bitwise_not(self.img_a_orig)
            desc = "Inversi Citra (Negatif)"
        elif preset_id == "thermal":
            therm = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            self.img_processed = cv2.cvtColor(therm, cv2.COLOR_BGR2RGB)
            desc = "Thermal False-Color (JET)"
        elif preset_id == "otsu":
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.img_processed = cv2.cvtColor(th, cv2.COLOR_GRAY2RGB)
            desc = "Otsu Adaptive Thresholding"
        elif preset_id == "emboss":
            kernel_emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
            em = cv2.filter2D(self.img_a_orig, -1, kernel_emboss)
            self.img_processed = em
            desc = "Emboss 3D Texture Filter"
        elif preset_id == "contour":
            _, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            res_c = self.img_a_orig.copy()
            cv2.drawContours(res_c, contours, -1, (232, 168, 124), 2)
            self.img_processed = res_c
            desc = f"Deteksi Kontur ({len(contours)} kurva)"
        else:
            self.img_processed = self.img_a_orig.copy()
            desc = "Normal"

        self.tag_proc.config(text=f"Filter: {desc}")
        self.refresh_active_view()
        self._update_specs_log(desc)

    def on_filter_param_change(self):
        if self.img_a_orig is None:
            return

        # Jika mode preset aktif adalah canny, update canny
        if self.active_filter_mode == "canny":
            self.apply_filter_preset("canny")
            return
        elif self.active_filter_mode == "blur":
            k = self.var_blur.get()
            if k % 2 == 0:
                k += 1
            if k < 1:
                k = 1
            if k == 1:
                self.img_processed = self.img_a_orig.copy()
                desc = "Gaussian Blur (Original / Off)"
            else:
                self.img_processed = cv2.GaussianBlur(self.img_a_orig, (k, k), 0)
                desc = f"Gaussian Smoothing (Kernel {k}x{k})"
            self.tag_proc.config(text=f"Filter: {desc}")
            self.refresh_active_view()
            self._update_specs_log(desc)
            return

        # Reset highlight preset button karena slider manual sedang dipakai
        for pid, b in self.filter_buttons.items():
            b.configure(bg=self.C_SUBPANEL, fg=self.C_TEXT, font=("Segoe UI", 8))

        res = self.img_a_orig.astype(np.float32)

        # Brightness & Contrast
        b = self.var_brightness.get()
        c = self.var_contrast.get()
        f_contrast = 131 * (c + 127) / (127 * (131 - c)) if c < 127 else 3.0
        res = f_contrast * (res - 128) + 128 + b
        res = np.clip(res, 0, 255)

        # Gamma
        gamma = self.var_gamma.get()
        if abs(gamma - 1.0) > 0.01:
            inv_gamma = 1.0 / max(0.1, gamma)
            res = 255.0 * ((res / 255.0) ** inv_gamma)

        res_uint8 = np.clip(res, 0, 255).astype(np.uint8)

        # Blur
        k = self.var_blur.get()
        if k > 1:
            if k % 2 == 0:
                k += 1
            res_uint8 = cv2.GaussianBlur(res_uint8, (k, k), 0)

        self.img_processed = res_uint8
        desc = f"Penyesuaian Spasial (B={b}, C={c}, γ={gamma:.2f})"
        self.tag_proc.config(text=desc)
        self.refresh_active_view()
        self._update_specs_log(desc)

    # =========================================================================
    # SECTION 4: SLICING INTENSITAS & BIT-PLANE DECOMPOSITION
    # =========================================================================
    def _build_section_slicing(self, parent):
        sf_c = tk.Canvas(parent, bg=self.C_PANEL, highlightthickness=0)
        sb = ttk.Scrollbar(parent, orient="vertical", command=sf_c.yview)
        f_s = tk.Frame(sf_c, bg=self.C_PANEL)
        f_s.bind("<Configure>", lambda e: sf_c.configure(scrollregion=sf_c.bbox("all")))
        sf_c.create_window((0, 0), window=f_s, anchor="nw", width=320)
        sf_c.configure(yscrollcommand=sb.set)
        sf_c.pack(side="left", fill="both", expand=True, padx=(2, 0))
        sb.pack(side="right", fill="y")

        lbl_head = tk.Label(f_s, text="Slicing Intensitas & Bit-Plane", bg=self.C_PANEL, fg=self.C_TEXT,
                            font=("Segoe UI", 11, "bold"))
        lbl_head.pack(anchor=tk.W, padx=12, pady=(12, 6))

        # Pilihan Mode
        f_type = ttk.LabelFrame(f_s, text="Mode Slicing Spasial", padding=(10, 6))
        f_type.pack(fill=tk.X, padx=10, pady=4)

        tk.Radiobutton(f_type, text="Gray-Level Intensity Slicing", variable=self.var_slicing_type,
                       value="Gray-Level Slicing", bg=self.C_PANEL, fg=self.C_TEXT, selectcolor=self.C_SUBPANEL,
                       command=self.on_slicing_change).pack(anchor=tk.W)
        tk.Radiobutton(f_type, text="Bit-Plane Slicing (Bits 0-7)", variable=self.var_slicing_type,
                       value="Bit-Plane Slicing", bg=self.C_PANEL, fg=self.C_TEXT, selectcolor=self.C_SUBPANEL,
                       command=self.on_slicing_change).pack(anchor=tk.W)

        # Parameter Gray-Level
        f_gl = ttk.LabelFrame(f_s, text="Ambang Batas Intensitas [A, B]", padding=(10, 6))
        f_gl.pack(fill=tk.X, padx=10, pady=6)

        self._create_slider_row(f_gl, "Batas Bawah (A):", self.var_slice_low, 0, 255, self.on_slicing_change)
        self._create_slider_row(f_gl, "Batas Atas (B):", self.var_slice_high, 0, 255, self.on_slicing_change)
        self._create_slider_row(f_gl, "Nilai Highlight:", self.var_slice_highlight, 0, 255, self.on_slicing_change)

        chk_bg = tk.Checkbutton(f_gl, text="Preserve Background (Pertahankan Latar)",
                                variable=self.var_slice_preserve_bg, bg=self.C_PANEL, fg=self.C_PEACH,
                                selectcolor=self.C_SUBPANEL, command=self.on_slicing_change)
        chk_bg.pack(anchor=tk.W, pady=4)

        # Parameter Bit-Plane
        f_bp = ttk.LabelFrame(f_s, text="Pilih Bit-Plane Spasial (0 - 7)", padding=(10, 6))
        f_bp.pack(fill=tk.X, padx=10, pady=6)

        f_bp_btns = tk.Frame(f_bp, bg=self.C_PANEL)
        f_bp_btns.pack(fill=tk.X)

        for bit in range(8):
            b = tk.Radiobutton(f_bp_btns, text=f"B-{bit}", variable=self.var_bit_plane, value=bit,
                               bg=self.C_PANEL, fg=self.C_TEXT, selectcolor=self.C_SUBPANEL,
                               command=self.on_slicing_change)
            b.grid(row=bit//4, column=bit%4, padx=2, pady=2, sticky="ew")
            f_bp_btns.grid_columnconfigure(bit%4, weight=1)

        # Rekonstruksi Bit Gabungan Populer
        f_rec = tk.Frame(f_bp, bg=self.C_PANEL)
        f_rec.pack(fill=tk.X, pady=(6, 2))

        btn_b76 = tk.Button(f_rec, text="Bit 7+6 MSB", font=("Segoe UI", 8), bg=self.C_SUBPANEL, fg=self.C_TEAL,
                            bd=1, relief=tk.SOLID, cursor="hand2", command=lambda: self._apply_multibit_slice([7, 6]))
        btn_b76.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)

        btn_b765 = tk.Button(f_rec, text="Bit 7+6+5 MSB", font=("Segoe UI", 8), bg=self.C_SUBPANEL, fg=self.C_TEAL,
                             bd=1, relief=tk.SOLID, cursor="hand2", command=lambda: self._apply_multibit_slice([7, 6, 5]))
        btn_b765.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=1)

        btn_reset_sl = tk.Button(f_s, text="🔄 Reset ke Citra Asli (Original)", font=("Segoe UI", 9),
                                 bg=self.C_PANEL, fg=self.C_PEACH, activebackground=self.C_BORDER,
                                 bd=1, relief=tk.SOLID, padx=10, pady=5, cursor="hand2",
                                 command=self.reset_to_original_image)
        btn_reset_sl.pack(fill=tk.X, padx=10, pady=(10, 4))

    def _apply_multibit_slice(self, bit_list):
        if self.img_a_orig is None:
            return
        gray = cv2.cvtColor(self.img_a_orig, cv2.COLOR_RGB2GRAY) if len(self.img_a_orig.shape) == 3 else self.img_a_orig
        res_g = np.zeros_like(gray)
        for bit in bit_list:
            res_g += ((gray >> bit) & 1) << bit
        self.img_processed = cv2.cvtColor(res_g, cv2.COLOR_GRAY2RGB)
        desc = f"Rekonstruksi Bit {'+'.join(map(str, bit_list))}"
        self.tag_proc.config(text=f"Slicing: {desc}")
        self.refresh_active_view()
        self._update_specs_log(desc)

    def on_slicing_change(self):
        if self.img_a_orig is None:
            return

        gray = cv2.cvtColor(self.img_a_orig, cv2.COLOR_RGB2GRAY) if len(self.img_a_orig.shape) == 3 else self.img_a_orig
        stype = self.var_slicing_type.get()

        if stype == "Gray-Level Slicing":
            a = min(self.var_slice_low.get(), self.var_slice_high.get())
            b = max(self.var_slice_low.get(), self.var_slice_high.get())
            hl = self.var_slice_highlight.get()

            mask = (gray >= a) & (gray <= b)

            if self.var_slice_preserve_bg.get():
                res_g = gray.copy()
                res_g[mask] = hl
            else:
                res_g = np.zeros_like(gray)
                res_g[mask] = hl

            self.img_processed = cv2.cvtColor(res_g, cv2.COLOR_GRAY2RGB)
            desc = f"Intensity Slicing [{a} - {b}]"
        else:
            bit = self.var_bit_plane.get()
            bit_slice = ((gray >> bit) & 1) * 255
            self.img_processed = cv2.cvtColor(bit_slice.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            desc = f"Bit-Plane Slice Bit-{bit}"

        self.tag_proc.config(text=f"Slicing: {desc}")
        self.refresh_active_view()
        self._update_specs_log(desc)

    # =========================================================================
    # SECTION 5: MATRIKS 10 TRANSFORMASI SIMULTAN (GALERI TAB)
    # =========================================================================
    def _build_section_gallery(self, parent):
        lbl_head = tk.Label(parent, text="Matriks 10 Transformasi Spasial", bg=self.C_PANEL, fg=self.C_TEXT,
                            font=("Segoe UI", 11, "bold"))
        lbl_head.pack(anchor=tk.W, padx=12, pady=(12, 6))

        btn_run = tk.Button(parent, text="🔬 Buka Matriks 10 Transformasi", font=("Segoe UI", 9, "bold"),
                            bg=self.C_PEACH, fg=self.C_BG, activebackground="#f0b890", bd=0,
                            padx=10, pady=8, cursor="hand2", command=self.render_gallery)
        btn_run.pack(fill=tk.X, padx=12, pady=8)

        tk.Label(parent, text="Menghasilkan 10 representasi citra simultan (Grayscale, Brightness, Contrast, Inversi, Gaussian Blur, Canny Edge, Thermal Jet, Slicing, dan Bit-Plane) dalam jendela komparasi ilmiah mandiri.",
                 bg=self.C_PANEL, fg=self.C_MUTED, font=("Segoe UI", 8), wraplength=300, justify=tk.LEFT).pack(anchor=tk.W, padx=12, pady=4)

    def render_gallery(self):
        if self.img_a_orig is None:
            return

        top = tk.Toplevel(self)
        top.title("SpectraVision DIP Lab • Matriks 10 Transformasi Citra (Jade Aurestha)")
        top.geometry("1140x640")
        top.configure(bg=self.C_BG)

        fig, axes = plt.subplots(2, 5, figsize=(12, 6), dpi=90)
        fig.patch.set_facecolor("#1E1E24")

        base = cv2.resize(self.img_a_orig, (320, 240))
        gray = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(blur, 100, 200)
        therm = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

        gal_dict = {
            "1. Original RGB": base,
            "2. Grayscale": cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB),
            "3. Brightness (+50)": np.clip(base.astype(np.float32) + 50, 0, 255).astype(np.uint8),
            "4. Contrast (*1.4)": np.clip(base.astype(np.float32) * 1.4 - 40, 0, 255).astype(np.uint8),
            "5. Inversi (Negatif)": cv2.bitwise_not(base),
            "6. Gaussian Blur": cv2.cvtColor(blur, cv2.COLOR_GRAY2RGB),
            "7. Canny Edge": cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB),
            "8. Thermal Jet": cv2.cvtColor(therm, cv2.COLOR_BGR2RGB),
            "9. Slicing [60-120]": cv2.cvtColor(np.where((gray >= 60) & (gray <= 120), 255, 0).astype(np.uint8), cv2.COLOR_GRAY2RGB),
            "10. Bit-6 Slicing": cv2.cvtColor((((gray >> 6) & 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        }

        for ax, (t, im) in zip(axes.ravel(), gal_dict.items()):
            ax.clear()
            ax.imshow(im)
            ax.set_title(t, color=self.C_PEACH, fontsize=8, fontweight='bold')
            ax.axis('off')

        fig.tight_layout()
        c_tk = FigureCanvasTkAgg(fig, master=top)
        c_tk.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.refresh_active_view()

    # =========================================================================
    # SECTION 6: TABEL & MATRIKS RGB (SLIDE 8 KURIKULUM PCD)
    # =========================================================================
    def _build_section_rgb_table(self, parent):
        container = tk.Frame(parent, bg=self.C_PANEL)
        container.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        lbl_head = tk.Label(container, text="Tabel Matriks RGB Spasial", bg=self.C_PANEL, fg=self.C_TEXT,
                            font=("Segoe UI", 11, "bold"))
        lbl_head.pack(anchor=tk.W, pady=(0, 4))

        tk.Label(container, text="Inspeksi nilai intensitas saluran Red, Green, Blue, dan Grayscale dari matriks ketetanggaan 3×3 (Sesuai Slide 8 PDF).",
                 bg=self.C_PANEL, fg=self.C_MUTED, font=("Segoe UI", 8), wraplength=310, justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 8))

        # Kontrol Sumber & Koordinat
        ctrl_box = tk.LabelFrame(container, text="Titik Koordinat Pusat (X, Y)", bg=self.C_PANEL, fg=self.C_PEACH,
                                 font=("Segoe UI", 8, "bold"), bd=1, relief=tk.SOLID)
        ctrl_box.pack(fill=tk.X, pady=(0, 8), ipady=4)

        # Baris 1: Sumber Citra
        f_src = tk.Frame(ctrl_box, bg=self.C_PANEL)
        f_src.pack(fill=tk.X, padx=6, pady=2)
        tk.Label(f_src, text="Sumber:", bg=self.C_PANEL, fg=self.C_MUTED, font=("Segoe UI", 8)).pack(side=tk.LEFT)
        combo_src = ttk.Combobox(f_src, textvariable=self.var_rgb_source, values=["Citra Asli", "Citra Hasil"],
                                 state="readonly", width=14, font=("Segoe UI", 8))
        combo_src.pack(side=tk.RIGHT, padx=2)
        combo_src.bind("<<ComboboxSelected>>", lambda e: self._update_rgb_table())

        # Baris 2: Spinbox X & Y + Tombol Pusat
        f_coord = tk.Frame(ctrl_box, bg=self.C_PANEL)
        f_coord.pack(fill=tk.X, padx=6, pady=4)

        tk.Label(f_coord, text="X:", bg=self.C_PANEL, fg=self.C_MUTED, font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self.spin_rgb_x = ttk.Spinbox(f_coord, from_=0, to=9999, textvariable=self.var_rgb_x, width=5,
                                      font=("Consolas", 8), command=self._update_rgb_table)
        self.spin_rgb_x.pack(side=tk.LEFT, padx=(2, 8))
        self.spin_rgb_x.bind("<Return>", lambda e: self._update_rgb_table())

        tk.Label(f_coord, text="Y:", bg=self.C_PANEL, fg=self.C_MUTED, font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self.spin_rgb_y = ttk.Spinbox(f_coord, from_=0, to=9999, textvariable=self.var_rgb_y, width=5,
                                      font=("Consolas", 8), command=self._update_rgb_table)
        self.spin_rgb_y.pack(side=tk.LEFT, padx=(2, 8))
        self.spin_rgb_y.bind("<Return>", lambda e: self._update_rgb_table())

        btn_center = tk.Button(f_coord, text="📍 Pusat", font=("Segoe UI", 8), bg=self.C_SUBPANEL, fg=self.C_PEACH,
                               bd=1, relief=tk.SOLID, padx=6, cursor="hand2", command=self._set_rgb_coord_center)
        btn_center.pack(side=tk.RIGHT, padx=2)

        # Kartu Pusat Piksel Terpilih
        self.f_center_info = tk.Frame(container, bg=self.C_SUBPANEL, bd=1, relief=tk.SOLID)
        self.f_center_info.pack(fill=tk.X, pady=(0, 8), ipady=3)

        self.lbl_rgb_swatch = tk.Label(self.f_center_info, text="      ", bg="#000000", bd=1, relief=tk.SOLID, width=4)
        self.lbl_rgb_swatch.pack(side=tk.LEFT, padx=8, pady=4)

        self.lbl_rgb_center_val = tk.Label(self.f_center_info, text="Pusat (0,0): R:-- G:-- B:-- | Gray:--",
                                           bg=self.C_SUBPANEL, fg=self.C_TEXT, font=("Consolas", 8, "bold"), justify=tk.LEFT)
        self.lbl_rgb_center_val.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Tabel Treeview Matriks 3x3
        f_tree = tk.Frame(container, bg=self.C_PANEL)
        f_tree.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        columns = ("pos", "coord", "r", "g", "b", "gray")
        self.tree_rgb = ttk.Treeview(f_tree, columns=columns, show="headings", height=8, selectmode="browse")
        self.tree_rgb.heading("pos", text="Pos")
        self.tree_rgb.heading("coord", text="(X, Y)")
        self.tree_rgb.heading("r", text="R")
        self.tree_rgb.heading("g", text="G")
        self.tree_rgb.heading("b", text="B")
        self.tree_rgb.heading("gray", text="Gray")

        self.tree_rgb.column("pos", width=42, anchor="center")
        self.tree_rgb.column("coord", width=62, anchor="center")
        self.tree_rgb.column("r", width=40, anchor="center")
        self.tree_rgb.column("g", width=40, anchor="center")
        self.tree_rgb.column("b", width=40, anchor="center")
        self.tree_rgb.column("gray", width=45, anchor="center")

        tree_scroll = ttk.Scrollbar(f_tree, orient="vertical", command=self.tree_rgb.yview)
        self.tree_rgb.configure(yscrollcommand=tree_scroll.set)
        self.tree_rgb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Formula Ilmiah dari Kurikulum
        lbl_formula = tk.Label(container, text="Formula Grayscale (Slide 8):  f_o(x,y) = (R + G + B) / 3",
                               bg=self.C_PANEL, fg=self.C_TEAL, font=("Segoe UI", 8, "italic"))
        lbl_formula.pack(anchor=tk.W, pady=(2, 6))

        # Tombol Aksi: Salin Matriks & Buka Dialog Lengkap
        btn_box = tk.Frame(container, bg=self.C_PANEL)
        btn_box.pack(fill=tk.X, pady=(0, 2))

        btn_copy = tk.Button(btn_box, text="📋 Salin Matriks", font=("Segoe UI", 8),
                             bg=self.C_SUBPANEL, fg=self.C_TEXT, activebackground=self.C_BORDER,
                             bd=1, relief=tk.SOLID, padx=8, pady=4, cursor="hand2",
                             command=self._copy_rgb_matrix_text)
        btn_copy.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 3))

        btn_dlg = tk.Button(btn_box, text="🔍 Dialog Matriks 3×3", font=("Segoe UI", 8, "bold"),
                            bg=self.C_PEACH, fg=self.C_BG, activebackground="#f0b890",
                            bd=0, relief=tk.FLAT, padx=8, pady=4, cursor="hand2",
                            command=self.show_rgb_table_dialog)
        btn_dlg.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(3, 0))

    def _set_rgb_coord_center(self):
        source_name = self.var_rgb_source.get()
        target_img = self.img_processed if (source_name == "Citra Hasil" and self.img_processed is not None) else self.img_a_orig
        if target_img is None:
            return
        h, w = target_img.shape[:2]
        self.var_rgb_x.set(w // 2)
        self.var_rgb_y.set(h // 2)
        self._update_rgb_table()

    def _update_rgb_table(self):
        source_name = self.var_rgb_source.get()
        target_img = self.img_processed if (source_name == "Citra Hasil" and self.img_processed is not None) else self.img_a_orig
        if target_img is None:
            return

        h, w = target_img.shape[:2]
        cx = max(0, min(w - 1, self.var_rgb_x.get()))
        cy = max(0, min(h - 1, self.var_rgb_y.get()))
        self.var_rgb_x.set(cx)
        self.var_rgb_y.set(cy)

        # Hapus baris lama di Treeview
        if hasattr(self, 'tree_rgb'):
            for item in self.tree_rgb.get_children():
                self.tree_rgb.delete(item)

        # Posisi relatif 3x3: (dy, dx)
        offsets = [
            (-1, -1, "TL"), (-1, 0, "TC"), (-1, 1, "TR"),
            ( 0, -1, "ML"), ( 0, 0, "C" ), ( 0, 1, "MR"),
            ( 1, -1, "BL"), ( 1, 0, "BC"), ( 1, 1, "BR")
        ]

        center_r, center_g, center_b = 0, 0, 0
        center_gray = 0

        for dy, dx, pos_tag in offsets:
            nx = cx + dx
            ny = cy + dy
            if 0 <= nx < w and 0 <= ny < h:
                px = target_img[ny, nx]
                if len(target_img.shape) == 3:
                    r, g, b = int(px[0]), int(px[1]), int(px[2])
                else:
                    r = g = b = int(px)
                gray = (r + g + b) // 3
                coord_str = f"({nx},{ny})"
            else:
                r, g, b = 0, 0, 0
                gray = 0
                coord_str = "Tepi"

            if pos_tag == "C":
                center_r, center_g, center_b = r, g, b
                center_gray = gray
                pos_display = "[C]"
            else:
                pos_display = pos_tag

            if hasattr(self, 'tree_rgb'):
                self.tree_rgb.insert("", tk.END, values=(pos_display, coord_str, r, g, b, gray))

        # Perbarui kotak warna dan label info pusat
        if hasattr(self, 'lbl_rgb_swatch'):
            hex_color = f"#{center_r:02x}{center_g:02x}{center_b:02x}"
            try:
                self.lbl_rgb_swatch.config(bg=hex_color)
            except Exception:
                pass
        if hasattr(self, 'lbl_rgb_center_val'):
            self.lbl_rgb_center_val.config(
                text=f"Pusat ({cx}, {cy}): R:{center_r:3d} G:{center_g:3d} B:{center_b:3d}\nGrayscale: {center_gray:3d}  |  Hex: #{center_r:02x}{center_g:02x}{center_b:02x}"
            )

    def _copy_rgb_matrix_text(self):
        source_name = self.var_rgb_source.get()
        target_img = self.img_processed if (source_name == "Citra Hasil" and self.img_processed is not None) else self.img_a_orig
        if target_img is None:
            return
        h, w = target_img.shape[:2]
        cx = max(0, min(w - 1, self.var_rgb_x.get()))
        cy = max(0, min(h - 1, self.var_rgb_y.get()))

        lines = [f"=== TABEL MATRIKS RGB 3x3 (Pusat X={cx}, Y={cy}) ===",
                 f"Sumber Citra: {source_name}",
                 f"Posisi\tKoordinat\tRed\tGreen\tBlue\tGrayscale (R+G+B)/3"]

        offsets = [
            (-1, -1, "TL"), (-1, 0, "TC"), (-1, 1, "TR"),
            ( 0, -1, "ML"), ( 0, 0, "C "), ( 0, 1, "MR"),
            ( 1, -1, "BL"), ( 1, 0, "BC"), ( 1, 1, "BR")
        ]
        for dy, dx, pos_tag in offsets:
            nx = cx + dx
            ny = cy + dy
            if 0 <= nx < w and 0 <= ny < h:
                px = target_img[ny, nx]
                if len(target_img.shape) == 3:
                    r, g, b = int(px[0]), int(px[1]), int(px[2])
                else:
                    r = g = b = int(px)
                gray = (r + g + b) // 3
                lines.append(f"{pos_tag}\t({nx}, {ny})\t{r}\t{g}\t{b}\t{gray}")
            else:
                lines.append(f"{pos_tag}\t(Tepi)\t0\t0\t0\t0")

        full_text = "\n".join(lines)
        self.clipboard_clear()
        self.clipboard_append(full_text)
        messagebox.showinfo("Tersalin", "Nilai matriks RGB 3×3 berhasil disalin ke clipboard!")

    def show_rgb_table_dialog(self):
        source_name = self.var_rgb_source.get()
        target_img = self.img_processed if (source_name == "Citra Hasil" and self.img_processed is not None) else self.img_a_orig
        if target_img is None:
            messagebox.showwarning("Peringatan", "Citra belum dimuat.")
            return

        h, w = target_img.shape[:2]
        cx = max(0, min(w - 1, self.var_rgb_x.get()))
        cy = max(0, min(h - 1, self.var_rgb_y.get()))

        top = tk.Toplevel(self)
        top.title("SpectraVision DIP Lab • Inspeksi Matriks Tensor RGB (Slide 8)")
        top.geometry("780x560")
        top.configure(bg=self.C_BG)

        # Header Info
        header_f = tk.Frame(top, bg=self.C_PANEL, padx=16, pady=12)
        header_f.pack(fill=tk.X)

        tk.Label(header_f, text="INSPEKSI MATRIKS TENSOR RGB 3×3 & GRAYSCALE",
                 bg=self.C_PANEL, fg=self.C_TEXT, font=("Segoe UI", 11, "bold")).pack(anchor=tk.W)
        tk.Label(header_f, text=f"Sumber: {source_name} | Pusat Spasial: ({cx}, {cy}) | Dimensi Citra: {w}×{h} px | Formula: f_o(x,y) = (R+G+B)/3",
                 bg=self.C_PANEL, fg=self.C_PEACH, font=("Segoe UI", 9)).pack(anchor=tk.W, pady=(2, 0))

        # Ekstraksi Matriks 3x3 R, G, B, Gray
        mat_r = np.zeros((3, 3), dtype=int)
        mat_g = np.zeros((3, 3), dtype=int)
        mat_b = np.zeros((3, 3), dtype=int)
        mat_gray = np.zeros((3, 3), dtype=int)

        table_data = []
        offsets = [
            (-1, -1, "TL (x-1, y-1)"), (-1, 0, "TC (x, y-1)"), (-1, 1, "TR (x+1, y-1)"),
            ( 0, -1, "ML (x-1, y)"),   ( 0, 0, "Pusat (x, y)"),  ( 0, 1, "MR (x+1, y)"),
            ( 1, -1, "BL (x-1, y+1)"), ( 1, 0, "BC (x, y+1)"), ( 1, 1, "BR (x+1, y+1)")
        ]

        for i, (dy, dx, name) in enumerate(offsets):
            r_idx, c_idx = i // 3, i % 3
            nx = cx + dx
            ny = cy + dy
            if 0 <= nx < w and 0 <= ny < h:
                px = target_img[ny, nx]
                if len(target_img.shape) == 3:
                    r, g, b = int(px[0]), int(px[1]), int(px[2])
                else:
                    r = g = b = int(px)
                gray = (r + g + b) // 3
                c_lbl = f"({nx}, {ny})"
            else:
                r, g, b = 0, 0, 0
                gray = 0
                c_lbl = "Di luar citra"

            mat_r[r_idx, c_idx] = r
            mat_g[r_idx, c_idx] = g
            mat_b[r_idx, c_idx] = b
            mat_gray[r_idx, c_idx] = gray
            table_data.append((name, c_lbl, r, g, b, gray))

        # Konten Utama: 4 Matriks Berdampingan
        content_f = tk.Frame(top, bg=self.C_BG, padx=16, pady=12)
        content_f.pack(fill=tk.BOTH, expand=True)

        grid_matrices_f = tk.Frame(content_f, bg=self.C_BG)
        grid_matrices_f.pack(side=tk.TOP, fill=tk.X, pady=(0, 12))

        channels = [
            ("Saluran Merah (R)", mat_r, "#ff6b6b"),
            ("Saluran Hijau (G)", mat_g, "#4ecdc4"),
            ("Saluran Biru (B)", mat_b, "#4d96ff"),
            ("Grayscale (R+G+B)/3", mat_gray, "#e8a87c")
        ]

        for title, mat, col in channels:
            box = tk.LabelFrame(grid_matrices_f, text=title, bg=self.C_PANEL, fg=col,
                                font=("Segoe UI", 8, "bold"), bd=1, relief=tk.SOLID, padx=8, pady=6)
            box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

            mat_text = (
                f"┌─────────────┐\n"
                f"│ {mat[0,0]:3d} {mat[0,1]:3d} {mat[0,2]:3d} │\n"
                f"│ {mat[1,0]:3d} {mat[1,1]:3d} {mat[1,2]:3d} │\n"
                f"│ {mat[2,0]:3d} {mat[2,1]:3d} {mat[2,2]:3d} │\n"
                f"└─────────────┘\n"
                f"Mean: {np.mean(mat):.1f}"
            )
            tk.Label(box, text=mat_text, bg=self.C_PANEL, fg=self.C_TEXT,
                     font=("Consolas", 9), justify=tk.CENTER).pack()

        # Tabel Lengkap 9 Piksel Ketetanggaan
        f_dlg_tree = tk.Frame(content_f, bg=self.C_PANEL, bd=1, relief=tk.SOLID)
        f_dlg_tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        dlg_cols = ("pos", "coord", "r", "g", "b", "gray", "hex")
        dlg_tree = ttk.Treeview(f_dlg_tree, columns=dlg_cols, show="headings", height=9)
        dlg_tree.heading("pos", text="Posisi Spasial (Slide 8)")
        dlg_tree.heading("coord", text="Koordinat (X, Y)")
        dlg_tree.heading("r", text="Red (R)")
        dlg_tree.heading("g", text="Green (G)")
        dlg_tree.heading("b", text="Blue (B)")
        dlg_tree.heading("gray", text="Grayscale (R+G+B)/3")
        dlg_tree.heading("hex", text="Kode Warna Hex")

        dlg_tree.column("pos", width=140, anchor="w")
        dlg_tree.column("coord", width=110, anchor="center")
        dlg_tree.column("r", width=80, anchor="center")
        dlg_tree.column("g", width=80, anchor="center")
        dlg_tree.column("b", width=80, anchor="center")
        dlg_tree.column("gray", width=140, anchor="center")
        dlg_tree.column("hex", width=100, anchor="center")

        for name, c_lbl, r, g, b, gray in table_data:
            hex_val = f"#{r:02x}{g:02x}{b:02x}"
            dlg_tree.insert("", tk.END, values=(name, c_lbl, r, g, b, gray, hex_val))

        dlg_scroll = ttk.Scrollbar(f_dlg_tree, orient="vertical", command=dlg_tree.yview)
        dlg_tree.configure(yscrollcommand=dlg_scroll.set)
        dlg_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        dlg_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Bottom Buttons
        bottom_f = tk.Frame(top, bg=self.C_BG, padx=16, pady=8)
        bottom_f.pack(side=tk.BOTTOM, fill=tk.X)

        btn_copy = tk.Button(bottom_f, text="📋 Salin Nilai Matriks", font=("Segoe UI", 9),
                             bg=self.C_SUBPANEL, fg=self.C_TEXT, activebackground=self.C_BORDER,
                             bd=1, relief=tk.SOLID, padx=14, pady=6, cursor="hand2",
                             command=self._copy_rgb_matrix_text)
        btn_copy.pack(side=tk.LEFT)

        btn_close = tk.Button(bottom_f, text="Tutup", font=("Segoe UI", 9, "bold"),
                              bg=self.C_PEACH, fg=self.C_BG, activebackground="#f0b890",
                              bd=0, relief=tk.FLAT, padx=18, pady=6, cursor="hand2",
                              command=top.destroy)
        btn_close.pack(side=tk.RIGHT)

    # =========================================================================
    # 4. BOTTOM STATUS BAR (TELEMETRI SPASIAL & SIGNATURE WATERMARK)
    # =========================================================================
    def _build_bottom_status_bar(self):
        bar = tk.Frame(self, bg=self.C_SUBPANEL, height=28, bd=0)
        bar.pack(side=tk.BOTTOM, fill=tk.X)

        sep_bar = tk.Frame(self, bg=self.C_BORDER, height=1)
        sep_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.lbl_stat_ready = tk.Label(bar, text="Status: Ready", bg=self.C_SUBPANEL, fg=self.C_TEAL, font=("Segoe UI", 8, "bold"))
        self.lbl_stat_ready.pack(side=tk.LEFT, padx=(12, 16))

        self.lbl_stat_cursor = tk.Label(bar, text="Cursor: X: ---, Y: ---", bg=self.C_SUBPANEL, fg=self.C_MUTED, font=("Consolas", 8))
        self.lbl_stat_cursor.pack(side=tk.LEFT, padx=12)

        self.lbl_stat_pixel = tk.Label(bar, text="Pixel Intensity: R: --- G: --- B: ---", bg=self.C_SUBPANEL, fg=self.C_MUTED, font=("Consolas", 8))
        self.lbl_stat_pixel.pack(side=tk.LEFT, padx=12)

        self.lbl_stat_zoom = tk.Label(bar, text="Zoom: Fit Screen", bg=self.C_SUBPANEL, fg=self.C_MUTED, font=("Segoe UI", 8))
        self.lbl_stat_zoom.pack(side=tk.LEFT, padx=12)

        # Signature Watermark Peneliti
        lbl_brand = tk.Label(bar, text="SpectraVision DIP Lab • Spatial Computing by Jade Aurestha • Python 3.14",
                             bg=self.C_SUBPANEL, fg=self.C_PEACH, font=("Segoe UI", 8, "bold"))
        lbl_brand.pack(side=tk.RIGHT, padx=16)

    def _on_canvas_mouse_move(self, event, is_proc=False):
        target_img = self.img_processed if (is_proc and self.img_processed is not None) else self.img_a_orig
        if target_img is None:
            return

        lbl = self.lbl_view_proc if is_proc else self.lbl_view_orig
        w_lbl = lbl.winfo_width()
        h_lbl = lbl.winfo_height()

        h_img, w_img = target_img.shape[:2]
        if w_lbl <= 0 or h_lbl <= 0:
            return

        if self.view_mode == "DUAL":
            w_target = max(320, w_lbl - 20) if w_lbl > 100 else 560
            h_target = max(240, h_lbl - 20) if h_lbl > 100 else 480
        else:
            w_target = max(200, w_lbl - 10) if w_lbl > 100 else 360
            h_target = max(160, h_lbl - 10) if h_lbl > 100 else 240

        scale = min(float(w_target) / w_img, float(h_target) / h_img)
        w_disp = int(w_img * scale)
        h_disp = int(h_img * scale)

        pad_x = (w_lbl - w_disp) // 2
        pad_y = (h_lbl - h_disp) // 2

        img_x = int((event.x - pad_x) / scale)
        img_y = int((event.y - pad_y) / scale)

        if 0 <= img_x < w_img and 0 <= img_y < h_img:
            px = target_img[img_y, img_x]
            if len(target_img.shape) == 3:
                r, g, b = int(px[0]), int(px[1]), int(px[2])
            else:
                r = g = b = int(px)
            self.lbl_stat_cursor.config(text=f"Cursor: X: {img_x:4d}, Y: {img_y:4d}")
            self.lbl_stat_pixel.config(text=f"Pixel Intensity: R: {r:3d} G: {g:3d} B: {b:3d}")
        else:
            self.lbl_stat_cursor.config(text="Cursor: X: ---, Y: ---")
            self.lbl_stat_pixel.config(text="Pixel Intensity: R: --- G: --- B: ---")

    def _on_canvas_click(self, event, is_proc=False):
        target_img = self.img_processed if (is_proc and self.img_processed is not None) else self.img_a_orig
        if target_img is None:
            return

        lbl = self.lbl_view_proc if is_proc else self.lbl_view_orig
        w_lbl = lbl.winfo_width()
        h_lbl = lbl.winfo_height()

        h_img, w_img = target_img.shape[:2]
        if w_lbl <= 0 or h_lbl <= 0:
            return

        if self.view_mode == "DUAL":
            w_target = max(320, w_lbl - 20) if w_lbl > 100 else 560
            h_target = max(240, h_lbl - 20) if h_lbl > 100 else 480
        else:
            w_target = max(200, w_lbl - 10) if w_lbl > 100 else 360
            h_target = max(160, h_lbl - 10) if h_lbl > 100 else 240

        scale = min(float(w_target) / w_img, float(h_target) / h_img)
        w_disp = int(w_img * scale)
        h_disp = int(h_img * scale)

        pad_x = (w_lbl - w_disp) // 2
        pad_y = (h_lbl - h_disp) // 2

        img_x = int((event.x - pad_x) / scale)
        img_y = int((event.y - pad_y) / scale)

        if 0 <= img_x < w_img and 0 <= img_y < h_img:
            self.var_rgb_source.set("Citra Hasil" if is_proc else "Citra Asli")
            self.var_rgb_x.set(img_x)
            self.var_rgb_y.set(img_y)
            self._update_rgb_table()
            self.lbl_stat_ready.config(text=f"Piksel dipilih ({'Hasil' if is_proc else 'Asli'}): ({img_x}, {img_y})")

    # =========================================================================
    # REFRESH VIEW & HELPER WIDGETS
    # =========================================================================
    def _on_window_configure(self, event):
        if event.widget == self:
            if self._resize_job is not None:
                self.after_cancel(self._resize_job)
            self._resize_job = self.after(150, self.refresh_active_view)

    def refresh_active_view(self):
        if self.img_a_orig is None:
            return

        target_proc = self.img_processed if self.img_processed is not None else self.img_a_orig

        if self.view_mode == "DUAL":
            self._display_on_widget(self.img_a_orig, self.lbl_view_orig, "dual_orig")
            self._display_on_widget(target_proc, self.lbl_view_proc, "dual_proc")
        else:
            self._display_on_widget(self.img_a_orig, self.lbl_3way_a, "3way_a")
            self._display_on_widget(self.img_b_orig, self.lbl_3way_b, "3way_b")
            self._display_on_widget(target_proc, self.lbl_3way_res, "3way_res")

    def _display_on_widget(self, img_rgb, widget, cache_key):
        if img_rgb is None:
            return

        w_w = widget.winfo_width()
        h_w = widget.winfo_height()

        if self.view_mode == "DUAL":
            w_target = max(320, w_w - 20) if w_w > 100 else 560
            h_target = max(240, h_w - 20) if h_w > 100 else 480
        else:
            w_target = max(200, w_w - 10) if w_w > 100 else 360
            h_target = max(160, h_w - 10) if h_w > 100 else 240

        h, w = img_rgb.shape[:2]
        scale = min(float(w_target) / w, float(h_target) / h)
        w_fit = max(1, int(w * scale))
        h_fit = max(1, int(h * scale))

        resized = cv2.resize(img_rgb, (w_fit, h_fit), interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(resized)
        tk_img = ImageTk.PhotoImage(pil_img)

        self.tk_cache[cache_key] = tk_img
        widget.configure(image=tk_img, text="")

    def _create_slider_row(self, parent, label_text, var, from_val, to_val, callback, step=1, is_float=False, odd_only=False):
        box = tk.Frame(parent, bg=self.C_PANEL)
        box.pack(fill=tk.X, padx=4, pady=2)

        lbl_t = tk.Label(box, text=label_text, bg=self.C_PANEL, fg=self.C_MUTED, font=("Segoe UI", 8))
        lbl_t.pack(side=tk.LEFT)

        init_v = f"{var.get():.2f}" if is_float else str(var.get())
        lbl_v = tk.Label(box, text=init_v, bg=self.C_SUBPANEL, fg=self.C_PEACH, font=("Consolas", 8, "bold"), width=5)
        lbl_v.pack(side=tk.RIGHT)

        def on_move(val):
            if is_float:
                fv = float(val)
                var.set(fv)
                lbl_v.config(text=f"{fv:.2f}")
            else:
                iv = int(round(float(val)))
                if odd_only or step == 2:
                    # Khusus kernel ukuran ganjil untuk konvolusi (1, 3, 5, 7, 9, ...)
                    if iv % 2 == 0:
                        iv = iv + 1
                    iv = max(int(from_val), min(int(to_val), iv))
                elif step > 1 and iv % step != 0:
                    iv = (iv // step) * step
                var.set(iv)
                lbl_v.config(text=str(iv))
            callback()

        sc = ttk.Scale(parent, from_=from_val, to=to_val, value=var.get(), command=on_move, style="Horizontal.TScale")
        sc.pack(fill=tk.X, padx=4, pady=(0, 4))
        return sc, lbl_v

    def _update_specs_log(self, active_op_name="None"):
        if self.img_processed is None:
            return

        h, w = self.img_processed.shape[:2]
        ch = 3 if len(self.img_processed.shape) == 3 else 1
        mem_mb = (self.img_processed.nbytes) / (1024 * 1024)

        gray = cv2.cvtColor(self.img_processed, cv2.COLOR_RGB2GRAY) if ch == 3 else self.img_processed
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        min_val, max_val = int(np.min(gray)), int(np.max(gray))

        hist, _ = np.histogram(gray.ravel(), bins=256, range=[0, 256])
        prob = hist / float(gray.size)
        prob = prob[prob > 0]
        entropy = -np.sum(prob * np.log2(prob))

        specs_text = (
            f"Dimensi Matriks : {w} × {h} px\n"
            f"Ruang Tensor    : {'RGB (3-Kanal)' if ch==3 else 'Gray (1-Kanal)'}\n"
            f"Rentang Dinamis : [{min_val} - {max_val}]\n"
            f"Mean Spasial (μ): {mean_val:.2f}\n"
            f"Std Deviasi (σ) : {std_val:.2f}\n"
            f"Entropi Shannon : {entropy:.3f} bits/px\n"
            f"Alokasi Memori  : {mem_mb:.2f} MB\n"
            f"Operasi Aktif   : {active_op_name[:18]}\n"
            f"Peneliti Utama  : Jade Aurestha"
        )
        self.lbl_specs_log.config(text=specs_text)

    # =========================================================================
    # FILE HANDLING & SYNTHETIC DATA GENERATION
    # =========================================================================
    def _generate_default_image_b(self, pattern_type="circle"):
        size = 500
        if pattern_type == "circle":
            y, x = np.ogrid[:size, :size]
            dist = np.sqrt((x - 250)**2 + (y - 250)**2)
            circle = np.clip(255 - dist * 0.9, 0, 255).astype(np.uint8)
            grid = np.zeros((size, size), dtype=np.uint8)
            grid[::30, :] = 90
            grid[:, ::30] = 90
            comb = cv2.add(circle, grid)
            self.filename_b = "Pola Sintetis Lingkaran & Grid"
        elif pattern_type == "checker":
            comb = np.zeros((size, size), dtype=np.uint8)
            for i in range(0, size, 50):
                for j in range(0, size, 50):
                    if (i // 50 + j // 50) % 2 == 0:
                        comb[i:i+50, j:j+50] = 220
                    else:
                        comb[i:i+50, j:j+50] = 30
            self.filename_b = "Pola Sintetis Kotak Catur"
        else:
            ramp = np.tile(np.linspace(0, 255, size, dtype=np.uint8), (size, 1))
            comb = ramp
            self.filename_b = "Pola Sintetis Gradien Horisontal"

        self.img_b_orig = cv2.cvtColor(comb, cv2.COLOR_GRAY2RGB)
        if hasattr(self, 'lbl_citra_b_status'):
            self.lbl_citra_b_status.config(text=f"Citra B: {self.filename_b}")

    def _cycle_synthetic_b(self):
        patterns = ["circle", "checker", "gradient"]
        curr = getattr(self, '_pattern_idx', 0)
        nxt = (curr + 1) % len(patterns)
        self._pattern_idx = nxt
        self._generate_default_image_b(patterns[nxt])
        self.var_arith_target.set("Antar Citra (Citra A & B)")
        self.set_view_mode("3WAY")
        self.on_arithmetic_change()
        messagebox.showinfo("Citra B Diperbarui", f"Citra B diganti dengan:\n{self.filename_b}")

    def _generate_synthetic_test_chart(self):
        """Membangkitkan Citra Kalibrasi Spektral Ilmiah (Zone Plate, Gradients, Color Bars)."""
        size = 512
        chart = np.zeros((size, size, 3), dtype=np.uint8)
        chart[:] = (26, 26, 32)

        # 1. Concentric Zone Rings di Bagian Tengah
        y, x = np.ogrid[:size, :size]
        dist = np.sqrt((x - 256)**2 + (y - 200)**2)
        rings = (np.sin(dist * 0.12) * 127 + 128).astype(np.uint8)
        chart[:, :, 0] = np.clip(rings * 0.75 + 30, 0, 255)
        chart[:, :, 1] = np.clip(rings * 0.85 + 30, 0, 255)
        chart[:, :, 2] = np.clip(rings + 30, 0, 255)

        # 2. Color Calibration Swatches di Bagian Atas
        swatches = [
            (232, 168, 124),  # Peach
            (78, 205, 196),   # Teal
            (255, 107, 107),  # Coral Red
            (255, 230, 109),  # Yellow
            (77, 150, 255),   # Blue
            (240, 240, 240)   # White
        ]
        sw_w = size // len(swatches)
        for i, color in enumerate(swatches):
            chart[10:45, i*sw_w:(i+1)*sw_w] = color

        # 3. Spatial Frequency Grating (Garis-garis Frekuensi)
        grid_zone = np.zeros((60, size, 3), dtype=np.uint8)
        for i in range(size):
            freq = int((np.sin((i / 5.0) ** 1.3) + 1) * 127)
            grid_zone[:, i] = (freq, freq, freq)
        chart[size-130:size-70, :] = grid_zone

        # 4. Grayscale Linear Gradient Ramp [0 - 255]
        ramp = np.tile(np.linspace(0, 255, size, dtype=np.uint8), (45, 1))
        chart[size-60:size-15, :, 0] = ramp
        chart[size-60:size-15, :, 1] = ramp
        chart[size-60:size-15, :, 2] = ramp

        # Watermark Ilmiah di Citra
        cv2.putText(chart, "SPECTRAVISION TEST MATRIX", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (248, 249, 250), 2)
        cv2.putText(chart, "Jade Aurestha", (size - 160, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (232, 168, 124), 1)

        self.filename_a = "Citra Kalibrasi Spektral (Sintetis)"
        self.img_a_orig = chart
        self.img_a_initial = chart.copy()
        self.img_processed = chart.copy()

        self.lbl_stat_ready.config(text=f"Loaded: {self.filename_a}")
        self.refresh_active_view()
        self.switch_right_tab(self.active_right_tab)

    def open_image_a_dialog(self):
        p = filedialog.askopenfilename(
            title="Pilih Citra A (SpectraVision DIP Lab)",
            filetypes=[("File Gambar", "*.jpg *.jpeg *.png *.bmp *.webp *.tif"), ("Semua File", "*.*")]
        )
        if p:
            self.load_image_a(p)

    def load_image_a(self, filepath):
        try:
            fb = np.fromfile(filepath, dtype=np.uint8)
            img_bgr = cv2.imdecode(fb, cv2.IMREAD_COLOR)
            if img_bgr is None:
                messagebox.showerror("Error", f"Gagal membaca format citra:\n{filepath}")
                return

            self.filepath_a = filepath
            self.filename_a = os.path.basename(filepath)
            self.img_a_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            self.img_a_initial = self.img_a_orig.copy()
            self.img_processed = self.img_a_orig.copy()

            self.lbl_stat_ready.config(text=f"Loaded: {self.filename_a}")
            self.refresh_active_view()
            self.switch_right_tab(self.active_right_tab)

        except Exception as e:
            messagebox.showerror("Error Membuka Citra", str(e))

    def open_image_b_dialog(self):
        p = filedialog.askopenfilename(
            title="Pilih Citra B (SpectraVision DIP Lab)",
            filetypes=[("File Gambar", "*.jpg *.jpeg *.png *.bmp *.webp *.tif"), ("Semua File", "*.*")]
        )
        if p:
            try:
                fb = np.fromfile(p, dtype=np.uint8)
                img_bgr = cv2.imdecode(fb, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    messagebox.showerror("Error", f"Gagal membaca citra B:\n{p}")
                    return

                self.filepath_b = p
                self.filename_b = os.path.basename(p)
                self.img_b_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                self.lbl_citra_b_status.config(text=f"Citra B: {self.filename_b}")
                self.var_arith_target.set("Antar Citra (Citra A & B)")
                self.set_view_mode("3WAY")
                self.on_arithmetic_change()
                messagebox.showinfo("Sukses", f"Citra B '{self.filename_b}' siap sebagai operan matriks 2!")

            except Exception as e:
                messagebox.showerror("Error", str(e))

    def save_processed_image(self):
        if self.img_processed is None:
            messagebox.showwarning("Peringatan", "Belum ada hasil citra untuk disimpan.")
            return

        sp = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("Bitmap", "*.bmp")],
            initialfile=f"SpectraVision_{self.filename_a}"
        )
        if sp:
            try:
                res_bgr = cv2.cvtColor(self.img_processed, cv2.COLOR_RGB2BGR) if len(self.img_processed.shape) == 3 else self.img_processed
                ext = os.path.splitext(sp)[1]
                is_success, buf = cv2.imencode(ext, res_bgr)
                if is_success:
                    buf.tofile(sp)
                    messagebox.showinfo("Tersimpan", f"Citra berhasil disimpan di:\n{sp}")
                else:
                    cv2.imwrite(sp, res_bgr)
                    messagebox.showinfo("Tersimpan", f"Citra berhasil disimpan di:\n{sp}")
            except Exception as e:
                messagebox.showerror("Gagal Menyimpan", str(e))

    def use_result_as_input_a(self):
        if self.img_processed is None:
            return
        # Pastikan format 3-kanal RGB
        if len(self.img_processed.shape) == 2:
            self.img_a_orig = cv2.cvtColor(self.img_processed, cv2.COLOR_GRAY2RGB)
        else:
            self.img_a_orig = self.img_processed.copy()

        self.filename_a = f"Pipeline_{self.filename_a}"
        self.lbl_stat_ready.config(text=f"Loaded: {self.filename_a}")
        self.refresh_active_view()
        self.switch_right_tab(self.active_right_tab)
        messagebox.showinfo("Pipeline", "Citra hasil ditetapkan sebagai Citra A masukan baru!")

    def reset_to_original_image(self):
        """Mereset citra hasil kembali persis seperti citra asli (original) awal serta mereset semua slider."""
        if self.img_a_initial is not None:
            self.img_a_orig = self.img_a_initial.copy()
        elif self.img_a_orig is not None:
            self.img_a_initial = self.img_a_orig.copy()
        else:
            return

        # Kembalikan hasil pemrosesan ke citra asli murni
        self.img_processed = self.img_a_orig.copy()

        # Reset semua nilai parameter slider ke kondisi netral/default
        self.var_brightness.set(0)
        self.var_contrast.set(0)
        self.var_gamma.set(1.0)
        self.var_blur.set(1)
        self.var_canny1.set(100)
        self.var_canny2.set(200)
        self.active_filter_mode = "sliders"
        self.var_add_scalar.set(40)
        self.var_sub_scalar.set(40)
        self.var_mul_scalar.set(1.5)
        self.var_div_scalar.set(2.0)
        self.var_slice_low.set(80)
        self.var_slice_high.set(180)
        self.var_slice_highlight.set(255)
        self.var_slice_preserve_bg.set(False)
        self.var_bit_plane.set(7)

        # Reset highlight tombol filter
        if hasattr(self, 'filter_buttons'):
            for pid, b in self.filter_buttons.items():
                b.configure(bg=self.C_SUBPANEL, fg=self.C_TEXT, font=("Segoe UI", 8))

        # Reset slider blur visual jika ada
        if hasattr(self, 'scale_blur') and hasattr(self, 'lbl_val_blur'):
            try:
                self.scale_blur.set(1)
                self.lbl_val_blur.config(text="1")
            except Exception:
                pass

        # Update label status & tag
        self.tag_proc.config(text="Citra Asli (Original)")
        self.lbl_stat_ready.config(text=f"Reset ke Citra Asli: {self.filename_a}")
        self._update_specs_log("Original (Reset)")

        # Render ulang kanvas
        self.refresh_active_view()
        if self.active_right_tab == "histogram":
            self._update_histogram_chart()

    def reset_all_parameters(self):
        self.reset_to_original_image()

    def sample_roi_matrix(self):
        if self.img_a_orig is None:
            return
        self.show_rgb_table_dialog()

    def _try_load_default_image(self):
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "assets", "photo.jpg"),
            os.path.join(os.path.dirname(__file__), "..", "assets", "profile.jpg"),
            os.path.join(os.path.dirname(__file__), "sample.jpg")
        ]
        for c in candidates:
            if os.path.exists(c):
                self.load_image_a(c)
                return

        self._generate_synthetic_test_chart()


# Alias backwards compatibility
PixelCraftStudio = SpectraVisionStudio


# =============================================================================
# ENTRY POINT PROGRAM
# =============================================================================
if __name__ == "__main__":
    print("Memulai SpectraVision DIP Laboratory by Jade Aurestha...")
    app = SpectraVisionStudio()
    app.mainloop()