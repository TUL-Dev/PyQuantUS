import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import scipy.interpolate as interpolate
from matplotlib.widgets import RectangleSelector, Cursor
import matplotlib.patches as patches

from PyQt6.QtWidgets import QDialog, QFileDialog, QHBoxLayout
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt
from PyQt6.uic.load_ui import loadUi

from pyquantus.seg_loading.options import get_seg_loaders
from pyquantus.data_objs.image import UltrasoundRfImage


def select_seg_helper(path_input, file_exts):
    file_exts = " ".join([f"*{ext}" for ext in file_exts])
    if not os.path.exists(path_input.text()):  # check if file path is manually typed
        file_name, _ = QFileDialog.getOpenFileName(None, "Open File", filter=file_exts)
        if file_name != "":  # If valid file is chosen
            path_input.setText(file_name)
        else:
            return

from .roi_selection_ui import Ui_constructRoi
class RoiSelectionGUI(QDialog, Ui_constructRoi):
    def __init__(self, image_data: UltrasoundRfImage):
        super(RoiSelectionGUI, self).__init__()
        super().__init__()
        self.setupUi(self)
        # loadUi(str(Path("pyquantus/gui/seg_loading/roi_selection.ui")), self)

        self.setLayout(self.full_screen_layout)
        self.full_screen_layout.removeItem(self.draw_roi_layout)
        self.hide_draw_roi_layout()
        self.full_screen_layout.removeItem(self.seg_loading_layout)
        self.hide_seg_loading_layout()
        self.full_screen_layout.removeItem(self.frame_preview_layout)
        self.hide_frame_preview_layout()
        self.full_screen_layout.setStretchFactor(self.side_bar_layout, 1)
        self.full_screen_layout.setStretchFactor(self.select_type_layout, 10)
        
        self.seg_loaders = get_seg_loaders()
        seg_loaders = [s.replace("_", " ").capitalize() for s in self.seg_loaders.keys()]
        self.seg_loaders_list = [s.replace("rf", "RF").replace("iq", "IQ").replace("roi", "ROI").replace("voi", "VOI") for s in seg_loaders]
        self.seg_loaders_list.insert(0, "Draw New")
        self.seg_type_dropdown.addItems(self.seg_loaders_list)
        
        self.image_path_input.setText(image_data.scan_name)
        self.phantom_path_input.setText(image_data.phantom_name)
        self.save_roi_button.hide()
        self.back_button.clicked.connect(self.set_go_back)
        self.accept_type_button.clicked.connect(self.accept_seg_type)
        self.choose_seg_path_button.clicked.connect(self.select_seg_helper)
        self.clear_seg_path_button.clicked.connect(self.seg_path_input.clear)
        self.user_draw_rectangle_button.clicked.connect(self.start_draw_rect)
        self.draw_rectangle_button.clicked.connect(self.draw_rect_clicked)
        self.draw_freehand_button.clicked.connect(self.start_draw_freehand)
        self.draw_roi_button.clicked.connect(self.draw_freehand_clicked)
        self.back_from_rectangle_button.clicked.connect(self.back_from_rect)
        self.undo_last_pt_button.clicked.connect(self.undo_last_pt)
        
        self.image_data = image_data
        self.go_back = False
        self.frame = 0; self.scattered_points = []; self.rect_coords = []
        self.points_plotted_x = []; self.points_plotted_y = []
        self.im_array = image_data.sc_bmode if image_data.sc_bmode is not None else image_data.bmode
        
        # Prepare B-Mode display plot
        self.bmode_layout = QHBoxLayout(self.im_display_frame)
        self.bmode_layout.setObjectName("bmode_layout")
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.bmode_layout.addWidget(self.canvas)
        self.selector = RectangleSelector(
            self.ax,
            self.draw_rect,
            useblit=True,
            props=dict(linestyle="-", color="cyan", fill=False),
        )
        self.selector.set_active(False)
        
    def set_go_back(self):
        self.go_back = True
        
    def select_seg_helper(self):
        select_seg_helper(self.seg_path_input, self.file_exts)
        self.select_seg_error_msg.hide()
        
    def back_to_select(self):
        self.full_screen_layout.removeItem(self.seg_loading_layout)
        self.hide_seg_loading_layout()
        self.full_screen_layout.removeItem(self.frame_preview_layout)
        self.hide_frame_preview_layout()
        self.full_screen_layout.removeItem(self.draw_roi_layout)
        self.hide_draw_roi_layout()
        self.full_screen_layout.addItem(self.select_type_layout)
        self.show_type_selection_layout()
        try:
            self.undo_last_roi()
        except AttributeError:
            pass
        self.full_screen_layout.setStretchFactor(self.side_bar_layout, 1)
        self.full_screen_layout.setStretchFactor(self.select_type_layout, 10)
        self.back_button.clicked.disconnect()
        self.back_button.clicked.connect(self.set_go_back)
        
    def hide_frame_preview_layout(self):
        self.select_frame_label.hide()
        self.im_preview.hide()
        self.frame_slider.hide()
        self.cur_frame_label.hide()
        self.of_frames_label.hide()
        self.total_frames_label.hide()
        self.accept_frame_button.hide()
        
    def show_frame_preview_layout(self):
        self.select_frame_label.show()
        self.im_preview.show()
        self.frame_slider.show()
        self.cur_frame_label.show()
        self.of_frames_label.show()
        self.total_frames_label.show()
        self.accept_frame_button.show()
        
    def move_to_draw(self):
        self.full_screen_layout.removeItem(self.seg_loading_layout)
        self.hide_seg_loading_layout()
        self.full_screen_layout.addItem(self.draw_roi_layout)
        self.show_draw_seg_start()
        self.full_screen_layout.setStretchFactor(self.side_bar_layout, 1)
        self.full_screen_layout.setStretchFactor(self.draw_roi_layout, 10)
        
    def accept_seg_type(self):
        self.seg_type = list(self.seg_loaders.keys())[self.seg_type_dropdown.currentIndex()-1] if self.seg_type_dropdown.currentIndex() else "Draw New"
        self.back_button.clicked.disconnect()
        self.back_button.clicked.connect(self.back_to_select)
        
        if self.seg_type != "Draw New":
            self.file_exts = self.seg_loaders[self.seg_type]["exts"]
            
            self.full_screen_layout.removeItem(self.select_type_layout)
            self.hide_type_selection_layout()
            self.full_screen_layout.addItem(self.seg_loading_layout)
            self.show_seg_loading_layout()
            self.full_screen_layout.setStretchFactor(self.side_bar_layout, 1)
            self.full_screen_layout.setStretchFactor(self.seg_loading_layout, 10)
            self.seg_path_label.setText(f"Input Path to Segmentation File\n ({', '.join(self.file_exts)})")
            
            self.seg_kwargs_box.setText("{\n\t'assert_scan': False,\t\t# Checks if the seg is initially from the same scan\n\t'assert_phantom': False,\t\t# Checks if the seg is initially from the same phantom\n}")
        else:
            if self.im_array.ndim == 3: # need to select frame
                self.full_screen_layout.removeItem(self.select_type_layout)
                self.hide_type_selection_layout()
                self.full_screen_layout.addItem(self.frame_preview_layout)
                self.show_frame_preview_layout()
                self.full_screen_layout.setStretchFactor(self.side_bar_layout, 1)
                self.full_screen_layout.setStretchFactor(self.seg_loading_layout, 10)
                
                self.im_data = np.array(self.im_array[self.frame]).reshape(self.im_array.shape[1], self.im_array.shape[2])
                self.im_data = np.require(self.im_data, np.uint8, 'C')
                self.bytes_line = self.im_data.strides[0]; self.ar_height, self.ar_width = self.im_data.shape
                self.q_im = QImage(self.im_data, self.ar_width, self.ar_height, self.bytes_line, QImage.Format.Format_Grayscale8)
                self.im_preview.setPixmap(QPixmap.fromImage(self.q_im).scaled(self.im_preview.width(), self.im_preview.height(), Qt.AspectRatioMode.IgnoreAspectRatio))
                
                self.total_frames_label.setText(str(self.im_array.shape[0]-1))
                self.frame_slider.setMinimum(0)
                self.frame_slider.setMaximum(self.im_array.shape[0]-1)
                self.frame_slider.valueChanged.connect(self.frameChanged)
                self.cur_frame_label.setText("0")
                
            elif self.im_array.ndim == 2: # only need to draw ROI
                self.full_screen_layout.removeItem(self.select_type_layout)
                self.hide_type_selection_layout()
                self.full_screen_layout.addItem(self.draw_roi_layout)
                self.show_draw_seg_start()
                self.full_screen_layout.setStretchFactor(self.side_bar_layout, 1)
                self.full_screen_layout.setStretchFactor(self.draw_roi_layout, 10)
                
                self.im_data = self.im_array
                self.plot_on_canvas()
            
    def hide_frame_selection_layout(self):
        self.select_frame_label.hide()
        
    def hide_type_selection_layout(self):
        self.select_type_label.hide()
        self.seg_type_dropdown.hide()
        self.accept_type_button.hide()
        
    def show_type_selection_layout(self):
        self.select_type_label.show()
        self.seg_type_dropdown.show()
        self.accept_type_button.show()
        
    def hide_seg_loading_layout(self):
        self.select_seg_label.hide()
        self.seg_path_label.hide()
        self.seg_path_input.hide()
        self.choose_seg_path_button.hide()
        self.clear_seg_path_button.hide()
        self.seg_kwargs_label.hide()
        self.seg_kwargs_box.hide()
        self.accept_seg_path_button.hide()
        self.loading_screen_label.hide()
        self.select_seg_error_msg.hide()
        
    def show_seg_loading_layout(self):
        self.select_seg_label.show()
        self.seg_path_label.show()
        self.seg_path_input.show()
        self.choose_seg_path_button.show()
        self.clear_seg_path_button.show()
        self.seg_kwargs_label.show()
        self.seg_kwargs_box.show()
        self.accept_seg_path_button.show()
        
    def hide_draw_roi_layout(self):
        self.physical_depth_label.hide(); self.physical_depth_val.hide()
        self.physical_width_label.hide(); self.physical_width_val.hide()
        self.physical_dims_label.hide(); self.pixel_depth_label.hide()
        self.pixel_depth_val.hide(); self.pixel_width_label.hide()
        self.pixel_width_val.hide(); self.pixel_dims_label.hide()
        self.construct_roi_label.hide(); self.edit_image_display_button.hide()
        self.physical_rect_dims_label.hide(); self.physical_rect_height_label.hide()
        self.physical_rect_height_val.hide(); self.physical_rect_width_label.hide()
        self.physical_rect_width_val.hide(); self.draw_roi_button.hide()
        self.accept_rectangle_button.hide(); self.back_from_freehand_button.hide()
        self.back_from_rectangle_button.hide(); self.close_roi_button.hide()
        self.draw_freehand_button.hide(); self.draw_rectangle_button.hide()
        self.redraw_roi_button.hide(); self.undo_last_pt_button.hide()
        self.user_draw_rectangle_button.hide(); self.draw_freehand_buttons.hide()
        self.im_display_frame.hide(); self.accept_freehand_button.hide()
        self.draw_options_buttons.hide(); self.draw_rect_buttons.hide()
        
    def show_draw_roi_layout(self):
        self.physical_depth_label.show(); self.physical_depth_val.show()
        self.physical_width_label.show(); self.physical_width_val.show()
        self.physical_dims_label.show(); self.pixel_depth_label.show()
        self.pixel_depth_val.show(); self.pixel_width_label.show()
        self.pixel_width_val.show(); self.pixel_dims_label.show()
        self.construct_roi_label.show(); self.edit_image_display_button.show()
        self.physical_rect_dims_label.show(); self.physical_rect_height_label.show()
        self.physical_rect_height_val.show(); self.physical_rect_width_label.show()
        self.physical_rect_width_val.show(); self.user_draw_rectangle_button.show()
        self.im_display_frame.show(); self.draw_options_buttons.hide()
        
    def show_draw_seg_start(self):
        self.physical_depth_label.show(); self.physical_depth_val.show()
        self.physical_width_label.show(); self.physical_width_val.show()
        self.physical_dims_label.show(); self.pixel_depth_label.show()
        self.pixel_depth_val.show(); self.pixel_width_label.show()
        self.pixel_width_val.show(); self.pixel_dims_label.show()
        self.draw_freehand_button.show(); self.user_draw_rectangle_button.show()
        self.im_display_frame.show(); self.draw_options_buttons.show()
        self.construct_roi_label.show()
        
    def start_draw_freehand(self):
        self.hide_initial_buttons()
        self.show_freehanded_buttons()
        
    def show_freehanded_buttons(self):
        self.undo_last_pt_button.show()
        self.close_roi_button.show()
        self.accept_freehand_button.show()
        self.back_from_freehand_button.show()
        self.draw_roi_button.show()
        self.draw_freehand_buttons.show()
        
    def start_draw_rect(self):
        self.hide_initial_buttons()
        self.show_rect_buttons()
        
    def hide_initial_buttons(self):
        self.draw_freehand_button.hide()
        self.user_draw_rectangle_button.hide()
        
    def hide_rect_buttons(self):
        self.draw_rect_buttons.hide()
        self.draw_rectangle_button.hide()
        self.back_from_rectangle_button.hide()
        self.accept_rectangle_button.hide()
        self.physical_rect_dims_label.hide()
        self.physical_rect_dims_label.hide()
        self.physical_rect_height_label.hide()
        self.physical_rect_width_label.hide()
        self.physical_rect_height_val.hide()
        self.physical_rect_width_val.hide()

    def show_rect_buttons(self):
        self.draw_rectangle_button.show()
        self.draw_rect_buttons.show()
        self.back_from_rectangle_button.show()
        self.accept_rectangle_button.show()
        self.physical_rect_dims_label.show()
        self.physical_rect_height_label.show()
        self.physical_rect_width_label.show()
        self.physical_rect_height_val.show()
        self.physical_rect_width_val.show()
        
    def draw_freehand_clicked(self):
        if self.draw_roi_button.isChecked():  # Set up b-mode to be drawn on
            self.cid = self.figure.canvas.mpl_connect(
                "button_press_event", self.interpolate_points
            )
        else:  # No longer let b-mode be drawn on
            if hasattr(self, "cid"):
                self.cid = self.figure.canvas.mpl_disconnect(self.cid)
        self.canvas.draw()
        
    def draw_rect_clicked(self):
        if self.draw_rectangle_button.isChecked():  # Set up b-mode to be drawn on
            self.selector.set_active(True)
            self.cid = self.figure.canvas.mpl_connect(
                "button_press_event", self.clear_rect
            )
        else:  # No longer let b-mode be drawn on
            self.cid = self.figure.canvas.mpl_disconnect(self.cid)
            self.selector.set_active(False)
        self.canvas.draw()
        
    def draw_rect(self, event1, event2):
        self.rect_coords = [
            int(event1.xdata),
            int(event1.ydata),
            int(event2.xdata),
            int(event2.ydata),
        ]
        self.plot_patch()
        
    def plot_patch(self):
        if len(self.rect_coords) > 0:
            left, bottom, right, top = self.rect_coords
            rect = patches.Rectangle(
                (left, bottom),
                (right - left),
                (top - bottom),
                linewidth=1,
                edgecolor="cyan",
                facecolor="none",
            )
            if len(self.ax.patches) > 0:
                self.ax.patches.pop()

            self.ax.add_patch(rect)

            mpl_pix_width = abs(right - left)
            lateral_res = self.image_data.lateral_res if self.image_data.sc_bmode is None else self.image_data.sc_lateral_res
            mm_width = mpl_pix_width * lateral_res
            # mm_width = lateral_res * im_pix_width  # (mm/pixel)*pixels
            self.physical_rect_width_val.setText(str(np.round(mm_width, decimals=2)))

            mpl_pix_height = abs(top - bottom)
            axial_res = self.image_data.axial_res if self.image_data.sc_bmode is None else self.image_data.sc_axial_res
            mm_height = mpl_pix_height * axial_res
            # mm_height = axial_res * im_pix_height  # (mm/pixel)*pixels
            self.physical_rect_height_val.setText(str(np.round(mm_height, decimals=2)))

            self.figure.subplots_adjust(
                left=0, right=1, bottom=0, top=1, hspace=0.2, wspace=0.2
            )
            self.ax.tick_params(bottom=False, left=False)
            self.canvas.draw()
        
    def back_from_rect(self):
        self.physical_rect_height_val.setText("0")
        self.physical_rect_width_val.setText("0")
        self.draw_rectangle_button.setChecked(False)
        self.undo_last_roi(); self.close_roi_button.hide()
        self.hide_rect_buttons()
        self.show_draw_seg_start()
        self.rect_coords = []
        self.selector.set_active(False)
        if len(self.ax.patches) > 0:
            self.ax.patches.pop()
        self.canvas.draw()
        
    def undo_last_pt(self):  # When drawing ROI, undo last point plotted
        if len(self.points_plotted_x) > 0 and self.draw_roi_button.isCheckable():
            scattered_point = self.scattered_points.pop()
            scattered_point.remove()
            self.points_plotted_x.pop(); self.points_plotted_y.pop()
            if len(self.points_plotted_x) > 0:
                old_spline = self.spline.pop(0)
                old_spline.remove()
                if len(self.points_plotted_x) > 1:
                    x_spline, y_spline = calculate_spline(
                        np.array(self.points_plotted_x) / self.im_data.shape[1], np.array(self.points_plotted_y) / self.im_data.shape[0]
                    )
                    x_spline *= self.im_data.shape[1]
                    y_spline *= self.im_data.shape[0]
                    x_spline = np.clip(x_spline, a_min=0, a_max=self.im_data.shape[1]-1)
                    y_spline = np.clip(y_spline, a_min=0, a_max=self.im_data.shape[0]-1)
                    self.spline_x = x_spline
                    self.spline_y = y_spline
                    self.spline = self.ax.plot(
                        self.spline_x,
                        self.spline_y,
                        color="cyan",
                        linewidth=0.75,
                    )
            self.canvas.draw()
            self.draw_roi_button.setChecked(True)
            self.draw_freehand_clicked()
        
    def undo_last_roi(self):  # Remove previously drawn roi and prepare user to draw a new one
        self.spline_x = np.array([])
        self.spline_y = np.array([])
        self.points_plotted_x = []
        self.points_plotted_y = []
        self.draw_roi_button.setChecked(False)
        self.draw_roi_button.setCheckable(True)
        self.close_roi_button.show()
        self.redraw_roi_button.hide()
        self.plot_on_canvas()
        
    def clear_rect(self, event):
        if len(self.ax.patches) > 0:
            rect = self.ax.patches[0]
            rect.remove()
            self.canvas.draw()
        
    def interpolate_points(self, event):  # Update ROI being drawn using spline using 2D interpolation
        if len(self.points_plotted_x) > 0 and self.points_plotted_x[-1] == int(event.xdata) and self.points_plotted_y[-1] == int(event.ydata):
            return

        self.points_plotted_x.append(int(event.xdata))
        self.points_plotted_y.append(int(event.ydata))
        plotted_points = len(self.points_plotted_x)

        if plotted_points > 1:
            if plotted_points > 2:
                old_spline = self.spline.pop(0)
                old_spline.remove()

            x_spline, y_spline = calculate_spline(
                np.array(self.points_plotted_x) / self.im_data.shape[1], np.array(self.points_plotted_y) / self.im_data.shape[0]
            )
            x_spline *= self.im_data.shape[1]
            y_spline *= self.im_data.shape[0]
            x_spline = np.clip(x_spline, a_min=0, a_max=self.im_data.shape[1]-1)
            y_spline = np.clip(y_spline, a_min=0, a_max=self.im_data.shape[0]-1)
            self.spline = self.ax.plot(
                x_spline, y_spline, color="cyan", zorder=1, linewidth=0.75
            )
            self.figure.subplots_adjust(
                left=0, right=1, bottom=0, top=1, hspace=0.2, wspace=0.2
            )
            self.ax.tick_params(bottom=False, left=False)
        self.scattered_points.append(
            self.ax.scatter(
                self.points_plotted_x[-1],
                self.points_plotted_y[-1],
                marker="o", # type: ignore
                s=0.5,
                c="red",
                zorder=500,
            )
        )
        self.canvas.draw()
        
    def plot_on_canvas(self):
        self.ax.clear()
        if self.image_data.sc_bmode is not None:
            width = self.im_data.shape[1]*self.image_data.sc_lateral_res
            height = self.im_data.shape[0]*self.image_data.sc_axial_res
        else:
            width = self.im_data.shape[1]*self.image_data.lateral_res
            height = self.im_data.shape[0]*self.image_data.axial_res
        aspect = width/height
        im = self.ax.imshow(self.im_data, cmap="gray")
        extent = im.get_extent()
        self.ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)
        self.figure.set_facecolor((0, 0, 0, 0)) #type: ignore
        self.ax.axis("off")
        
        if hasattr(self, 'spline_x') and len(self.spline_x):
            self.spline = self.ax.plot(self.spline_x, self.spline_y, 
                                       color="cyan", zorder=1, linewidth=0.75)
        elif len(self.points_plotted_x) > 0:
            self.scattered_points.append(
                self.ax.scatter(
                    self.points_plotted_x[-1],
                    self.points_plotted_y[-1],
                    marker="o", #type: ignore
                    s=0.5,
                    c="red",
                    zorder=500,
                )
            )
            if len(self.points_plotted_x) > 1:
                x_spline, y_spline = calculate_spline(
                    np.array(self.points_plotted_x) / self.im_data.shape[1], np.array(self.points_plotted_y) / self.im_data.shape[0]
                )
                x_spline *= self.im_data.shape[1]
                y_spline *= self.im_data.shape[0]
                x_spline = np.clip(x_spline, a_min=0, a_max=self.im_data.shape[1]-1)
                y_spline = np.clip(y_spline, a_min=0, a_max=self.im_data.shape[0]-1)
                self.spline = self.ax.plot(
                    x_spline, y_spline, color="cyan", zorder=1, linewidth=0.75
                )

        self.figure.subplots_adjust(
            left=0, right=1, bottom=0, top=1, hspace=0.2, wspace=0.2
        )
        plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        self.canvas.draw()  # Refresh canvas
    
        
def calculate_spline(xpts, ypts):  # 2D spline interpolation
    cv = []
    for i in range(len(xpts)):
        cv.append([xpts[i], ypts[i]])
    cv = np.array(cv)
    if len(xpts) == 2:
        tck, _ = interpolate.splprep(cv.T, s=0.0, k=1)
    elif len(xpts) == 3:
        tck, _ = interpolate.splprep(cv.T, s=0.0, k=2)
    else:
        tck, _ = interpolate.splprep(cv.T, s=0.0, k=3)
    x, y = np.array(interpolate.splev(np.linspace(0, 1, 1000), tck))
    return x, y
