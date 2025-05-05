import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import scipy.interpolate as interpolate
from matplotlib.widgets import RectangleSelector, Cursor
import matplotlib.patches as patches

from PyQt6.QtWidgets import QDialog, QFileDialog
from PyQt6.QtGui import QImage
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

class RoiSelectionGUI(QDialog):
    def __init__(self, image_data: UltrasoundRfImage):
        super(RoiSelectionGUI, self).__init__()
        loadUi(str(Path("pyquantus/gui/seg_loading/roi_selection.ui")), self)

        self.setLayout(self.full_screen_layout)
        self.full_screen_layout.removeItem(self.draw_roi_layout)
        self.hide_draw_roi_layout()
        self.full_screen_layout.removeItem(self.seg_loading_layout)
        self.hide_seg_loading_layout()
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
        
        self.image_data = image_data
        self.go_back = False
        
    def set_go_back(self):
        self.go_back = True
        
    def select_seg_helper(self):
        select_seg_helper(self.seg_path_input, self.file_exts)
        self.select_seg_error_msg.hide()
        
    def back_from_load(self):
        self.full_screen_layout.removeItem(self.seg_loading_layout)
        self.hide_seg_loading_layout()
        self.full_screen_layout.addItem(self.select_type_layout)
        self.show_type_selection_layout()
        self.full_screen_layout.setStretchFactor(self.side_bar_layout, 1)
        self.full_screen_layout.setStretchFactor(self.select_type_layout, 10)
        self.back_button.clicked.disconnect()
        self.back_button.clicked.connect(self.set_go_back)
        
    def show_draw_seg_start(self):
        self.physical_depth_label.show(); self.physical_depth_val.show()
        self.physical_width_label.show(); self.physical_width_val.show()
        self.physical_dims_label.show(); self.pixel_depth_label.show()
        self.pixel_depth_val.show(); self.pixel_width_label.show()
        self.pixel_width_val.show(); self.pixel_dims_label.show()
        self.draw_freehand_button.show(); self.user_draw_rectangle_button.show()
        self.im_display_frame.show(); self.draw_options_buttons.show()
        self.construct_roi_label.show()
        
    def move_to_draw(self):
        self.full_screen_layout.removeItem(self.seg_loading_layout)
        self.hide_seg_loading_layout()
        self.full_screen_layout.addItem(self.draw_roi_layout)
        self.show_draw_seg_start()
        self.full_screen_layout.setStretchFactor(self.side_bar_layout, 1)
        self.full_screen_layout.setStretchFactor(self.draw_roi_layout, 10)
        
        # self.back_button.clicked.disconnect()
        # self.back_button.clicked.connect(self.back_from_load)
        
    def accept_seg_type(self):
        self.seg_type = list(self.seg_loaders.keys())[self.seg_type_dropdown.currentIndex()-1]
        self.back_button.clicked.disconnect()
        self.back_button.clicked.connect(self.back_from_load)
        
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
            raise NotImplementedError("Haven't implemented custom segmentation drawing yet!")
        
        # self.hideRectButtons()
        # self.hideFreehandedButtons()
        # self.hideLoadedRoiButtons()
        # self.acceptLoadedRoiButton.clicked.connect(self.acceptROI)
        # self.acceptRectangleButton.clicked.connect(self.acceptRect)
        # self.undoLoadedRoiButton.clicked.connect(self.undoRoiLoad)

        # self.ultrasoundImage = UltrasoundImage()

        # # Prepare B-Mode display plot
        # self.horizontalLayout = QHBoxLayout(self.imDisplayFrame)
        # self.horizontalLayout.setObjectName("horizontalLayout")
        # self.figure = plt.figure()
        # self.canvas = FigureCanvas(self.figure)
        # self.ax = self.figure.add_subplot(111)
        # self.horizontalLayout.addWidget(self.canvas)

        # self.editImageDisplayGUI = EditImageDisplayGUI()
        # self.editImageDisplayGUI.contrastVal.valueChanged.connect(self.changeContrast)
        # self.editImageDisplayGUI.brightnessVal.valueChanged.connect(
        #     self.changeBrightness
        # )
        # self.editImageDisplayGUI.sharpnessVal.valueChanged.connect(self.changeSharpness)

        # self.analysisParamsGUI = AnalysisParamsGUI()

        # self.scatteredPoints = []
        # self.utcData: UtcData
        # self.lastGui: SelectImageSection.SelectImageGUI_UtcTool2dIQ

        # # self.crosshairCursor = Cursor(
        # #     self.ax, color="gold", linewidth=0.4, useblit=True
        # # )
        # self.selector = RectangleSelector(
        #     self.ax,
        #     self.drawRect,
        #     useblit=True,
        #     props=dict(linestyle="-", color="cyan", fill=False),
        # )
        # self.selector.set_active(False)
        # self.cid: int

        # self.editImageDisplayButton.clicked.connect(self.openImageEditor)
        # self.drawRoiButton.clicked.connect(self.recordDrawRoiClicked)
        # self.userDrawRectangleButton.clicked.connect(self.recordDrawRectClicked)
        # self.undoLastPtButton.clicked.connect(self.undoLastPt)
        # self.closeRoiButton.clicked.connect(self.closeInterpolation)
        # self.redrawRoiButton.clicked.connect(self.undoLastRoi)
        # self.acceptRoiButton.clicked.connect(self.acceptROI)
        # self.backButton.clicked.connect(self.backToWelcomeScreen)
        # self.drawFreehandButton.clicked.connect(self.drawFreehandRoi)
        # self.drawRectangleButton.clicked.connect(self.startDrawRectRoi)
        # self.loadRoiButton.clicked.connect(self.openLoadRoiWindow)
        # self.backFromFreehandButton.clicked.connect(self.backFromFreehand)
        # self.backFromRectangleButton.clicked.connect(self.backFromRect)
        # self.saveRoiButton.clicked.connect(self.saveRoi)
    
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
        self.accept_rectangle_buttons.hide(); self.back_from_freehand_button.hide()
        self.back_from_rectangle_button.hide(); self.close_roi_button.hide()
        self.draw_freehand_button.hide(); self.draw_rectangle_button.hide()
        self.redraw_roi_button.hide(); self.undo_last_pt_button.hide()
        self.user_draw_rectangle_button.hide(); self.draw_freehand_buttons.hide()
        self.im_display_frame.hide(); self.accept_roi_buttons.hide()
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
        self.physical_rect_width_val.show(); self.draw_roi_button.show()
        self.accept_rectangle_buttons.show(); self.back_from_freehand_button.show()
        self.back_from_rectangle_button.show(); self.close_roi_button.show()
        self.draw_freehand_button.show(); self.draw_rectangle_button.show()
        self.redraw_roi_button.show(); self.undo_last_pt_button.show()
        self.user_draw_rectangle_button.show(); self.draw_freehand_buttons.hide()
        self.im_display_frame.show(); self.accept_roi_buttons.show()
        self.draw_options_buttons.hide(); self.draw_rect_buttons.hide()
        
    # def hideInitialButtons(self):
    #     self.drawFreehandButton.hide()
    #     self.drawRectangleButton.hide()
    #     self.loadRoiButton.hide()
        
    # def showInitialButtons(self):
    #     self.drawFreehandButton.show()
    #     self.drawRectangleButton.show()
    #     self.loadRoiButton.show()
        
    # def hideFreehandedButtons(self):
    #     self.undoLastPtButton.hide()
    #     self.closeRoiButton.hide()
    #     self.acceptRoiButton.hide()
    #     self.backFromFreehandButton.hide()
    #     self.drawRoiButton.hide()
    #     self.redrawRoiButton.hide()
        
    # def showFreehandedButtons(self):
    #     self.undoLastPtButton.show()
    #     self.closeRoiButton.show()
    #     self.acceptRoiButton.show()
    #     self.backFromFreehandButton.show()
    #     self.drawRoiButton.show()
        
    # def hideRectButtons(self):
    #     self.userDrawRectangleButton.hide()
    #     self.backFromRectangleButton.hide()
    #     self.acceptRectangleButton.hide()
    #     self.physicalRectDimsLabel.hide()
    #     self.physicalRectHeightLabel.hide()
    #     self.physicalRectWidthLabel.hide()
    #     self.physicalRectHeightVal.hide()
    #     self.physicalRectWidthVal.hide()
        
    # def showRectButtons(self):
    #     self.userDrawRectangleButton.show()
    #     self.backFromRectangleButton.show()
    #     self.acceptRectangleButton.show()
    #     self.physicalRectDimsLabel.show()
    #     self.physicalRectHeightLabel.show()
    #     self.physicalRectWidthLabel.show()
    #     self.physicalRectHeightVal.show()
    #     self.physicalRectWidthVal.show()
        
    # def hideLoadedRoiButtons(self):
    #     self.undoLoadedRoiButton.hide()
    #     self.acceptLoadedRoiButton.hide()
        
    # def showLoadedRoiButtons(self):
    #     self.undoLoadedRoiButton.show()
    #     self.acceptLoadedRoiButton.show()

    # def saveRoi(self):
    #     del self.saveRoiGUI
    #     self.saveRoiGUI = SaveRoiGUI()
    #     self.acceptRect(moveOn=False)
    #     self.saveRoiGUI.splineX = self.utcData.splineX
    #     self.saveRoiGUI.splineY = self.utcData.splineY
    #     self.saveRoiGUI.frame = self.frame
    #     self.saveRoiGUI.imName = self.imagePathInput.text()
    #     self.saveRoiGUI.phantomName = self.phantomPathInput.text()
    #     self.saveRoiGUI.show()

    # def undoRoiLoad(self):
    #     self.undoLastRoi(); self.closeRoiButton.hide()
    #     self.hideLoadedRoiButtons()
    #     self.showInitialButtons()
    #     self.utcData.rectCoords = []
        
    # def openLoadRoiWindow(self):
    #     self.loadRoiGUI.chooseRoiGUI = self
    #     self.loadRoiGUI.show()

    # def backFromFreehand(self):
    #     self.undoLastRoi()
    #     self.hideFreehandedButtons()
    #     self.showInitialButtons()
    #     self.drawRoiButton.setChecked(False)
    #     self.recordDrawRoiClicked()

    # def backFromRect(self):
    #     self.physicalRectHeightVal.setText("0")
    #     self.physicalRectWidthVal.setText("0")
    #     self.userDrawRectangleButton.setChecked(False)
    #     self.undoLastRoi(); self.closeRoiButton.hide()
    #     self.hideRectButtons()
    #     self.showInitialButtons()
    #     self.utcData.rectCoords = []
    #     self.selector.set_active(False)
    #     if len(self.ax.patches) > 0:
    #         self.ax.patches.pop()
    #     self.canvas.draw()

    # def drawFreehandRoi(self):
    #     self.hideInitialButtons()
    #     self.showFreehandedButtons()

    # def startDrawRectRoi(self):
    #     self.hideInitialButtons()
    #     self.showRectButtons()

    # def backToWelcomeScreen(self):
    #     self.lastGui.show()
    #     self.lastGui.resize(self.size())
    #     self.hide()

    # def changeContrast(self):
    #     self.editImageDisplayGUI.contrastValDisplay.setValue(
    #         int(self.editImageDisplayGUI.contrastVal.value() * 10)
    #     )
    #     self.updateBModeSettings()

    # def changeBrightness(self):
    #     self.editImageDisplayGUI.brightnessValDisplay.setValue(
    #         int(self.editImageDisplayGUI.brightnessVal.value() * 10)
    #     )
    #     self.updateBModeSettings()

    # def changeSharpness(self):
    #     self.editImageDisplayGUI.sharpnessValDisplay.setValue(
    #         int(self.editImageDisplayGUI.sharpnessVal.value() * 10)
    #     )
    #     self.updateBModeSettings()

    # def openImageEditor(self):
    #     if self.editImageDisplayGUI.isVisible():
    #         self.editImageDisplayGUI.hide()
    #     else:
    #         self.editImageDisplayGUI.show()

    # def setFilenameDisplays(self, imageName, phantomName):
    #     self.imagePathInput.show()
    #     self.phantomPathInput.show()
    #     self.imagePathInput.setText(imageName)
    #     self.phantomPathInput.setText(phantomName)

    # def plotOnCanvas(self):  # Plot current image on GUI
    #     self.ax.clear()
    #     quotient = self.utcData.depth / self.utcData.width
    #     self.ax.imshow(self.utcData.finalBmode, aspect=quotient*(self.utcData.finalBmode.shape[1]/self.utcData.finalBmode.shape[0]))
    #     self.figure.set_facecolor((0, 0, 0, 0)) #type: ignore
    #     self.ax.axis("off")

    #     try:
    #         if self.utcData.numSamplesDrOut == 1400:
    #             # Preset 1 boundaries for 20220831121844_IQ.bin
    #             self.ax.plot([148.76, 154.22], [0, 500], c="purple")  # left boundary
    #             self.ax.plot([0, 716], [358.38, 386.78], c="purple")  # bottom boundary
    #             self.ax.plot([572.47, 509.967], [0, 500], c="purple")  # right boundary

    #         elif self.utcData.numSamplesDrOut == 1496:
    #             # Preset 2 boundaries for 20220831121752_IQ.bin
    #             self.ax.plot([146.9, 120.79], [0, 500], c="purple")  # left boundary
    #             self.ax.plot([0, 644.76], [462.41, 500], c="purple")  # bottom boundary
    #             self.ax.plot([614.48, 595.84], [0, 500], c="purple")  # right boundary

    #         # elif self.ImDisplayInfo.numSamplesDrOut != -1:
    #         #     print("No preset found!")
    #     except (AttributeError, UnboundLocalError):
    #         pass

    #     if hasattr(self.utcData, 'splineX') and len(self.utcData.splineX):
    #         self.spline = self.ax.plot(self.utcData.splineX, self.utcData.splineY, 
    #                                    color="cyan", zorder=1, linewidth=0.75)
    #     elif len(self.pointsPlottedX) > 0:
    #         self.scatteredPoints.append(
    #             self.ax.scatter(
    #                 self.pointsPlottedX[-1],
    #                 self.pointsPlottedY[-1],
    #                 marker="o", #type: ignore
    #                 s=0.5,
    #                 c="red",
    #                 zorder=500,
    #             )
    #         )
    #         if len(self.pointsPlottedX) > 1:
    #             xSpline, ySpline = calculateSpline(
    #                 np.array(self.pointsPlottedX) / self.utcData.pixWidth, np.array(self.pointsPlottedY) / self.utcData.pixDepth
    #             )
    #             xSpline *= self.utcData.pixWidth
    #             ySpline *= self.utcData.pixDepth
    #             xSpline = np.clip(xSpline, a_min=0, a_max=self.utcData.pixWidth-1)
    #             ySpline = np.clip(ySpline, a_min=0, a_max=self.utcData.pixDepth-1)
    #             self.spline = self.ax.plot(
    #                 xSpline, ySpline, color="cyan", zorder=1, linewidth=0.75
    #             )

    #     self.figure.subplots_adjust(
    #         left=0, right=1, bottom=0, top=1, hspace=0.2, wspace=0.2
    #     )
    #     # self.crosshairCursor.set_active(False)
    #     plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    #     self.canvas.draw()  # Refresh canvas

    # def openImageVerasonics(
    #     self, imageFilePath, phantomFilePath
    # ):  # Open initial image given data and phantom files previously inputted
    #     raise NotImplementedError("Not updated with refactor")
    #     # tmpLocation = imageFilePath.split("/")
    #     # dataFileName = tmpLocation[-1]
    #     # dataFileLocation = imageFilePath[: len(imageFilePath) - len(dataFileName)]
    #     # tmpPhantLocation = phantomFilePath.split("/")
    #     # phantFileName = tmpPhantLocation[-1]
    #     # phantFileLocation = phantomFilePath[: len(phantomFilePath) - len(phantFileName)]

    #     # (imArray, imgDataStruct, imgInfoStruct, refDataStruct, refInfoStruct,) = vera.getImage(
    #     #     dataFileName, dataFileLocation, phantFileName, phantFileLocation
    #     # )
    #     # self.AnalysisInfo.verasonics = True

    #     # from src.Utils.roiFuncs import computeSpecWindowsIQ, computeSpecWindowsRF
    #     # self.AnalysisInfo.computeSpecWindows = computeSpecWindowsIQ

    #     # self.processImage(
    #     #     imArray, imgDataStruct, refDataStruct, imgInfoStruct, refInfoStruct
    #     # )

    #     # self.editImageDisplayGUI.brightnessVal.setValue(1)
    #     # self.editImageDisplayGUI.sharpnessVal.setValue(1)

    # def openImageCanon(
    #     self, imageFilePath, phantomFilePath
    # ):
    #     # Open both images and record relevant data
    #     imgDataStruct, imgInfoStruct, refDataStruct, refInfoStruct = canonIqParser(imageFilePath, phantomFilePath)

    #     scConfig = ScConfig()
    #     scConfig.width = imgInfoStruct.width1
    #     scConfig.tilt = imgInfoStruct.tilt1
    #     scConfig.startDepth = imgInfoStruct.startDepth1
    #     scConfig.endDepth = imgInfoStruct.endDepth1
    #     scConfig.numSamplesDrOut = imgInfoStruct.numSamplesDrOut
    #     self.utcData.scConfig = scConfig

    #     self.ultrasoundImage.bmode = imgDataStruct.bMode
    #     self.ultrasoundImage.scBmode = imgDataStruct.scBmodeStruct.scArr
    #     self.ultrasoundImage.xmap = imgDataStruct.scBmodeStruct.xmap
    #     self.ultrasoundImage.ymap = imgDataStruct.scBmodeStruct.ymap
    #     self.ultrasoundImage.axialResRf = imgInfoStruct.depth / imgDataStruct.rf.shape[0]
    #     self.ultrasoundImage.lateralResRf = self.ultrasoundImage.axialResRf * (
    #         imgDataStruct.rf.shape[0]/imgDataStruct.rf.shape[1]
    #     ) # placeholder

    #     self.processImage(
    #         imgDataStruct, refDataStruct, imgInfoStruct, refInfoStruct
    #     )

    #     self.editImageDisplayGUI.contrastVal.setValue(1)
    #     self.editImageDisplayGUI.brightnessVal.setValue(1.4)
    #     self.editImageDisplayGUI.sharpnessVal.setValue(3)

    # def openImageTerason(self, imageFilePath, phantomFilePath):
    #     imgDataStruct, imgInfoStruct, refDataStruct, refInfoStruct = terasonRfParser(
    #         imageFilePath, phantomFilePath
    #     )

    #     self.ultrasoundImage.bmode = imgDataStruct.bMode
    #     self.ultrasoundImage.axialResRf = imgInfoStruct.axialRes
    #     self.ultrasoundImage.lateralResRf = imgInfoStruct.lateralRes

    #     self.processImage(
    #         imgDataStruct, refDataStruct, imgInfoStruct, refInfoStruct
    #     )

    #     self.editImageDisplayGUI.contrastVal.setValue(1)
    #     self.editImageDisplayGUI.brightnessVal.setValue(1)
    #     self.editImageDisplayGUI.sharpnessVal.setValue(1)

    # def processImage(
    #     self, imgDataStruct, refDataStruct, imgInfoStruct, refInfoStruct
    # ):
    #     self.ultrasoundImage.rf = imgDataStruct.rf
    #     self.ultrasoundImage.phantomRf = refDataStruct.rf
    #     if self.ultrasoundImage.rf.shape != self.ultrasoundImage.phantomRf.shape:
    #         print("WARNING: RF and phantom RF are not the same size")
    #         print(f"\tRF shape: {self.ultrasoundImage.rf.shape}")
    #         print(f"\tPhantom RF shape: {self.ultrasoundImage.phantomRf.shape}")
        

    #     analysisConfig = AnalysisConfig()
    #     analysisConfig.analysisFreqBand = [imgInfoStruct.lowBandFreq, imgInfoStruct.upBandFreq]
    #     analysisConfig.transducerFreqBand = [imgInfoStruct.minFrequency, imgInfoStruct.maxFrequency]
    #     analysisConfig.samplingFrequency = imgInfoStruct.samplingFrequency
    #     analysisConfig.centerFrequency = imgInfoStruct.centerFrequency

    #     utcAnalysis = UtcAnalysis()
    #     utcAnalysis.ultrasoundImage = self.ultrasoundImage
    #     utcAnalysis.config = analysisConfig

    #     self.utcData.utcAnalysis = utcAnalysis
    #     self.utcData.depth = imgInfoStruct.depth
    #     self.utcData.width = imgInfoStruct.width
        
    #     self.utcData.convertImagesToRGB()

    #     self.displayInitialImage()

    # def updateImageDisplay(self, cvIm):
    #     enhancer = ImageEnhance.Contrast(cvIm)
    #     imOutput = enhancer.enhance(self.editImageDisplayGUI.contrastVal.value())
    #     bright = ImageEnhance.Brightness(imOutput)
    #     imOutput = bright.enhance(self.editImageDisplayGUI.brightnessVal.value())
    #     sharp = ImageEnhance.Sharpness(imOutput)
    #     imOutput = sharp.enhance(self.editImageDisplayGUI.sharpnessVal.value())
    #     return np.array(imOutput)


    # def displayInitialImage(self):
    #     flippedIm = np.flipud(self.utcData.finalBmode).astype(np.uint8)

    #     qIm = QImage(
    #         flippedIm.data,
    #         flippedIm.shape[1],
    #         flippedIm.shape[0],
    #         flippedIm.strides[0],
    #         QImage.Format.Format_RGB888,
    #     )

    #     qIm.mirrored().save(
    #         os.path.join("Junk", "bModeImRaw.png")
    #     )  # Save as .png file

    #     if hasattr(self.utcData, 'scConfig'):
    #         flippedIm = np.flipud(self.utcData.bmode).astype(np.uint8)

    #         qIm = QImage(
    #             flippedIm.data,
    #             flippedIm.shape[1],
    #             flippedIm.shape[0],
    #             flippedIm.strides[0],
    #             QImage.Format.Format_RGB888,
    #         )

    #         qIm.mirrored().save(
    #             os.path.join("Junk", "bModeImRawPreSc.png")
    #         )  # Save as .png file

    #     self.utcData.utcAnalysis.initAnalysisConfig()

    #     self.physicalDepthVal.setText(
    #         str(np.round(self.utcData.depth, decimals=2))
    #     )
    #     self.physicalWidthVal.setText(
    #         str(np.round(self.utcData.width, decimals=2))
    #     )
    #     self.pixelWidthVal.setText(str(self.utcData.finalBmode.shape[1]))
    #     self.pixelDepthVal.setText(str(self.utcData.finalBmode.shape[0]))
    #     self.plotOnCanvas()

    # def recordDrawRoiClicked(self):
    #     if self.drawRoiButton.isChecked():  # Set up b-mode to be drawn on
    #         self.cid = self.figure.canvas.mpl_connect(
    #             "button_press_event", self.interpolatePoints
    #         )
    #         # self.crosshairCursor.set_active(True)
    #     else:  # No longer let b-mode be drawn on
    #         if hasattr(self, "cid"):
    #             self.cid = self.figure.canvas.mpl_disconnect(self.cid)
    #         # self.crosshairCursor.set_active(False)
    #     self.canvas.draw()

    # def recordDrawRectClicked(self):
    #     if self.userDrawRectangleButton.isChecked():  # Set up b-mode to be drawn on
    #         self.selector.set_active(True)
    #         self.cid = self.figure.canvas.mpl_connect(
    #             "button_press_event", self.clearRect
    #         )
    #     else:  # No longer let b-mode be drawn on
    #         self.cid = self.figure.canvas.mpl_disconnect(self.cid)
    #         self.selector.set_active(False)
    #     self.canvas.draw()

    # def undoLastPt(self):  # When drawing ROI, undo last point plotted
    #     if len(self.pointsPlottedX) > 0 and self.drawRoiButton.isCheckable():
    #         scatteredPoint = self.scatteredPoints.pop()
    #         scatteredPoint.remove()
    #         self.pointsPlottedX.pop()
    #         self.pointsPlottedY.pop()
    #         if len(self.pointsPlottedX) > 0:
    #             oldSpline = self.spline.pop(0)
    #             oldSpline.remove()
    #             if len(self.pointsPlottedX) > 1:
    #                 xSpline, ySpline = calculateSpline(
    #                     np.array(self.pointsPlottedX) / self.utcData.pixWidth, np.array(self.pointsPlottedY) / self.utcData.pixDepth
    #                 )
    #                 xSpline *= self.utcData.pixWidth
    #                 ySpline *= self.utcData.pixDepth
    #                 xSpline = np.clip(xSpline, a_min=0, a_max=self.utcData.pixWidth-1)
    #                 ySpline = np.clip(ySpline, a_min=0, a_max=self.utcData.pixDepth-1)
    #                 self.utcData.splineX = xSpline
    #                 self.utcData.splineY = ySpline
    #                 self.spline = self.ax.plot(
    #                     self.utcData.splineX,
    #                     self.utcData.splineY,
    #                     color="cyan",
    #                     linewidth=0.75,
    #                 )
    #         self.canvas.draw()
    #         self.drawRoiButton.setChecked(True)
    #         self.recordDrawRoiClicked()

    # def closeInterpolation(self):  # Finish drawing ROI
    #     if len(self.pointsPlottedX) > 2:
    #         self.ax.clear()
    #         quotient = self.utcData.depth / self.utcData.width
    #         self.ax.imshow(self.utcData.finalBmode, aspect=quotient*(self.utcData.finalBmode.shape[1]/self.utcData.finalBmode.shape[0]))
    #         if self.pointsPlottedX[0] != self.pointsPlottedX[-1] and self.pointsPlottedY[0] != self.pointsPlottedY[-1]:
    #             self.pointsPlottedX.append(self.pointsPlottedX[0])
    #             self.pointsPlottedY.append(self.pointsPlottedY[0])
    #         xSpline, ySpline = calculateSpline(
    #             np.array(self.pointsPlottedX) / self.utcData.pixWidth, np.array(self.pointsPlottedY) / self.utcData.pixDepth
    #         )
    #         xSpline *= self.utcData.pixWidth
    #         ySpline *= self.utcData.pixDepth
    #         xSpline = np.clip(xSpline, a_min=0, a_max=self.utcData.pixWidth-1)
    #         ySpline = np.clip(ySpline, a_min=0, a_max=self.utcData.pixDepth-1)
    #         self.utcData.splineX = xSpline
    #         self.utcData.splineY = ySpline
            
    #         self.utcData.splineX = np.clip(self.utcData.splineX, a_min=0, a_max=self.utcData.pixWidth-1)
    #         self.utcData.splineY = np.clip(self.utcData.splineY, a_min=0, a_max=self.utcData.pixDepth-1)

    #         try:
    #             if self.utcData.numSamplesDrOut == 1400:
    #                 self.utcData.splineX = np.clip(self.utcData.splineX, a_min=148, a_max=573)
    #                 self.utcData.splineY = np.clip(self.utcData.splineY, a_min=0.5, a_max=387)
    #             elif self.utcData.numSamplesDrOut == 1496:
    #                 self.utcData.splineX = np.clip(self.utcData.splineX, a_min=120, a_max=615)
    #                 self.utcData.splineY = np.clip(self.utcData.splineY, a_min=0.5, a_max=645)
    #             # elif self.ImDisplayInfo.numSamplesDrOut != -1:
    #             #     print("Preset not found!")
    #             #     return
    #         except (AttributeError, UnboundLocalError):
    #             pass

    #         self.drawRoiButton.setChecked(False)
    #         self.drawRoiButton.setCheckable(False)
    #         self.redrawRoiButton.setHidden(False)
    #         self.closeRoiButton.setHidden(True)
    #         self.cid = self.figure.canvas.mpl_disconnect(self.cid)
    #         # self.crosshairCursor.set_active(False)
    #         self.plotOnCanvas()

    # def undoLastRoi(
    #     self,
    # ):  # Remove previously drawn roi and prepare user to draw a new one
    #     self.utcData.splineX = np.array([])
    #     self.utcData.splineY = np.array([])
    #     self.pointsPlottedX = []
    #     self.pointsPlottedY = []
    #     self.drawRoiButton.setChecked(False)
    #     self.drawRoiButton.setCheckable(True)
    #     self.closeRoiButton.setHidden(False)
    #     self.redrawRoiButton.setHidden(True)
    #     self.plotOnCanvas()

    # def updateBModeSettings(
    #     self,
    # ):  # Updates background photo when image settings are modified
    #     cvIm = Image.open(os.path.join("Junk", "bModeImRaw.png"))
    #     self.utcData.finalBmode = self.updateImageDisplay(cvIm)

    #     if hasattr(self.utcData, 'scConfig'):
    #         cvIm = Image.open(os.path.join("Junk", "bModeImRawPreSc.png"))
    #         self.utcData.bmode = self.updateImageDisplay(cvIm)
        
    #     self.plotOnCanvas()

    # def clearRect(self, event):
    #     if len(self.ax.patches) > 0:
    #         rect = self.ax.patches[0]
    #         rect.remove()
    #         self.canvas.draw()

    # def interpolatePoints(
    #     self, event
    # ):  # Update ROI being drawn using spline using 2D interpolation
    #     try:
    #         if self.utcData.numSamplesDrOut == 1400:
    #             # Preset 1 boundaries for 20220831121844_IQ.bin
    #             leftSlope = (500 - 0) / (154.22 - 148.76)
    #             pointSlopeLeft = (event.ydata - 0) / (event.xdata - 148.76)
    #             if pointSlopeLeft <= 0 or leftSlope < pointSlopeLeft:
    #                 return

    #             bottomSlope = (386.78 - 358.38) / (716 - 0)
    #             pointSlopeBottom = (event.ydata - 358.38) / (event.xdata - 0)
    #             rightSlope = (500 - 0) / (509.967 - 572.47)
    #             pointSlopeRight = (event.ydata - 0) / (event.xdata - 572.47)

    #         elif self.utcData.numSamplesDrOut == 1496:
    #             # Preset 2 boundaries for 20220831121752_IQ.bin
    #             leftSlope = (500 - 0) / (120.79 - 146.9)
    #             pointSlopeLeft = (event.ydata - 0) / (event.xdata - 146.9)
    #             if pointSlopeLeft > leftSlope and pointSlopeLeft <= 0:
    #                 return

    #             bottomSlope = (500 - 462.41) / (644.76 - 0)
    #             pointSlopeBottom = (event.ydata - 462.41) / (event.xdata - 0)
    #             rightSlope = (500 - 0) / (595.84 - 614.48)
    #             pointSlopeRight = (event.ydata - 0) / (event.xdata - 614.48)

    #         # elif self.ImDisplayInfo.numSamplesDrOut != -1:
    #         #     print("Preset not found!")
    #         #     return

    #         if pointSlopeBottom > bottomSlope: # type: ignore
    #             return
    #         if pointSlopeRight >= 0 or pointSlopeRight < rightSlope: # type: ignore
    #             return
    #     except (AttributeError, UnboundLocalError):
    #         pass

    #     if len(self.pointsPlottedX) > 0 and self.pointsPlottedX[-1] == int(event.xdata) and self.pointsPlottedY[-1] == int(event.ydata):
    #         return

    #     self.pointsPlottedX.append(int(event.xdata))
    #     self.pointsPlottedY.append(int(event.ydata))
    #     plottedPoints = len(self.pointsPlottedX)

    #     if plottedPoints > 1:
    #         if plottedPoints > 2:
    #             oldSpline = self.spline.pop(0)
    #             oldSpline.remove()

    #         xSpline, ySpline = calculateSpline(
    #             np.array(self.pointsPlottedX) / self.utcData.pixWidth, np.array(self.pointsPlottedY) / self.utcData.pixDepth
    #         )
    #         xSpline *= self.utcData.pixWidth
    #         ySpline *= self.utcData.pixDepth
    #         xSpline = np.clip(xSpline, a_min=0, a_max=self.utcData.pixWidth-1)
    #         ySpline = np.clip(ySpline, a_min=0, a_max=self.utcData.pixDepth-1)
    #         self.spline = self.ax.plot(
    #             xSpline, ySpline, color="cyan", zorder=1, linewidth=0.75
    #         )
    #         self.figure.subplots_adjust(
    #             left=0, right=1, bottom=0, top=1, hspace=0.2, wspace=0.2
    #         )
    #         self.ax.tick_params(bottom=False, left=False)
    #     self.scatteredPoints.append(
    #         self.ax.scatter(
    #             self.pointsPlottedX[-1],
    #             self.pointsPlottedY[-1],
    #             marker="o", # type: ignore
    #             s=0.5,
    #             c="red",
    #             zorder=500,
    #         )
    #     )
    #     self.canvas.draw()

    # def drawRect(self, event1, event2):
    #     try:
    #         if self.utcData.numSamplesDrOut == 1400:
    #             # Preset 1 boundaries for 20220831121844_IQ.bin
    #             leftSlope = (500 - 0) / (154.22 - 148.76)
    #             pointSlopeLeft = (event1.ydata - 0) / (event1.xdata - 148.76)
    #             if pointSlopeLeft <= 0 or leftSlope < pointSlopeLeft:
    #                 return
    #             pointSlopeLeft = (event2.ydata - 0) / (event2.xdata - 148.76)
    #             if pointSlopeLeft <= 0 or leftSlope < pointSlopeLeft:
    #                 return

    #             bottomSlope = (386.78 - 358.38) / (716 - 0)
    #             pointSlopeBottom = (event1.ydata - 358.38) / (event1.xdata - 0)
    #             if pointSlopeBottom > bottomSlope:
    #                 return
    #             pointSlopeBottom = (event2.ydata - 358.38) / (event2.xdata - 0)
    #             if pointSlopeBottom > bottomSlope:
    #                 return
    #             rightSlope = (500 - 0) / (509.967 - 572.47)
    #             pointSlopeRight = (event1.ydata - 0) / (event1.xdata - 572.47)
    #             if pointSlopeRight >= 0 or pointSlopeRight < rightSlope:
    #                 return
    #             pointSlopeRight = (event2.ydata - 0) / (event2.xdata - 572.47)
    #             if pointSlopeRight >= 0 or pointSlopeRight < rightSlope:
    #                 return

    #         elif self.utcData.numSamplesDrOut == 1496:
    #             # Preset 2 boundaries for 20220831121752_IQ.bin
    #             leftSlope = (500 - 0) / (120.79 - 146.9)
    #             pointSlopeLeft = (event1.ydata - 0) / (event1.xdata - 146.9)
    #             if pointSlopeLeft > leftSlope and pointSlopeLeft <= 0:
    #                 return
    #             pointSlopeLeft = (event2.ydata - 0) / (event2.xdata - 146.9)
    #             if pointSlopeLeft > leftSlope and pointSlopeLeft <= 0:
    #                 return

    #             bottomSlope = (500 - 462.41) / (644.76 - 0)
    #             pointSlopeBottom = (event1.ydata - 462.41) / (event1.xdata - 0)
    #             if pointSlopeBottom > bottomSlope:
    #                 return
    #             pointSlopeBottom = (event2.ydata - 462.41) / (event2.xdata - 0)
    #             if pointSlopeBottom > bottomSlope:
    #                 return
    #             rightSlope = (500 - 0) / (595.84 - 614.48)
    #             pointSlopeRight = (event1.ydata - 0) / (event1.xdata - 614.48)
    #             if pointSlopeRight >= 0 or pointSlopeRight < rightSlope:
    #                 return
    #             pointSlopeRight = (event2.ydata - 0) / (event2.xdata - 614.48)
    #             if pointSlopeRight >= 0 or pointSlopeRight < rightSlope:
    #                 return

    #         # elif self.ImDisplayInfo.numSamplesDrOut != -1:
    #         #     print("Preset not found!")
    #         #     return

    #     except (AttributeError, UnboundLocalError):
    #         pass

    #     self.utcData.rectCoords = [
    #         int(event1.xdata),
    #         int(event1.ydata),
    #         int(event2.xdata),
    #         int(event2.ydata),
    #     ]
    #     self.plotPatch()

    # def plotPatch(self):
    #     if len(self.utcData.rectCoords) > 0:
    #         left, bottom, right, top = self.utcData.rectCoords
    #         rect = patches.Rectangle(
    #             (left, bottom),
    #             (right - left),
    #             (top - bottom),
    #             linewidth=1,
    #             edgecolor="cyan",
    #             facecolor="none",
    #         )
    #         if len(self.ax.patches) > 0:
    #             self.ax.patches.pop()

    #         self.ax.add_patch(rect)

    #         mplPixWidth = abs(right - left)
    #         imPixWidth = mplPixWidth * self.utcData.lateralRes
    #         mmWidth = self.utcData.lateralRes * imPixWidth  # (mm/pixel)*pixels
    #         self.physicalRectWidthVal.setText(str(np.round(mmWidth, decimals=2)))

    #         mplPixHeight = abs(top - bottom)
    #         imPixHeight = mplPixHeight * self.utcData.axialRes
    #         mmHeight = self.utcData.axialRes * imPixHeight  # (mm/pixel)*pixels
    #         self.physicalRectHeightVal.setText(str(np.round(mmHeight, decimals=2)))

    #         self.figure.subplots_adjust(
    #             left=0, right=1, bottom=0, top=1, hspace=0.2, wspace=0.2
    #         )
    #         self.ax.tick_params(bottom=False, left=False)
    #         self.canvas.draw()

    # def acceptRect(self, moveOn=True):
    #     if len(self.ax.patches) == 1:
    #         left, bottom = self.ax.patches[0].get_xy()
    #         left = int(left)
    #         bottom = int(bottom)
    #         width = int(self.ax.patches[0].get_width())
    #         height = int(self.ax.patches[0].get_height())
    #         self.pointsPlottedX = (
    #             list(range(left, left + width))
    #             + list(np.ones(height).astype(int) * (left + width - 1))
    #             + list(range(left + width - 1, left - 1, -1))
    #             + list(np.ones(height).astype(int) * left)
    #         )
    #         self.pointsPlottedY = (
    #             list(np.ones(width).astype(int) * bottom)
    #             + list(range(bottom, bottom + height))
    #             + list(np.ones(width).astype(int) * (bottom + height - 1))
    #             + list(range(bottom + height - 1, bottom - 1, -1))
    #         )
    #         self.utcData.splineX = np.array(
    #             self.pointsPlottedX
    #         )  # Image boundaries already addressed at plotting phase
    #         self.utcData.splineY = np.array(
    #             self.pointsPlottedY
    #         )  # Image boundaries already addressed at plotting phase
    #         if moveOn:
    #             self.acceptROI()

    # def acceptROI(self):
    #     if len(self.utcData.splineX) > 1 and len(self.utcData.splineX) == len(self.utcData.splineY):
    #         self.analysisParamsGUI.utcData = self.utcData
    #         self.analysisParamsGUI.initParams()
    #         self.analysisParamsGUI.lastGui = self
    #         self.analysisParamsGUI.frame = self.frame
    #         self.analysisParamsGUI.setFilenameDisplays(
    #             self.imagePathInput.text(),
    #             self.phantomPathInput.text(),
    #         )
    #         # self.analysisParamsGUI.plotRoiPreview()
    #         self.analysisParamsGUI.show()
    #         self.editImageDisplayGUI.hide()
    #         self.analysisParamsGUI.resize(self.size())
    #         self.hide()


# def calculateSpline(xpts, ypts):  # 2D spline interpolation
#     cv = []
#     for i in range(len(xpts)):
#         cv.append([xpts[i], ypts[i]])
#     cv = np.array(cv)
#     if len(xpts) == 2:
#         tck, _ = interpolate.splprep(cv.T, s=0.0, k=1)
#     elif len(xpts) == 3:
#         tck, _ = interpolate.splprep(cv.T, s=0.0, k=2)
#     else:
#         tck, _ = interpolate.splprep(cv.T, s=0.0, k=3)
#     x, y = np.array(interpolate.splev(np.linspace(0, 1, 1000), tck))
#     return x, y
