# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'roi_selection.ui'
##
## Created by: Qt User Interface Compiler version 6.9.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLayout, QLineEdit,
    QPushButton, QSizePolicy, QSpacerItem, QTextEdit,
    QVBoxLayout, QWidget)

class Ui_constructRoi(object):
    def setupUi(self, constructRoi):
        if not constructRoi.objectName():
            constructRoi.setObjectName(u"constructRoi")
        constructRoi.resize(1512, 829)
        constructRoi.setMinimumSize(QSize(0, 0))
        constructRoi.setStyleSheet(u"QWidget {\n"
"	background: rgb(42, 42, 42);\n"
"}")
        self.horizontalLayoutWidget_4 = QWidget(constructRoi)
        self.horizontalLayoutWidget_4.setObjectName(u"horizontalLayoutWidget_4")
        self.horizontalLayoutWidget_4.setGeometry(QRect(-500, -110, 2182, 871))
        self.full_screen_layout = QHBoxLayout(self.horizontalLayoutWidget_4)
        self.full_screen_layout.setObjectName(u"full_screen_layout")
        self.full_screen_layout.setContentsMargins(0, 0, 0, 0)
        self.side_bar_layout = QVBoxLayout()
        self.side_bar_layout.setSpacing(0)
        self.side_bar_layout.setObjectName(u"side_bar_layout")
        self.side_bar_layout.setSizeConstraint(QLayout.SetMaximumSize)
        self.sidebar = QWidget(self.horizontalLayoutWidget_4)
        self.sidebar.setObjectName(u"sidebar")
        self.sidebar.setMinimumSize(QSize(341, 601))
        self.sidebar.setMaximumSize(QSize(241, 601))
        self.sidebar.setStyleSheet(u"QWidget {\n"
"	background-color: rgb(28, 0, 101);\n"
"}")
        self.imageSelectionSidebar = QFrame(self.sidebar)
        self.imageSelectionSidebar.setObjectName(u"imageSelectionSidebar")
        self.imageSelectionSidebar.setGeometry(QRect(0, 0, 341, 121))
        self.imageSelectionSidebar.setMinimumSize(QSize(341, 121))
        self.imageSelectionSidebar.setMaximumSize(QSize(341, 121))
        self.imageSelectionSidebar.setStyleSheet(u"QFrame {\n"
"	background-color: rgb(99, 0, 174);\n"
"	border: 1px solid black;\n"
"}")
        self.imageSelectionSidebar.setFrameShape(QFrame.StyledPanel)
        self.imageSelectionSidebar.setFrameShadow(QFrame.Raised)
        self.imageSelectionLabelSidebar = QLabel(self.imageSelectionSidebar)
        self.imageSelectionLabelSidebar.setObjectName(u"imageSelectionLabelSidebar")
        self.imageSelectionLabelSidebar.setGeometry(QRect(70, 0, 191, 51))
        self.imageSelectionLabelSidebar.setStyleSheet(u"QLabel {\n"
"	font-size: 21px;\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"	border: 0px;\n"
"	font-weight: bold;\n"
"}")
        self.imageSelectionLabelSidebar.setAlignment(Qt.AlignCenter)
        self.imageLabel = QLabel(self.imageSelectionSidebar)
        self.imageLabel.setObjectName(u"imageLabel")
        self.imageLabel.setGeometry(QRect(-60, 40, 191, 51))
        self.imageLabel.setStyleSheet(u"QLabel {\n"
"	font-size: 16px;\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"	border: 0px;\n"
"	font-weight: bold;\n"
"}")
        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.phantomLabel = QLabel(self.imageSelectionSidebar)
        self.phantomLabel.setObjectName(u"phantomLabel")
        self.phantomLabel.setGeometry(QRect(-50, 70, 191, 51))
        self.phantomLabel.setStyleSheet(u"QLabel {\n"
"	font-size: 16px;\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"	border: 0px;\n"
"	font-weight: bold\n"
"}")
        self.phantomLabel.setAlignment(Qt.AlignCenter)
        self.image_path_input = QLabel(self.imageSelectionSidebar)
        self.image_path_input.setObjectName(u"image_path_input")
        self.image_path_input.setGeometry(QRect(100, 40, 241, 51))
        self.image_path_input.setStyleSheet(u"QLabel {\n"
"	font-size: 14px;\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"	border: 0px;\n"
"}")
        self.image_path_input.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.phantom_path_input = QLabel(self.imageSelectionSidebar)
        self.phantom_path_input.setObjectName(u"phantom_path_input")
        self.phantom_path_input.setGeometry(QRect(100, 70, 241, 51))
        self.phantom_path_input.setStyleSheet(u"QLabel {\n"
"	font-size: 14px;\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"	border: 0px;\n"
"}")
        self.phantom_path_input.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.roiSidebar = QFrame(self.sidebar)
        self.roiSidebar.setObjectName(u"roiSidebar")
        self.roiSidebar.setGeometry(QRect(0, 120, 341, 121))
        self.roiSidebar.setMaximumSize(QSize(341, 121))
        self.roiSidebar.setStyleSheet(u"QFrame {\n"
"	background-color: rgb(99, 0, 174);\n"
"	border: 1px solid black;\n"
"}")
        self.roiSidebar.setFrameShape(QFrame.StyledPanel)
        self.roiSidebar.setFrameShadow(QFrame.Raised)
        self.roiSidebarLabel = QLabel(self.roiSidebar)
        self.roiSidebarLabel.setObjectName(u"roiSidebarLabel")
        self.roiSidebarLabel.setGeometry(QRect(0, 40, 341, 51))
        self.roiSidebarLabel.setStyleSheet(u"QLabel {\n"
"	font-size: 21px;\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"	border: 0px;\n"
"	font-weight: bold;\n"
"}")
        self.roiSidebarLabel.setAlignment(Qt.AlignCenter)
        self.rfAnalysisSidebar = QFrame(self.sidebar)
        self.rfAnalysisSidebar.setObjectName(u"rfAnalysisSidebar")
        self.rfAnalysisSidebar.setGeometry(QRect(0, 360, 341, 121))
        self.rfAnalysisSidebar.setMinimumSize(QSize(341, 121))
        self.rfAnalysisSidebar.setMaximumSize(QSize(341, 121))
        self.rfAnalysisSidebar.setStyleSheet(u"QFrame {\n"
"	background-color:  rgb(49, 0, 124);\n"
"	border: 1px solid black;\n"
"}")
        self.rfAnalysisSidebar.setFrameShape(QFrame.StyledPanel)
        self.rfAnalysisSidebar.setFrameShadow(QFrame.Raised)
        self.rfAnalysisLabel = QLabel(self.rfAnalysisSidebar)
        self.rfAnalysisLabel.setObjectName(u"rfAnalysisLabel")
        self.rfAnalysisLabel.setGeometry(QRect(0, 30, 341, 51))
        self.rfAnalysisLabel.setStyleSheet(u"QLabel {\n"
"	font-size: 21px;\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"	border: 0px;\n"
"	font-weight: bold;\n"
"}")
        self.rfAnalysisLabel.setAlignment(Qt.AlignCenter)
        self.exportResultsSidebar = QFrame(self.sidebar)
        self.exportResultsSidebar.setObjectName(u"exportResultsSidebar")
        self.exportResultsSidebar.setGeometry(QRect(0, 480, 341, 121))
        self.exportResultsSidebar.setMinimumSize(QSize(341, 121))
        self.exportResultsSidebar.setMaximumSize(QSize(341, 121))
        self.exportResultsSidebar.setStyleSheet(u"QFrame {\n"
"	background-color:  rgb(49, 0, 124);\n"
"	border: 1px solid black;\n"
"}")
        self.exportResultsSidebar.setFrameShape(QFrame.StyledPanel)
        self.exportResultsSidebar.setFrameShadow(QFrame.Raised)
        self.exportResultsLabel = QLabel(self.exportResultsSidebar)
        self.exportResultsLabel.setObjectName(u"exportResultsLabel")
        self.exportResultsLabel.setGeometry(QRect(20, 30, 301, 51))
        self.exportResultsLabel.setStyleSheet(u"QLabel {\n"
"	font-size: 21px;\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"	border: 0px;\n"
"	font-weight: bold;\n"
"}")
        self.exportResultsLabel.setAlignment(Qt.AlignCenter)
        self.analysisParamsSidebar = QFrame(self.sidebar)
        self.analysisParamsSidebar.setObjectName(u"analysisParamsSidebar")
        self.analysisParamsSidebar.setGeometry(QRect(0, 240, 341, 121))
        self.analysisParamsSidebar.setMaximumSize(QSize(341, 121))
        self.analysisParamsSidebar.setStyleSheet(u"QFrame {\n"
"	background-color: rgb(49, 0, 124);\n"
"	border: 1px solid black;\n"
"}")
        self.analysisParamsSidebar.setFrameShape(QFrame.StyledPanel)
        self.analysisParamsSidebar.setFrameShadow(QFrame.Raised)
        self.analysisParamsLabel = QLabel(self.analysisParamsSidebar)
        self.analysisParamsLabel.setObjectName(u"analysisParamsLabel")
        self.analysisParamsLabel.setGeometry(QRect(0, 30, 341, 51))
        self.analysisParamsLabel.setStyleSheet(u"QLabel {\n"
"	font-size: 21px;\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"	border: 0px;\n"
"	font-weight:bold;\n"
"}")
        self.analysisParamsLabel.setAlignment(Qt.AlignCenter)

        self.side_bar_layout.addWidget(self.sidebar)

        self.gridFrame = QFrame(self.horizontalLayoutWidget_4)
        self.gridFrame.setObjectName(u"gridFrame")
        self.gridFrame.setMaximumSize(QSize(341, 16777215))
        self.gridFrame.setStyleSheet(u"QFrame {\n"
"	background-color: rgb(28, 0, 101);\n"
"}")
        self.backButtonGrid = QGridLayout(self.gridFrame)
        self.backButtonGrid.setObjectName(u"backButtonGrid")
        self.backButtonGrid.setSizeConstraint(QLayout.SetMinAndMaxSize)
        self.backButtonGrid.setContentsMargins(10, 10, 10, 10)
        self.save_roi_button = QPushButton(self.gridFrame)
        self.save_roi_button.setObjectName(u"save_roi_button")
        self.save_roi_button.setMinimumSize(QSize(131, 41))
        self.save_roi_button.setMaximumSize(QSize(131, 41))
        self.save_roi_button.setStyleSheet(u"QPushButton {\n"
"	color: white;\n"
"	font-size: 16px;\n"
"	background: rgb(90, 37, 255);\n"
"	border-radius: 15px;\n"
"}")

        self.backButtonGrid.addWidget(self.save_roi_button, 1, 2, 1, 1)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.backButtonGrid.addItem(self.verticalSpacer, 0, 0, 1, 1)

        self.back_button = QPushButton(self.gridFrame)
        self.back_button.setObjectName(u"back_button")
        self.back_button.setMinimumSize(QSize(131, 41))
        self.back_button.setMaximumSize(QSize(131, 41))
        self.back_button.setStyleSheet(u"QPushButton {\n"
"	color: white;\n"
"	font-size: 16px;\n"
"	background: rgb(90, 37, 255);\n"
"	border-radius: 15px;\n"
"}")

        self.backButtonGrid.addWidget(self.back_button, 1, 0, 1, 1)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.backButtonGrid.addItem(self.horizontalSpacer, 1, 1, 1, 1)


        self.side_bar_layout.addWidget(self.gridFrame)


        self.full_screen_layout.addLayout(self.side_bar_layout)

        self.draw_roi_layout = QVBoxLayout()
        self.draw_roi_layout.setSpacing(0)
        self.draw_roi_layout.setObjectName(u"draw_roi_layout")
        self.draw_roi_layout.setContentsMargins(30, 10, 30, 10)
        self.draw_roi_heading_layout = QVBoxLayout()
        self.draw_roi_heading_layout.setSpacing(5)
        self.draw_roi_heading_layout.setObjectName(u"draw_roi_heading_layout")
        self.draw_roi_title_layout = QHBoxLayout()
        self.draw_roi_title_layout.setObjectName(u"draw_roi_title_layout")
        self.pix_dim_layout_cm = QVBoxLayout()
        self.pix_dim_layout_cm.setObjectName(u"pix_dim_layout_cm")
        self.physical_dims_label = QLabel(self.horizontalLayoutWidget_4)
        self.physical_dims_label.setObjectName(u"physical_dims_label")
        self.physical_dims_label.setStyleSheet(u"QLabel {\n"
"	font-size: 18px;\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"}")
        self.physical_dims_label.setTextFormat(Qt.AutoText)
        self.physical_dims_label.setScaledContents(False)
        self.physical_dims_label.setAlignment(Qt.AlignCenter)
        self.physical_dims_label.setWordWrap(True)

        self.pix_dim_layout_cm.addWidget(self.physical_dims_label)

        self.pix_dim_grid_cm = QGridLayout()
        self.pix_dim_grid_cm.setObjectName(u"pix_dim_grid_cm")
        self.physical_depth_label = QLabel(self.horizontalLayoutWidget_4)
        self.physical_depth_label.setObjectName(u"physical_depth_label")
        self.physical_depth_label.setMinimumSize(QSize(129, 0))
        self.physical_depth_label.setStyleSheet(u"QLabel {\n"
"	font-size: 14px;\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"}")
        self.physical_depth_label.setTextFormat(Qt.AutoText)
        self.physical_depth_label.setScaledContents(False)
        self.physical_depth_label.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.physical_depth_label.setWordWrap(True)

        self.pix_dim_grid_cm.addWidget(self.physical_depth_label, 1, 0, 1, 1)

        self.physical_width_val = QLabel(self.horizontalLayoutWidget_4)
        self.physical_width_val.setObjectName(u"physical_width_val")
        self.physical_width_val.setStyleSheet(u"QLabel {\n"
"	font-size: 14px;\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"}")
        self.physical_width_val.setTextFormat(Qt.AutoText)
        self.physical_width_val.setScaledContents(False)
        self.physical_width_val.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.physical_width_val.setWordWrap(True)

        self.pix_dim_grid_cm.addWidget(self.physical_width_val, 0, 1, 1, 1)

        self.physical_depth_val = QLabel(self.horizontalLayoutWidget_4)
        self.physical_depth_val.setObjectName(u"physical_depth_val")
        self.physical_depth_val.setStyleSheet(u"QLabel {\n"
"	font-size: 14px;\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"}")
        self.physical_depth_val.setTextFormat(Qt.AutoText)
        self.physical_depth_val.setScaledContents(False)
        self.physical_depth_val.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.physical_depth_val.setWordWrap(True)

        self.pix_dim_grid_cm.addWidget(self.physical_depth_val, 1, 1, 1, 1)

        self.physical_width_label = QLabel(self.horizontalLayoutWidget_4)
        self.physical_width_label.setObjectName(u"physical_width_label")
        self.physical_width_label.setMinimumSize(QSize(129, 0))
        self.physical_width_label.setStyleSheet(u"QLabel {\n"
"	font-size: 14px;\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"}")
        self.physical_width_label.setTextFormat(Qt.AutoText)
        self.physical_width_label.setScaledContents(False)
        self.physical_width_label.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.physical_width_label.setWordWrap(True)

        self.pix_dim_grid_cm.addWidget(self.physical_width_label, 0, 0, 1, 1)


        self.pix_dim_layout_cm.addLayout(self.pix_dim_grid_cm)

        self.pix_dim_layout_cm.setStretch(0, 1)
        self.pix_dim_layout_cm.setStretch(1, 2)

        self.draw_roi_title_layout.addLayout(self.pix_dim_layout_cm)

        self.construct_roi_label = QLabel(self.horizontalLayoutWidget_4)
        self.construct_roi_label.setObjectName(u"construct_roi_label")
        self.construct_roi_label.setStyleSheet(u"QLabel {\n"
"	font-size: 29px;\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"}")
        self.construct_roi_label.setTextFormat(Qt.AutoText)
        self.construct_roi_label.setScaledContents(False)
        self.construct_roi_label.setAlignment(Qt.AlignCenter)
        self.construct_roi_label.setWordWrap(True)

        self.draw_roi_title_layout.addWidget(self.construct_roi_label)

        self.pix_dim_layout = QVBoxLayout()
        self.pix_dim_layout.setObjectName(u"pix_dim_layout")
        self.pixel_dims_label = QLabel(self.horizontalLayoutWidget_4)
        self.pixel_dims_label.setObjectName(u"pixel_dims_label")
        self.pixel_dims_label.setStyleSheet(u"QLabel {\n"
"	font-size: 18px;\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"}")
        self.pixel_dims_label.setTextFormat(Qt.AutoText)
        self.pixel_dims_label.setScaledContents(False)
        self.pixel_dims_label.setAlignment(Qt.AlignCenter)
        self.pixel_dims_label.setWordWrap(True)

        self.pix_dim_layout.addWidget(self.pixel_dims_label)

        self.pix_dim_grid = QGridLayout()
        self.pix_dim_grid.setObjectName(u"pix_dim_grid")
        self.pixel_width_label = QLabel(self.horizontalLayoutWidget_4)
        self.pixel_width_label.setObjectName(u"pixel_width_label")
        self.pixel_width_label.setStyleSheet(u"QLabel {\n"
"	font-size: 14px;\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"}")
        self.pixel_width_label.setTextFormat(Qt.AutoText)
        self.pixel_width_label.setScaledContents(False)
        self.pixel_width_label.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.pixel_width_label.setWordWrap(True)

        self.pix_dim_grid.addWidget(self.pixel_width_label, 0, 0, 1, 1)

        self.pixel_depth_label = QLabel(self.horizontalLayoutWidget_4)
        self.pixel_depth_label.setObjectName(u"pixel_depth_label")
        self.pixel_depth_label.setStyleSheet(u"QLabel {\n"
"	font-size: 14px;\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"}")
        self.pixel_depth_label.setTextFormat(Qt.AutoText)
        self.pixel_depth_label.setScaledContents(False)
        self.pixel_depth_label.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.pixel_depth_label.setWordWrap(True)

        self.pix_dim_grid.addWidget(self.pixel_depth_label, 1, 0, 1, 1)

        self.pixel_width_val = QLabel(self.horizontalLayoutWidget_4)
        self.pixel_width_val.setObjectName(u"pixel_width_val")
        self.pixel_width_val.setStyleSheet(u"QLabel {\n"
"	font-size: 14px;\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"}")
        self.pixel_width_val.setTextFormat(Qt.AutoText)
        self.pixel_width_val.setScaledContents(False)
        self.pixel_width_val.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.pixel_width_val.setWordWrap(True)

        self.pix_dim_grid.addWidget(self.pixel_width_val, 0, 1, 1, 1)

        self.pixel_depth_val = QLabel(self.horizontalLayoutWidget_4)
        self.pixel_depth_val.setObjectName(u"pixel_depth_val")
        self.pixel_depth_val.setStyleSheet(u"QLabel {\n"
"	font-size: 14px;\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"}")
        self.pixel_depth_val.setTextFormat(Qt.AutoText)
        self.pixel_depth_val.setScaledContents(False)
        self.pixel_depth_val.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.pixel_depth_val.setWordWrap(True)

        self.pix_dim_grid.addWidget(self.pixel_depth_val, 1, 1, 1, 1)


        self.pix_dim_layout.addLayout(self.pix_dim_grid)

        self.pix_dim_layout.setStretch(0, 1)
        self.pix_dim_layout.setStretch(1, 2)

        self.draw_roi_title_layout.addLayout(self.pix_dim_layout)

        self.draw_roi_title_layout.setStretch(0, 1)
        self.draw_roi_title_layout.setStretch(1, 2)
        self.draw_roi_title_layout.setStretch(2, 1)

        self.draw_roi_heading_layout.addLayout(self.draw_roi_title_layout)


        self.draw_roi_layout.addLayout(self.draw_roi_heading_layout)

        self.draw_roi_buttons = QFrame(self.horizontalLayoutWidget_4)
        self.draw_roi_buttons.setObjectName(u"draw_roi_buttons")
        self.draw_roi_buttons.setMaximumSize(QSize(804, 16777215))
        self.horizontalLayout_2 = QHBoxLayout(self.draw_roi_buttons)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.draw_freehand_button = QPushButton(self.draw_roi_buttons)
        self.draw_freehand_button.setObjectName(u"draw_freehand_button")
        self.draw_freehand_button.setMinimumSize(QSize(221, 41))
        self.draw_freehand_button.setMaximumSize(QSize(221, 41))
        self.draw_freehand_button.setStyleSheet(u"QPushButton {\n"
"	color: white;\n"
"	font-size: 16px;\n"
"	background: rgb(90, 37, 255);\n"
"	border-radius: 15px;\n"
"}\n"
"")
        self.draw_freehand_button.setCheckable(True)
        self.draw_freehand_button.setChecked(False)

        self.horizontalLayout_2.addWidget(self.draw_freehand_button)

        self.draw_roi_button = QPushButton(self.draw_roi_buttons)
        self.draw_roi_button.setObjectName(u"draw_roi_button")
        self.draw_roi_button.setMinimumSize(QSize(141, 41))
        self.draw_roi_button.setMaximumSize(QSize(141, 41))
        self.draw_roi_button.setStyleSheet(u"QPushButton {\n"
"	color: white;\n"
"	font-size: 16px;\n"
"	background: rgb(90, 37, 255);\n"
"	border-radius: 15px;\n"
"}\n"
"QPushButton:checked {\n"
"	color:white; \n"
"	font-size: 16px;\n"
"	background: rgb(45, 0, 110);\n"
"	border-radius: 15px;\n"
"}\n"
"")
        self.draw_roi_button.setCheckable(True)
        self.draw_roi_button.setChecked(False)

        self.horizontalLayout_2.addWidget(self.draw_roi_button)

        self.undo_last_pt_button = QPushButton(self.draw_roi_buttons)
        self.undo_last_pt_button.setObjectName(u"undo_last_pt_button")
        self.undo_last_pt_button.setMinimumSize(QSize(141, 41))
        self.undo_last_pt_button.setStyleSheet(u"QPushButton {\n"
"	color: white;\n"
"	font-size: 16px;\n"
"	background: rgb(90, 37, 255);\n"
"	border-radius: 15px;\n"
"}")
        self.undo_last_pt_button.setCheckable(False)

        self.horizontalLayout_2.addWidget(self.undo_last_pt_button)

        self.redraw_roi_button = QPushButton(self.draw_roi_buttons)
        self.redraw_roi_button.setObjectName(u"redraw_roi_button")
        self.redraw_roi_button.setMinimumSize(QSize(141, 41))
        self.redraw_roi_button.setMaximumSize(QSize(141, 41))
        self.redraw_roi_button.setStyleSheet(u"QPushButton {\n"
"	color: white;\n"
"	font-size: 16px;\n"
"	background: rgb(90, 37, 255);\n"
"	border-radius: 15px;\n"
"}")
        self.redraw_roi_button.setCheckable(False)

        self.horizontalLayout_2.addWidget(self.redraw_roi_button)

        self.close_roi_button = QPushButton(self.draw_roi_buttons)
        self.close_roi_button.setObjectName(u"close_roi_button")
        self.close_roi_button.setMinimumSize(QSize(141, 41))
        self.close_roi_button.setMaximumSize(QSize(141, 41))
        self.close_roi_button.setStyleSheet(u"QPushButton {\n"
"	color: white;\n"
"	font-size: 16px;\n"
"	background: rgb(90, 37, 255);\n"
"	border-radius: 15px;\n"
"}")
        self.close_roi_button.setCheckable(False)

        self.horizontalLayout_2.addWidget(self.close_roi_button)

        self.back_from_freehand_button = QPushButton(self.draw_roi_buttons)
        self.back_from_freehand_button.setObjectName(u"back_from_freehand_button")
        self.back_from_freehand_button.setMinimumSize(QSize(141, 41))
        self.back_from_freehand_button.setMaximumSize(QSize(141, 41))
        self.back_from_freehand_button.setStyleSheet(u"QPushButton {\n"
"	color: white;\n"
"	font-size: 16px;\n"
"	background: rgb(90, 37, 255);\n"
"	border-radius: 15px;\n"
"}")
        self.back_from_freehand_button.setCheckable(False)

        self.horizontalLayout_2.addWidget(self.back_from_freehand_button)

        self.draw_rectangle_button = QPushButton(self.draw_roi_buttons)
        self.draw_rectangle_button.setObjectName(u"draw_rectangle_button")
        self.draw_rectangle_button.setMinimumSize(QSize(221, 41))
        self.draw_rectangle_button.setMaximumSize(QSize(221, 41))
        self.draw_rectangle_button.setStyleSheet(u"QPushButton {\n"
"	color: white;\n"
"	font-size: 16px;\n"
"	background: rgb(90, 37, 255);\n"
"	border-radius: 15px;\n"
"}\n"
"")
        self.draw_rectangle_button.setCheckable(True)
        self.draw_rectangle_button.setChecked(False)

        self.horizontalLayout_2.addWidget(self.draw_rectangle_button)

        self.user_draw_rectangle_button = QPushButton(self.draw_roi_buttons)
        self.user_draw_rectangle_button.setObjectName(u"user_draw_rectangle_button")
        self.user_draw_rectangle_button.setMinimumSize(QSize(241, 41))
        self.user_draw_rectangle_button.setMaximumSize(QSize(241, 41))
        self.user_draw_rectangle_button.setStyleSheet(u"QPushButton {\n"
"	color: white;\n"
"	font-size: 16px;\n"
"	background: rgb(90, 37, 255);\n"
"	border-radius: 15px;\n"
"}\n"
"QPushButton:checked {\n"
"	color:white; \n"
"	font-size: 16px;\n"
"	background: rgb(45, 0, 110);\n"
"	border-radius: 15px;\n"
"}\n"
"")
        self.user_draw_rectangle_button.setCheckable(True)
        self.user_draw_rectangle_button.setChecked(False)

        self.horizontalLayout_2.addWidget(self.user_draw_rectangle_button)

        self.back_from_rectangle_button = QPushButton(self.draw_roi_buttons)
        self.back_from_rectangle_button.setObjectName(u"back_from_rectangle_button")
        self.back_from_rectangle_button.setMinimumSize(QSize(241, 41))
        self.back_from_rectangle_button.setMaximumSize(QSize(241, 41))
        self.back_from_rectangle_button.setStyleSheet(u"QPushButton {\n"
"	color: white;\n"
"	font-size: 16px;\n"
"	background: rgb(90, 37, 255);\n"
"	border-radius: 15px;\n"
"}\n"
"")
        self.back_from_rectangle_button.setCheckable(True)
        self.back_from_rectangle_button.setChecked(False)

        self.horizontalLayout_2.addWidget(self.back_from_rectangle_button)

        self.undo_loaded_roi_button = QPushButton(self.draw_roi_buttons)
        self.undo_loaded_roi_button.setObjectName(u"undo_loaded_roi_button")
        self.undo_loaded_roi_button.setMinimumSize(QSize(271, 41))
        self.undo_loaded_roi_button.setMaximumSize(QSize(271, 41))
        self.undo_loaded_roi_button.setStyleSheet(u"QPushButton {\n"
"	color: white;\n"
"	font-size: 16px;\n"
"	background: rgb(90, 37, 255);\n"
"	border-radius: 15px;\n"
"}\n"
"")
        self.undo_loaded_roi_button.setCheckable(True)
        self.undo_loaded_roi_button.setChecked(False)

        self.horizontalLayout_2.addWidget(self.undo_loaded_roi_button)

        self.accept_roi_buttons = QPushButton(self.draw_roi_buttons)
        self.accept_roi_buttons.setObjectName(u"accept_roi_buttons")
        self.accept_roi_buttons.setMinimumSize(QSize(141, 41))
        self.accept_roi_buttons.setMaximumSize(QSize(141, 41))
        self.accept_roi_buttons.setStyleSheet(u"QPushButton {\n"
"	color: white;\n"
"	font-size: 16px;\n"
"	background: rgb(90, 37, 255);\n"
"	border-radius: 15px;\n"
"}")
        self.accept_roi_buttons.setCheckable(False)

        self.horizontalLayout_2.addWidget(self.accept_roi_buttons)

        self.accept_loaded_roi_buttons = QPushButton(self.draw_roi_buttons)
        self.accept_loaded_roi_buttons.setObjectName(u"accept_loaded_roi_buttons")
        self.accept_loaded_roi_buttons.setMinimumSize(QSize(271, 41))
        self.accept_loaded_roi_buttons.setMaximumSize(QSize(271, 41))
        self.accept_loaded_roi_buttons.setStyleSheet(u"QPushButton {\n"
"	color: white;\n"
"	font-size: 16px;\n"
"	background: rgb(90, 37, 255);\n"
"	border-radius: 15px;\n"
"}\n"
"")
        self.accept_loaded_roi_buttons.setCheckable(True)
        self.accept_loaded_roi_buttons.setChecked(False)

        self.horizontalLayout_2.addWidget(self.accept_loaded_roi_buttons)

        self.load_roi_button = QPushButton(self.draw_roi_buttons)
        self.load_roi_button.setObjectName(u"load_roi_button")
        self.load_roi_button.setMinimumSize(QSize(221, 41))
        self.load_roi_button.setMaximumSize(QSize(221, 41))
        self.load_roi_button.setStyleSheet(u"QPushButton {\n"
"	color: white;\n"
"	font-size: 16px;\n"
"	background: rgb(90, 37, 255);\n"
"	border-radius: 15px;\n"
"}\n"
"")
        self.load_roi_button.setCheckable(True)
        self.load_roi_button.setChecked(False)

        self.horizontalLayout_2.addWidget(self.load_roi_button)

        self.accept_rectangle_buttons = QPushButton(self.draw_roi_buttons)
        self.accept_rectangle_buttons.setObjectName(u"accept_rectangle_buttons")
        self.accept_rectangle_buttons.setMinimumSize(QSize(241, 41))
        self.accept_rectangle_buttons.setMaximumSize(QSize(241, 41))
        self.accept_rectangle_buttons.setStyleSheet(u"QPushButton {\n"
"	color: white;\n"
"	font-size: 16px;\n"
"	background: rgb(90, 37, 255);\n"
"	border-radius: 15px;\n"
"}\n"
"")
        self.accept_rectangle_buttons.setCheckable(True)
        self.accept_rectangle_buttons.setChecked(False)

        self.horizontalLayout_2.addWidget(self.accept_rectangle_buttons)


        self.draw_roi_layout.addWidget(self.draw_roi_buttons)

        self.im_display_frame = QFrame(self.horizontalLayoutWidget_4)
        self.im_display_frame.setObjectName(u"im_display_frame")
        self.im_display_frame.setMinimumSize(QSize(501, 321))
        self.im_display_frame.setMaximumSize(QSize(16777215, 16777215))
        self.im_display_frame.setFrameShape(QFrame.StyledPanel)
        self.im_display_frame.setFrameShadow(QFrame.Raised)

        self.draw_roi_layout.addWidget(self.im_display_frame, 0, Qt.AlignHCenter|Qt.AlignVCenter)

        self.rect_dims_layout = QHBoxLayout()
        self.rect_dims_layout.setObjectName(u"rect_dims_layout")
        self.physical_rect_dims_label = QLabel(self.horizontalLayoutWidget_4)
        self.physical_rect_dims_label.setObjectName(u"physical_rect_dims_label")
        self.physical_rect_dims_label.setMinimumSize(QSize(200, 0))
        self.physical_rect_dims_label.setStyleSheet(u"QLabel {\n"
"	font-size: 18px;\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"}")
        self.physical_rect_dims_label.setTextFormat(Qt.AutoText)
        self.physical_rect_dims_label.setScaledContents(False)
        self.physical_rect_dims_label.setAlignment(Qt.AlignCenter)
        self.physical_rect_dims_label.setWordWrap(True)

        self.rect_dims_layout.addWidget(self.physical_rect_dims_label)

        self.physical_rect_width_label = QLabel(self.horizontalLayoutWidget_4)
        self.physical_rect_width_label.setObjectName(u"physical_rect_width_label")
        self.physical_rect_width_label.setStyleSheet(u"QLabel {\n"
"	font-size: 14px;\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"}")
        self.physical_rect_width_label.setTextFormat(Qt.AutoText)
        self.physical_rect_width_label.setScaledContents(False)
        self.physical_rect_width_label.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.physical_rect_width_label.setWordWrap(True)

        self.rect_dims_layout.addWidget(self.physical_rect_width_label, 0, Qt.AlignRight)

        self.physical_rect_width_val = QLabel(self.horizontalLayoutWidget_4)
        self.physical_rect_width_val.setObjectName(u"physical_rect_width_val")
        self.physical_rect_width_val.setStyleSheet(u"QLabel {\n"
"	font-size: 14px;\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"}")
        self.physical_rect_width_val.setTextFormat(Qt.AutoText)
        self.physical_rect_width_val.setScaledContents(False)
        self.physical_rect_width_val.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.physical_rect_width_val.setWordWrap(True)

        self.rect_dims_layout.addWidget(self.physical_rect_width_val)

        self.physical_rect_height_label = QLabel(self.horizontalLayoutWidget_4)
        self.physical_rect_height_label.setObjectName(u"physical_rect_height_label")
        self.physical_rect_height_label.setStyleSheet(u"QLabel {\n"
"	font-size: 14px;\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"}")
        self.physical_rect_height_label.setTextFormat(Qt.AutoText)
        self.physical_rect_height_label.setScaledContents(False)
        self.physical_rect_height_label.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.physical_rect_height_label.setWordWrap(True)

        self.rect_dims_layout.addWidget(self.physical_rect_height_label, 0, Qt.AlignRight)

        self.physical_rect_height_val = QLabel(self.horizontalLayoutWidget_4)
        self.physical_rect_height_val.setObjectName(u"physical_rect_height_val")
        self.physical_rect_height_val.setStyleSheet(u"QLabel {\n"
"	font-size: 14px;\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"}")
        self.physical_rect_height_val.setTextFormat(Qt.AutoText)
        self.physical_rect_height_val.setScaledContents(False)
        self.physical_rect_height_val.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.physical_rect_height_val.setWordWrap(True)

        self.rect_dims_layout.addWidget(self.physical_rect_height_val)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.rect_dims_layout.addItem(self.horizontalSpacer_2)

        self.edit_image_display_button = QPushButton(self.horizontalLayoutWidget_4)
        self.edit_image_display_button.setObjectName(u"edit_image_display_button")
        self.edit_image_display_button.setMinimumSize(QSize(181, 41))
        self.edit_image_display_button.setMaximumSize(QSize(181, 41))
        self.edit_image_display_button.setStyleSheet(u"QPushButton {\n"
"	color: white;\n"
"	font-size: 12px;\n"
"	background: rgb(90, 37, 255);\n"
"	border-radius: 15px;\n"
"}")

        self.rect_dims_layout.addWidget(self.edit_image_display_button)

        self.rect_dims_layout.setStretch(0, 1)
        self.rect_dims_layout.setStretch(1, 1)
        self.rect_dims_layout.setStretch(2, 2)
        self.rect_dims_layout.setStretch(3, 1)
        self.rect_dims_layout.setStretch(4, 2)
        self.rect_dims_layout.setStretch(6, 1)

        self.draw_roi_layout.addLayout(self.rect_dims_layout)

        self.draw_roi_layout.setStretch(0, 1)
        self.draw_roi_layout.setStretch(2, 10)
        self.draw_roi_layout.setStretch(3, 1)

        self.full_screen_layout.addLayout(self.draw_roi_layout)

        self.seg_loading_layout = QVBoxLayout()
        self.seg_loading_layout.setSpacing(20)
        self.seg_loading_layout.setObjectName(u"seg_loading_layout")
        self.seg_loading_layout.setContentsMargins(30, 30, 30, 30)
        self.select_seg_label = QLabel(self.horizontalLayoutWidget_4)
        self.select_seg_label.setObjectName(u"select_seg_label")
        self.select_seg_label.setStyleSheet(u"QLabel {\n"
"	font-size: 29px;\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"}")
        self.select_seg_label.setTextFormat(Qt.AutoText)
        self.select_seg_label.setScaledContents(False)
        self.select_seg_label.setAlignment(Qt.AlignCenter)
        self.select_seg_label.setWordWrap(True)

        self.seg_loading_layout.addWidget(self.select_seg_label)

        self.chooseImgLayout = QVBoxLayout()
        self.chooseImgLayout.setObjectName(u"chooseImgLayout")
        self.chooseImgLayout.setContentsMargins(20, -1, 20, -1)
        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.chooseImgLayout.addItem(self.verticalSpacer_2)

        self.seg_path_label = QLabel(self.horizontalLayoutWidget_4)
        self.seg_path_label.setObjectName(u"seg_path_label")
        self.seg_path_label.setStyleSheet(u"QLabel {\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"	color: white;\n"
"	font-size: 17px;\n"
"}")
        self.seg_path_label.setAlignment(Qt.AlignCenter)
        self.seg_path_label.setTextInteractionFlags(Qt.NoTextInteraction)

        self.chooseImgLayout.addWidget(self.seg_path_label)

        self.seg_path_input = QLineEdit(self.horizontalLayoutWidget_4)
        self.seg_path_input.setObjectName(u"seg_path_input")
        self.seg_path_input.setMinimumSize(QSize(201, 31))
        self.seg_path_input.setMaximumSize(QSize(401, 31))
        self.seg_path_input.setStyleSheet(u"QLineEdit {\n"
"	background-color: rgb(249, 249, 249);\n"
"	color: black;\n"
"}")

        self.chooseImgLayout.addWidget(self.seg_path_input, 0, Qt.AlignHCenter|Qt.AlignVCenter)

        self.chooseImageButtonsLayout = QHBoxLayout()
        self.chooseImageButtonsLayout.setSpacing(1)
        self.chooseImageButtonsLayout.setObjectName(u"chooseImageButtonsLayout")
        self.choose_seg_path_button = QPushButton(self.horizontalLayoutWidget_4)
        self.choose_seg_path_button.setObjectName(u"choose_seg_path_button")
        self.choose_seg_path_button.setMinimumSize(QSize(131, 41))
        self.choose_seg_path_button.setMaximumSize(QSize(131, 41))
        self.choose_seg_path_button.setStyleSheet(u"QPushButton {\n"
"	color: white;\n"
"	font-size: 16px;\n"
"	background: rgb(90, 37, 255);\n"
"	border-radius: 15px;\n"
"}")

        self.chooseImageButtonsLayout.addWidget(self.choose_seg_path_button, 0, Qt.AlignRight)

        self.clear_seg_path_button = QPushButton(self.horizontalLayoutWidget_4)
        self.clear_seg_path_button.setObjectName(u"clear_seg_path_button")
        self.clear_seg_path_button.setMinimumSize(QSize(131, 41))
        self.clear_seg_path_button.setMaximumSize(QSize(131, 41))
        self.clear_seg_path_button.setStyleSheet(u"QPushButton {\n"
"	color: white;\n"
"	font-size: 16px;\n"
"	background: rgb(90, 37, 255);\n"
"	border-radius: 15px;\n"
"}")

        self.chooseImageButtonsLayout.addWidget(self.clear_seg_path_button, 0, Qt.AlignLeft)


        self.chooseImgLayout.addLayout(self.chooseImageButtonsLayout)

        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.chooseImgLayout.addItem(self.verticalSpacer_3)


        self.seg_loading_layout.addLayout(self.chooseImgLayout)

        self.seg_kwargs_label = QLabel(self.horizontalLayoutWidget_4)
        self.seg_kwargs_label.setObjectName(u"seg_kwargs_label")
        self.seg_kwargs_label.setStyleSheet(u"QLabel {\n"
"	font-size: 18px;\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"}")
        self.seg_kwargs_label.setTextFormat(Qt.AutoText)
        self.seg_kwargs_label.setScaledContents(False)
        self.seg_kwargs_label.setAlignment(Qt.AlignCenter)
        self.seg_kwargs_label.setWordWrap(True)

        self.seg_loading_layout.addWidget(self.seg_kwargs_label)

        self.seg_kwargs_box = QTextEdit(self.horizontalLayoutWidget_4)
        self.seg_kwargs_box.setObjectName(u"seg_kwargs_box")
        self.seg_kwargs_box.setStyleSheet(u"QTextEdit {\n"
"	background: rgb(108, 108, 108);\n"
"	color: white;\n"
"}")

        self.seg_loading_layout.addWidget(self.seg_kwargs_box)

        self.accept_seg_path_button = QPushButton(self.horizontalLayoutWidget_4)
        self.accept_seg_path_button.setObjectName(u"accept_seg_path_button")
        self.accept_seg_path_button.setMinimumSize(QSize(131, 41))
        self.accept_seg_path_button.setMaximumSize(QSize(131, 41))
        self.accept_seg_path_button.setStyleSheet(u"QPushButton {\n"
"	color: white;\n"
"	font-size: 16px;\n"
"	background: rgb(90, 37, 255);\n"
"	border-radius: 15px;\n"
"}")

        self.seg_loading_layout.addWidget(self.accept_seg_path_button, 0, Qt.AlignHCenter|Qt.AlignVCenter)

        self.loading_screen_label = QLabel(self.horizontalLayoutWidget_4)
        self.loading_screen_label.setObjectName(u"loading_screen_label")
        self.loading_screen_label.setStyleSheet(u"QLabel {\n"
"	color: rgb(0, 255, 0);\n"
"	font-size: 20px;\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"}")
        self.loading_screen_label.setAlignment(Qt.AlignCenter)

        self.seg_loading_layout.addWidget(self.loading_screen_label)

        self.select_seg_error_msg = QLabel(self.horizontalLayoutWidget_4)
        self.select_seg_error_msg.setObjectName(u"select_seg_error_msg")
        self.select_seg_error_msg.setStyleSheet(u"QLabel {\n"
"	color: rgb(255, 0, 23);\n"
"	font-size: 20px;\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"}")
        self.select_seg_error_msg.setAlignment(Qt.AlignCenter)

        self.seg_loading_layout.addWidget(self.select_seg_error_msg)

        self.seg_loading_layout.setStretch(0, 2)
        self.seg_loading_layout.setStretch(4, 3)
        self.seg_loading_layout.setStretch(6, 2)

        self.full_screen_layout.addLayout(self.seg_loading_layout)

        self.select_type_layout = QVBoxLayout()
        self.select_type_layout.setSpacing(50)
        self.select_type_layout.setObjectName(u"select_type_layout")
        self.select_type_layout.setContentsMargins(30, -1, 30, -1)
        self.select_type_label = QLabel(self.horizontalLayoutWidget_4)
        self.select_type_label.setObjectName(u"select_type_label")
        self.select_type_label.setStyleSheet(u"QLabel {\n"
"	font-size: 29px;\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"}")
        self.select_type_label.setTextFormat(Qt.AutoText)
        self.select_type_label.setScaledContents(False)
        self.select_type_label.setAlignment(Qt.AlignCenter)
        self.select_type_label.setWordWrap(True)

        self.select_type_layout.addWidget(self.select_type_label, 0, Qt.AlignHCenter|Qt.AlignVCenter)

        self.seg_type_dropdown = QComboBox(self.horizontalLayoutWidget_4)
        self.seg_type_dropdown.setObjectName(u"seg_type_dropdown")
        self.seg_type_dropdown.setMinimumSize(QSize(180, 41))
        self.seg_type_dropdown.setMaximumSize(QSize(16777215, 16777215))
        font = QFont()
        font.setPointSize(16)
        self.seg_type_dropdown.setFont(font)
        self.seg_type_dropdown.setStyleSheet(u"QComboBox {\n"
"	color: white;\n"
"}")

        self.select_type_layout.addWidget(self.seg_type_dropdown)

        self.accept_type_button = QPushButton(self.horizontalLayoutWidget_4)
        self.accept_type_button.setObjectName(u"accept_type_button")
        self.accept_type_button.setMinimumSize(QSize(131, 41))
        self.accept_type_button.setMaximumSize(QSize(131, 41))
        self.accept_type_button.setStyleSheet(u"QPushButton {\n"
"	color: white;\n"
"	font-size: 16px;\n"
"	background: rgb(90, 37, 255);\n"
"	border-radius: 15px;\n"
"}")

        self.select_type_layout.addWidget(self.accept_type_button, 0, Qt.AlignHCenter|Qt.AlignVCenter)

        self.verticalSpacer_10 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.select_type_layout.addItem(self.verticalSpacer_10)

        self.select_type_layout.setStretch(0, 2)
        self.select_type_layout.setStretch(1, 2)
        self.select_type_layout.setStretch(2, 2)
        self.select_type_layout.setStretch(3, 1)

        self.full_screen_layout.addLayout(self.select_type_layout)

        self.full_screen_layout.setStretch(0, 1)

        self.retranslateUi(constructRoi)

        QMetaObject.connectSlotsByName(constructRoi)
    # setupUi

    def retranslateUi(self, constructRoi):
        constructRoi.setWindowTitle(QCoreApplication.translate("constructRoi", u"Select Region of Interest", None))
#if QT_CONFIG(tooltip)
        self.sidebar.setToolTip(QCoreApplication.translate("constructRoi", u"<html><head/><body><p><br/></p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.imageSelectionLabelSidebar.setText(QCoreApplication.translate("constructRoi", u"Image Selection:", None))
        self.imageLabel.setText(QCoreApplication.translate("constructRoi", u"Image:", None))
        self.phantomLabel.setText(QCoreApplication.translate("constructRoi", u"Phantom:", None))
        self.image_path_input.setText(QCoreApplication.translate("constructRoi", u"Sample filename ", None))
        self.phantom_path_input.setText(QCoreApplication.translate("constructRoi", u"Sample filename ", None))
        self.roiSidebarLabel.setText(QCoreApplication.translate("constructRoi", u"Region of Interest (ROI) Selection", None))
        self.rfAnalysisLabel.setText(QCoreApplication.translate("constructRoi", u"Radio Frequency Data Analysis", None))
        self.exportResultsLabel.setText(QCoreApplication.translate("constructRoi", u"Export Results", None))
        self.analysisParamsLabel.setText(QCoreApplication.translate("constructRoi", u"Analysis Parameter Selection", None))
        self.save_roi_button.setText(QCoreApplication.translate("constructRoi", u"Save ROI", None))
        self.back_button.setText(QCoreApplication.translate("constructRoi", u"Back", None))
        self.physical_dims_label.setText(QCoreApplication.translate("constructRoi", u"Physical Dims (cm):", None))
        self.physical_depth_label.setText(QCoreApplication.translate("constructRoi", u"Depth:", None))
        self.physical_width_val.setText(QCoreApplication.translate("constructRoi", u"0", None))
        self.physical_depth_val.setText(QCoreApplication.translate("constructRoi", u"0", None))
        self.physical_width_label.setText(QCoreApplication.translate("constructRoi", u"Width:", None))
        self.construct_roi_label.setText(QCoreApplication.translate("constructRoi", u"Construct Region of Interest (ROI):", None))
        self.pixel_dims_label.setText(QCoreApplication.translate("constructRoi", u"Pixel Dims:", None))
        self.pixel_width_label.setText(QCoreApplication.translate("constructRoi", u"Width:", None))
        self.pixel_depth_label.setText(QCoreApplication.translate("constructRoi", u"Depth:", None))
        self.pixel_width_val.setText(QCoreApplication.translate("constructRoi", u"0", None))
        self.pixel_depth_val.setText(QCoreApplication.translate("constructRoi", u"0", None))
        self.draw_freehand_button.setText(QCoreApplication.translate("constructRoi", u"Draw Freehand", None))
        self.draw_roi_button.setText(QCoreApplication.translate("constructRoi", u"Draw ROI", None))
        self.undo_last_pt_button.setText(QCoreApplication.translate("constructRoi", u"Undo Last Point", None))
        self.redraw_roi_button.setText(QCoreApplication.translate("constructRoi", u"Redraw ROI", None))
        self.close_roi_button.setText(QCoreApplication.translate("constructRoi", u"Close ROI", None))
        self.back_from_freehand_button.setText(QCoreApplication.translate("constructRoi", u"Back", None))
        self.draw_rectangle_button.setText(QCoreApplication.translate("constructRoi", u"Draw Rectangle", None))
        self.user_draw_rectangle_button.setText(QCoreApplication.translate("constructRoi", u"Draw Rectangle", None))
        self.back_from_rectangle_button.setText(QCoreApplication.translate("constructRoi", u"Back", None))
        self.undo_loaded_roi_button.setText(QCoreApplication.translate("constructRoi", u"Undo", None))
        self.accept_roi_buttons.setText(QCoreApplication.translate("constructRoi", u"Accept ROI", None))
        self.accept_loaded_roi_buttons.setText(QCoreApplication.translate("constructRoi", u"Accept ROI", None))
        self.load_roi_button.setText(QCoreApplication.translate("constructRoi", u"Load ROI", None))
        self.accept_rectangle_buttons.setText(QCoreApplication.translate("constructRoi", u"Accept ROI", None))
        self.physical_rect_dims_label.setText(QCoreApplication.translate("constructRoi", u"Rect. Dims (cm):", None))
        self.physical_rect_width_label.setText(QCoreApplication.translate("constructRoi", u"Width:", None))
        self.physical_rect_width_val.setText(QCoreApplication.translate("constructRoi", u"0", None))
        self.physical_rect_height_label.setText(QCoreApplication.translate("constructRoi", u"Height:", None))
        self.physical_rect_height_val.setText(QCoreApplication.translate("constructRoi", u"0", None))
        self.edit_image_display_button.setText(QCoreApplication.translate("constructRoi", u"Edit Image Display", None))
        self.select_seg_label.setText(QCoreApplication.translate("constructRoi", u"Select Segmentation File to Load:", None))
        self.seg_path_label.setText(QCoreApplication.translate("constructRoi", u"Input Path to Image file\n"
" (.rf, .rfd, .mat, .bin)", None))
        self.choose_seg_path_button.setText(QCoreApplication.translate("constructRoi", u"Choose File", None))
        self.clear_seg_path_button.setText(QCoreApplication.translate("constructRoi", u"Clear Path", None))
        self.seg_kwargs_label.setText(QCoreApplication.translate("constructRoi", u"\n"
"Segmentation Loading Options:", None))
        self.accept_seg_path_button.setText(QCoreApplication.translate("constructRoi", u"Accept", None))
        self.loading_screen_label.setText(QCoreApplication.translate("constructRoi", u"LOADING....", None))
        self.select_seg_error_msg.setText(QCoreApplication.translate("constructRoi", u"ERROR: At least one dimension of phantom data\n"
"smaller than corresponding dimension\n"
"of image data", None))
        self.select_type_label.setText(QCoreApplication.translate("constructRoi", u"Select Segmentation Type:", None))
        self.accept_type_button.setText(QCoreApplication.translate("constructRoi", u"Accept", None))
    # retranslateUi

