# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'select_image.ui'
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

class Ui_selectImage(object):
    def setupUi(self, selectImage):
        if not selectImage.objectName():
            selectImage.setObjectName(u"selectImage")
        selectImage.resize(1512, 832)
        selectImage.setMinimumSize(QSize(201, 31))
        selectImage.setMaximumSize(QSize(16777215, 16777215))
        selectImage.setStyleSheet(u"QWidget {\n"
"	background: rgb(42, 42, 42);\n"
"}")
        self.horizontalLayoutWidget = QWidget(selectImage)
        self.horizontalLayoutWidget.setObjectName(u"horizontalLayoutWidget")
        self.horizontalLayoutWidget.setGeometry(QRect(70, 10, 1545, 844))
        self.full_screen_layout = QHBoxLayout(self.horizontalLayoutWidget)
        self.full_screen_layout.setObjectName(u"full_screen_layout")
        self.full_screen_layout.setContentsMargins(0, 0, 0, 0)
        self.side_bar_layout = QVBoxLayout()
        self.side_bar_layout.setSpacing(0)
        self.side_bar_layout.setObjectName(u"side_bar_layout")
        self.side_bar_layout.setSizeConstraint(QLayout.SetMaximumSize)
        self.sidebar = QWidget(self.horizontalLayoutWidget)
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
        self.roiSidebar = QFrame(self.sidebar)
        self.roiSidebar.setObjectName(u"roiSidebar")
        self.roiSidebar.setGeometry(QRect(0, 120, 341, 121))
        self.roiSidebar.setMaximumSize(QSize(341, 121))
        self.roiSidebar.setStyleSheet(u"QFrame {\n"
"	background-color: rgb(49, 0, 124);\n"
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

        self.gridFrame = QFrame(self.horizontalLayoutWidget)
        self.gridFrame.setObjectName(u"gridFrame")
        self.gridFrame.setMinimumSize(QSize(341, 0))
        self.gridFrame.setMaximumSize(QSize(341, 16777215))
        self.gridFrame.setStyleSheet(u"QFrame {\n"
"	background-color: rgb(28, 0, 101);\n"
"}")
        self.backButtonGrid = QGridLayout(self.gridFrame)
        self.backButtonGrid.setObjectName(u"backButtonGrid")
        self.backButtonGrid.setSizeConstraint(QLayout.SetMinAndMaxSize)
        self.backButtonGrid.setContentsMargins(10, 10, 10, 10)
        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.backButtonGrid.addItem(self.verticalSpacer, 0, 0, 1, 1)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.backButtonGrid.addItem(self.horizontalSpacer, 1, 1, 1, 1)

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


        self.side_bar_layout.addWidget(self.gridFrame)


        self.full_screen_layout.addLayout(self.side_bar_layout)

        self.img_selection_layout = QVBoxLayout()
        self.img_selection_layout.setSpacing(20)
        self.img_selection_layout.setObjectName(u"img_selection_layout")
        self.img_selection_layout.setContentsMargins(30, 30, 30, 30)
        self.select_data_label = QLabel(self.horizontalLayoutWidget)
        self.select_data_label.setObjectName(u"select_data_label")
        self.select_data_label.setStyleSheet(u"QLabel {\n"
"	font-size: 29px;\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"}")
        self.select_data_label.setTextFormat(Qt.AutoText)
        self.select_data_label.setScaledContents(False)
        self.select_data_label.setAlignment(Qt.AlignCenter)
        self.select_data_label.setWordWrap(True)

        self.img_selection_layout.addWidget(self.select_data_label)

        self.chooseImgPhantLayout = QHBoxLayout()
        self.chooseImgPhantLayout.setObjectName(u"chooseImgPhantLayout")
        self.chooseImgLayout = QVBoxLayout()
        self.chooseImgLayout.setObjectName(u"chooseImgLayout")
        self.chooseImgLayout.setContentsMargins(20, -1, 20, -1)
        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.chooseImgLayout.addItem(self.verticalSpacer_2)

        self.image_path_label = QLabel(self.horizontalLayoutWidget)
        self.image_path_label.setObjectName(u"image_path_label")
        self.image_path_label.setStyleSheet(u"QLabel {\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"	color: white;\n"
"	font-size: 17px;\n"
"}")
        self.image_path_label.setAlignment(Qt.AlignCenter)
        self.image_path_label.setTextInteractionFlags(Qt.NoTextInteraction)

        self.chooseImgLayout.addWidget(self.image_path_label)

        self.image_path_input = QLineEdit(self.horizontalLayoutWidget)
        self.image_path_input.setObjectName(u"image_path_input")
        self.image_path_input.setMinimumSize(QSize(201, 31))
        self.image_path_input.setMaximumSize(QSize(401, 31))
        self.image_path_input.setStyleSheet(u"QLineEdit {\n"
"	background-color: rgb(249, 249, 249);\n"
"	color: black;\n"
"}")

        self.chooseImgLayout.addWidget(self.image_path_input, 0, Qt.AlignHCenter|Qt.AlignVCenter)

        self.chooseImageButtonsLayout = QHBoxLayout()
        self.chooseImageButtonsLayout.setObjectName(u"chooseImageButtonsLayout")
        self.choose_image_path_button = QPushButton(self.horizontalLayoutWidget)
        self.choose_image_path_button.setObjectName(u"choose_image_path_button")
        self.choose_image_path_button.setMinimumSize(QSize(131, 41))
        self.choose_image_path_button.setMaximumSize(QSize(131, 41))
        self.choose_image_path_button.setStyleSheet(u"QPushButton {\n"
"	color: white;\n"
"	font-size: 16px;\n"
"	background: rgb(90, 37, 255);\n"
"	border-radius: 15px;\n"
"}")

        self.chooseImageButtonsLayout.addWidget(self.choose_image_path_button)

        self.clear_image_path_button = QPushButton(self.horizontalLayoutWidget)
        self.clear_image_path_button.setObjectName(u"clear_image_path_button")
        self.clear_image_path_button.setMinimumSize(QSize(131, 41))
        self.clear_image_path_button.setMaximumSize(QSize(131, 41))
        self.clear_image_path_button.setStyleSheet(u"QPushButton {\n"
"	color: white;\n"
"	font-size: 16px;\n"
"	background: rgb(90, 37, 255);\n"
"	border-radius: 15px;\n"
"}")

        self.chooseImageButtonsLayout.addWidget(self.clear_image_path_button)


        self.chooseImgLayout.addLayout(self.chooseImageButtonsLayout)

        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.chooseImgLayout.addItem(self.verticalSpacer_3)


        self.chooseImgPhantLayout.addLayout(self.chooseImgLayout)

        self.choosePhantomLayout = QVBoxLayout()
        self.choosePhantomLayout.setObjectName(u"choosePhantomLayout")
        self.choosePhantomLayout.setContentsMargins(20, -1, 20, -1)
        self.verticalSpacer_4 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.choosePhantomLayout.addItem(self.verticalSpacer_4)

        self.phantom_path_label = QLabel(self.horizontalLayoutWidget)
        self.phantom_path_label.setObjectName(u"phantom_path_label")
        self.phantom_path_label.setStyleSheet(u"QLabel {\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"	color: white;\n"
"	font-size: 17px;\n"
"}")
        self.phantom_path_label.setAlignment(Qt.AlignCenter)
        self.phantom_path_label.setTextInteractionFlags(Qt.NoTextInteraction)

        self.choosePhantomLayout.addWidget(self.phantom_path_label)

        self.phantom_path_input = QLineEdit(self.horizontalLayoutWidget)
        self.phantom_path_input.setObjectName(u"phantom_path_input")
        self.phantom_path_input.setMinimumSize(QSize(201, 31))
        self.phantom_path_input.setMaximumSize(QSize(401, 31))
        self.phantom_path_input.setStyleSheet(u"QLineEdit {\n"
"	background-color: rgb(249, 249, 249);\n"
"	color: black;\n"
"}")

        self.choosePhantomLayout.addWidget(self.phantom_path_input, 0, Qt.AlignHCenter|Qt.AlignVCenter)

        self.choosePhantomButtonsLayout = QHBoxLayout()
        self.choosePhantomButtonsLayout.setObjectName(u"choosePhantomButtonsLayout")
        self.choose_phantom_path_button = QPushButton(self.horizontalLayoutWidget)
        self.choose_phantom_path_button.setObjectName(u"choose_phantom_path_button")
        self.choose_phantom_path_button.setMinimumSize(QSize(131, 41))
        self.choose_phantom_path_button.setMaximumSize(QSize(131, 41))
        self.choose_phantom_path_button.setStyleSheet(u"QPushButton {\n"
"	color: white;\n"
"	font-size: 16px;\n"
"	background: rgb(90, 37, 255);\n"
"	border-radius: 15px;\n"
"}")

        self.choosePhantomButtonsLayout.addWidget(self.choose_phantom_path_button)

        self.clear_phantom_path_button = QPushButton(self.horizontalLayoutWidget)
        self.clear_phantom_path_button.setObjectName(u"clear_phantom_path_button")
        self.clear_phantom_path_button.setMinimumSize(QSize(131, 41))
        self.clear_phantom_path_button.setMaximumSize(QSize(131, 41))
        self.clear_phantom_path_button.setStyleSheet(u"QPushButton {\n"
"	color: white;\n"
"	font-size: 16px;\n"
"	background: rgb(90, 37, 255);\n"
"	border-radius: 15px;\n"
"}")

        self.choosePhantomButtonsLayout.addWidget(self.clear_phantom_path_button)


        self.choosePhantomLayout.addLayout(self.choosePhantomButtonsLayout)

        self.verticalSpacer_5 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.choosePhantomLayout.addItem(self.verticalSpacer_5)


        self.chooseImgPhantLayout.addLayout(self.choosePhantomLayout)


        self.img_selection_layout.addLayout(self.chooseImgPhantLayout)

        self.analysis_kwargs_label = QLabel(self.horizontalLayoutWidget)
        self.analysis_kwargs_label.setObjectName(u"analysis_kwargs_label")
        self.analysis_kwargs_label.setStyleSheet(u"QLabel {\n"
"	font-size: 18px;\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"}")
        self.analysis_kwargs_label.setTextFormat(Qt.AutoText)
        self.analysis_kwargs_label.setScaledContents(False)
        self.analysis_kwargs_label.setAlignment(Qt.AlignCenter)
        self.analysis_kwargs_label.setWordWrap(True)

        self.img_selection_layout.addWidget(self.analysis_kwargs_label)

        self.analysis_kwargs_box = QTextEdit(self.horizontalLayoutWidget)
        self.analysis_kwargs_box.setObjectName(u"analysis_kwargs_box")
        self.analysis_kwargs_box.setStyleSheet(u"QTextEdit {\n"
"	background: rgb(108, 108, 108);\n"
"	color: white;\n"
"}")

        self.img_selection_layout.addWidget(self.analysis_kwargs_box)

        self.generate_image_button = QPushButton(self.horizontalLayoutWidget)
        self.generate_image_button.setObjectName(u"generate_image_button")
        self.generate_image_button.setMinimumSize(QSize(131, 41))
        self.generate_image_button.setMaximumSize(QSize(131, 41))
        self.generate_image_button.setStyleSheet(u"QPushButton {\n"
"	color: white;\n"
"	font-size: 16px;\n"
"	background: rgb(90, 37, 255);\n"
"	border-radius: 15px;\n"
"}")

        self.img_selection_layout.addWidget(self.generate_image_button, 0, Qt.AlignHCenter|Qt.AlignVCenter)

        self.loading_screen_label = QLabel(self.horizontalLayoutWidget)
        self.loading_screen_label.setObjectName(u"loading_screen_label")
        self.loading_screen_label.setStyleSheet(u"QLabel {\n"
"	color: rgb(0, 255, 0);\n"
"	font-size: 20px;\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"}")
        self.loading_screen_label.setAlignment(Qt.AlignCenter)

        self.img_selection_layout.addWidget(self.loading_screen_label)

        self.select_image_error_msg = QLabel(self.horizontalLayoutWidget)
        self.select_image_error_msg.setObjectName(u"select_image_error_msg")
        self.select_image_error_msg.setStyleSheet(u"QLabel {\n"
"	color: rgb(255, 0, 23);\n"
"	font-size: 20px;\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"}")
        self.select_image_error_msg.setAlignment(Qt.AlignCenter)

        self.img_selection_layout.addWidget(self.select_image_error_msg)

        self.img_selection_layout.setStretch(0, 2)
        self.img_selection_layout.setStretch(1, 2)
        self.img_selection_layout.setStretch(4, 3)
        self.img_selection_layout.setStretch(6, 2)

        self.full_screen_layout.addLayout(self.img_selection_layout)

        self.select_type_layout = QVBoxLayout()
        self.select_type_layout.setSpacing(50)
        self.select_type_layout.setObjectName(u"select_type_layout")
        self.select_type_layout.setContentsMargins(30, -1, 30, -1)
        self.select_type_label = QLabel(self.horizontalLayoutWidget)
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

        self.scan_type_dropdown = QComboBox(self.horizontalLayoutWidget)
        self.scan_type_dropdown.setObjectName(u"scan_type_dropdown")
        self.scan_type_dropdown.setMinimumSize(QSize(180, 41))
        self.scan_type_dropdown.setMaximumSize(QSize(16777215, 16777215))
        font = QFont()
        font.setPointSize(16)
        self.scan_type_dropdown.setFont(font)
        self.scan_type_dropdown.setStyleSheet(u"QComboBox {\n"
"	color: white;\n"
"}")

        self.select_type_layout.addWidget(self.scan_type_dropdown)

        self.accept_type_button = QPushButton(self.horizontalLayoutWidget)
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
        self.full_screen_layout.setStretch(1, 5)
        self.full_screen_layout.setStretch(2, 5)

        self.retranslateUi(selectImage)

        QMetaObject.connectSlotsByName(selectImage)
    # setupUi

    def retranslateUi(self, selectImage):
        selectImage.setWindowTitle(QCoreApplication.translate("selectImage", u"Select Ultrasound Image", None))
#if QT_CONFIG(tooltip)
        self.sidebar.setToolTip(QCoreApplication.translate("selectImage", u"<html><head/><body><p><br/></p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.imageSelectionLabelSidebar.setText(QCoreApplication.translate("selectImage", u"Image Selection:", None))
        self.imageLabel.setText(QCoreApplication.translate("selectImage", u"Image:", None))
        self.phantomLabel.setText(QCoreApplication.translate("selectImage", u"Phantom:", None))
        self.roiSidebarLabel.setText(QCoreApplication.translate("selectImage", u"Region of Interest (ROI) Selection", None))
        self.rfAnalysisLabel.setText(QCoreApplication.translate("selectImage", u"Radio Frequency Data Analysis", None))
        self.exportResultsLabel.setText(QCoreApplication.translate("selectImage", u"Export Results", None))
        self.analysisParamsLabel.setText(QCoreApplication.translate("selectImage", u"Analysis Parameter Selection", None))
        self.back_button.setText(QCoreApplication.translate("selectImage", u"Back", None))
        self.select_data_label.setText(QCoreApplication.translate("selectImage", u"Select Data and Phantom Files to Generate Ultrasound Image:", None))
        self.image_path_label.setText(QCoreApplication.translate("selectImage", u"Input Path to Image file\n"
" (.rf, .rfd, .mat, .bin)", None))
        self.choose_image_path_button.setText(QCoreApplication.translate("selectImage", u"Choose File", None))
        self.clear_image_path_button.setText(QCoreApplication.translate("selectImage", u"Clear Path", None))
        self.phantom_path_label.setText(QCoreApplication.translate("selectImage", u"Input Path to Phantom file\n"
" (.rf, .rfd, .mat, .bin)", None))
        self.choose_phantom_path_button.setText(QCoreApplication.translate("selectImage", u"Choose File", None))
        self.clear_phantom_path_button.setText(QCoreApplication.translate("selectImage", u"Clear Path", None))
        self.analysis_kwargs_label.setText(QCoreApplication.translate("selectImage", u"\n"
"Image Loading Options:", None))
        self.generate_image_button.setText(QCoreApplication.translate("selectImage", u"Generate Image", None))
        self.loading_screen_label.setText(QCoreApplication.translate("selectImage", u"LOADING....", None))
        self.select_image_error_msg.setText(QCoreApplication.translate("selectImage", u"ERROR: At least one dimension of phantom data\n"
"smaller than corresponding dimension\n"
"of image data", None))
        self.select_type_label.setText(QCoreApplication.translate("selectImage", u"Select Data Type:", None))
        self.accept_type_button.setText(QCoreApplication.translate("selectImage", u"Accept", None))
    # retranslateUi

