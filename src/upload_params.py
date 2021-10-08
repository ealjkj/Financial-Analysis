# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'designercideTX.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

class Communicate(QObject):
    set_browsed_text = pyqtSignal()



class LabelToDrop(QLabel):
    def __init__(self, parent):
        super().__init__()
        self.c = Communicate()

    def setPixmap(self, image: QPixmap) -> None:
        return super().setPixmap(image)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        file_path = event.mimeData().urls()[0].toLocalFile()
        if file_path.split('.')[-1] == 'json':
            event.accept()
        else:
            event.ignore()
    
    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        file_path = event.mimeData().urls()[0].toLocalFile()
        if file_path.split('.')[-1] == 'json':
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:

        self.file_path = event.mimeData().urls()[0].toLocalFile()

        if self.file_path.split('.')[-1] == 'json':
            fileInfo = QFileInfo(self.file_path)
            iconProvider = QFileIconProvider()
            icon = iconProvider.icon(fileInfo)
            event.setDropAction(Qt.CopyAction)
            pixmap = icon.pixmap(icon.actualSize(QSize(64, 64)))
            self.setPixmap(pixmap)
            self.setStyleSheet('border-style: none')
            event.accept()
            self.c.set_browsed_text.emit()
        else:
            event.accept()
        

    def set_image(self, file_path):
        self.setPixmap(QPixmap(file_path))
        


class Ui_Dialog_Upload(object):
    def setupUi(self, Dialog):
        if not Dialog.objectName():
            Dialog.setObjectName(u"upload_dialog")

        Dialog.resize(650, 500)
        self.verticalLayout = QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.drop_files_label = LabelToDrop(Dialog)
        self.drop_files_label.setObjectName(u"drop_files_label")
        self.drop_files_label.setStyleSheet(u"color: black;\n"
"border: 5px dashed black;\n  border-radius: 12px; ")
        
        self.drop_files_label.c.set_browsed_text.connect(self.change_text)

        self.drop_files_label.setAcceptDrops(True)
        self.drop_files_label.setAlignment(Qt.AlignCenter)


        self.verticalLayout_2.addWidget(self.drop_files_label)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.browse_files_edit = QLineEdit(Dialog)
        self.browse_files_edit.setObjectName(u"browse_files_edit")
        self.horizontalLayout.addWidget(self.browse_files_edit)

        self.browse_files_button = QPushButton(Dialog)
        self.browse_files_button.setObjectName(u"browse_files_button")
        self.browse_files_button.clicked.connect(lambda: self.browse_files(Dialog))

        self.horizontalLayout.addWidget(self.browse_files_button)


        self.verticalLayout_2.addLayout(self.horizontalLayout)

        self.verticalLayout_2.setStretch(0, 6)
        self.verticalLayout_2.setStretch(1, 1)

        self.verticalLayout.addLayout(self.verticalLayout_2)

        self.buttonBox = QDialogButtonBox(Dialog)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)

        self.verticalLayout.addWidget(self.buttonBox)


        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(lambda: self.new_accept(Dialog))
        self.buttonBox.rejected.connect(Dialog.reject)

        QMetaObject.connectSlotsByName(Dialog)
    # setupUi


    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"Dialog", None))
        self.browse_files_button.setText(QCoreApplication.translate("Dialog", u"Browse files", None))
        self.drop_files_label.setText(QCoreApplication.translate("Dialog", u"Drop your json file here", None))
    # retranslateUi

    def new_accept(self, Dialog):
        Dialog.accept()
        self.file_path = self.drop_files_label.file_path

    def change_text(self):
        self.browse_files_edit.setText(self.drop_files_label.file_path)
        
    def browse_files(self, Dialog):
        print(Dialog)
        fname = QFileDialog.getOpenFileName(Dialog, 'Open file')
        self.browse_files_edit.setText(fname[0])





