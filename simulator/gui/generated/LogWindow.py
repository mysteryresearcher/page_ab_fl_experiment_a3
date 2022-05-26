# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file './../forms/./../forms/LogWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_LogWindow(object):
    def setupUi(self, LogWindow):
        LogWindow.setObjectName("LogWindow")
        LogWindow.resize(677, 300)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/root/journal.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        LogWindow.setWindowIcon(icon)
        self.centralWidget = QtWidgets.QWidget(LogWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralWidget)
        self.verticalLayout.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName("verticalLayout")
        self.txtMain = QtWidgets.QTextEdit(self.centralWidget)
        self.txtMain.setReadOnly(True)
        self.txtMain.setObjectName("txtMain")
        self.verticalLayout.addWidget(self.txtMain)
        LogWindow.setCentralWidget(self.centralWidget)
        self.mainToolBar = QtWidgets.QToolBar(LogWindow)
        self.mainToolBar.setObjectName("mainToolBar")
        LogWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.mainToolBar)
        self.actionClean = QtWidgets.QAction(LogWindow)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/root/gnome_clear.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionClean.setIcon(icon1)
        self.actionClean.setObjectName("actionClean")
        self.actionSysInfo = QtWidgets.QAction(LogWindow)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/root/system_info.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionSysInfo.setIcon(icon2)
        self.actionSysInfo.setObjectName("actionSysInfo")
        self.actionTorchInfo = QtWidgets.QAction(LogWindow)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/root/pytorch_logo.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionTorchInfo.setIcon(icon3)
        self.actionTorchInfo.setObjectName("actionTorchInfo")
        self.actionGarbageCollector = QtWidgets.QAction(LogWindow)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/root/garbage.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionGarbageCollector.setIcon(icon4)
        self.actionGarbageCollector.setObjectName("actionGarbageCollector")
        self.actionMemInfo = QtWidgets.QAction(LogWindow)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(":/root/memory.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionMemInfo.setIcon(icon5)
        self.actionMemInfo.setObjectName("actionMemInfo")
        self.actionCmdLineGeneration = QtWidgets.QAction(LogWindow)
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(":/root/cmdline.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionCmdLineGeneration.setIcon(icon6)
        self.actionCmdLineGeneration.setObjectName("actionCmdLineGeneration")
        self.actionCmdLineSingleGeneration = QtWidgets.QAction(LogWindow)
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap(":/root/cmdline_green.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionCmdLineSingleGeneration.setIcon(icon7)
        self.actionCmdLineSingleGeneration.setObjectName("actionCmdLineSingleGeneration")
        self.actionExperimentInfoGeneration = QtWidgets.QAction(LogWindow)
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap(":/root/experiment.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionExperimentInfoGeneration.setIcon(icon8)
        self.actionExperimentInfoGeneration.setObjectName("actionExperimentInfoGeneration")
        self.actionExitLogWindow = QtWidgets.QAction(LogWindow)
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap(":/root/exit.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionExitLogWindow.setIcon(icon9)
        self.actionExitLogWindow.setObjectName("actionExitLogWindow")
        self.mainToolBar.addSeparator()
        self.mainToolBar.addAction(self.actionClean)
        self.mainToolBar.addAction(self.actionSysInfo)
        self.mainToolBar.addAction(self.actionTorchInfo)
        self.mainToolBar.addAction(self.actionGarbageCollector)
        self.mainToolBar.addAction(self.actionMemInfo)
        self.mainToolBar.addAction(self.actionCmdLineGeneration)
        self.mainToolBar.addAction(self.actionCmdLineSingleGeneration)
        self.mainToolBar.addAction(self.actionExperimentInfoGeneration)
        self.mainToolBar.addAction(self.actionExitLogWindow)

        self.retranslateUi(LogWindow)
        QtCore.QMetaObject.connectSlotsByName(LogWindow)

    def retranslateUi(self, LogWindow):
        _translate = QtCore.QCoreApplication.translate
        LogWindow.setWindowTitle(_translate("LogWindow", "Log window"))
        self.txtMain.setHtml(_translate("LogWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.actionClean.setText(_translate("LogWindow", "clean"))
        self.actionSysInfo.setText(_translate("LogWindow", "sysinfo"))
        self.actionSysInfo.setToolTip(_translate("LogWindow", "System information"))
        self.actionTorchInfo.setText(_translate("LogWindow", "torchinfo"))
        self.actionTorchInfo.setToolTip(_translate("LogWindow", "PyTorch information"))
        self.actionGarbageCollector.setText(_translate("LogWindow", "gccollector"))
        self.actionGarbageCollector.setToolTip(_translate("LogWindow", "Launch garbage collector"))
        self.actionMemInfo.setText(_translate("LogWindow", "meminfo"))
        self.actionMemInfo.setToolTip(_translate("LogWindow", "Information about used CPU/GPU memory"))
        self.actionCmdLineGeneration.setText(_translate("LogWindow", "cmdline"))
        self.actionCmdLineGeneration.setToolTip(_translate("LogWindow", "Command line generation for experiments"))
        self.actionCmdLineSingleGeneration.setText(_translate("LogWindow", "cmdline"))
        self.actionCmdLineSingleGeneration.setToolTip(_translate("LogWindow", "Single line command line generation for experiments"))
        self.actionExperimentInfoGeneration.setText(_translate("LogWindow", "experimentInfo"))
        self.actionExperimentInfoGeneration.setToolTip(_translate("LogWindow", "Configured experiment description generation"))
        self.actionExitLogWindow.setText(_translate("LogWindow", "exit"))
        self.actionExitLogWindow.setToolTip(_translate("LogWindow", "Exit from log window"))
import resources_rc


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    LogWindow = QtWidgets.QMainWindow()
    ui = Ui_LogWindow()
    ui.setupUi(LogWindow)
    LogWindow.show()
    sys.exit(app.exec_())
