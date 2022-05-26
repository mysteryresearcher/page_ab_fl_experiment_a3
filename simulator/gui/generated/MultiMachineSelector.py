# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file './../forms/./../forms/MultiMachineSelector.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MultiMachineSelector(object):
    def setupUi(self, MultiMachineSelector):
        MultiMachineSelector.setObjectName("MultiMachineSelector")
        MultiMachineSelector.resize(797, 666)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/root/network.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MultiMachineSelector.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MultiMachineSelector)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.loHorB = QtWidgets.QHBoxLayout()
        self.loHorB.setObjectName("loHorB")
        self.edtMachine = QtWidgets.QLineEdit(self.centralwidget)
        self.edtMachine.setObjectName("edtMachine")
        self.loHorB.addWidget(self.edtMachine)
        self.btnAddMachine = QtWidgets.QPushButton(self.centralwidget)
        self.btnAddMachine.setObjectName("btnAddMachine")
        self.loHorB.addWidget(self.btnAddMachine)
        self.verticalLayout_2.addLayout(self.loHorB)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.tblMachines = QtWidgets.QTableWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tblMachines.sizePolicy().hasHeightForWidth())
        self.tblMachines.setSizePolicy(sizePolicy)
        self.tblMachines.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tblMachines.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectItems)
        self.tblMachines.setObjectName("tblMachines")
        self.tblMachines.setColumnCount(10)
        self.tblMachines.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tblMachines.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tblMachines.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tblMachines.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tblMachines.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tblMachines.setHorizontalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.tblMachines.setHorizontalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.tblMachines.setHorizontalHeaderItem(6, item)
        item = QtWidgets.QTableWidgetItem()
        self.tblMachines.setHorizontalHeaderItem(7, item)
        item = QtWidgets.QTableWidgetItem()
        self.tblMachines.setHorizontalHeaderItem(8, item)
        item = QtWidgets.QTableWidgetItem()
        self.tblMachines.setHorizontalHeaderItem(9, item)
        self.verticalLayout.addWidget(self.tblMachines)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.loHorB_2 = QtWidgets.QHBoxLayout()
        self.loHorB_2.setObjectName("loHorB_2")
        self.btnRefreshMachineStatus = QtWidgets.QPushButton(self.centralwidget)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/root/refresh.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btnRefreshMachineStatus.setIcon(icon1)
        self.btnRefreshMachineStatus.setObjectName("btnRefreshMachineStatus")
        self.loHorB_2.addWidget(self.btnRefreshMachineStatus)
        self.btnCleanAll = QtWidgets.QPushButton(self.centralwidget)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/root/gnome_clear.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btnCleanAll.setIcon(icon2)
        self.btnCleanAll.setObjectName("btnCleanAll")
        self.loHorB_2.addWidget(self.btnCleanAll)
        self.btnRemoveSelected = QtWidgets.QPushButton(self.centralwidget)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/root/delete.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btnRemoveSelected.setIcon(icon3)
        self.btnRemoveSelected.setObjectName("btnRemoveSelected")
        self.loHorB_2.addWidget(self.btnRemoveSelected)
        self.btnExit = QtWidgets.QPushButton(self.centralwidget)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/root/exit.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btnExit.setIcon(icon4)
        self.btnExit.setObjectName("btnExit")
        self.loHorB_2.addWidget(self.btnExit)
        self.verticalLayout_2.addLayout(self.loHorB_2)
        self.verticalLayout_3.addLayout(self.verticalLayout_2)
        MultiMachineSelector.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MultiMachineSelector)
        self.statusbar.setObjectName("statusbar")
        MultiMachineSelector.setStatusBar(self.statusbar)

        self.retranslateUi(MultiMachineSelector)
        QtCore.QMetaObject.connectSlotsByName(MultiMachineSelector)

    def retranslateUi(self, MultiMachineSelector):
        _translate = QtCore.QCoreApplication.translate
        MultiMachineSelector.setWindowTitle(_translate("MultiMachineSelector", "Extra machines resource selector"))
        self.edtMachine.setToolTip(_translate("MultiMachineSelector", "Please insert <hostname|ip>:port into machine name field"))
        self.btnAddMachine.setText(_translate("MultiMachineSelector", "Add Machine"))
        item = self.tblMachines.horizontalHeaderItem(0)
        item.setText(_translate("MultiMachineSelector", "Host"))
        item = self.tblMachines.horizontalHeaderItem(1)
        item.setText(_translate("MultiMachineSelector", "Ip"))
        item = self.tblMachines.horizontalHeaderItem(2)
        item.setText(_translate("MultiMachineSelector", "Port"))
        item = self.tblMachines.horizontalHeaderItem(3)
        item.setText(_translate("MultiMachineSelector", "#GPUs"))
        item = self.tblMachines.horizontalHeaderItem(4)
        item.setText(_translate("MultiMachineSelector", "Use CPU:-1"))
        item = self.tblMachines.horizontalHeaderItem(5)
        item.setText(_translate("MultiMachineSelector", "Use GPU:0"))
        item = self.tblMachines.horizontalHeaderItem(6)
        item.setText(_translate("MultiMachineSelector", "Use GPU:1"))
        item = self.tblMachines.horizontalHeaderItem(7)
        item.setText(_translate("MultiMachineSelector", "Use GPU:2"))
        item = self.tblMachines.horizontalHeaderItem(8)
        item.setText(_translate("MultiMachineSelector", "Use GPU:3"))
        item = self.tblMachines.horizontalHeaderItem(9)
        item.setText(_translate("MultiMachineSelector", "Online"))
        self.btnRefreshMachineStatus.setText(_translate("MultiMachineSelector", "Refresh machine status"))
        self.btnCleanAll.setText(_translate("MultiMachineSelector", "Remove all machines"))
        self.btnRemoveSelected.setText(_translate("MultiMachineSelector", "Remove selected machines"))
        self.btnExit.setText(_translate("MultiMachineSelector", "Close window"))
import resources_rc


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MultiMachineSelector = QtWidgets.QMainWindow()
    ui = Ui_MultiMachineSelector()
    ui.setupUi(MultiMachineSelector)
    MultiMachineSelector.show()
    sys.exit(app.exec_())
