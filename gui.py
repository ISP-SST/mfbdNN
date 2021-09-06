#!/usr/bin/python3

"""
A simple GUI front end to process and display live images at the SST

Author: Tomas Hillberg <hillberg@astro.su.se>
"""

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel


def main():

    app = QApplication(sys.argv)

    w = QWidget()
    w.resize(250, 150)
    w.move(300, 300)
    w.setWindowTitle('Bla')
    w.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

