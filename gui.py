#!/usr/bin/python3

"""
A simple GUI frontend to process and display live images at the SST

Author: Tomas Hillberg <hillberg@astro.su.se>
"""

import sys
from PyQt5.QtWidgets import (QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QApplication, QMainWindow, QMessageBox, QFrame, QLabel)
from PyQt5.QtGui import (QPixmap, QImage)
from PyQt5.QtCore import (Qt, pyqtSlot, QSettings)
from PyQt5.QtNetwork import QTcpSocket, QAbstractSocket
from PyQt5.QtCore import QDataStream, QIODevice

import numpy as np
import astropy.io.fits as fts
import encdec_sst as mfbdnn
from skimage.transform import resize

testfiles = [ 'test/Chromis-N_2016-09-19T09:28:36_00027_12.00ms_G10.00_3999_4000_+1258_momfbd.fits',
              'test/Chromis-N_2016-09-19T09:28:36_00047_12.00ms_G10.00_3934_3934_-1331_raw.fits',
              'test/Chromis-N_2016-09-19T09:28:36_00027_12.00ms_G10.00_3999_4000_+1258_raw.fits',
              'test/Chromis-N_2016-09-19T09:28:36_00064_12.00ms_G10.00_3934_3934_-391_momfbd.fits',
              'test/Chromis-N_2016-09-19T09:28:36_00047_12.00ms_G10.00_3934_3934_-1331_momfbd.fits',
              'test/Chromis-N_2016-09-19T09:28:36_00064_12.00ms_G10.00_3934_3934_-391_raw.fits' ]

class Gui( QMainWindow ):

    def __init__( self ):
        super().__init__()
        
        self.verbose = True

        self.mainWidget = QWidget( self )
        self.displayLabel = QLabel( self )
        self.currentFile = 1
        self.currentFrame = 0
        self.nFrames = 0
        self.initUI()
        self.readSettings()
        
        self.testfile = testfiles[ self.currentFile ]
        if self.verbose: print( "=> TestFile: " + self.testfile )
        
        self.tcpSocket = QTcpSocket(self)
        self.blockSize = 4
        self.tcpSocket.readyRead.connect( self.tcpRead )
        self.tcpSocket.error.connect( self.tcpError )
        self.tcpConnect( "localhost", 15000 )

    def tcpConnect( self, host, port, timeout=1000 ):
        self.tcpSocket.disconnectFromHost()
        self.tcpSocket.connectToHost( host, port, QIODevice.ReadWrite )
        if self.tcpSocket.waitForConnected( timeout ):
            print( "Connected to" + host + ":" + str(port) )
            self.tcpSocket.write( b'get exposure\r\n' )
        else:
            print( "Failed to connected to" + host + ":" + str(port) )
            
    def tcpRead( self ):
#        instr = QDataStream( self.tcpSocket )
#        instr.setVersion( QDataStream.Qt_5_0 )
#        if self.tcpSocket.bytesAvailable() < self.blockSize:
#            return
        
        # Print response to terminal, we could use it anywhere else we wanted.
#        print( "Reading stream " + str(self.tcpSocket.bytesAvailable()) + " bytes available" )
#        print( instr.readString() )
        conn = self.tcpSocket.sender();
        #list = QStringList list;
        while conn.canReadLine() :
            data = conn.readLine();
            print( "Data: " + str(data) )
            #print( "Data: " + data.str() )

    def tcpError( self, socketError ):
        if socketError == QAbstractSocket.RemoteHostClosedError:
            pass
        else:
            print( self, "The following error occurred: %s." % self.tcpSocket.errorString() )
            

    def initUI(self):

        okButton = QPushButton( "OK" )
        cancelButton = QPushButton( "Cancel" )
        
        okButton.clicked.connect( self.nextFrame )
        
        self.settingsFrame = QFrame( self )
        self.settingsFrame.setFrameStyle( QFrame.Panel | QFrame.Raised );
        self.settingsFrame.setLineWidth( 1 );
        self.settingsLayoutV = QVBoxLayout()
        self.settingsLayoutV.addWidget( okButton )
        self.settingsLayoutV.addStretch( 1 )
        self.settingsLayoutV.addWidget( cancelButton )
        self.settingsFrame.setLayout( self.settingsLayoutV )
        
        #self.displayFrame = QFrame( self )
        #self.displayFrame.setFrameStyle( QFrame.StyledPanel | QFrame.Raised );
        #self.displayFrame.setLineWidth( 1 );
        self.mainLayout = QHBoxLayout()
        self.mainLayout.addWidget( self.settingsFrame )
        self.mainLayout.addStretch(1)
        #self.mainLayout.addWidget( self.displayFrame )
        self.mainLayout.addWidget( self.displayLabel )
        self.displayLabel.setText( "bla " )

        self.mainWidget.setLayout( self.mainLayout )
        #self.mainWidget.setStyleSheet("background-color:steelblue;");
        
        self.setCentralWidget( self.mainWidget )
        
        self.setGeometry( 300, 300, 300, 150 )
        self.setWindowTitle( 'SST - MFBD Neural Network display' )
        self.show()
        
    def readSettings( self ):
        settings = QSettings( "SST", "MFBDNN-GUI" )
        geo = settings.value( "geometry" )
        if geo:
            self.restoreGeometry( geo );
        state = settings.value( "windowState" )
        if state:
            self.restoreState( state );
        
    def saveSettings( self ):
        settings = QSettings( "SST", "MFBDNN-GUI" )
        settings.setValue( "geometry", self.saveGeometry() );
        settings.setValue( "windowState", self.saveState() );
        
    def closeEvent( self, event ):
        self.saveSettings()
        super().closeEvent( event );
        
    @pyqtSlot()
    def nextFrame( self ):
        if self.nFrames:
            self.currentFrame = (self.currentFrame + 1) % self.nFrames
        else:
            self.currentFrame = 0
        self.displayTest( self.currentFrame )
        
    def displayTest( self, frame_id=0 ):
        self.testfile = testfiles[ self.currentFile ]
        myFits = fts.open( self.testfile )

        imgData = myFits[0].data[:]
        img = 0
        if len( imgData.shape ) == 2:
            self.nFrames = 1
            img = imgData[:,:]
            nX = imgData.shape[0]
            nY = imgData.shape[1]
        elif len( imgData.shape ) == 3:
            self.nFrames = imgData.shape[0]
            img = imgData[ frame_id, :, : ]
            nX = imgData.shape[1]
            nY = imgData.shape[2]
        else:
            self.nFrames = 0
            return
        #img =  np.zeros( (int(nX/2), int(nY/2)), np.ubyte )
        img >>= 8
        img = resize( img, (600, 400) )

        #img >>= 2
        print( "=> Shape {}   {}".format(img.shape,img[0,0]) )
        #grayImg = QImage( img, img.shape[0], img.shape[1], QImage.Format_Mono )
        #for x in range(1,img.shape[0]):
        #    for y in range(1,img.shape[1]):
        #        img[x,y] = y # (125+x) % 1024  # + y*nX #% 255
        #        img[x,y] = imgData[ frame_id]
                #grayPixel = qRgba( gray, gray, gray, alpha )
                #grayImg.setPixel(x, y, int(img[x,y]) )
        #img = np.transpose( img, (1,0) ).copy()
        #print( "=> Type {}".format(img.) )
        #qimg = QImage()
        #qimg.loadFromData( img )
        #pix_img = qimg.scaled(66, 66, Qt.KeepAspectRatio)
        
        qimage = QImage( img, img.shape[1], img.shape[0], QImage.Format_Grayscale8 )  #Format_MonoLSB ) # QImage.Format_Grayscale8 ) #QImage.Format_Mono )
        qpixmap = QPixmap( qimage )

        
        self.displayLabel.setPixmap( qpixmap )
        
        # dataType, ims = mfbdnn.check_type( myFits[0].data[:] )
        if self.verbose:
            print( "=> Shape {}".format(img.shape) )
            print( "=> currentFrame {}".format( self.currentFrame ) )
            

def main():

    app = QApplication( sys.argv )
    gui = Gui()
    sys.exit( app.exec_() )


if __name__ == '__main__':
    main()

