import numpy as np
import torch
import torch.nn as nn
import time
import model_encdec as model
import astropy.io.fits as fts
import sys
import warnings
import argparse
import torch.multiprocessing as mp
import os
import sys
# print('version 1.1')

#--------------------------------------------------------------------------------------
def update_progress(progress, cada,total, tiempo=0):
    barLength = 20 # Modify this to change the length of the progress bar
    status = " in {2}s ({3}s is remaining) - Deconvolving {0}/{1} scans   ".format(cada, total, int(tiempo),int((tiempo/cada)*(total-cada)))
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = " in {2}s - {0}/{1} scans: Deconvolved.                   \t\n".format(cada, total, int(tiempo),int(tiempo/cada)*int(total-cada))
    block = int(round(barLength*progress))
    text = "\r[{0}] {1:3.1f}% {2}".format( "="*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(200*'')
    sys.stdout.flush()
    sys.stdout.write(text)
    sys.stdout.flush()

#--------------------------------------------------------------------------------------
def rebin(a, new_shape):
    M, N = a.shape
    m, n = new_shape
    if m<M:
        return a.reshape((m,int(M/m),n,int(N/n))).mean(3).mean(1)
    else:
        return np.repeat(np.repeat(a, m/M, axis=0), n/N, axis=1)

def rebin7factor(original, factor):
    # print(original.shape)
    nf, m, n = original.shape
    nueva = np.zeros((7,int(m*factor),int(factor*n)))
    for iframe in range(7):
            nueva[iframe,:,:] = rebin(original[iframe,:,:], (int(m*factor),int(factor*n)))
    return nueva

def rebin7factor_scans(original, factor):
    # print(original.shape)
    ns, nf, m, n = original.shape
    nueva = np.zeros((original.shape[0],7,int(m*factor),int(factor*n)))
    for iscan in range(original.shape[0]):
        for iframe in range(7):
                nueva[iscan,iframe,:,:] = rebin(original[iscan,iframe,:,:], (int(m*factor),int(factor*n)))
    return nueva

def rebinfactor_scans(original, factor):
    # print(original.shape)
    ns, m, n = original.shape
    nueva = np.zeros((original.shape[0],int(m*factor),int(factor*n)))
    for iscan in range(original.shape[0]):
        nueva[iscan,:,:] = rebin(original[iscan,:,:], (int(m*factor),int(factor*n)))
    return nueva        
            

#--------------------------------------------------------------------------------------
def writefits(name, d):
    io = fts.PrimaryHDU(d)
    io.writeto(name, overwrite=True)

#--------------------------------------------------------------------------------------
def buil7frame(array):
    if len(array.shape) == 3:
        
        # print('buil7frame =>',array.shape[0])
        if array.shape[0] > 7: 
            # New order: std
            stdarray = []
            for ii in range(array.shape[0]):
                stdarray.append(np.std(array[ii,200:-200,200:-200]))
            indices = np.argsort(stdarray)[::-1]
            array = array[indices,:,:]

        # The encdec can only use 7 frames
        narray =  np.zeros((7,array.shape[1],array.shape[2]))
        for ii in range(7):
            narray[ii,:,:] = array[ii%array.shape[0],:,:]

    if len(array.shape) == 4:
        narray =  np.zeros((array.shape[0],7,array.shape[2],array.shape[3]))

        if array.shape[1] > 7:
            print('red:: neural network : Taking best frames (7 of {0})'.format(array.shape[1]))

        for kk in range(array.shape[0]):

            # New order: std
            if array.shape[1] > 7:
                stdarray = []
                for jj in range(array.shape[1]):
                    stdarray.append(np.std(array[kk,jj,200:-200,200:-200]))
                indices = np.argsort(stdarray)[::-1]
                array[kk,:,:,:] = array[kk,indices,:,:]

            # Only take first 7 frames
            for ii in range(7):
                # print('buil7frame ==>',ii,ii%array.shape[1])
                narray[kk,ii,:,:] = array[kk,ii%array.shape[1],:,:]
    return narray

#--------------------------------------------------------------------------------------
def check_type(array):
    if len(array.shape) == 2:
        print('red:: neural network : data type - single frame')
        mytype = 0
        # Expand, repeat and expand
        array = np.expand_dims(array, axis=0)
        array = np.expand_dims(array, axis=0)

    elif len(array.shape) == 3:
        print('red:: neural network : data type - several frames')
        mytype = 1
        # Repeat and expand
        array = buil7frame(array)
        # array = rebin7factor(array,0.5)
        array = np.expand_dims(array, axis=0)

    elif len(array.shape) == 4:
        print('red:: neural network : data type - several scans')
        print('red:: neural network : Loading and preparing data')
        mytype = 2
        # Repeat 7 per scan
        array = buil7frame(array)
        # array = rebin7factor_scans(array,0.5)

    else:
        print('=> WARNING: Unknown format')
        mytype = 99
        sys.exit()

    return mytype, array

#--------------------------------------------------------------------------------------
class deep_mfbd(object):
    def __init__(self, plot_option, verbose, threads, mode, input_name):
        self.cuda = torch.cuda.is_available()
        self.n_frames = 7
        self.plot_option = plot_option
        self.depth = self.n_frames-1
        self.verbose = verbose
        self.num_threads = threads
        self.device = torch.device("cuda" if self.cuda else "cpu")
        warnings.filterwarnings("ignore")
        
        if self.verbose: print("=> Pytorch version {}".format(torch.__version__))
        if self.verbose: print("=> Cuda {}".format(self.cuda))
        if self.verbose: print("=> Device {}".format(self.device))
        if self.cuda:
            if self.verbose: print("=> GPU: {}".format(torch.cuda.get_device_name(0)))

        self.model = model.deconv_block(n_frames=self.n_frames).to(self.device)
        # self.model = nn.DataParallel(model)


        # Priority: command order:
        if mode == 'CHROMIS':
            self.checkpoint = 'encdec_network/CHROMIS.pth.tar'
        elif mode == 'CRISP':
            self.checkpoint = 'encdec_network/CRISP.pth.tar'
        elif mode is None:
            if self.verbose: print('=> Checking the instrumentation:')
            # Checking the instrumentation:
            myfile = fts.open(input_name)
            try:
                wave = float(myfile[0].header['WAVELNTH'])
            except:
                print('=> No wavelength information.')
                sys.exit()

            if self.verbose: print('=> WAVELNTH: {0}'.format(myfile[0].header['WAVELNTH']))

            # If it's less than 500 nm it's CHROMIS data, otherwise CRISP.
            wave = float(myfile[0].header['WAVELNTH'])
            if wave > 500.:
                self.checkpoint = 'encdec_network/CRISP.pth.tar'
                print('red:: neural network : CRISP data detected')
            if wave < 500.:
                self.checkpoint = 'encdec_network/CHROMIS.pth.tar'
                print('red:: neural network : CHROMIS data detected')
        else:
            print('=> Incorrect mode: write CRIP or CHROMIS')
            sys.exit()




        # self.checkpoint = 'trained_encdec/2019-06-10-19:56.pth.tar'

        if self.verbose: print("=> loading checkpoint '{}'".format(self.checkpoint))
        if (self.cuda):
            checkpoint = torch.load(self.checkpoint)
        else:
            checkpoint = torch.load(self.checkpoint, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint['state_dict'])        
        if self.verbose: print("=> loaded checkpoint '{}'".format(self.checkpoint))

        self.model.share_memory()
        # torch.set_num_threads(threads)
        # if self.verbose: print("=> # threads {}".format(torch.get_num_threads()))

    #--------------------------------------------------------------------------------------
    def test(self, input_name, output_name):

        self.model.eval()

        myfile = fts.open(input_name)
        if self.verbose: print("=> Shape {}".format(myfile[0].data.shape))



        myfile0 = myfile[0].data[:]
        # print(myfile0.shape)
        mytype, ims = check_type(myfile0)
        # mytype, ims = check_type(myfile[0].data)

        if mytype == 2:

            outfull = np.zeros((ims.shape[0],ims.shape[2],ims.shape[3]))
            print('red:: neural network : Estimating scans')
            start = time.time()
            for ik in range(ims.shape[0]):
                # print(ims[ik:ik+1,:,:,:].mean())
                normalization = 400./ims[ik:ik+1,:,:,:].mean()
                # data = torch.from_numpy((ims[ik:ik+1,:,:,:]*normalization / 1e3).astype('float32')).to(self.device)
                data = torch.from_numpy(( ims[ik:ik+1,:,:,:]*normalization / 1e3   ).astype('float32')).to(self.device)

                with torch.no_grad():
                    # if self.verbose: print('next(model.parameters()).is_cuda',next(self.model.parameters()).is_cuda)            
                    # print('=> Estimating scan {}'.format(ik))
                    # update_progress(float(ik)/ims.shape[0], ik,ims.shape[0])

                    out = self.model(data)              

                    final = time.time()-start
                    # print(final)
                    update_progress(float(ik+1)/ims.shape[0], ik+1,ims.shape[0],final)

                    # print('Elapsed time : {0:3.1f} ms'.format((time.time()-start)*1e3))
                    if self.verbose: print('Elapsed time : {0:3.3f} s'.format((time.time()-start)))

                    outfull[ik,:,:] = 1e3 * np.squeeze(out.to("cpu").data.numpy()) / normalization
                    if self.verbose: print('Out shape: {0}'.format(outfull.shape))
            
            print('red:: neural network : Saving data')
            # outfull = rebinfactor_scans(outfull,2.0)
            writefits(output_name,outfull)

                
            if self.plot_option is True:
                import matplotlib.pyplot as pl
                arcsec_per_px = 0.059

                for jj in range(outfull.shape[0]):
                    # print(jj)
                    pl.close('all')
                    fig, ax = pl.subplots(ncols=2, nrows=2, figsize=(10,10))
                    mini, maxi = np.min(outfull[jj,:,:]), np.max(outfull[jj,:,:])
                    ax[0,0].imshow(ims[jj,0,:,:], extent=(0,960*arcsec_per_px,0,960*arcsec_per_px),cmap='gray',vmin=mini,vmax=maxi)                
                    ax[0,1].imshow(outfull[jj,:,:], extent=(0,960*arcsec_per_px,0,960*arcsec_per_px),cmap='gray',vmin=mini,vmax=maxi)
                    ax[1,0].imshow(ims[jj,0,100:500,100:500], extent=(0,400*arcsec_per_px,0,400*arcsec_per_px),cmap='gray',vmin=mini,vmax=maxi)                
                    ax[1,1].imshow(outfull[jj,100:500,100:500], extent=(0,400*arcsec_per_px,0,400*arcsec_per_px),cmap='gray',vmin=mini,vmax=maxi)
                    ax[0,0].set_title('Frame')
                    ax[0,1].set_title('NN')
                    pl.savefig('test_encode_{0}.pdf'.format(jj))

        else:
            normalization = 400./ims.mean()
            data = torch.from_numpy((  ims*normalization / 1e3 ).astype('float32')).to(self.device)

            with torch.no_grad():
                # if self.verbose: print('next(model.parameters()).is_cuda',next(self.model.parameters()).is_cuda)            
                start = time.time()
                if self.verbose: print('=> Estimating the output...')
                out = self.model(data)              
                # print('Elapsed time : {0:3.1f} ms'.format((time.time()-start)*1e3))
                if self.verbose: print('Elapsed time : {0:3.3f} s'.format((time.time()-start)))

                out = 1e3 * np.squeeze(out.to("cpu").data.numpy())/normalization
                print('red:: neural network : Saving data')
                # out = rebin(out,(int(out.shape[0]*2),(int(out.shape[1]*2))))
                # ims = rebin7factor(ims[0,:,:,:],2)
                if self.verbose: print('Out shape:',out.shape)
                
                writefits(output_name,out)

                if self.plot_option is True:
                    import matplotlib.pyplot as pl
                    arcsec_per_px = 0.059

                    pl.close('all')
                    fig, ax = pl.subplots(ncols=2, nrows=2, figsize=(10,10))
                    mini, maxi = np.min(ims), np.max(ims)
                    maxi /= 1
                    # print(np.std(ims[0,0,100:500,100:500]),np.std(out[100:500,100:500]))
                    ax[0,0].imshow(ims[0,:,:], extent=(0,960*arcsec_per_px,0,960*arcsec_per_px),cmap='gray',vmin=mini,vmax=maxi)                
                    ax[0,1].imshow(out, extent=(0,960*arcsec_per_px,0,960*arcsec_per_px),cmap='gray',vmin=mini,vmax=maxi)
                    ax[1,0].imshow(ims[0,100:500,100:500], extent=(0,400*arcsec_per_px,0,400*arcsec_per_px),cmap='gray',vmin=mini,vmax=maxi)                
                    ax[1,1].imshow(out[100:500,100:500], extent=(0,400*arcsec_per_px,0,400*arcsec_per_px),cmap='gray',vmin=mini,vmax=maxi)
                    ax[0,0].set_title('Frame {0:3.1f}'.format(100*np.std(ims[0,100:500,100:500])/np.mean(ims[0,100:500,100:500])))
                    ax[0,1].set_title('NN {0:3.1f}'.format(100*np.std(out[100:500,100:500])/np.mean(out[100:500,100:500])))
                    # print(out.min())
                    pl.savefig('test_encode.pdf', bbox_inches='tight')
            
#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------
if (__name__ == '__main__'):

    # EXAMPLE: 
    # python encdec_sst.py -i /scratch/mats/2016.09.19/CRISP-testtt/tmp_out.fits -o salida.fits -p True -v True -m CRISP
    # python encdec_sst.py -i /scratch/mats/2016.09.19/CRISP-testtt/tmp_out_3.fits -o salida.fits -p True -v True -m CRISP
    # python encdec_sst.py -i /scratch/mats/2016.09.19/CHROMIS-jan19/collected/09:28:36/Chromis-N_2016-09-19T09:28:36_00064_12.00ms_G10.00_3934_3934_-391_raw.fits -o salida.fits -p True -v True -m CHROMIS
    
    # python encdec_sst.py -i /scratch/mats/2016.09.19/CHROMIS-jan19/collected/09:28:36/Chromis-N_2016-09-19T09:28:36_00047_12.00ms_G10.00_3934_3934_-1331_raw.fits -o salida.fits -p True -v True -m CHROMIS
    # python encdec_sst.py -i /scratch/mats/2016.09.19/CHROMIS-jan19/collected/09:28:36/Chromis-N_2016-09-19T09:28:36_00027_12.00ms_G10.00_3999_4000_+1258_raw.fits -o salida.fits -p True -v True -m CHROMIS

    # source activate pt_gpu
    # cd /scratch/carlos/GPUDEEPL/PYTORCH/learned_mfbd

    parser = argparse.ArgumentParser()
    # parser.add_argument("echo")
    parser.add_argument('-i','--input', help='input')
    parser.add_argument('-o','--out', help='out')
    parser.add_argument('-t','--threads', help='threads', default=1)
    parser.add_argument('-m','--mode', help='mode')
    parser.add_argument('-v','--verbose', help='verbose', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('-p','--plot', help='plot', default=False, type=lambda x: (str(x).lower() == 'true'))
    parsed = vars(parser.parse_args())
    # print(parsed)

    plot_option = parsed['plot']
    verbose = parsed['verbose']
    input_name = parsed['input']
    output_name = parsed['out']
    mode = parsed['mode']
    threads = int(parsed['threads'])

    deep_mfbd_network = deep_mfbd(plot_option, verbose, threads, mode, input_name)
    deep_mfbd_network.test(input_name, output_name)
