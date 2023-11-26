import numpy as np
import pandas as pd
import numpy.random as npr
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io
import scipy.signal
import torch
from PIL import Image
from torchvision.transforms import ToTensor
import os
import math


class retinotopy_analysis:

    def get_data_location(path, file_num):
        '''
        creates the full filename given file path  
        This is a helper function to use in load_dat.
        
        out = ath_azimuth_LR, path_azimuth_RL, path_elevation_UD, path_elevation_DU
        
        With out default to: 
            out = dict(dtype=dtype, shape = shape, fnum = None)
        '''
        #file_name = 'Frames_1_640_540_uint16_000{}.dat'.format(file_num)
        path_azimuth_LR = path + 'Frames_1_640_540_uint16_000{}.dat'.format(file_num)
        path_azimuth_RL = path + 'Frames_1_640_540_uint16_000{}.dat'.format(file_num + 1)
        path_elevation_UD = path + 'Frames_1_640_540_uint16_000{}.dat'.format(file_num + 2)
        path_elevation_DU = path + 'Frames_1_640_540_uint16_000{}.dat'.format(file_num + 3)
        return path_azimuth_LR, path_azimuth_RL, path_elevation_UD, path_elevation_DU

    def _parse_binary_fname(fname,lastidx=None, dtype = 'uint16', shape = None, sep = '_'):
        '''
        Gets the data type and the shape from the filename 
        This is a helper function to use in load_dat.
        
        out = _parse_binary_fname(fname)
        
        With out default to: 
            out = dict(dtype=dtype, shape = shape, fnum = None)
        '''
        fn = os.path.splitext(os.path.basename(fname))[0]
        fnsplit = fn.split(sep)
        fnum = None
        if lastidx is None:
            # find the datatype first (that is the first dtype string from last)
            lastidx = -1
            idx = np.where([not f.isnumeric() for f in fnsplit])[0]
            for i in idx[::-1]:
                try:
                    dtype = np.dtype(fnsplit[i])
                    lastidx = i
                except TypeError:
                    pass
        if dtype is None:
            dtype = np.dtype(fnsplit[lastidx])
        # further split in those before and after lastidx
        before = [f for f in fnsplit[:lastidx] if f.isdigit()]
        after = [f for f in fnsplit[lastidx:] if f.isdigit()]
        if shape is None:
            # then the shape are the last 3
            shape = [int(t) for t in before[-3:]]
        if len(after)>0:
            fnum = [int(t) for t in after]
        return dtype,shape,fnum

    def load_dat(filename,
                nframes = None,
                offset = 0,
                shape = None,
                dtype='uint16'): 
        '''
        Loads frames from a binary file.
        
        Inputs:
            filename (str)       : fileformat convention, file ends in _NCHANNELS_H_W_DTYPE.dat
            nframes (int)        : number of frames to read (default is None: the entire file)
            offset (int)         : offset frame number (default 0)
            shape (list|tuple)   : dimensions (NCHANNELS, HEIGHT, WIDTH) default is None
            dtype (str)          : datatype (default uint16) 
        Returns:
            An array with size (NFRAMES,NCHANNELS, HEIGHT, WIDTH).

        Example:
            dat = load_dat(filename)
            
        ''' 
        if not os.path.isfile(filename):
            raise OSError('File {0} not found.'.format(filename))
        if shape is None or dtype is None: # try to get it from the filename
            dtype,shape,_ = retinotopy_analysis._parse_binary_fname(filename,
                                                shape = shape,
                                                dtype = dtype)
        if type(dtype) is str:
            dt = np.dtype(dtype)
        else:
            dt = dtype

        if nframes is None:
            # Get the number of samples from the file size
            nframes = int(os.path.getsize(filename)/(np.prod(shape)*dt.itemsize))
        framesize = int(np.prod(shape))

        offset = int(offset)
        with open(filename,'rb') as fd:
            fd.seek(offset*framesize*int(dt.itemsize))
            buf = np.fromfile(fd,dtype = dt, count=framesize*nframes)
        buf = buf.reshape((-1,*shape),
                        order='C')
        return buf

    def create_tensor_from_dat(filename):
        '''
            Loads binary data and converts it to a pytorch tensor for easy use.
            
            Inputs:
                filename (str)       : fileformat convention, file ends in _NCHANNELS_H_W_DTYPE.dat
            Returns:
                A tensor with size (NFRAMES, HEIGHT, WIDTH).

            Example:
                tensor_azimuth1 = create_tensor_from_dat(path_azimuth_LR)
                
            ''' 
        np_stack = retinotopy_analysis.load_dat(filename)
        np_stack = np.array(np_stack, dtype=np.float32)
        np_stack.transpose(0,1,3,2)
        tensor_stack = torch.from_numpy(np_stack).unsqueeze(1).float()
        tensor_470 = tensor_stack[:,0,0,150:450,150:450]
        return tensor_470

    def corrected_az_elev(tensor_azimuthLR, tensor_azimuthRL, tensor_elevationUD, tensor_elevationDU):
        tensor_azimuth = (tensor_azimuthLR - tensor_azimuthRL)
        tensor_elevation = (tensor_elevationUD - tensor_elevationDU)
        return tensor_azimuth, tensor_elevation
    
    def create_stimulus_corrected_movie_pt(tensor, baseline_start, baseline_end):
        '''
            Correct the movie by subtracting the mean of frames before the start of stimulation in PyTorch.

            Inputs:
                tensor (torch.Tensor): A 3D tensor representing the movie data (time x width x height).
                baseline_start (int): The start frame of the baseline period.
                baseline_end (int): The end frame of the baseline period.

            Returns:
                torch.Tensor: A 3D tensor representing the corrected movie.
            '''
        # Calculate the baseline mean
        baseline_mean = tensor[baseline_start:baseline_end, :, :].mean(dim=0)

        # Subtract the baseline mean from all frames
        corrected_movie = tensor - baseline_mean

        return corrected_movie

    def extract_first_harmonic_phase(data, peak_frequency, bandwidth, sample_rate):
        '''
            Extract the phase of the first harmonic component within a specific frequency band.

            Inputs:
                data (torch.Tensor): A 3D tensor representing the data (time x width x height).
                peak_frequency (float): Peak frequency for the Fourier transform.
                bandwidth (tuple): The frequency band (low, high).
                sample_rate (float): The sampling rate of the data.

            Returns:
                torch.Tensor: A 2D tensor representing the phase map.
        '''
        # Fourier transform
        ft_data = torch.fft.fftn(data, dim=[0])

        # Frequency resolution
        freq_resolution = torch.fft.fftfreq(data.size(0), d=1.0/sample_rate)

        # Find indices within the specified bandwidth
        bandwidth_indices = (freq_resolution >= bandwidth[0]) & (freq_resolution <= bandwidth[1])

        # Extract the first harmonic within the bandwidth
        first_harmonic_band = ft_data[bandwidth_indices, :, :]

        # Calculate the phase of the first harmonic
        phase_map = torch.angle(first_harmonic_band)

        return phase_map

    def get_phase_map(corrected_movie, sample_rate, peak_freq):
        calcium_data = corrected_movie
        #sample_rate = 30.0
        #peak_freq = 0.043
        min_band = peak_freq - peak_freq/2
        max_band = peak_freq + peak_freq/2
        phase_map = retinotopy_analysis.extract_first_harmonic_phase(calcium_data, peak_freq, (min_band, max_band), sample_rate)
        return phase_map
    
    def visualSignMap(phasemap1, phasemap2):
        '''
        calculate visual sign map from two orthogonally oriented phase maps
        '''

        if phasemap1.shape != phasemap2.shape:
            raise LookupError("'phasemap1' and 'phasemap2' should have same size.")

        gradmap1 = np.gradient(phasemap1)
        gradmap2 = np.gradient(phasemap2)

        # gradmap1 = ni.filters.median_filter(gradmap1,100.)
        # gradmap2 = ni.filters.median_filter(gradmap2,100.)

        graddir1 = np.zeros(np.shape(gradmap1[0]))
        # gradmag1 = np.zeros(np.shape(gradmap1[0]))

        graddir2 = np.zeros(np.shape(gradmap2[0]))
        # gradmag2 = np.zeros(np.shape(gradmap2[0]))

        for i in range(phasemap1.shape[0]):
            for j in range(phasemap2.shape[1]):
                graddir1[i, j] = math.atan2(gradmap1[1][i, j], gradmap1[0][i, j])
                graddir2[i, j] = math.atan2(gradmap2[1][i, j], gradmap2[0][i, j])

                # gradmag1[i,j] = np.sqrt((gradmap1[1][i,j]**2)+(gradmap1[0][i,j]**2))
                # gradmag2[i,j] = np.sqrt((gradmap2[1][i,j]**2)+(gradmap2[0][i,j]**2))

        vdiff = np.multiply(np.exp(1j * graddir1), np.exp(-1j * graddir2))

        areamap = np.sin(np.angle(vdiff))

        return areamap