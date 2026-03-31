
import sys
sys.path.insert(1, '.type3detect./') # make sure to use the code in this repo

import matplotlib.dates as mdates
import datetime
import matplotlib.pyplot as plt
import numpy as np
import astropy.io.fits as fits
import scipy
from scipy import interpolate,optimize
from astropy.time import Time
import matplotlib as mpl
import os
# try to use the precise epoch
mpl.rcParams['date.epoch']='1970-01-01T00:00:00'
try:
    mdates.set_epoch('1970-01-01T00:00:00')
except:
    pass
from type3detect import radioTools as rt
from skimage.transform import probabilistic_hough_line

# frequency [MHz]


def read_fits(fname):
    
    hdu = fits.open(fname)
    dyspec = np.array(hdu[0].data)
    f_fits = np.array(hdu[1].data['FREQ'][0])
    t_fits = np.array(hdu[1].data['TIME'][0])
    return (dyspec,t_fits,f_fits,hdu)

def read_osra(fname): #fname is file full path name. e.g "/net/lyot/scratch3/vocks/OSRA/1998/CD_232/980820_232.roh"

    f1= 800.0 -np.array(range(256))*400./256.
    f2 = 400.0 -np.array(range(256))*200./255.
    f3 = 170.0 -np.array(range(256))*70./256.
    f4 = 100.0 -np.array(range(256))*60./255.
    f_fits = np.concatenate ((f1,f2,f3,f4))

    #get file size
    file_stats = os.stat(fname)
    print(file_stats) #prints out all parameters
    print(file_stats.st_size) #prints only the file size
    print(file_stats.st_size / 1040) # compare with (a=fstat(lun) & a1=a.size/1040 & na=long(a1)) from read_pots_roh.pro idl
    a1 = int( file_stats.st_size / 1040 + 0.5) #reason for adding  0.5 is for rounding up errors. recognise and interpret a1 as  floating

    #create numpy array
    t_fits = np.zeros(a1, dtype=object)
    dyspec=np.zeros((a1,1024), dtype=np.ubyte)
    file = open(fname, "rb")
    for i in range(a1):
        data_chunk = file.read(1040)
        np_data_chunk = np.frombuffer(data_chunk, dtype=np.uint8)
    
        year = int(np_data_chunk[0] / 16) * 10 + (np_data_chunk[0] & 15)
        if year > 50:
            year = year + 1900
        else:
            year = year + 2000

        # Extract time information
        month = int(np_data_chunk[1] / 16) * 10 + (np_data_chunk[1] & 15)
        day = int(np_data_chunk[2] / 16) * 10 + (np_data_chunk[2] & 15)
        hour = int(np_data_chunk[3] / 16) * 10 + (np_data_chunk[3] & 15)
        minute = int(np_data_chunk[4] / 16) * 10 + (np_data_chunk[4] & 15)
        second = int(np_data_chunk[5] / 16) * 10 + (np_data_chunk[5] & 15)
        microsecond = 100000 * np_data_chunk[6]

        print( datetime.datetime(year, month, day, hour, minute, second, microsecond))

        t_fits[i] = datetime.datetime(year, month, day, hour, minute, second, microsecond)

        # Only take the relevant part of the data chunk for f2
        dyspec[i,:] = np_data_chunk[16:] 


    close(file)
    return (dyspec,t_fits,f_fits)
#............................................................................ 
def read_osraf2(fname): #fname is file full path name. e.g "/net/lyot/scratch3/vocks/OSRA/1998/CD_232/980820_232.roh"
    f2 = 400.0 -np.array(range(256))*200./255.
    f_fits = f2

    # Simulate file size and stats
    file_stats = os.stat(fname)
    a1 = int(file_stats.st_size / 1040 + 0.5)
    
    # Create a dummy start time
    dummy_start = np.datetime64('2025-02-01T15:23:15.00')
    t_fits = dummy_start + np.linspace(0, 1, a1).astype('timedelta64[D]')

    # Initialize the spectrum array for f2 range only
    dyspec = np.zeros((a1, 256), dtype=np.ubyte)

    # Simulate reading the file
    file = open(fname, "rb")
    for i in range(a1):  # Line 53
        data_chunk = file.read(1040)  # Ensure this is indented
        np_data_chunk = np.frombuffer(data_chunk, dtype=np.uint8)

        year = int(np_data_chunk[0] / 16) * 10 + (np_data_chunk[0] & 15)
        if year > 50:
            year = year + 1900
        else:
            year = year + 2000
        # Extract time information
        month = int(np_data_chunk[1] / 16) * 10 + (np_data_chunk[1] & 15)
        day = int(np_data_chunk[2] / 16) * 10 + (np_data_chunk[2] & 15)
        hour = int(np_data_chunk[3] / 16) * 10 + (np_data_chunk[3] & 15)
        minute = int(np_data_chunk[4] / 16) * 10 + (np_data_chunk[4] & 15)
        second = int(np_data_chunk[5] / 16) * 10 + (np_data_chunk[5] & 15)
        microsecond = 100000 * np_data_chunk[6]
        t_fits[i] = datetime.datetime(year, month, day, hour, minute, second, microsecond)
        # Only take the relevant part of the data chunk for f2
        dyspec[i, :] = np_data_chunk[272: 528]  # Adjust the slice to match the f2 range

    file.close()
    return (dyspec,t_fits,f_fits)
    

#.....................................................................................................
def idx_val_pos(f_fits, target):
    return np.abs(np.array(f_fits)-target).argmin()

def cut_low(dyspec,f_fits,f_low_cut_val=21):
    idx_cut = idx_val_pos(f_fits, f_low_cut_val)
    dyspec=dyspec[:,idx_cut:]
    f_fits = f_fits[idx_cut:]
    return (dyspec,f_fits)
    
    
def preproc(dyspec,gauss_sigma=1.5, background_normalize=True):
    # const background removal and gaussian smooth
    
    if background_normalize:
        data_fits_new_tmp = (dyspec / np.nanmean(
                np.sort(dyspec, 0)[
                int(dyspec.shape[0] * 0.1):int(dyspec.shape[0] * 0.3), :], 0))-1
    else:
        data_fits_new_tmp = dyspec
    data_fits_new  = scipy.ndimage.gaussian_filter(data_fits_new_tmp,
        gauss_sigma, order=0, output=None,  cval=0.0, truncate=5.0,mode='nearest')
    return (data_fits_new_tmp,data_fits_new)
    
def preproc2(dyspec, gauss_sigma=(1.5, 0), background_normalize=True):
    if background_normalize:
        denom = np.nanmean(
            np.sort(dyspec, 0)[
            int(dyspec.shape[0] * 0.1):int(dyspec.shape[0] * 0.3), :], 0)
        
        # Replace zeros/NaNs in denominator to avoid divide-by-zero
        denom = np.where((denom == 0) | np.isnan(denom), np.nan, denom)
        
        data_fits_new_tmp = (dyspec / denom) - 1
        data_fits_new_tmp = np.nan_to_num(data_fits_new_tmp, nan=0.0)
    else:
        data_fits_new_tmp = dyspec


    # Fix: Pass sigma as a tuple (time_sigma, freq_sigma)
    # This ensures only the time coordinate (axis 0) is smoothed
    data_fits_new = scipy.ndimage.gaussian_filter(
        data_fits_new_tmp,
        sigma=gauss_sigma, 
        order=0, 
        output=None, 
        cval=0.0, 
        truncate=5.0, 
        mode='nearest'
    )
    return (data_fits_new_tmp, data_fits_new)

def binarization(data_fits_new,N_order=6,peak_r=0.99):
    # with high order local-max method 
    bmap = np.ones_like(data_fits_new)
    N_pad = N_order
    local_max_arr = np.pad(data_fits_new,((N_pad,N_pad),(0,0)))
    for idx in range(N_pad-1):
        bmap=bmap* ((peak_r*local_max_arr[N_pad+idx+1:-N_pad+idx+1,:]<
                            local_max_arr[N_pad+idx:-N_pad+idx,:]) & 
                    (peak_r*local_max_arr[N_pad-idx-1:-N_pad-idx-1,:]<
                            local_max_arr[N_pad-idx:-N_pad-idx,:]) )
    return bmap

def hough_detect(bmap,dyspec,threshold=50,line_gap=10,line_length=25,
            theta=np.linspace(np.pi/2-np.pi/8,np.pi/2-1/180*np.pi,300)):
    lines = probabilistic_hough_line(bmap, threshold=threshold,line_gap=line_gap,line_length=line_length,
                                 theta=theta)
    return lines

    
norm = np.linalg.norm
def point_to_line_distance(p1,p2,p3):
    d = np.abs(norm(np.cross(p2-p1, p1-p3)))/norm(p2-p1)
    return d

def line_grouping(lines,min_dist=3): # pix
    # group the detected lines into group in regard of events
    lines = sorted(lines, key=lambda i: i[0][1])
    line_sets = [[lines[0]]]
    for idx,line in enumerate(lines[0:-1]):
        (A,B),(C,D) = np.array([lines[idx], lines[idx+1] ])
        if np.min([point_to_line_distance(A,B,C),point_to_line_distance(A,B,D)])< 3:
            line_sets[len(line_sets)-1].append(lines[idx+1])
        else:
            line_sets.append([lines[idx+1]])
    
    return line_sets



def get_info_from_linegroupt_fits_cutout(line_sets,t_fits_cutout,f_fits):

    # mapping from t and f to index of x and y
    t_idx_arr = np.arange(0, t_fits_cutout.shape[0])
    f_idx_arr = np.arange(0, f_fits.shape[0])
    t_interf = interpolate.interp1d(t_idx_arr, t_fits_cutout)
    f_interf = interpolate.interp1d(f_idx_arr, f_fits)
    
    v_beam = []
    f_range_burst = []
    t_range_burst = []
    model_curve_set = []
    t_set_arr_set = []
    f_set_arr_set = []
    t_set_arr = []
    f_set_arr = []
    t_model_arr = []
    f_model_arr = []

    for lines in line_sets:
        if len(lines)==1:
            continue
        try:
            x_set=[]
            y_set=[]
            for line in lines:
                x_set.append(line[0][1])
                x_set.append(line[1][1])
                y_set.append(line[0][0])
                y_set.append(line[1][0])

            t_set_arr = (t_interf(x_set) - np.min(t_fits))*24*3600
            f_set_arr = f_interf(y_set)

            popt, pcov = optimize.curve_fit(rt.freq_drift_f_t,
                t_set_arr, f_set_arr, p0=(0.1,np.min(t_set_arr)-3./3600/24), method="lm")

            t_model_arr  = np.linspace(rt.freq_drift_t_f(np.min(f_set_arr),*popt) ,
                                       rt.freq_drift_t_f(np.max(f_set_arr),*popt),50) 
            f_model_arr = rt.freq_drift_f_t(t_model_arr,popt[0],popt[1])

            t_model_arr = t_model_arr/(24*3600)+np.min(t_fits)


            model_curve_set.append([t_model_arr,f_model_arr])
            t_range_burst.append( [rt.freq_drift_t_f(np.min(f_set_arr),*popt)[0]/(24*3600)+np.min(t_fits) ,
                                   rt.freq_drift_t_f(np.max(f_set_arr),*popt)[0]/(24*3600)+np.min(t_fits) ] )
            f_range_burst.append([np.min(f_set_arr),np.max(f_set_arr)])
            v_beam.append(popt[0])
            t_set_arr_set.append(t_set_arr)
            f_set_arr_set.append(f_set_arr)
        except:
            pass

    
    return (v_beam, f_range_burst, t_range_burst, model_curve_set,
            t_set_arr_set,f_set_arr_set,
           t_model_arr,f_model_arr)

def get_info_from_linegroup(line_sets,t_fits,f_fits):

    # mapping from t and f to index of x and y
    t_idx_arr = np.arange(0, t_fits.shape[0])
    f_idx_arr = np.arange(0, f_fits.shape[0])
    t_interf = interpolate.interp1d(t_idx_arr, t_fits)
    f_interf = interpolate.interp1d(f_idx_arr, f_fits)
    
    v_beam = []
    f_range_burst = []
    t_range_burst = []
    model_curve_set = []
    t_set_arr_set = []
    f_set_arr_set = []
    t_set_arr = []
    f_set_arr = []
    t_model_arr = []
    f_model_arr = []

    for lines in line_sets:
        if len(lines)==1:
            continue
        try:
            x_set=[]
            y_set=[]
            for line in lines:
                x_set.append(line[0][1])
                x_set.append(line[1][1])
                y_set.append(line[0][0])
                y_set.append(line[1][0])

            t_set_arr = (t_interf(x_set) - np.min(t_fits))*24*3600
            f_set_arr = f_interf(y_set)

            popt, pcov = optimize.curve_fit(rt.freq_drift_f_t,
                t_set_arr, f_set_arr, p0=(0.1,np.min(t_set_arr)-3./3600/24), method="lm")

            t_model_arr  = np.linspace(rt.freq_drift_t_f(np.min(f_set_arr),*popt) ,
                                       rt.freq_drift_t_f(np.max(f_set_arr),*popt),50) 
            f_model_arr = rt.freq_drift_f_t(t_model_arr,popt[0],popt[1])

            t_model_arr = t_model_arr/(24*3600)+np.min(t_fits)


            model_curve_set.append([t_model_arr,f_model_arr])
            t_range_burst.append( [rt.freq_drift_t_f(np.min(f_set_arr),*popt)[0]/(24*3600)+np.min(t_fits) ,
                                   rt.freq_drift_t_f(np.max(f_set_arr),*popt)[0]/(24*3600)+np.min(t_fits) ] )
            f_range_burst.append([np.min(f_set_arr),np.max(f_set_arr)])
            v_beam.append(popt[0])
            t_set_arr_set.append(t_set_arr)
            f_set_arr_set.append(f_set_arr)
        except:
            pass

    
    return (v_beam, f_range_burst, t_range_burst, model_curve_set,
            t_set_arr_set,f_set_arr_set,
           t_model_arr,f_model_arr)

def append_into_json(old_json, v_beam, f_range_burst, t_range_burst):
    
    event_detail = []
    for idx,v_cur in enumerate(v_beam):
        event_detail.append({
            'v_beam':v_cur,
            'freq_range':((f_range_burst[idx])),
            'time_range':((t_range_burst[idx])),
            'str_time':mdates.num2date(t_range_burst[idx][0]).strftime("%H:%M:%S")})
    
    old_json['event']={
        'detection': True,
        'type':'III',
        'detail': event_detail}
    
    return old_json
