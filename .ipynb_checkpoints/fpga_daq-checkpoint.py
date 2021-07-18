import numpy as np
from numpy import random as rand
import matplotlib 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#The convolution operator models the effect of a linear time-invariant 
#system on a signal. Another way to think baout this is that convolution 
#is a measurement of effect of one signal on the other signal.

def running_mean(x, N):
    return np.convolve(x, np.ones((N,))/N)[(N-1):]

# This function takes the raw data from the ROACH-FPGA and extracts correlated 
# data (i.e. reflectometry measurement).
# The inputs are plate (.txt file), material (.txt file), and N_MULT (an integer) 
# which is the multiplication factor for your given setup. 

def refl_save(plate,sample,N_MULT):
    
    PLATE = np.loadtxt(plate,skiprows=1)
    SAMPL = np.loadtxt(sample,skiprows=1)
    L_MEAN = 1
    N_INDIV = 2 #Specifies appropriate column to read over in the txt file from the ROACH-FPGA

    line_size = np.size(PLATE[0])
    nsamp =  np.size(PLATE,0) 

    # Initialize arrays
    arr_f = np.zeros(nsamp)
    arr_index_diff = np.zeros(nsamp)
    
    amp_AB=np.zeros(nsamp)
    amp_AB_SAMPL = np.zeros(nsamp)
    
    
    amp_AA=np.zeros(nsamp)
    amp_BB=np.zeros(nsamp) 
    
    amp_AA_SAMPL = np.zeros(nsamp)
    amp_BB_SAMPL= np.zeros(nsamp)
    
    
    phase=np.zeros(nsamp)
    phase_SAMPL=np.zeros(nsamp)
    amp_var=np.zeros(nsamp)
    amp_var_SAMPL = np.zeros(nsamp)
    
    amp_diff_var = np.zeros(nsamp)
    amp_diff_phase=np.zeros(nsamp)
    amp_diff_AB = np.zeros(nsamp)
    
    amp_R_var = np.zeros(nsamp)
    amp_R_cross = np.zeros(nsamp)

    amp_R_receive = np.zeros(nsamp)
    amp_p_cross = np.zeros(nsamp)

    # Define indices for reading the .txt file 
    i_AA_begin = int(N_INDIV + (1-1)*(line_size-N_INDIV)/4)
    i_AA_end= int(N_INDIV + (2-1)*(line_size-N_INDIV)/4) -1
    i_BB_begin = int(N_INDIV + (2-1)*(line_size-N_INDIV)/4)
    i_BB_end= int(N_INDIV + (3-1)*(line_size-N_INDIV)/4) -1
    i_AB_begin = int(N_INDIV + (3-1)*(line_size-N_INDIV)/4)
    i_AB_end= int(N_INDIV + (4-1)*(line_size-N_INDIV)/4) -1
    i_phase_begin = int(N_INDIV + (4-1)*(line_size-N_INDIV)/4)
    i_phase_end= int(N_INDIV + (5-1)*(line_size-N_INDIV)/4) -1

    i=int(0)
    replacement_val = 1

    while (i < nsamp):

        arr_f[i] = PLATE[i][0]
        index_signal = PLATE[i][1] # use same index singal for both datasets.  
        arr_index_diff[i]=int(abs(PLATE[i][1]-SAMPL[i][1]))
        arr_AA = np.array(running_mean(PLATE[i][i_AA_begin : i_AA_end],L_MEAN))
        arr_BB = np.array(running_mean(PLATE[i][i_BB_begin : i_BB_end],L_MEAN))        
        arr_AB = np.array(running_mean(PLATE[i][i_AB_begin : i_AB_end],L_MEAN))
        arr_AA_SAMPL = np.array(running_mean(SAMPL[i][i_AA_begin : i_AA_end],L_MEAN))
        arr_BB_SAMPL = np.array(running_mean(SAMPL[i][i_BB_begin : i_BB_end],L_MEAN))
        arr_AB_SAMPL = np.array(running_mean(SAMPL[i][i_AB_begin : i_AB_end],L_MEAN))

        arr_AA[np.abs(arr_AA)==0.]=replacement_val
        arr_BB[np.abs(arr_BB)==0.]=replacement_val
        arr_AB[np.abs(arr_AB)==0.]=replacement_val
        arr_AA_SAMPL[np.abs(arr_AA_SAMPL)==0.]=replacement_val
        arr_BB_SAMPL[np.abs(arr_BB_SAMPL)==0.]=replacement_val
        arr_AB_SAMPL[np.abs(arr_AB_SAMPL)==0.]=replacement_val

        arr_phase = np.array( PLATE[i][i_phase_begin : i_phase_end] ) 
        arr_phase_SAMPL = np.array( SAMPL[i][i_phase_begin : i_phase_end] ) 
        
        #Convert signal to power. I also divide the sample's cross-correlation (AB) 
        # by its auto-correlation (AA) to remove any noise from the electronics since AA
        # is just the auto-correlation of the receiver. 
        
        arr_p_cross=(arr_phase_SAMPL-arr_phase) # Reflected phase
        arr_R_cross=np.divide((arr_AB_SAMPL/arr_AA_SAMPL),(arr_AB/arr_AA)) # Cross correlated signal
        arr_R_receive=np.divide(arr_BB_SAMPL,arr_BB) # Cross correlated signal at receiver
        arr_R_source =np.divide(arr_AA_SAMPL,arr_AA) # Cross correlated signal at soure

        n_channels = np.size(arr_AB)
        
        amp_AB[i] = np.power(arr_AB[int(n_channels/2)], 1)
        amp_AB_SAMPL[i] = np.power(arr_AB_SAMPL[int(n_channels/2)], 1)
        amp_AA[i] = arr_AA[int(n_channels/2)]
        amp_AA_SAMPL[i] = arr_AA_SAMPL[int(n_channels/2)]
        amp_BB[i] = arr_BB[int(n_channels/2)]
        amp_BB_SAMPL[i] = arr_BB_SAMPL[int(n_channels/2)]
        
        phase[i] = np.remainder(arr_phase[int(n_channels/2)],360.)
        phase_SAMPL[i] = arr_phase_SAMPL[int(n_channels/2)]
        amp_p_cross[i] = arr_p_cross[int(n_channels/2)]
        amp_R_cross[i] = arr_R_cross[int(n_channels/2)]
        i = i+1

    ARR_Y = np.power(amp_R_cross[1:],2)
    arr_f = (N_MULT/1000.)*arr_f[1:]

    return arr_f,ARR_Y