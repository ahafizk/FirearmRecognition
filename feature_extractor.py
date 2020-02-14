import numpy as np
from scipy.stats import entropy
from scipy import signal
from math import log
from filter import *
'''
Time Domain Features:
spectral_centroid()
mean()
std()
max()
min()
energy()
variance()
avg_resultant_magnitude()
entropy()

Frequency Domain Features:

get_max_fft_energy()
frequency_domain_entropy()

spectral_centroid() --need to check this function


'''

def sma(x, y, z):
    '''calculate the signal magnitude area or SMA'''
    # sum = np.mean(np.abs(x)+np.abs(y)+np.abs(z))
    # length = len(x)
    return [np.mean(np.abs(x)+np.abs(y)+np.abs(z))]

def mean(x,y,z):
    '''calculate the mean features of x, y and z'''
    return [np.mean(x),np.mean(y),np.mean(z)]

def std(x,y,z):
    '''calculate standard deviation of x, y, z'''
    return [np.std(x),np.std(y),np.std(z)]

def t_max(x,y,z):
    return [np.max(x), np.max(y),np.max(z)]

def t_min(x,y,z):
    return [np.min(x), np.min(y), np.min(z)]

def mad(x,y,z):
    return [np.median(x),np.median(y),np.median(z)]

def energy(x,y,z):
    '''Energy measure. Sum of the squares divided by the number of values'''
    length = len(x)
    xx = np.mean(np.square(x))

    yy = np.mean(np.square(y))

    zz = np.mean(np.square(z))

    return [xx,yy,zz]

def varinace(x,y,z):
    return [np.var(x),np.var(y),np.var(z)]


def avg_resultant_magnitude(x,y,z):
    xx = np.square(x)
    yy = np.square(y)
    zz = np.square(z)
    return [np.mean(np.sqrt(xx+yy+zz))]


def entropy( arr):
    '''
    calculate entropy of an array
    Basically time domain signals entropy calculated.
    '''
    disct_arr = np.unique(arr)

    n_labels = len(disct_arr)

    if n_labels <= 1:
        return 0

    counts = np.bincount(arr)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

    # Compute standard entropy.
    for i in probs:
        ent -= i * log(i, base=n_classes)
    return ent

#--- below are frequency domain features ----


def spectral_centroid(wavedata, window_size, sample_rate):
    '''
    need to check this function. stft was removed from scipy.signal package.
    :param wavedata:
    :param window_size:
    :param sample_rate:
    :return:
    '''
    magnitude_spectrum = signal.stft(wavedata, window_size)

    timebins, freqbins = np.shape(magnitude_spectrum)

    # when do these blocks begin (time in seconds)?
    timestamps = (np.arange(0, timebins - 1) * (timebins / float(sample_rate)))

    sc = []

    for t in range(timebins - 1):
        power_spectrum = np.abs(magnitude_spectrum[t]) ** 2

        sc_t = np.sum(power_spectrum * np.arange(1, freqbins + 1)) / np.sum(power_spectrum)

        sc.append(sc_t)

    sc = np.asarray(sc)
    sc = np.nan_to_num(sc)

    return sc, np.asarray(timestamps)

def f_energy(x,y,z):
    length = len(x)
    # p = (np.abs(np.fft.rfft(x)))
    xx = np.mean(np.square(np.abs(np.fft.rfft(x,len(x)))))
    # print xx
    yy = np.mean(np.square(np.abs(np.fft.rfft(y,len(y)))))
    zz = np.mean(np.square(np.abs(np.fft.rfft(z,len(z)))))
    return [xx,yy,zz]

def get_max_fft_energy(data):
    p = (np.abs(np.fft.rfft(data)))
    e = np.sum(np.square(p)) / len(data)
    max_mag = np.max(p)
    return max_mag, e

def frequency_domain_entropy(x,y,z):
    '''calculate frequency domain entropy with the help of scipy.stats entropy function
    (Power) Spectral Entropy --calculated in this way
        http://stackoverflow.com/questions/30418391/what-is-frequency-domain-entropy-in-fft-result-and-how-to-calculate-it
    '''
    fx = np.fft.rfft(x,n=len(x),norm='ortho')
    fy = np.fft.rfft(y,n=len(y),norm='ortho')
    fz = np.fft.rfft(z,n=len(z),norm='ortho')
    px = np.square(fx)/len(fx)
    py = np.square(fy)/len(fy)
    pz = np.square(fz)/len(fz)
    px_n = px/np.sum(px)
    py_n = py/np.sum(py)
    pz_n = pz/np.sum(pz)

    return entropy(px_n),entropy(py_n),entropy(pz_n)

# see this paper Comparative study on classifying human activities with miniature inertial
# and magnetic sensors for
def get_features(x,y,z):
    '''
    1    tBodyAcc - Mean - 1
2    tBodyAcc - Mean - 2
3    tBodyAcc - Mean - 3
4    tBodyAcc - STD - 1
5    tBodyAcc - STD - 2
6    tBodyAcc - STD - 3
7    tBodyAcc - Mad - 1
8    tBodyAcc - Mad - 2
9    tBodyAcc - Mad - 3
10    tBodyAcc - Max - 1
11    tBodyAcc - Max - 2
12    tBodyAcc - Max - 3
13    tBodyAcc - Min - 1
14    tBodyAcc - Min - 2
15    tBodyAcc - Min - 3
16    tBodyAcc - SMA - 1
17    tBodyAcc - Energy - 1
18    tBodyAcc - Energy - 2
19    tBodyAcc - Energy - 3

282,283,284 - frequency domain energy
    :param x:
    :param y:
    :param z:
    :return:
    '''
    fet =[]
    fet.extend(mean(x,y,z))  # 3


    fet.extend(std(x,y,z))   # 3


    fet.extend(mad(x,y,z))   # 3


    fet.extend(t_max(x,y,z))   # 3


    fet.extend(t_min(x,y,z))   # 3


    fet.extend(sma(x,y,z)) # 1


    fet.extend(energy(x,y,z)) # 3



    fet.extend(f_energy(x, y, z)) # 3

    return fet

def filter_data(x,fs,cutoff):
    fsignal = signal.medfilt(x, kernel_size=5)
    lp = LowPassFilter()
    fsignal = lp.lowpass_filter(fsignal, fs, cutoff, order=3)
    return fsignal

if __name__=='__main__':
    #example of entropy calculation sigma * np.random.randn(...) + mu
    ex,ey,ez = frequency_domain_entropy(np.random.randn(50),np.random.randn(50),np.random.randn(50))
