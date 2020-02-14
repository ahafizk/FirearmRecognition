__author__ = 'hafiz'
import math
import numpy as np
def get_numframes(siglen,frame_length, step):
    slen = int(siglen)
    frame_length = round(frame_length)
    step = round(step)
    if slen <= frame_length:
        num_frames = 1
    else:
        num_frames = 1 + math.ceil((1.0 * slen - frame_length) / step)
    return  num_frames

def generate_frames_1D(signal,frame_length, step):
    siglen = len(signal)
    num_frames = int(get_numframes(siglen, frame_length,step))
    # print num_frames
    #add padding into rest of the frames when partial signal exists
    pad_length = int((num_frames - 1) * step + frame_length)
    # print pad_length
    # print siglen
    psignal = np.concatenate((signal,np.zeros((pad_length-siglen,))))
    # print psignal
    #generate index for all the frames using first frames signals indices
    first_frame_index_array = np.tile(np.arange(0, frame_length), (num_frames, 1))
    other_frame_idices = np.arange(0, num_frames * step, step)
    other_frame_idices_array = np.tile(other_frame_idices, (frame_length, 1)).T
    # get the indices of all frames with respect to signal's corresponding positions
    indices = np.array(first_frame_index_array + other_frame_idices_array, dtype=np.int32)
    # create frames using indices
    frames = psignal[indices]
    return frames

def generate_frames_2D(signal,frame_length, step):
    '''
    assume signal is a 2D array and
    :param signal: 2D signal array
    :param frame_length: lenght of each frame to be generated
    :param step: overlap size of the frames
    :return: created frames
    '''

    row,col = signal.shape
    siglen = col
    num_frames = get_numframes(siglen, frame_length,step)
    #add padding into rest of the frames when partial signal exists
    pad_length = int((num_frames - 1) * step + frame_length)
    psignal = np.concatenate((signal,np.zeros((row,pad_length-siglen))),axis=1)

    #generate index for all the frames using first frames signals indices
    first_frame_index_array = np.tile(np.arange(0, frame_length), (num_frames, 1))
    other_frame_idices = np.arange(0, num_frames * step, step)
    other_frame_idices_array = np.tile(other_frame_idices, (frame_length,  1)).T
    # get the indices of all frames with respect to signal's corresponding positions
    indices = np.array(first_frame_index_array + other_frame_idices_array, dtype=np.int32)
    # create frames using indices
    # print indices
    frames = (psignal.T)[indices,:]
    # frames = psignal[indices]
    return frames

if __name__=='__main__':
    sig = np.array([1,2,3,4,5,6,7,8,9])
    sig_2d = np.array([[1,2,3,4,5,6,7,8,9],[9,8,7,6,5,4,3,2,1],[1,4,7,2,5,8,3,6,9]])
    frame_length = 4
    step = 2
    print generate_frames_1D(sig,frame_length,step)
    frames = generate_frames_2D(sig_2d,frame_length,step)
    print frames[0].T
    print frames[1].Ts