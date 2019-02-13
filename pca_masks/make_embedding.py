import os.path as op
from nilearn import image, input_data, surface, signal
import pandas as pd
import numpy as np

#def main(derivatives,
         #ds='ds-02'):

derivatives = '/home/shared/2018/subcortex/bias_task/'
ds = 'ds-02'

subjects = ['{:02d}'.format(i) for i in range(1, 16)]
subjects.pop(3)
subjects.pop(0)

mean_rs = []

for subject in subjects:
    rs = np.load(op.join(derivatives,
                         ds,
                         'surface_correlations',
                         'sub-{subject}_rs_l.npz'.format(subject=subject)))['arr_0']

    print(subject, rs.shape)
    subj_r = rs.mean(0)
    mean_rs.append(subj_r)

all_mean_rs = np.mean(mean_rs, 0)








if __name__ == '__main__':
    main('/home/shared/2018/subcortex/bias_task')

