import numpy as np
import os.path as op
derivatives = '/home/shared/2018/subcortex/bias_task'
ds = 'ds-02'

embedding = np.load('/home/shared/2018/subcortex/bias_task/ds-02/zooi/embedding.npy').squeeze()


quantiles = np.quantile(embedding, [0.33, 0.67])

mask0 = embedding < quantiles[0]
mask2 = embedding > quantiles[1]
mask1 = 1 - mask0 - mask2

masks = np.array([mask0, mask1, mask2])

subjects = ['{:02d}'.format(i) for i in range(1, 16)]
subjects.pop(3)
subjects.pop(0)

subject_rs = []

for subject in subjects:
    print(subject)
    stn_l = np.load(op.join(derivatives,
                         ds,
                         'surface_correlations',
                         'sub-{subject}_stn_l.npz'.format(subject=subject)))['arr_0']

    surf = np.load(op.join(derivatives,
                         ds,
                         'surface_correlations',
                         'sub-{subject}_surfs.npz'.format(subject=subject)))['arr_0']


    rs = []
    for run in range(len(stn_l)):
       tc = masks.dot(stn_l[run].T).T
       tc /= tc.std(0)
       rs.append(tc.T.dot(surf[run]) / len(tc))

    rs = np.mean(rs,0)

    subject_rs.append(rs)
