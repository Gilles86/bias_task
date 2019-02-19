import numpy as np
import os.path as op
from nilearn import image, input_data

derivatives = '/home/shared/2018/subcortex/bias_task'
ds = 'ds-02'

# Make maskers
mask_l = op.join(derivatives,
                 ds,
                 'mean_mask_mni_space',
                 '_mask_stnl',
                 'sub-01_desc-stnl_mask_resampled_trans_merged_mean.nii.gz')

mask_r = op.join(derivatives,
                 ds,
                 'mean_mask_mni_space',
                 '_mask_stnr',
                 'sub-01_desc-stnr_mask_resampled_trans_merged_mean.nii.gz')

mask_l = image.math_img('mask > 0.3', mask=mask_l)
mask_r = image.math_img('mask > 0.3', mask=mask_r)

tmp = op.join(derivatives,
              ds,
              'fmriprep',
              'sub-01',
              'sub-01_task-randomdotmotion_run-01_space-T1w_desc-preproc_bold.nii.gz')

masker_l = input_data.NiftiMasker(mask_l)
masker_l.fit(tmp)
masker_r = input_data.NiftiMasker(mask_r)
masker_r.fit(tmp)


# LEFT
# Make left correlations
embedding = masker_l.transform(op.join(derivatives,
                                     ds,
                                     'embedding',
                                     'group_desc-stnl_embedding.nii.gz'))
                                     
quantiles_l = np.quantile(embedding, [0.33, 0.67])

mask0 = embedding < quantiles_l[0]
mask2 = embedding > quantiles_l[1]
mask1 = 1 - mask0 - mask2

masks = np.array([mask0, mask1, mask2])

subjects = ['{:02d}'.format(i) for i in range(1, 16)]
subjects.pop(3)
subjects.pop(0)

subject_rs = []

for subject in subjects:
    print('STN_L ', subject)
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

subject_rs = np.array(subject_rs)

np.save(op.join(derivatives,
                ds,
                'embedding',
                'group_correlations_embedding_stnl.npy'),
        subject_rs)

# RIGHT
embedding = masker_r.transform(op.join(derivatives,
                                     ds,
                                     'embedding',
                                     'group_desc-stnr_embedding.nii.gz'))
                                     
quantiles_r = np.quantile(embedding, [0.33, 0.67])

mask0 = embedding < quantiles_r[0]
mask2 = embedding > quantiles_r[1]
mask1 = 1 - mask0 - mask2

masks = np.array([mask0, mask1, mask2])

subjects = ['{:02d}'.format(i) for i in range(1, 16)]
subjects.pop(3)
subjects.pop(0)

subject_rs = []

for subject in subjects:
    print('STN_R ', subject)
    stn_r = np.load(op.join(derivatives,
                         ds,
                         'surface_correlations',
                         'sub-{subject}_stn_r.npz'.format(subject=subject)))['arr_0']

    surf = np.load(op.join(derivatives,
                         ds,
                         'surface_correlations',
                         'sub-{subject}_surfs.npz'.format(subject=subject)))['arr_0']


    rs = []
    for run in range(len(stn_r)):
       tc = masks.dot(stn_r[run].T).T
       tc /= tc.std(0)
       rs.append(tc.T.dot(surf[run]) / len(tc))

    rs = np.mean(rs,0)

    subject_rs.append(rs)

subject_rs = np.array(subject_rs)

np.save(op.join(derivatives,
                ds,
                'embedding',
                'group_correlations_embedding_stnr.npy'),
        subject_rs)
