import os.path as op
from nilearn import image, input_data, surface, signal
import pandas as pd
import numpy as np
include = [u'dvars',u'framewise_displacement', u'a_comp_cor_00', u'a_comp_cor_01', u'a_comp_cor_02', u'a_comp_cor_03',
       u'a_comp_cor_04', u'a_comp_cor_05', u'cosine00', u'cosine01', u'cosine02', u'cosine03', u'cosine04', u'cosine05', u'cosine06',u'cosine07', u'cosine08', u'cosine09', u'cosine10', u'cosine11', u'cosine12', u'cosine13', u'cosine14', u'cosine15', u'trans_x', u'trans_y', u'trans_z', u'rot_x', u'rot_y', u'rot_z']

def main(derivatives,
         ds='ds-02'):
    subjects = ['{:02d}'.format(i) for i in range(1, 16)]
    subjects.pop(3)


    mask_l = op.join(derivatives, 
                   ds,
                   'mean_mask_mni_space/_mask_stnl',
                   'sub-01_desc-stnl_mask_resampled_trans_merged_mean.nii.gz')
    mask_r = op.join(derivatives, 
                   ds,
                   'mean_mask_mni_space/_mask_stnr',
                   'sub-01_desc-stnr_mask_resampled_trans_merged_mean.nii.gz')

    mask_l = image.math_img('mask > 0.5', mask=mask_l)
    mask_r = image.math_img('mask > 0.5', mask=mask_r)

    #mask = image.load_img(mask)
    masker_l = input_data.NiftiMasker(mask_l, detrend=True, standardize=True)
    masker_r = input_data.NiftiMasker(mask_r, detrend=True, standardize=True)

    data_stn_l = []
    data_stn_r = []
    
    for subject in subjects:
        surfs = []
        data_stn_l = []
        data_stn_r = []

        if subject == '07':
            n_runs = 2
        else:
            n_runs = 3

        for run in range(1, n_runs+1):
            lh = surface.load_surf_data(op.join(derivatives,
                        ds,
                        'fmriprep/sub-{subject}/func/sub-{subject}_task-randomdotmotion_run-{run:02d}_space-fsaverage_hemi-L.func.gii'.format(**locals()))).T
            rh = surface.load_surf_data(op.join(derivatives,
                        ds,
                        'fmriprep/sub-{subject}/func/sub-{subject}_task-randomdotmotion_run-{run:02d}_space-fsaverage_hemi-R.func.gii'.format(**locals()))).T


            confounds = pd.read_table(op.join(derivatives, ds, 'fmriprep/sub-{subject}/func/sub-{subject}_task-randomdotmotion_run-{run:02d}_desc-confounds_regressors.tsv'.format(**locals())))
            confounds.fillna(method='bfill', inplace=True)    

            im = op.join(derivatives, ds, 'fmriprep/sub-{subject}/func/sub-{subject}_task-randomdotmotion_run-{run:02d}_space-T1w_desc-preproc_bold.nii.gz'.format(**locals()))

            surf = np.concatenate((lh, rh), 1)
            surf = signal.clean(surf, standardize=True, confounds=confounds[include].values)

            surfs.append(surf)

            d = masker_l.fit_transform(im, confounds=confounds[include].values)
            data_stn_l.append(d)

            d = masker_r.fit_transform(im, confounds=confounds[include].values)
            data_stn_r.append(d)

        surfs = np.array(surfs)
        data_stn_l = np.array(data_stn_l)
        data_stn_r = np.array(data_stn_r)

        rs_l = np.array([data_stn_l[i].T.dot(surfs[i]) / data_stn_l.shape[1] for i in range(n_runs)])
        rs_r = np.array([data_stn_r[i].T.dot(surfs[i]) / data_stn_r.shape[1] for i in range(n_runs)])

        np.savez_compressed(op.join(derivatives, ds, 'zooi', 'sub-{subject}_surfs.npz'.format(**locals())), surfs)
        np.savez_compressed(op.join(derivatives, ds, 'zooi', 'sub-{subject}_stn_l.npz'.format(**locals())), data_stn_l)
        np.savez_compressed(op.join(derivatives, ds, 'zooi', 'sub-{subject}_stn_r.npz'.format(**locals())), data_stn_r)
        np.savez_compressed(op.join(derivatives, ds, 'zooi', 'sub-{subject}_rs_l.npz'.format(**locals())), rs_l)
        np.savez_compressed(op.join(derivatives, ds, 'zooi', 'sub-{subject}_rs_r.npz'.format(**locals())), rs_r)


if __name__ == '__main__':
    main('/home/shared/2018/subcortex/bias_task')

