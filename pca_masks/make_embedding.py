import os.path as op
from nilearn import image, input_data, surface, signal
import pandas as pd
import numpy as np
from sklearn import manifold

#def main(derivatives,
         #ds='ds-02'):

derivatives = '/home/shared/2018/subcortex/bias_task/'
ds = 'ds-02'

subjects = ['{:02d}'.format(i) for i in range(1, 16)]
subjects.pop(3)
subjects.pop(0)

overall_means = {}

for hemi in ['l', 'r']:
    mean_rs = []
    for subject in subjects:
        print(subject)
        with np.load(op.join(derivatives,
                           ds,
                           'surface_correlations',
                             'sub-{subject}_rs_{hemi}.npz'.format(**locals()))) as f:

            subj_r = f['arr_0'].mean(0)

        mean_rs.append(subj_r)
    overall_means[hemi] = np.mean(mean_rs, 0)

embedder = manifold.SpectralEmbedding(1)
embedding_l = embedder.fit_transform(overall_means['l'])
embedding_r = embedder.fit_transform(overall_means['r'])

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

masker_l.inverse_transform(embedding_l.T).to_filename(op.join(derivatives,
                                                            ds,
                                                            'embedding',
                                                            'group_desc-stnl_embedding.nii.gz'))

masker_r.inverse_transform(embedding_r.T).to_filename(op.join(derivatives,
                                                            ds,
                                                            'embedding',
                                                            'group_desc-stnr_embedding.nii.gz'))


#if __name__ == '__main__':
    #main('/home/shared/2018/subcortex/bias_task')

