import os.path as op
import pandas as pd
from nilearn import image
from nistats.first_level_model import FirstLevelModel

ds = 'ds-02'
derivatives = '/home/shared/2018/subcortex/bias_task'

subjects = ['{:02d}'.format(s) for s in range(1, 16)]
subjects.pop(0) # Remove 1
subjects.pop(2) # Remove 4
# 2, 3, 5, 6, 7...
models = []

for subject in subjects:
    print('subject {}'.format(subject))
    if subject == '07':
        runs = ['{:02d}'.format(i) for i in range(1,3)]
    else:
        runs = ['{:02d}'.format(i) for i in range(1,4)]

    include = [u'dvars',u'framewise_displacement', u'a_comp_cor_00', u'a_comp_cor_01', u'a_comp_cor_02', u'a_comp_cor_03', u'a_comp_cor_04', u'a_comp_cor_05', u'cosine00', u'cosine01', u'cosine02', u'cosine03', u'cosine04', u'cosine05', u'cosine06',u'cosine07', u'cosine08', u'cosine09', u'cosine10', u'cosine11', u'cosine12', u'cosine13', u'cosine14', u'cosine15', u'trans_x', u'trans_y', u'trans_z', u'rot_x', u'rot_y', u'rot_z']

    images = []
    confounds = []
    behavior = []
    masks = []


    for run in runs:
        masks.append(op.join(derivatives, ds, 'fmriprep', 'sub-{subject}', 'func',
                              'sub-{subject}_task-randomdotmotion_run-{run}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz').format(**locals()))

        images.append(op.join(derivatives, ds, 'fmriprep', 'sub-{subject}', 'func',
                              'sub-{subject}_task-randomdotmotion_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz').format(**locals()))
        confounds.append(pd.read_table(op.join(derivatives, ds, 'fmriprep', 'sub-{subject}', 'func',
                              'sub-{subject}_task-randomdotmotion_run-{run}_desc-confounds_regressors.tsv').format(**locals())))

        behavior.append(pd.read_table(op.join(derivatives, ds, 'event_files',
                                         'sub-{subject}_task-randomdotmotion_run-{run}_events.tsv').format(**locals())))

    confounds = [c[include].fillna(method='bfill') for c in confounds]

    model = FirstLevelModel(t_r=3,
                            mask=masks[0],
                            drift_model=None, # Already done by fmriprep
                            smoothing_fwhm=5.0,
                            hrf_model='spm + derivative',
                            n_jobs=10,
                            subject_label=subject)

    model.fit(images, 
              behavior,
              confounds)

    print(model.design_matrices_[0].columns)

    difficulty = model.compute_contrast('hard - easy', output_type='z_score')
    left_right_cue = model.compute_contrast('cue_left - cue_right', output_type='z_score')
    left_right_response = model.compute_contrast('response_left - response_right', output_type='z_score')
    error = model.compute_contrast('error', output_type='z_score')
    cue = model.compute_contrast('cue_left + cue_right - 2 * cue_neutral')

    template = op.join(derivatives, ds, 'glm', 'individual_zmaps', 'sub-{subject}_desc-{contrast}_contrast.nii.gz')

    difficulty.to_filename(template.format(subject=subject, contrast='difficulty'))
    left_right_cue.to_filename(template.format(subject=subject, contrast='left_right_cue'))
    left_right_response.to_filename(template.format(subject=subject, contrast='left_right_response'))
    error.to_filename(template.format(subject=subject, contrast='error'))
    cue.to_filename(template.format(subject=subject, contrast='cue'))

    models.append(model)

