import os.path as op
import os
import pandas as pd
from nilearn import image
from nistats.first_level_model import FirstLevelModel
from nistats.second_level_model import SecondLevelModel
from nipype.interfaces import fsl
import argparse

derivatives = '/home/shared/2018/subcortex/bias_task/'
#def main(derivatives):

subjects = ['{:02d}'.format(s) for s in range(1, 20)]
ds_ = ['ds-01']  * len(subjects)
subjects += ['{:02d}'.format(s) for s in range(1, 16) if s != 4]
ds_ += ['ds-02'] * (len(subjects) - len(ds_))

models = []

for ds, subject in zip(ds_, subjects):
    print('subject {}, ds {}'.format(subject, ds))
    runs = ['{:02d}'.format(i) for i in range(1,4)]
    if ds == 'ds-01':
        if subject == '06':
            runs = ['{:02d}'.format(i) for i in range(1,3)]
    elif ds == 'ds-02':
        if subject == '07':
            runs = ['{:02d}'.format(i) for i in range(1,3)]


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

        behavior[-1]['duration'] = 0
    

    confounds = [c[include].fillna(method='bfill') for c in confounds]

    model = FirstLevelModel(t_r=3,
                            mask=masks[0],
                            drift_model=None, # Already done by fmriprep
                            smoothing_fwhm=5.0,
                            hrf_model='spm + derivative',
                            n_jobs=10,
                            subject_label='{}.{}'.format(ds, subject))

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

mask = fsl.Info.standard_image('MNI152_T1_2mm_brain_mask.nii.gz')

confounds = pd.DataFrame({'ds':ds_, 'subject_label':['{}.{}'.format(ds, subject) for ds, subject in zip(ds_, subjects)]})
confounds['ds'] = confounds['ds'].map({'ds-01':0, 'ds-02':1})

model2 = SecondLevelModel(mask)
model2.fit(models, confounds=confounds)

difficulty = model2.compute_contrast(first_level_contrast='hard - easy',
                                     second_level_contrast='intercept',
                                     output_type='z_score')
left_right_cue = model2.compute_contrast(first_level_contrast='cue_left - cue_right',
                                         second_level_contrast='intercept',
                                         output_type='z_score')
left_right_response = model2.compute_contrast(first_level_contrast='response_left - response_right',
                                              second_level_contrast='intercept',
                                              output_type='z_score')
error = model2.compute_contrast(first_level_contrast='error',
                                second_level_contrast='intercept',
                                output_type='z_score')

cue = model2.compute_contrast(first_level_contrast='cue_left + cue_right - 2 * cue_neutral',
                              second_level_contrast='intercept',
                              output_type='z_score')


if not op.exists(op.join(derivatives, 'modelfitting', 'glm_2')):
    os.makedirs(op.join(derivatives, 'modelfitting' 'glm_2'))

template = op.join(derivatives, 'modelfitting', 'glm_2', 'sub-{subject}_desc-{contrast}_contrast.nii.gz')
difficulty.to_filename(template.format(subject='group', contrast='difficulty'))
left_right_cue.to_filename(template.format(subject='group', contrast='left_right_cue'))
left_right_response.to_filename(template.format(subject='group', contrast='left_right_response'))
error.to_filename(template.format(subject='group', contrast='error'))
cue.to_filename(template.format(subject='group', contrast='cue'))

#if __name__ == '__main__':
    #main('/home/shared/2018/subcortex/bias_task/')
