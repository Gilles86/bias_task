from nipype.interfaces import io as nio
from nipype.interfaces import utility as niu
from nipype.algorithms.confounds import TSNR
import nipype.pipeline.engine as pe
import os.path as op
from nipype.interfaces import fsl
from niworkflows.interfaces import bids
import argparse


def main(derivatives,
         ds):

    func_template = {'func': op.join(derivatives,
                             ds,
                             'fmriprep',
                             'sub-{subject}',
                             'func',
                             'sub-{subject}_task-randomdotmotion_run-*_space-T1w_desc-preproc_bold.nii.gz'),
                     'boldref':op.join(derivatives,
                                 ds,
                                 'fmriprep',
                                 'sub-{subject}',
                                 'func',
                                 'sub-{subject}_task-randomdotmotion_run-*_space-T1w_boldref.nii.gz'),
}

    if ds == 'ds-01':
        mask_fn = 'sub-{subject}_space-FLASH_desc-{mask}_space-T1w.nii.gz'
        subjects = ['{:02d}'.format(si) for si in range(1,20)]
    elif ds == 'ds-02':
        mask_fn = 'sub-{subject}_desc-{mask}_mask.nii.gz'
        subjects = ['{:02d}'.format(si) for si in range(1,16)]
        subjects.pop(3)
        subjects.pop(0)

    mask_template = {'mask' : op.join(derivatives,
                              ds,
                              'conjunct_masks',
                              'sub-{subject}',
                              'anat',
                              mask_fn)}


    wf = pe.Workflow(name='get_tsnr_{}'.format(ds),
                     base_dir='/tmp/workflow_folders')

    identity = pe.Node(niu.IdentityInterface(fields=['subject']),
                         name='identity')
    identity.iterables = [('subject', subjects)]

    
    func_selector = pe.Node(nio.SelectFiles(func_template),
                                name='func_selector')
    func_selector.inputs.subject = '01'

    mask_identity = pe.Node(niu.IdentityInterface(fields=['mask']),
                            name='mask_identity')
    mask_identity.iterables = [('mask', ['stnl', 'stnr'])]

    mask_selector = pe.Node(nio.SelectFiles(mask_template),
                                name='mask_selector')
    wf.connect(identity, 'subject', mask_selector, 'subject')
    wf.connect(mask_identity, 'mask', mask_selector, 'mask')


    tsnr = pe.MapNode(TSNR(regress_poly=2),
                      iterfield=['in_file'],
                      name='tsnr')

    wf.connect(identity, 'subject', func_selector, 'subject')
    wf.connect(func_selector, 'func', tsnr, 'in_file')


    def resample_to_img(source_file, target_file):
        import os.path as op
        from nilearn import image
        from nipype.utils.filemanip import split_filename

        _, fn, ext = split_filename(source_file)

        new_fn = op.abspath('{}_resampled{}'.format(fn, ext))
        im = image.resample_to_img(source_file, target_file, interpolation='nearest')

        im.to_filename(new_fn)

        return new_fn


    resampler = pe.Node(niu.Function(function=resample_to_img,
                                     input_names=['source_file',
                                                  'target_file'],
                                     output_names=['resampled_file']),
                        name='resampler')

    wf.connect(mask_selector, 'mask', resampler, 'source_file')
    wf.connect(func_selector, 'boldref', resampler, 'target_file')

    extractor = pe.MapNode(fsl.ImageMeants(),
                           iterfield=['in_file'],
                           name='extractor')
    wf.connect(resampler, 'resampled_file', extractor, 'mask')
    wf.connect(tsnr, 'tsnr_file', extractor, 'in_file')

    
    ds_tsnrmaps = pe.MapNode(bids.DerivativesDataSink(base_directory=op.join(derivatives, ds),
                                                   suffix='tsnr',
                                                   out_path_base='tsnr'),
                             iterfield=['source_file', 'in_file'],
                          name='datasink_tsnrmaps')
    wf.connect(func_selector, 'func', ds_tsnrmaps, 'source_file')
    wf.connect(tsnr, 'tsnr_file', ds_tsnrmaps, 'in_file')


    ds_values_stn = pe.MapNode(bids.DerivativesDataSink(base_directory=op.join(derivatives, ds),
                                                   suffix='tsnr',
                                                   out_path_base='tsnr'),
                             iterfield=['source_file', 'in_file'],
                          name='ds_values_stnr')
    wf.connect(func_selector, 'func', ds_values_stn, 'source_file')
    wf.connect(extractor, 'out_file', ds_values_stn, 'in_file')
    wf.connect(mask_identity, 'mask', ds_values_stn, 'desc')

    wf.run(plugin='MultiProc',
           plugin_args={'n_procs':6})


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('ds', type=str,)
    args = parser.parse_args()

    main('/home/shared/2018/subcortex/bias_task/',
         args.ds)
