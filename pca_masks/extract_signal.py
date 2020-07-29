import argparse
import os.path as op
import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio 
import nipype.interfaces.utility as niu 
import nipype.interfaces.ants as ants 
from niworkflows.interfaces.bids import DerivativesDataSink


def main(derivatives, ds):

    if ds == 'ds-01':
        subjects = ['{:02d}'.format(s) for s in range(1, 20)]
    elif ds == 'ds-02':
        subjects = ['{:02d}'.format(s) for s in range(1, 16)]
        subjects.pop(3) # Remove 4

    subjects = subjects
    wf_folder = '/tmp/workflow_folders'

    templates = {'preproc':op.join(derivatives, ds, 'fmriprep', 'sub-{subject}', 'func',
                                   'sub-{subject}_task-randomdotmotion_run-*_space-T1w_desc-preproc_bold.nii.gz')}

    if ds == 'ds-01':
        templates['individual_mask'] = op.join(derivatives, ds, 'conjunct_masks', 'sub-{subject}', 'anat',
                                               'sub-{subject}_space-FLASH_desc-{mask}_space-T1w.nii.gz')

    elif ds =='ds-02':
        templates['individual_mask'] = op.join(derivatives, ds, 'conjunct_masks', 'sub-{subject}', 'anat',
                                               'sub-{subject}_desc-{mask}_mask.nii.gz')

    wf = pe.Workflow(name='extract_signal_masks_{}'.format(ds),
                     base_dir=wf_folder)

    mask_identity = pe.Node(niu.IdentityInterface(fields=['mask']),
                            name='mask_identity')
    mask_identity.iterables = [('mask', ['stnl', 'stnr'])]

    selector = pe.Node(nio.SelectFiles(templates),
                       name='selector')

    selector.iterables = [('subject', subjects)]
    wf.connect(mask_identity, 'mask', selector, 'mask')

    def extract_signal(preproc, mask):
        from nilearn import image
        from nilearn import input_data
        from nipype.utils.filemanip import split_filename
        import os.path as op
        import pandas as pd

        _, fn, ext = split_filename(preproc)
        masker = input_data.NiftiMasker(mask, standardize='psc')

        data = pd.DataFrame(masker.fit_transform(preproc))

        new_fn = op.abspath('{}_signal.csv'.format(fn))
        data.to_csv(new_fn)

        return new_fn

    extract_signal_node = pe.MapNode(niu.Function(function=extract_signal,
                                     input_names=['preproc', 'mask'],
                                     output_names=['signal']),
                         iterfield=['preproc'],
                        name='extract_signal_node')

    wf.connect(selector, 'preproc', extract_signal_node, 'preproc')
    wf.connect(selector, 'individual_mask', extract_signal_node, 'mask')

    datasink_signal = pe.MapNode(DerivativesDataSink(base_directory=op.join(derivatives, ds),
                                                      out_path_base='extracted_signal'),
                                 iterfield=['source_file', 'in_file'],
                                  name='datasink_signal')

    wf.connect(selector, 'preproc', datasink_signal, 'source_file')
    wf.connect(extract_signal_node, 'signal', datasink_signal, 'in_file')
    wf.connect(mask_identity, 'mask', datasink_signal, 'desc')


    wf.run(plugin='MultiProc',
           plugin_args={'n_procs':8})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ds', type=str,)
    args = parser.parse_args()

    main('/home/shared/2018/subcortex/bias_task/',
         args.ds)
