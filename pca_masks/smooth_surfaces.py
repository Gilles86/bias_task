import os.path as op
from nipype.interfaces import freesurfer
from nipype.interfaces import utility as niu
import nipype.pipeline.engine as pe
from niworkflows.interfaces import bids



def main(derivatives,
         ds):
    subjects = ['{:02d}'.format(i) for i in range(1, 16)]
    subjects.pop(3)
    subjects.pop(0)

    input_files = []
    hemispheres = []

    for subject in subjects:

        if subject == '07':
            runs = ['{:02d}'.format(i) for i in range(1,3)]
        else:
            runs = ['{:02d}'.format(i) for i in range(1,4)]

        for run in runs:
            for hemi in ['lh', 'rh']:
                hemi_ = {'lh':'L', 'rh':'R'}[hemi]
                input_files.append(op.join(derivatives,
                                          ds,
                                          'fmriprep',
                                          'sub-{subject}',
                                          'func',
                                          'sub-{subject}_task-randomdotmotion_run-{run}_space-fsaverage_hemi-{hemi_}.func.gii').format(**locals()))
                hemispheres.append(hemi)
     
    derivatives_dir = op.join(derivatives, ds)

    wf = smooth_surf_wf('smooth_wf_{ds}'.format(**locals()),
                        input_files,
                        hemispheres,
                        derivatives_dir)

    wf.run(plugin='MultiProc',
           plugin_args={'n_procs':10})

def smooth_surf_wf(name,
                   in_files,
                   hemispheres,
                   derivatives_dir,
                   smooth_fwhm=5.0,
                   base_dir='/tmp/workflow_folders'):

    wf = pe.Workflow(name=name,
                     base_dir=base_dir)

    input_node = pe.Node(niu.IdentityInterface(fields=['in-files',
                                          'hemispheres']),
                                  name='input_node')

    freesurfer_dir = op.join(derivatives_dir, 'freesurfer')

    input_node.inputs.in_files = in_files
    input_node.inputs.hemispheres = hemispheres

    smoother = pe.MapNode(freesurfer.SurfaceSmooth(subjects_dir=freesurfer_dir),
                          iterfield=['in_file',
                                     'hemi'],
                          name='smoother')
    smoother.inputs.subject_id = 'fsaverage'
    smoother.inputs.fwhm = smooth_fwhm

    wf.connect(input_node, 'in_files', smoother, 'in_file')
    wf.connect(input_node, 'hemispheres', smoother, 'hemi')

    ds = pe.MapNode(bids.DerivativesDataSink(desc='smoothed',
                                             out_path_base='smoothed_surfaces',
                                             base_directory=op.join(derivatives_dir)),
                    iterfield=['source_file',
                               'in_file',
                               'extra_values'],
                    name='datasink')
    wf.connect(input_node, 'in_files', ds, 'source_file')
    wf.connect(smoother, 'out_file', ds, 'in_file')
    wf.connect(input_node, ('hemispheres', format_hemi), ds, 'extra_values')



    return wf

def format_hemi(hemi):
    if type(hemi) is list:
        return [[format_hemi(h)] for h in hemi]
    else:
        return 'hemi-{}'.format(hemi)


if __name__ == '__main__':
    main('/home/shared/2018/subcortex/bias_task',
         'ds-02')
