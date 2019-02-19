import os.path as op
import os
import nipype
from nipype.interfaces import ants
from nipype.interfaces import fsl
from nipype.interfaces import io as nio
from nipype.interfaces import utility as niu
import nipype.pipeline.engine as pe

def resample_to_img(source_img,
                    target_img):

    from nilearn import image
    from nipype.utils.filemanip import split_filename
    import os.path as op

    new_img = image.resample_to_img(source_img,
                                    target_img,
                                    interpolation='nearest')

    p, fn, ext = split_filename(source_img)

    new_fn = op.abspath('{fn}_resampled{ext}'.format(fn=fn,
                                                     ext=ext))
    new_img.to_filename(new_fn)

    return new_fn

def main(derivatives_dir,
         ds,
         wf_folders):

    template_mask = op.join(derivatives_dir,
                            ds,
                            'conjunct_masks',
                            'sub-{subject}',
                            'anat',
                            'sub-{subject}_desc-{mask}_mask.nii.gz')

    template_transform = op.join(derivatives_dir,
                                 ds,
                                 'fmriprep',
                                 'sub-{subject}',
                                 'anat',
                                 'sub-{subject}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5')

    t1w_template = op.join(derivatives_dir,
                         ds,
                         'fmriprep',
                         'sub-{subject}',
                         'anat',
                         'sub-{subject}_desc-preproc_T1w.nii.gz')

    templates = {'mask':template_mask,
                'transform':template_transform,
                 't1w':t1w_template}

    selector = pe.MapNode(nio.SelectFiles(templates),
                          iterfield=['subject'],
                          name='selector')

    subjects = ['{:02d}'.format(i) for i in list(range(1, 16))]
    subjects.pop(3)

    selector.inputs.subject = subjects
    selector.iterables = [('mask', ['stnl', 'stnr'])]

    wf = pe.Workflow(name='transform_stn_masks',
                     base_dir=wf_folders)

    reorient_mask = pe.MapNode(niu.Function(function=resample_to_img,
                                            input_names=['source_img',
                                                      'target_img'],
                                         output_names='reoriented_image'),
                               iterfield=['source_img',
                                          'target_img'],
                            name='reorient_mask')
    wf.connect(selector, 'mask', reorient_mask, 'source_img')
    wf.connect(selector, 't1w', reorient_mask, 'target_img')

    transformer = pe.MapNode(ants.ApplyTransforms(interpolation='NearestNeighbor'),
                             iterfield=['input_image', 'transforms'],
                             name='transformer')

    wf.config = { "execution": { "crashdump_dir": op.join(os.environ['HOME'], 'crashdumps') }}
    
    wf.connect(reorient_mask, 'reoriented_image', transformer, 'input_image')
    wf.connect(selector, 'transform', transformer, 'transforms')
    transformer.inputs.reference_image = fsl.Info.standard_image('MNI152_T1_0.5mm.nii.gz')

    merge_masks = pe.Node(fsl.Merge(dimension='t'),
                          name='merge_masks')
    wf.connect(transformer, 'output_image', merge_masks, 'in_files')

    mean_mask = pe.Node(fsl.MeanImage(),
                        name='mean_mask')
    wf.connect(merge_masks, 'merged_file', mean_mask, 'in_file')

    base_dir = op.join(derivatives_dir, ds)
    ds = pe.Node(nio.DataSink(base_directory=base_dir),
                 name='datasink')

    wf.connect(transformer, 'output_image', ds, 'individual_masks_mni_space')
    wf.connect(mean_mask, 'out_file', ds, 'mean_mask_mni_space')
                 

    wf.run(plugin='MultiProc',
           plugin_args={'n_procs':4})

if __name__ == '__main__':
    main('/home/shared/2018/subcortex/bias_task/',
         'ds-02',
         '/tmp/workflow_folders')
