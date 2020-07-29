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

    wf_folder = '/tmp/workflow_folders'

    identity = pe.Node(niu.IdentityInterface(fields=['mask']),
                       name='identity')

    templates = {'pca_map':op.join(derivatives, ds, 'pca_mni', '{mask}_pca.nii.gz'),
                 't1w':op.join(derivatives, ds, 'fmriprep', 'sub-{subject}', 'anat', 'sub-{subject}_desc-preproc_T1w.nii.gz'),
                 'mni2t1w':op.join(derivatives, ds, 'fmriprep', 'sub-{subject}', 'anat',
                                   'sub-{subject}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5')}

    if ds == 'ds-01':
        templates['individual_mask'] = op.join(derivatives, ds, 'conjunct_masks', 'sub-{subject}', 'anat',
                                               'sub-{subject}_space-FLASH_desc-{mask}_space-T1w.nii.gz')

    elif ds =='ds-02':
        templates['individual_mask'] = op.join(derivatives, ds, 'conjunct_masks', 'sub-{subject}', 'anat',
                                               'sub-{subject}_desc-{mask}_mask.nii.gz')

    wf = pe.Workflow(name='make_pca_masks_{}'.format(ds),
                     base_dir=wf_folder)

    selector = pe.Node(nio.SelectFiles(templates),
                       name='selector')
    selector.iterables = [('mask', ['stnl', 'stnr']),
                          ('subject', subjects)]

    individual_pca_map = pe.Node(ants.ApplyTransforms(num_threads=4),
                                 name='individual_pca_map')

    wf.connect(selector, 't1w', individual_pca_map, 'reference_image')
    wf.connect(selector, 'pca_map', individual_pca_map, 'input_image')
    wf.connect(selector, 'mni2t1w', individual_pca_map, 'transforms')

    def make_pca_mask(pca_map, mask):
        from nilearn import image
        from nipype.utils.filemanip import split_filename
        import os.path as op

        _, fn, ext = split_filename(mask)

        pca_map = image.load_img(pca_map)
        mask = image.load_img(mask)

        pca_map = image.resample_to_img(pca_map, mask, interpolation='nearest')

        new_mask = image.math_img('pca_map * (mask > 0)', 
                                  pca_map=pca_map,
                                  mask=mask)

        tmp = new_mask.get_data()
        tmp[tmp!=0] -=  tmp[tmp!=0].min() - 1e-4
        tmp[tmp!=0] /=  tmp[tmp!=0].max()

        new_mask = image.new_img_like(new_mask, tmp)

        new_mask.to_filename(op.abspath('{}_map{}'.format(fn, ext)))

        return new_mask.get_filename()

    make_mask = pe.Node(niu.Function(function=make_pca_mask,
                                     input_names=['pca_map', 'mask'],
                                     output_names=['mask']),
                        name='make_mask')

    wf.connect(individual_pca_map, 'output_image', make_mask, 'pca_map')
    wf.connect(selector, 'individual_mask', make_mask, 'mask')

    def make_submask(mask):
        from nilearn import image
        import numpy as np
        import os.path as op
        from nipype.utils.filemanip import split_filename

        _, fn, ext = split_filename(mask)

        im = image.load_img(mask)

        data = im.get_data()
        percentiles = np.percentile(data[data !=0 ], [33, 66])
		
        mask1 = image.math_img('(im > 0) & (im < {})'.format(percentiles[0]), im=im)
        mask2 = image.math_img('(im > {}) & (im < {})'.format(*percentiles), im=im)
        mask3 = image.math_img('(im > {})'.format(percentiles[1]), im=im)

        fn1 = op.abspath('{}_maskA{}'.format(fn, ext))
        fn2 = op.abspath('{}_maskB{}'.format(fn, ext))
        fn3 = op.abspath('{}_maskC{}'.format(fn, ext))

        mask1.to_filename(fn1)
        mask2.to_filename(fn2)
        mask3.to_filename(fn3)

        return fn3, fn2, fn1

    make_submasksnode = pe.Node(niu.Function(function=make_submask,
                                             input_names=['mask'],
                                             output_names=['submasks']),
                             name='make_submasks')
    
    wf.connect(make_mask, 'mask', make_submasksnode, 'mask')

    datasink_whole_mask = pe.Node(DerivativesDataSink(base_directory=op.join(derivatives, ds),
                                                      space='T1w',
                                                      suffix='roi',
                                                      out_path_base='pca_masks'),
                                  name='datasink_whole_mask')
    datasink_whole_mask.base_path = 'pca_masks'
    def remove_space(input):
        return input.replace('_space-FLASH', '')

    wf.connect(selector, ('individual_mask', remove_space), datasink_whole_mask, 'source_file')
    wf.connect(make_mask, 'mask', datasink_whole_mask, 'in_file')

    datasink_submasks = pe.MapNode(DerivativesDataSink(base_directory=op.join(derivatives, ds),
                                                       space='T1w',
                                                      out_path_base='pca_masks'),
                                   iterfield=['suffix', 'in_file'],
                                   name='datasink_submasks')
    datasink_submasks.base_path = 'pca_masks'
    datasink_submasks.inputs.suffix = ['subroi-A_roi', 'subroi-B_roi', 'subroi-C_roi']
    wf.connect(selector, ('individual_mask', remove_space), datasink_submasks, 'source_file')
    wf.connect(make_submasksnode, 'submasks', datasink_submasks, 'in_file')



    wf.run(plugin='MultiProc',
           plugin_args={'n_procs':8})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ds', type=str,)
    args = parser.parse_args()

    main('/home/shared/2018/subcortex/bias_task/',
         args.ds)
