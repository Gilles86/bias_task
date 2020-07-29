import nipype.pipeline.engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces import fsl
import glob
import os.path as op
from nilearn import image
from niworkflows.interfaces.bids import DerivativesDataSink
from nipype.interfaces import io as nio

derivatives = '/home/shared/2018/subcortex/bias_task'

glm = 'glm_5'


zmap = op.join(derivatives, 'both', 'modelfitting', glm, '*.nii.gz')
zmap = glob.glob(zmap)

wf = pe.Workflow(name='threshold_wf_{}'.format(glm),
                 base_dir='/tmp/workflow_folders')


inputnode = pe.Node(niu.IdentityInterface(fields=['zmap']),
                    name='zmap')

inputnode.inputs.zmap = zmap

mask = fsl.Info.standard_image('MNI152_T1_1mm_brain_mask_dil.nii.gz')
resampled_mask = image.resample_to_img(mask, zmap[0], interpolation='nearest')
resampled_mask.to_filename(op.join(wf.base_dir, 'mask.nii.gz'))

n_voxels = (resampled_mask.get_data() > 0).sum()

smooth_est = pe.MapNode(fsl.SmoothEstimate(),
                        iterfield=['zstat_file'],
                        name='smooth_estimate')

smooth_est.inputs.mask_file = resampled_mask.get_filename()

wf.connect(inputnode, 'zmap', smooth_est, 'zstat_file')

cluster = pe.MapNode(fsl.Cluster(threshold=3.1,
                                 volume=n_voxels,
                                 pthreshold=0.05,
                                 out_pval_file=True,
                                 out_threshold_file=True,
                                 out_index_file=True,
                                 out_localmax_txt_file=True),
                     iterfield=['in_file', 'dlh',],
                     name='cluster')
wf.connect(inputnode, 'zmap', cluster, 'in_file')
wf.connect(smooth_est, 'dlh', cluster, 'dlh')
cluster.inputs.volume = n_voxels


def invert_zmap(zmap):
    from nilearn import image
    from nipype.utils.filemanip import split_filename
    import os.path as op
    
    _, fn, ext = split_filename(zmap)

    zmap_ = image.math_img('-zmap', zmap=zmap)

    zmap_.to_filename(op.abspath('{}_inv{}'.format(fn, ext)))

    return zmap_.get_filename()

invert_zmap_node = pe.MapNode(niu.Function(function=invert_zmap,
                                           input_names=['zmap'],
                                           output_names=['inv_zmap']),
                              iterfield=['zmap'],
                              name='invert_zmap_node')

cluster_neg = pe.MapNode(fsl.Cluster(threshold=2.6,
                                 volume=n_voxels,
                                 pthreshold=0.05,
                                 out_pval_file=True,
                                 out_threshold_file=True,
                                     out_index_file=True,
                                 out_localmax_txt_file=True),
                     iterfield=['in_file', 'dlh',],
                     name='cluster_neg')
wf.connect(inputnode, 'zmap', invert_zmap_node, 'zmap')
wf.connect(invert_zmap_node, 'inv_zmap', cluster_neg, 'in_file')
wf.connect(smooth_est, 'dlh', cluster_neg, 'dlh')
cluster_neg.inputs.volume = n_voxels

datasink = pe.Node(nio.DataSink(base_directory=op.join(derivatives, 'both', 'thr2.6')), name='datasink_thr')
datasink.inputs.regexp_substitutions = [('/_cluster(|_neg)[0-9]+/', '/')]
wf.connect(cluster, 'threshold_file', datasink, 'thresholded_zmap')
wf.connect(cluster, 'pval_file', datasink, 'thresholded_zmap.@pvals')
wf.connect(cluster, 'localmax_txt_file', datasink, 'thresholded_zmap.@localmax')
wf.connect(cluster, 'index_file', datasink, 'thresholded_zmap.@index_file')

wf.connect(cluster_neg, 'threshold_file', datasink, 'thresholded_zmap.@neg')
wf.connect(cluster_neg, 'pval_file', datasink, 'thresholded_zmap.@pvals_neg')
wf.connect(cluster_neg, 'localmax_txt_file', datasink, 'thresholded_zmap.@localmax_neg')
wf.connect(cluster_neg, 'index_file', datasink, 'thresholded_zmap.@indexfile_neg')

wf.run()
