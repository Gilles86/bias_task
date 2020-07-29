import os
import os.path as op
from nipype.interfaces import io as nio
from nipype.interfaces import fsl
from nipype.interfaces import ants
from nipype.interfaces import utility as niu
import nipype.pipeline.engine as pe
from niworkflows.interfaces import bids
from niworkflows.interfaces.registration import FLIRTRPT
from niworkflows.interfaces.registration import ANTSRegistrationRPT		
from nipype.interfaces.c3 import C3dAffineTool

derivatives = '/home/shared/2018/subcortex/bias_task/ds-01'
sourcedata = '/home/raw_data/2018/subcortex/bias_task/sourcedata/ds-01/'

flash_templates = {'FLASH':op.join(sourcedata, 'sub-{subject}', 'anat', 'sub-{subject}_echo-{echo}_FLASH.nii.gz'),}
t1w_templates = {'T1w':op.join(derivatives, 'fmriprep', 'sub-{subject}', 'anat',
                               'sub-{subject}_desc-preproc_T1w.nii.gz'),
                 'aseg':op.join(derivatives, 'fmriprep', 'sub-{subject}', 'anat',
                                'sub-{subject}_desc-aseg_dseg.nii.gz'),
                 'bold':op.join(derivatives, 'fmriprep', 'sub-{subject}', 'func',
                                'sub-{subject}_task-randomdotmotion_run-01_space-T1w_boldref.nii.gz')}
mask_templates = {'mask':op.join(derivatives, 'conjunct_masks', 'sub-{subject}', 'anat',
                              'sub-{subject}_space-FLASH_desc-{mask}_mask.nii.gz')}

wf = pe.Workflow(name='transform_masks',
                 base_dir='/tmp/workflow_folders')

inputnode = pe.Node(niu.IdentityInterface(fields=['subject']),
                    name='inputnode')
inputnode.iterables = [('subject', ['{:02d}'.format(i) for i in range(1, 20)])]
inputnode.iterables = [('subject', ['{:02d}'.format(i) for i in [10]])]

flash_selector = pe.MapNode(nio.SelectFiles(flash_templates),
                      iterfield=['echo'],
                      name='flash_selector')
flash_selector.inputs.echo = [1,2,3]
wf.connect(inputnode, 'subject', flash_selector, 'subject')

t1w_selector = pe.Node(nio.SelectFiles(t1w_templates),
                      name='t1w_selector')
wf.connect(inputnode, 'subject', t1w_selector, 'subject')


mask_selector = pe.Node(nio.SelectFiles(mask_templates),
                        name='mask_selector')
wf.connect(inputnode, 'subject', mask_selector, 'subject')
mask_selector.iterables = [('mask', ['stnl', 'stnr'])]

merger = pe.Node(fsl.Merge(dimension='t'),
                name='merger')
wf.connect(flash_selector, 'FLASH', merger, 'in_files')

meaner = pe.Node(fsl.MeanImage(), 
                 name='meaner')
wf.connect(merger, 'merged_file', meaner, 'in_file')

n4_correct = pe.Node(ants.N4BiasFieldCorrection(),
					 n_procs=8,
					name='n4correct')
wf.connect(meaner, 'out_file', n4_correct, 'input_image')


reg = pe.Node(FLIRTRPT(generate_report=True, cost_func='normcorr',dof=6), # DOF was 6 for subject 10
	        	name='flirt')
#reg.inputs.schedule = op.join(os.getenv('FSLDIR'), 'etc/flirtsch/bbr.sch')

wf.connect(n4_correct, 'output_image', reg, 'in_file')
#wf.connect(t1w_selector, 'T1w', reg, 'reference')
wf.connect(t1w_selector, 'bold', reg, 'reference')
#wf.connect(get_wm, 'wm', reg, 'wm_seg')

convert_fsl_to_ants = pe.Node(C3dAffineTool(), name='convert_fsl_to_ants_t1_to_epi')
convert_fsl_to_ants.inputs.fsl2ras = True
convert_fsl_to_ants.inputs.itk_transform = True

wf.connect(reg, 'out_matrix_file', convert_fsl_to_ants, 'transform_file')
wf.connect(n4_correct, 'output_image', convert_fsl_to_ants, 'source_file')
wf.connect(t1w_selector, 'bold', convert_fsl_to_ants, 'reference_file')

ds_registration = pe.Node(bids.DerivativesDataSink(base_directory=derivatives,
												   out_path_base='FLASH_to_T1w'),
						  name='ds_registration')
#wf.connect(reg, 'warped_image', ds_registration, 'in_file')
wf.connect(reg, 'out_file', ds_registration, 'in_file')
wf.connect(t1w_selector, 'T1w', ds_registration, 'source_file')

ds_registration_report = pe.Node(bids.DerivativesDataSink(base_directory=derivatives,
                                                   out_path_base='FLASH_to_T1w'),
                          name='ds_registration_report')
wf.connect(reg, 'out_report', ds_registration_report, 'in_file')
wf.connect(t1w_selector, 'T1w', ds_registration_report, 'source_file')



mask_transformer = pe.MapNode(ants.ApplyTransforms(interpolation='MultiLabel'),
                              iterfield=['input_image'],
                              name='applier')

#wf.connect(reg, 'composite_transform', mask_transformer, 'transforms')
wf.connect(convert_fsl_to_ants, 'itk_transform', mask_transformer, 'transforms')
wf.connect(mask_selector, 'mask', mask_transformer, 'input_image')
wf.connect(t1w_selector, 'T1w', mask_transformer, 'reference_image')


ds_mask = pe.Node(bids.DerivativesDataSink(base_directory=derivatives,
                                           out_path_base='conjunct_masks',
                                           space='T1w'),

                          name='ds_mask')
wf.connect(mask_transformer, 'output_image', ds_mask, 'in_file')
wf.connect(mask_selector, 'mask', ds_mask, 'source_file')

wf.run(plugin='MultiProc', plugin_args={'n_procs':12})
