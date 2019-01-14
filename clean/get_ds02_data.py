import os
import shutil
import argparse
import glob
from natsort import natsorted
from nilearn import image
import json

def main(mode='anat'):
    subjects = ['{:02d}'.format(s) for s in range(1,16)]

    subjects.pop(2)
    subjects.pop(2)

    mapping = {'01':'005',
               '02':'025',
               '03':'015',
               '04':'002',
               '05':'006',
               '06':'053',
               '07':'041',
               '08':'106',
               '09':'104',
               '10':'109',
               '11':'108',
               '12':'024',
               '13':'001',
               '14':'031',
               '15':'105'}


    if mode == 'anat':
        for subject in subjects:
            subject_ = mapping[subject]

            dir = '/home/raw_data/2018/subcortex/bias_task/raw/ds-02/anat/sub-{subject}/ses-1/anat/wb/source/'.format(**locals())
            new_dir = '/home/raw_data/2018/subcortex/bias_task/sourcedata/ds-02/sub-{subject}/anat/'.format(**locals())

            if not os.path.exists(new_dir):
                os.makedirs(new_dir)


            inv1 = os.path.join(dir, 'sub-{subject_}_ses-1_acq-wb_inv-1_part-mag_mprage.nii'.format(**locals()))
            inv1_ph = os.path.join(dir, 'sub-{subject_}_ses-1_acq-wb_inv-1_part-ph_mprage.nii'.format(**locals()))

            shutil.copy(inv1, os.path.join(new_dir, 'sub-{subject}_inv-1_part-mag_MP2RAGE.nii'.format(**locals())))
            shutil.copy(inv1, os.path.join(new_dir, 'sub-{subject}_inv-1_part-phase_MP2RAGE.nii'.format(**locals())))

            inv2 = []
            inv2_ph = []

            for echo in range(1,5):
                inv2.append(os.path.join(dir, 'sub-{subject_}_ses-1_acq-wb_inv-2_echo-{echo}_part-mag_mprage.nii'.format(**locals())))
                inv2_ph.append(os.path.join(dir, 'sub-{subject_}_ses-1_acq-wb_inv-2_echo-{echo}_part-ph_mprage.nii'.format(**locals())))

                shutil.copy(inv2[-1], os.path.join(new_dir, 'sub-{subject}_inv-2_echo-{echo}_part-mag_MP2RAGE.nii'.format(**locals())))
                shutil.copy(inv2_ph[-1], os.path.join(new_dir, 'sub-{subject}_inv-2_echo-{echo}_part-phase_MP2RAGE.nii'.format(**locals())))

            t1w = os.path.join(dir, 'sub-{subject_}_ses-1_acq-wb_mod-t1w_mprage.nii'.format(**locals()))
            shutil.copy(t1w, os.path.join(new_dir, 'sub-{subject}_T1UNI.nii'.format(**locals())))

    elif mode == 'func':
        dir = '/home/raw_data/2018/subcortex/bias_task/raw/ds-02/func'

        for subject in subjects:

            for modality in ['func', 'fmap']:
                new_dir = os.path.join('/home/raw_data/2018/subcortex/bias_task/sourcedata/ds-02/sub-{subject}/{modality}'.format(**locals()))
                if not os.path.exists(new_dir):
                    os.makedirs(new_dir)

            subject_ = mapping[subject]
            template = os.path.join(dir, 'STNREP_S{subject}*_B0_*.nii'.format(**locals()))

            # B0 MAP
            B0 = glob.glob(template)
            if len(B0) == 2:
                B0 = [b0 for b0 in B0 if 'shimmed' in b0 ]

            if len(B0) != 1:
                raise Exception('Problem with B0')

            B0 = B0[0]
            magnitude = image.index_img(B0, 0)
            phase = image.index_img(B0, 1)

            magnitude.to_filename('/home/raw_data/2018/subcortex/bias_task/sourcedata/ds-02/sub-{subject}/fmap/sub-{subject}_magnitude1.nii'.format(**locals()))
            phase.to_filename('/home/raw_data/2018/subcortex/bias_task/sourcedata/ds-02/sub-{subject}/fmap/sub-{subject}_phasediff.nii'.format(**locals()))


            # FUNCTIONAL
            template = os.path.join(dir, 'STNREP_S{subject}*_TE14_*.nii'.format(**locals()))
            func = glob.glob(template)
            func = natsorted(func)
           
            fmap_meta = {"EchoTime1": 0.0,
                         "EchoTime2": 0.00438,
                         "IntendedFor":[]}
                         
            for ix, f in enumerate(func):
                run = '{:02d}'.format(ix+1)
                template = "func/sub-{subject}_task-randomdotmotion_run-{run}_bold.nii.gz".format(**locals())

                shutil.copy(f, os.path.join('/home/raw_data/2018/subcortex/bias_task/sourcedata/ds-02/', 
                                            'sub-{subject}', template).format(**locals()))

                fmap_meta['IntendedFor'].append(template)
            
            json_file = '/home/raw_data/2018/subcortex/bias_task/sourcedata/ds-02/sub-{subject}/fmap/sub-{subject}_phasediff.json'.format(**locals())

            with open(json_file, 'w') as outfile:
                json.dump(fmap_meta, outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str,)
    args = parser.parse_args()

    main(args.mode)
