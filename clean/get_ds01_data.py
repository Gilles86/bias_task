import os
import shutil
import argparse
import glob
from natsort import natsorted
from nilearn import image
import json

def main(mode='anat'):
    subjects = ['{:02d}'.format(s) for s in range(1,20)]

    #subjects.pop(2)
    #subjects.pop(2)

    mapping = {'01': 'BI3T',
                    '02': 'DA9T',
                    '03': 'FMFT',
                    '04': 'GAIT',
                    '05': 'HCBT',
                    '06': 'KCAT',
                    '07': 'KP6T',
                    '08': 'LV2T',
                    '09': 'MRCT',
                    '10': 'NM3T',
                    '11': 'PF5T',
                    '12': 'RSIT',
                    '13': 'SPGT',
                    '14': 'TS6T',
                    '15': 'UM2T',
                    '16': 'VL1T',
                    '17': 'WSFT',
                    '18': 'WW2T',
                    '19': 'ZK4T'}


    if mode == 'anat':
        for subject in subjects:
            subject_ = mapping[subject]

            dir = '/home/raw_data/2018/subcortex/bias_task/raw/ds-01/anat/clean/{subject_}'.format(**locals())

            print(dir, os.path.exists(dir))

            new_dir = '/home/raw_data/2018/subcortex/bias_task/sourcedata/ds-01/sub-{subject}/anat/'.format(**locals())

            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            
            inv1 = os.path.join(dir, 'INV1.nii'.format(**locals()))
            shutil.copy(inv1, os.path.join(new_dir, 'sub-{subject}_inv-1_part-mag_MP2RAGE.nii'.format(**locals())))

            inv2 = os.path.join(dir, 'INV2.nii'.format(**locals()))
            shutil.copy(inv2, os.path.join(new_dir, 'sub-{subject}_inv-2_part-mag_MP2RAGE.nii'.format(**locals())))

            t1w = os.path.join(dir, 'UNI_Images.nii')
            shutil.copy(t1w, os.path.join(new_dir, 'sub-{subject}_T1UNI.nii'.format(**locals())))

            t1map = os.path.join(dir, 'T1_Images')
            shutil.copy(t1w, os.path.join(new_dir, 'sub-{subject}_T1map.nii'.format(**locals())))

            for ix, te in enumerate([11.22, 20.39, 29.57]):

                echo = ix+1

                #old_fn = os.path.join('/home/raw_data/2018/subcortex/bias_task/raw/ds-01/anat/FLASH', 'sub-{subject}_FLASH_echo_{te}.nii.gz'.format(**locals()))
                old_fn = os.path.join('/home/raw_data/2018/subcortex/bias_task/raw/ds-01/structural_2std/{subject_}/FLASH_magnitude',
                             'e{te}.nii.gz').format(**locals())
                print(old_fn, os.path.exists(old_fn))

                new_fn = os.path.join(new_dir, 'sub-{subject}_echo-{echo}_FLASH.nii.gz'.format(**locals()))
                shutil.copy(old_fn, new_fn)

    elif mode == 'func':
        dir = '/home/raw_data/2018/subcortex/bias_task/raw/ds-01/func'

        for subject in subjects:

            # FUNCTIONAL
            fmap_meta = {"EchoTime1": 0.00600,
                         "EchoTime2": 0.00702,
                         "IntendedFor":[]}
                         
            for run in [1,2,3]:
                fmap_meta['IntendedFor'].append('func/sub-{subject}_task-randomdotmotion_run-{run:02d}_bold.nii'.format(**locals()))


            json_file = '/home/raw_data/2018/subcortex/bias_task/sourcedata/ds-01/sub-{subject}/fmap/sub-{subject}_phasediff.json'.format(**locals())

            with open(json_file, 'w') as outfile:
                json.dump(fmap_meta, outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str,)
    args = parser.parse_args()

    main(args.mode)
