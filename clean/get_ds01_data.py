import os
import shutil
import argparse
import glob
from natsort import natsorted
from nilearn import image
import json

def main(mode='anat'):
    subjects = ['{:02d}'.format(s) for s in range(1,20)]

    subjects.pop(2)
    subjects.pop(2)

    mapping = {'01':'KCAT',
               '02':'PF5T',
               '03':'WW2T',
               '04':'WSFT',
               '05':'KP6T',
               '06':'LV2T',
               '07':'FMFT',
               '08':'HCBT',
               '09':'RSIT',
               '10':'TS6T',
               '11':'UM2T',
               '12':'BI3T',
               '13':'MRCT',
               '14':'NM3T',
               '15':'SC1T',
               '16':'SPGT',
               '17':'ZK4T',
               '18':'DA9T',
               '19':'VL1T'}


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
            shutil.copy(inv1, os.path.join(new_dir, 'sub-{subject}_inv-2_part-mag_MP2RAGE.nii'.format(**locals())))

            t1w = os.path.join(dir, 'UNI_Images.nii')
            shutil.copy(t1w, os.path.join(new_dir, 'sub-{subject}_T1UNI.nii'.format(**locals())))

            t1map = os.path.join(dir, 'T1_Images')
            shutil.copy(t1w, os.path.join(new_dir, 'sub-{subject}_T1map.nii'.format(**locals())))

            for ix, te in enumerate([11.22, 20.39, 29.57]):

                old_fn = os.path.join('/home/raw_data/2018/subcortex/bias_task/raw/ds-01/anat/FLASH', 'sub-{subject}_FLASH_echo_{te}.nii.gz'.format(**locals()))
                print(old_fn, os.path.exists(old_fn))

                new_fn = os.path.join(new_dir, 'sub-{subject}_echo-{echo}_FLASH.nii.gz'.format(**locals()))
                shutil.copy(old_fn, new_fn)

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
