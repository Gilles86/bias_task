import argparse
import os.path as op
from itertools import product
import glob
import os
import shutil

def main(raw_dir,
         derivatives,
         ds='ds-02'):

    raw_dir = op.join(raw_dir, ds)

    if ds == 'ds-01':
        subjects = ['{:02d}'.format(s) for s in range(1,20)]

        masks_dir = op.join(raw_dir, 'masks')

        d =  op.join(masks_dir,
                  'sub-{subject}',
                  'masks',
                  'FLASH_space')
        stn_l = op.join(d, 'sub-{subject}_mask-STh_L.nii.gz')
        stn_r = op.join(d, 'sub-{subject}_mask-STh_R.nii.gz')

        mask = 'stn'

        for subject in subjects:
            for hemi in ['l', 'r']:
                old_fn = op.join(d, 'sub-{subject}_mask-STh_{hemi}.nii.gz').format(subject=subject,
                                                                                   hemi=hemi.capitalize())

                new_dir = op.join(derivatives, ds, 'conjunct_masks', 'sub-{subject}', 'anat').format(**locals())
                if not op.exists(new_dir):
                    os.makedirs(new_dir)

                new_fn = op.join(new_dir, 'sub-{subject}_space-FLASH_desc-{mask}{hemi}_mask.nii.gz').format(**locals())
                shutil.copy(old_fn, new_fn)

    if ds == 'ds-02':

        subjects = ['{:02d}'.format(s) for s in range(1,16)]

        subjects.pop(3)
        subjects = ['03']

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
        print(mapping)
        
        masks = ['stn', 'sn', 'rn']
        hemispheres = ['l', 'r']
        masks = ['stn']



        for subject, mask, hemi in product(subjects, masks, hemispheres):
            if subject == '03':
                session = 2
            else:
                session = 1
            print(subject)
            old_subject = mapping[subject]

            new_dir = op.join(derivatives, ds, 'conjunct_masks', 'sub-{subject}', 'anat').format(**locals())
            if not op.exists(new_dir):
                os.makedirs(new_dir)

            new_fn = op.join(new_dir, 'sub-{subject}_desc-{mask}{hemi}_mask.nii.gz').format(**locals())

            for rater in ['aaxbkx', 'aaxbix', 'aaxlfx', 'aaxgjgx', 'aaxmmx', 'aaxbos', 'aaxgdh']:
                old_fn = op.join(raw_dir,
                                 'anat',
                                 'sub-{subject}',
                                 'ses-{session}',
                                 'derivatives',
                                 '{mask}',
                                 'conjunct_masks',
                                 'sub-{old_subject}_ses-{session}_mask-{mask}_hem-{hemi}_rat-{rater}_calc-conj.nii.gz')
                old_fn = old_fn.format(**locals())
                print(old_fn)
                old_fn = glob.glob(old_fn)

                if len(old_fn) > 0:
                    print(subject, old_subject, old_fn, len(old_fn), new_fn)
                    assert(len(old_fn) == 1)
                    shutil.copy(old_fn[0], new_fn)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ds', type=str,)
    args = parser.parse_args()

    main('/home/raw_data/2018/subcortex/bias_task/raw',
         '/home/shared/2018/subcortex/bias_task/',
         args.ds)

