import os
import shutil
import argparse
import glob
from natsort import natsorted
from nilearn import image
import json
from nipype.utils.filemanip import split_filename

def main(ds):
    if ds == 'ds-02':
        subjects = ['{:02d}'.format(s) for s in range(1,16)]

        subjects.pop(3)

    print(subjects)


    old_dir = os.path.join('/home/raw_data/2018/subcortex/bias_task/sourcedata/{ds}'.format(ds=ds))
    new_dir = os.path.join('/home/shared/2018/subcortex/bias_task/{ds}/fmriprep_data'.format(ds=ds))

    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    jsons = glob.glob(os.path.join(old_dir, '*.json'))
    for j in jsons:
        shutil.copy(j, new_dir)

    for subject in subjects:
        for d in ['fmap', 'func', 'anat']:
            sub_dir = os.path.join(new_dir, 'sub-{subject}', d).format(**locals())
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)

        func_dir = os.path.join(old_dir, 'sub-{subject}', 'func').format(subject=subject)
        fns_func = glob.glob(os.path.join(func_dir, '*'))
        
        new_func_dir = os.path.join(new_dir, 'sub-{subject}', 'func').format(**locals())

        for fn in fns_func:

            im = image.load_img(fn)
            d, fn = os.path.split(fn)
            im.header.set_xyzt_units(t=8)
            print(fn)
            im.to_filename(os.path.join(new_func_dir, fn))


        # FMAPS
        fmap_dir = os.path.join(old_dir, 'sub-{subject}', 'fmap').format(subject=subject)
        fns_fmap = glob.glob(os.path.join(fmap_dir, '*'))
        new_fmap_dir = os.path.join(new_dir, 'sub-{subject}', 'fmap').format(**locals())
        for fn in fns_fmap:
            shutil.copy(fn, new_fmap_dir)


        t1w = '/home/shared/2018/subcortex/bias_task/{ds}/masked_mp2rages/sub-{subject}/anat/sub-{subject}_desc-masked_T1w.nii'.format(**locals())
        print(t1w)

        shutil.copy(t1w, os.path.join(new_dir, 'sub-{subject}', 'anat', 'sub-{subject}_T1w.nii').format(**locals()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ds', type=str,)
    args = parser.parse_args()

    main(args.ds)
