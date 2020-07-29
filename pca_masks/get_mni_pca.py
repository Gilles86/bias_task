from nilearn import image
import glob
import re
import pandas as pd
import os.path as op
import numpy as np
from sklearn import decomposition
import os
import argparse


def main(derivatives, ds):

	if ds == 'ds-01':
		reg = re.compile('.*/_mask_(?P<mask>.+)/_transformer[0-9]+/sub-(?P<subject>.+)_space-FLASH_desc-.+_space-T1w_resampled_trans.nii.gz')
		fns = glob.glob(op.join(derivatives, 'ds-01/individual_masks_mni_space/_mask_*/_transformer*/sub-*_space-FLASH_desc-*_space-T1w_resampled_trans.nii.gz'))

	elif ds == 'ds-02':

		reg = re.compile('.*/_mask_(?P<mask>.+)/_transformer[0-9]+/sub-(?P<subject>.+)_desc-.+_resampled_trans.nii.gz')
		fns = glob.glob(op.join(derivatives,
								'ds-02/individual_masks_mni_space/_mask_*/_transformer*/sub-*_desc-*_mask_resampled_trans.nii.gz'))


	df = []

	for fn in fns:
		print(fn)
		im = image.load_img(fn)
		d = pd.DataFrame(np.where(im.get_data()), index=['x', 'y', 'z']).T

		d['subject'] = reg.match(fn).groupdict()['subject']
		d['mask'] = reg.match(fn).groupdict()['mask']


		df.append(d)
		
	df = pd.concat(df)

	mean_coordinates = df.groupby('mask').mean()
	df[['x', 'y', 'z']] -= df[['x', 'y', 'z']].mean()

	x, y, z = np.meshgrid(np.arange(im.shape[0]),
						 np.arange(im.shape[1]),
						 np.arange(im.shape[2]), indexing='ij')

	pca = decomposition.PCA()

	if not op.exists(op.join(derivatives, ds, 'pca_mni')):
		os.makedirs(op.join(derivatives, ds, 'pca_mni'))

	for mask in ['stnl', 'stnr']:
		pca.fit(df.loc[df['mask'] == mask, ['x', 'y', 'z']])
		
		if pca.components_[0, 2] < 0:
			pca.components_[0] *= -1

		pca1ness = pca.transform(np.array([x.ravel(), y.ravel(), z.ravel()]).T)[:, 0]
		pca1ness = pca1ness.reshape(im.shape)
		pca1ness = image.new_img_like(im, pca1ness)
		pca1ness.to_filename(op.join(derivatives, ds, 'pca_mni', '{}_pca.nii.gz'.format(mask)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ds', type=str,)
    args = parser.parse_args()

    main('/home/shared/2018/subcortex/bias_task/',
         args.ds)
