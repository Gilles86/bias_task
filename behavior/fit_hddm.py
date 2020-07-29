import pandas as pd
import os.path as op
import os
import numpy as np
from patsy import dmatrix  # for generation of (regression) design matrices
import hddm
from joblib import Parallel, delayed
import argparse

def main(derivatives, ds):

	df = pd.read_pickle(op.join(derivatives, 'behavior.pkl'))
	df = df[df.ds == ds]

	def get_cue_congruency(row):
		if row.cue == 'neutral':
			return 'neutral'
		elif row.stimulus == row.cue:
			return 'congruent'
		else:
			return 'incongruent'

	def z_link_func(x):
		return 1 / (1 + np.exp(-x))

	df['cue congruency'] = df.apply(get_cue_congruency, 1)

	df['response'] = df.correct.map({0:'error', 1:'correct'})
	df['rt'] = df['rt'] / 1000.
	df['cue_regressor'] = df['cue congruency'].map({'congruent':1, 'neutral':0, 'incongruent':-1})
	df['subj_idx'] = df['subject']

	z_reg = {'model': 'z ~ 0 + cue_regressor', 'link_func': z_link_func}
	v_reg = {'model': 'v ~ 0 + C(difficulty)', 'link_func':lambda x: x}



	model = hddm.HDDMRegressor(df[df.rt > 0.15],
							   [z_reg, v_reg],
							   include='z',
							   group_only_regressors=False)

	def fit_model(i):
		model = hddm.HDDMRegressor(df[df.rt > 0.15],
								   [z_reg, v_reg],
								   include='z',
								   group_only_regressors=False)

		model.sample(20000, 10000, dbname='/tmp/traces{}.db'.format(i), db='pickle')

		return model

	results = Parallel(n_jobs=6)(delayed(fit_model)(i) for i in range(6))

	traces = pd.concat([r.get_traces() for r in results])

	if not op.exists(op.join(derivatives, 'ddm_results')):
		os.makedirs(op.join(derivatives, 'ddm_results'))


	traces.to_pickle(op.join(derivatives, 'ddm_results', 'traces_{}.pkl'.format(ds)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ds', type=str,)
    args = parser.parse_args()
    main('/home/shared/2018/subcortex/bias_task/',
         args.ds)
