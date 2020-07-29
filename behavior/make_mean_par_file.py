import pandas as pd
import os.path as op
import numpy as np

derivatives = '/home/shared/2018/subcortex/bias_task/'

df1 = pd.read_pickle(op.join(derivatives, 'ddm_results', 'traces_ds-01.pkl'))
df1['dataset'] = 'ds-01'

df2 = pd.read_pickle(op.join(derivatives, 'ddm_results', 'traces_ds-02.pkl'))
df2['dataset'] = 'ds-02'

df = pd.concat((df1, df2))

import re
reg = re.compile('(?P<variable>.+)_subj.(?P<subject>.+)')

tmp = df.melt(id_vars='dataset')
tmp['tmp'] = tmp.variable.apply(lambda x: reg.match(x).groupdict() if reg.match(x) else None)

tmp = tmp[~tmp.tmp.isnull()]
tmp['variable'] = tmp.tmp.apply(lambda x: x['variable'])
tmp['subject'] = tmp.tmp.apply(lambda x: x['subject'])
tmp.drop('tmp', axis=1, inplace=True)
df = tmp

beh = pd.read_pickle(op.join(derivatives, 'behavior.pkl')).rename(columns={'ds':'dataset'})

behavior_difficulty = beh.pivot_table(index=['dataset', 'subject'], columns='difficulty', values=['correct', 'rt'])
difference =pd.concat([(behavior_difficulty.swaplevel(axis=1)['easy'] - behavior_difficulty.swaplevel(axis=1)['hard'])], keys=['difference'], axis=1).swaplevel(axis=1)
behavior_difficulty = pd.concat((behavior_difficulty, difference), axis=1)

behavior_cue = beh.pivot_table(index=['dataset', 'subject'], columns='cue congruency', values=['correct', 'rt'])
difference =pd.concat([(behavior_cue.swaplevel(axis=1)['incongruent'] - behavior_cue.swaplevel(axis=1)['congruent'])], keys=['difference'], axis=1).swaplevel(axis=1)
behavior_cue = pd.concat((behavior_cue, difference), axis=1)

mean_behavior = pd.concat((behavior_difficulty, behavior_cue), keys=['difficulty', 'cue'], axis=1)

mean_pars = df.pivot_table(index=['dataset', 'subject'], columns='variable', values='value', aggfunc='mean')
mean_pars = df.pivot_table(index=['dataset', 'subject'], columns='variable', values='value', aggfunc='mean')
mean_pars['z_ddm_scale'] = 1 / (1 + np.exp(-mean_pars['z_cue_regressor']))
mean_pars['difficulty_effect'] = mean_pars['v_C(difficulty)[easy]'] - mean_pars['v_C(difficulty)[hard]']

mean_behavior.columns = [' '.join(col).strip() for col in mean_behavior.columns.values]

mean_all = pd.concat((mean_behavior, mean_pars), keys=['behavior', 'ddm'], axis=1)
mean_all.columns = [' '.join(col).strip() for col in mean_all.columns.values]

mean_all.to_pickle(op.join(derivatives, 'all_subjectwise_parameters.pkl'))
