import pandas as pd
import os.path as op

derivatives = '/home/shared/2018/subcortex/bias_task/'
ds = 'ds-02'

subjects = ['{:02d}'.format(subject) for subject in range(1, 16)]
subjects.pop(3)

df = []
for subject in subjects:
    d = pd.read_table('/home/raw_data/2018/subcortex/bias_task/raw/{ds}/behavior/S{subject}_log_STNREP_dot_bias_fmri.txt'.format(**locals()), sep=',')

    for run, d_ in d.groupby('block'):
        run  = '{:02d}'.format(run)
        tmp1 = d[['difficulty', 'onset_stim']].copy().rename(columns={'onset_stim':'onset', 
                                                                     'difficulty':'trial_type'})

        tmp2 = d.loc[d.correct == 0, ['onset_stim']].copy().rename(columns={'onset_stim':'onset'}) 
        tmp2['trial_type'] = 'error'

        tmp3 = d[['cue', 'onset_cue']].copy().rename(columns={'onset_cue':'onset', 
                                                              'cue':'trial_type'})
        tmp3['trial_type'] = tmp3.trial_type.apply(lambda x: 'cue_{}'.format(x))

        tmp4 = d[['response', 'onset_stim']].copy().rename(columns={'onset_stim':'onset', 
                                                                    'response':'trial_type'})
        tmp4 = tmp4[tmp4.trial_type != -1]
        tmp4['trial_type'] = tmp4.trial_type.map({1:'response_left', 2:'response_right'})

        tmp = pd.concat((tmp1, tmp2, tmp3, tmp4))

        tmp.to_csv(op.join(derivatives, ds, 'event_files',
                           'sub-{subject}_task-randomdotmotion_run-{run}_events.tsv').format(**locals()),
                   sep='\t', index=False)
