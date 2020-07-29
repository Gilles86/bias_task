import pandas as pd
import os.path as op
import argparse

def main(derivatives, ds):

    from_bids_to_leipzig_mapping = {'06': '1', '11': '2', '18': '3', '17': '4', '07': '5', '08': '6', '03': '7', '05': '8', '12': '9', '14': '10', '15': '11', '01': '12', '09': '13', '10': '14', '13': '16', '19': '17', '04': '18', '02': '19', '16': '20'}

    if ds == 'ds-02':
        subjects = ['{:02d}'.format(subject) for subject in range(1, 16)]
        subjects.pop(3)
        fn_template = 'S{subject_}_log_STNREP_dot_bias_fmri.txt'
    elif ds == 'ds-01':
        subjects = ['{:02d}'.format(subject) for subject in range(1, 20)]
        fn_template = 'S{subject_}_log_bias.txt'

    df = []
    for subject in subjects:
        
        if ds == 'ds-01':
            subject_ = from_bids_to_leipzig_mapping[subject]
        else:
            subject_ = subject


        d = pd.read_table(('/home/raw_data/2018/subcortex/bias_task/raw/{ds}/behavior/'+fn_template).format(**locals()), sep=',')


        for run, d_ in d.groupby('block'):
            print(subject, run)
            run  = '{:02d}'.format(run)
            tmp1 = d_[['difficulty', 'onset_stim']].copy().rename(columns={'onset_stim':'onset', 
                                                                         'difficulty':'trial_type'})

            tmp2 = d_.loc[d.correct == 0, ['onset_stim']].copy().rename(columns={'onset_stim':'onset'}) 
            tmp2['trial_type'] = 'error'

            tmp3 = d_[['cue', 'onset_cue']].copy().rename(columns={'onset_cue':'onset', 
                                                                  'cue':'trial_type'})
            tmp3['trial_type'] = tmp3.trial_type.apply(lambda x: 'cue_{}'.format(x))

            tmp4 = d_[['response', 'onset_stim']].copy().rename(columns={'onset_stim':'onset', 
                                                                        'response':'trial_type'})
            tmp4 = tmp4[tmp4.trial_type != -1]
            tmp4['trial_type'] = tmp4.trial_type.map({1:'response_left', 2:'response_right'})

            tmp = pd.concat((tmp1, tmp2, tmp3, tmp4))

            tmp['onset'] -= 3.0 # Slice-time correction and off-by-one-error in script

            tmp.to_csv(op.join(derivatives, ds, 'event_files',
                                   'sub-{subject}_task-randomdotmotion_run-{run}_events.tsv').format(**locals()),
                           sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ds', type=str,)
    args = parser.parse_args()

    main('/home/shared/2018/subcortex/bias_task/',
         args.ds)
