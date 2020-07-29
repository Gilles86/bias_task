import pandas as pd
from_bids_to_leipzig_mapping = {'06': '1', '11': '2', '18': '3', '17': '4', '07': '5', '08': '6', '03': '7', '05': '8', '12': '9', '14': '10', '15': '11', '01': '12', '09': '13', '10': '14', '13': '16', '19': '17', '04': '18', '02': '19', '16': '20'}


df = []
for ds in ['ds-01', 'ds-02']:
    if ds == 'ds-02':
        subjects = ['{:02d}'.format(subject) for subject in range(1, 16)]
        subjects.pop(3)
        fn_template = 'S{subject_}_log_STNREP_dot_bias_fmri.txt'
    elif ds == 'ds-01':
        subjects = ['{:02d}'.format(subject) for subject in range(1, 20)]
        fn_template = 'S{subject_}_log_bias.txt'

    for subject in subjects:
        
        if ds == 'ds-01':
            subject_ = from_bids_to_leipzig_mapping[subject]
        else:
            subject_ = subject


        d = pd.read_table(('/home/raw_data/2018/subcortex/bias_task/raw/{ds}/behavior/'+fn_template).format(**locals()), sep=',')

        d['subject'] = subject
        d['ds'] = ds
        df.append(d)


df = pd.concat(df)

def get_cue_congruency(row):
    if row.cue == 'neutral':
        return 'neutral'
    elif row.stimulus == row.cue:
        return 'congruent'
    else:
        return 'incongruent'


df['cue congruency'] = df.apply(get_cue_congruency, 1)

df.to_pickle('/home/shared/2018/subcortex/bias_task/behavior.pkl')
