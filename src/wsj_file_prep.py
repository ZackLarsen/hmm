

import os
import csv
import pandas as pd


# Mac OS X:
homedir = os.path.dirname('/Users/zacklarsen/Zack_master/Projects/Work Projects/hmm/')
datadir = os.path.join(homedir, 'data/')
WSJ_head = os.path.join(datadir, 'WSJ_head.txt')
WSJ_train = os.path.join(datadir, 'WSJ-train.txt')
WSJ_test = os.path.join(datadir, 'WSJ-test.txt')


# From text file
observation_state_list = []
sentence_count = 0
lowercase = True

with open(WSJ_train) as infile:
  for line in infile:
    line = line.strip('\n')
    chars = line.split(' ')
    if len(chars) == 3:
      observation = chars[0].lower()
      state = chars[1]
      observation_state_list.append((observation, state))
    elif len(chars) != 3:
      sentence_count += 1
      observation_state_list.append(('<EOS>', '<EOS>'))
      observation_state_list.append(('<START>', '<START>'))

observation_state_list.insert(0, ('<START>', '<START>'))
observation_state_list.pop()


observation_state_list[:45]
observation_state_list[-5:]


os_df = pd.DataFrame(observation_state_list)
os_df['start'] = os_df[0].map({'<START>' : 1})
os_df['cumsum'] = os_df['start'].cumsum()
os_df


cutoff = int(round(os_df['cumsum'].max()*0.7))
cutoff

test_start = os_df[os_df['cumsum'] == cutoff].index.values[0]
test_start

# Now we can get rid of the cumsum and start columns:
os_df.drop(['start','cumsum'], axis=1, inplace=True)

train_df = os_df.iloc[:test_start,:]
test_df = os_df.iloc[test_start:,:]

train_df
test_df

train_df.to_csv(os.path.join(datadir, 'wsj_train.csv'), index=False)
test_df.to_csv(os.path.join(datadir, 'wsj_test.csv'), index=False)


