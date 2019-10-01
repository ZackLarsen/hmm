### Hello! 

Here we are taking the raw text files with the observation token and hidden state part-of-speech (POS) tags and adding in START and EOS markers to demarcate the boundaries between sentences (sequences). This way we do not consider bigrams such as "EOS,START", which would occur very frequently but not make sense in building our model:

```python

import os
import csv
import pandas as pd

# Setting relevant file paths (Mac OS X):
homedir = os.path.dirname('~/Projects/hmm/')
datadir = os.path.join(homedir, 'data/')
WSJ_train = os.path.join(datadir, 'WSJ-train.txt')
WSJ_test = os.path.join(datadir, 'WSJ-test.txt')

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

os_df = pd.DataFrame(observation_state_list)
os_df['start'] = os_df[0].map({'<START>' : 1})
os_df['cumsum'] = os_df['start'].cumsum()

cutoff = int(round(os_df['cumsum'].max()*0.7))

test_start = os_df[os_df['cumsum'] == cutoff].index.values[0]

# Now we can get rid of the cumsum and start columns:
os_df.drop(['start','cumsum'], axis=1, inplace=True)

train_df = os_df.iloc[:test_start,:]
test_df = os_df.iloc[test_start:,:]

train_df.to_csv(os.path.join(datadir, 'wsj_train.csv'), index=False)
test_df.to_csv(os.path.join(datadir, 'wsj_test.csv'), index=False)

```

