# %%
import pickle
import pandas as pd
import random
import numpy as np
import os
random.seed(0)
np.random.seed(0)
import torchaudio, torch
from tqdm import tqdm

# %%
audiocategories = ['breathing-deep','breathing-shallow', 'vowel-a', 'vowel-o', 'vowel-e', 'cough-heavy', 'cough-shallow', 'counting-normal', 'counting-fast']
symptoms=['fever', 'cold', 'cough', 'mp', 'loss_of_smell', 'st', 'ftg', 'diarrhoea']
positive_statuses = ['positive_asymp', 'positive_moderate', 'positive_mild']
negative_statuses = ['healthy', 'resp_illness_not_identified']
datadir='data'
if not os.path.exists(datadir):
    os.mkdir(datadir)

# %%
metadata = pd.read_csv('/home/data/Coswara-Data/combined_data.csv')
metadata = metadata[metadata.a>15]
metadata.reset_index(inplace=True, drop=True)
metadata = metadata[metadata.a<=90]
metadata.reset_index(inplace=True, drop=True)

metadata['record_date'] = pd.to_datetime(metadata['record_date'], format='%Y-%m-%d')

omicron_list = metadata[metadata.record_date>'2021-10-01']
omicron_list = omicron_list.reset_index(drop=True)
pre_omicron_list = metadata[~(metadata.record_date>'2021-10-01')]
pre_omicron_list = pre_omicron_list.reset_index(drop=True)

# %%
if not os.path.exists('{}/misc/folders_withaudio.pkl'.format(datadir)):
    folder_list = None

    all_folders = {}
    for item in audiocategories:
        os.system("find /home/data/Coswara-Data/extracted_data/ -name '{}.wav' >temp".format(item))
        temp = open('temp').readlines()
        temp = ['/'.join(line.split('/')[:-1]) for line in temp]
        folders = {}
        for line in temp:
            folders[line.split('/')[-1]] = line
            all_folders[line.split('/')[-1]] = line
        if folder_list:
            folder_list = folder_list.intersection(set(list(folders.keys())))
        else:
            folder_list = set(list(folders.keys()))
        os.remove('temp')
    folders = {}
    for key in folder_list:
        folders[key] = all_folders[key]

    with open('{}/misc/folders_withaudio.pkl'.format(datadir),'wb') as f:
        pickle.dump(folders,f)
else:
    folders = pickle.load(open('{}/misc/folders_withaudio.pkl'.format(datadir),'rb'))


# %%
good_set=None
for audiocategory in audiocategories:
    assert os.path.exists('{}/misc/good_set_{}'.format(datadir,audiocategory)), "run check_for_files script on {} to create good_set_{}".format(audiocategory, audiocategory)
    lst = [line.strip() for line in open('{}/misc/good_set_{}'.format(datadir,audiocategory)).readlines()]
    if good_set:
        good_set = good_set.intersection(set(lst))
    else:
        good_set = set(lst)
good_set = list(good_set)


# %%
clean_coswara_files_ids = pickle.load(open('/home/neerajs/neerajs/work/neeks/codes/coswara/PLoS/data/feature_extracted_subject_ids.pickle','rb'))
subject_ids = list(clean_coswara_files_ids.keys())
commonset = []
for audiocategory in audiocategories:
    sel_ids = [subject for subject in subject_ids if audiocategory in clean_coswara_files_ids[subject]]
    if len(commonset) == 0:
        commonset = sel_ids
    else:
        commonset = [item for item in sel_ids if item in commonset]

pre_omicron_list = pre_omicron_list[pre_omicron_list.id.isin(commonset)]
pre_omicron_list = pre_omicron_list.reset_index(drop=True)

pre_omicron_list = pre_omicron_list[pre_omicron_list.id.isin(good_set)]
pre_omicron_list = pre_omicron_list.reset_index(drop=True)

omicron_list = omicron_list[omicron_list.id.isin(good_set)]
omicron_list = omicron_list.reset_index(drop=True)

updated_metadata = pd.concat([pre_omicron_list,omicron_list])
updated_metadata.reset_index(drop=True)

all_pos_neg_data = updated_metadata[updated_metadata.covid_status.isin(positive_statuses+negative_statuses)]
all_pos_neg_data = all_pos_neg_data.reset_index(drop=True)


# %%
def sample_list(lst, num, var):
    stats = lst.covid_status.value_counts()
    nsamples = [stats[item] for item in var]
    ids = []
    for item in var:
        ns = int(np.round(num*stats[item]/sum(nsamples))) 
        keys = list(lst[lst.covid_status==item].id)
        random.shuffle(keys)
        ids.extend(keys[:ns])
    random.shuffle(ids)
    return ids[:num]


# %%

test_omicron_positives = sample_list(omicron_list, 50, positive_statuses)
test_preomicron_positives = sample_list(pre_omicron_list, 50, positive_statuses)
test_preomicron_negatives = sample_list(pre_omicron_list, 150, negative_statuses)

test_omicron_orig_negatives = list(omicron_list[omicron_list.covid_status.isin(negative_statuses)].id)
temp = pre_omicron_list[pre_omicron_list.covid_status.isin(negative_statuses)]
temp = temp.reset_index(drop=True)
temp = temp[~temp.id.isin(test_preomicron_negatives)]
temp = temp.reset_index(drop=True)
test_omicron_addl_negatives = sample_list(temp, 150-len(test_omicron_orig_negatives), negative_statuses)
test_omicron_negatives = test_omicron_orig_negatives + test_omicron_addl_negatives

# %%
all_test_ids = test_preomicron_positives + test_preomicron_negatives + test_omicron_positives + test_omicron_negatives

dev_metadata = all_pos_neg_data[~all_pos_neg_data.id.isin(all_test_ids)]
dev_metadata = dev_metadata.reset_index(drop=True)
test1_metadata = all_pos_neg_data[all_pos_neg_data.id.isin(test_preomicron_positives+test_preomicron_negatives)]
test1_metadata = test1_metadata.reset_index(drop=True)
test2_metadata = all_pos_neg_data[all_pos_neg_data.id.isin(test_omicron_positives+test_omicron_negatives)]
test2_metadata = test2_metadata.reset_index(drop=True)


dev_pos = dev_metadata[dev_metadata.covid_status.isin(positive_statuses)]
dev_pos = dev_pos.reset_index(drop=True)
val_pos = sample_list(dev_pos, int(len(dev_pos)*0.2), positive_statuses)
dev_neg = dev_metadata[dev_metadata.covid_status.isin(negative_statuses)]
dev_neg = dev_neg.reset_index(drop=True)
val_neg = sample_list(dev_neg, int(len(dev_neg)*0.2), negative_statuses)

val_metadata = dev_metadata[dev_metadata.id.isin(val_pos+val_neg)]
val_metadata = val_metadata.reset_index(drop=True)
train_metadata = dev_metadata[~dev_metadata.id.isin(val_pos+val_neg)]
train_metadata = train_metadata.reset_index(drop=True)

print(dev_metadata.covid_status.value_counts())
print(test1_metadata.covid_status.value_counts())
print(test2_metadata.covid_status.value_counts())


# %%

for audiocategory in audiocategories:
    if not os.path.exists('{}/{}'.format(datadir,audiocategory)):
        os.mkdir('{}/{}'.format(datadir,audiocategory))

test1_metadata.to_csv(open('{}/test1_metadata.csv'.format(datadir),'w'))
test2_metadata.to_csv(open('{}/test2_metadata.csv'.format(datadir),'w'))
dev_metadata.to_csv(open('{}/dev_metadata.csv'.format(datadir),'w'))

with open('{}/test1'.format(datadir),'w') as f:
    for idx,row in test1_metadata.iterrows():
        status = 'p' if row['covid_status'] in positive_statuses else 'n'
        f.write('{} {}\n'.format(row['id'], status))
for audiocategory in audiocategories:
    with open('{}/{}/test1.scp'.format(datadir, audiocategory),'w') as f:
        for idx,row in test1_metadata.iterrows():
            f.write('{} {}/{}.wav\n'.format(row['id'], folders[row['id']], audiocategory))

with open('{}/test2'.format(datadir),'w') as f:
    for idx,row in test2_metadata.iterrows():
        status = 'p' if row['covid_status'] in positive_statuses else 'n'
        f.write('{} {}\n'.format(row['id'], status))
for audiocategory in audiocategories:
    with open('{}/{}/test2.scp'.format(datadir, audiocategory),'w') as f:
        for idx,row in test2_metadata.iterrows():
            f.write('{} {}/{}.wav\n'.format(row['id'], folders[row['id']], audiocategory))

with open('{}/dev'.format(datadir),'w') as f:
    for idx,row in dev_metadata.iterrows():
        status = 'p' if row['covid_status'] in positive_statuses else 'n'
        f.write('{} {}\n'.format(row['id'], status))
for audiocategory in audiocategories:
    with open('{}/{}/dev.scp'.format(datadir, audiocategory),'w') as f:
        for idx,row in dev_metadata.iterrows():
            f.write('{} {}/{}.wav\n'.format(row['id'], folders[row['id']], audiocategory))

with open('{}/val'.format(datadir),'w') as f:
    for idx,row in val_metadata.iterrows():
        status = 'p' if row['covid_status'] in positive_statuses else 'n'
        f.write('{} {}\n'.format(row['id'], status))
for audiocategory in audiocategories:
    with open('{}/{}/val.scp'.format(datadir, audiocategory),'w') as f:
        for idx,row in val_metadata.iterrows():
            f.write('{} {}/{}.wav\n'.format(row['id'], folders[row['id']], audiocategory))

with open('{}/train'.format(datadir),'w') as f:
    for idx,row in train_metadata.iterrows():
        status = 'p' if row['covid_status'] in positive_statuses else 'n'
        f.write('{} {}\n'.format(row['id'], status))
for audiocategory in audiocategories:
    with open('{}/{}/train.scp'.format(datadir, audiocategory),'w') as f:
        for idx,row in train_metadata.iterrows():
            f.write('{} {}/{}.wav\n'.format(row['id'], folders[row['id']], audiocategory))


