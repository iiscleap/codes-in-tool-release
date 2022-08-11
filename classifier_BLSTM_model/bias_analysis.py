from email import header
import pandas as pd
import numpy as np
import os
import scipy
import local.scoring as scoring
from pdb import set_trace as bp

def save_bias_analysis_results(save_dir, df_sample, df_score):
    df_sample.to_csv(os.path.join(save_dir, 'df_sampled_'+str(seed)), header=False, index=False, sep=' ')
    df_score_sampled = df_score.loc[df_score.id.isin(df_sample.id.values)]
    df_score_sampled.to_csv(os.path.join(save_dir, 'df_score_'+str(seed)+'.txt'), header=False, index=False, sep=' ')
    scoring.scoring(os.path.join(save_dir, 'df_sampled_'+str(seed)), os.path.join(save_dir, 'df_score_'+str(seed)+'.txt'), os.path.join(save_dir, 'score_'+str(seed)+'.pkl'))

def save_MWU_test_results(df_sample, df_score):
    x = df_score.score.values
    df_score_sampled = df_score.loc[df_score.id.isin(df_sample.id.values)]
    y = df_score_sampled.score.values
    _, p = scipy.stats.mannwhitneyu(x, y)
    #print(f'p value: {p}')
    return p

def save_MWU_greater_alternate_hypothesis(df_pos, df_neg, df_score):
    df1 = df_score.loc[df_score.id.isin(df_pos.id.values)]
    df2 = df_score.loc[df_score.id.isin(df_neg.id.values)]
    x = df1.score.values
    y = df2.score.values
    #_, p = scipy.stats.mannwhitneyu(x, y)
    _, p = scipy.stats.mannwhitneyu(x, y, alternative="greater")
    return p

work_dir = 'bias_analysis'
csv_metadata_path = 'data/test1_test2_metadata.csv'
test1_path, test2_path = 'data/test1', 'data/test2'
df_test1 = pd.read_csv(test1_path, header=None, delimiter=' ')
df_test1.columns = ['id', 'label']
test1_ids = df_test1.id.values
df_test2 = pd.read_csv(test2_path, header=None, delimiter=' ')
df_test2.columns = ['id', 'label']
test2_ids = df_test2.id.values
gnd_truth_path ='data/test1_test2'
#gnd_truth_path = 'data/test1'
score_paths = ['/home/debottamd/COSWARA_SPACE/JASA_models/debarpan_fusion_result/results/fusion_combs/test1_scores.txt', '/home/debottamd/COSWARA_SPACE/JASA_models/debarpan_fusion_result/results/fusion_combs/test2_scores.txt']
#score_paths = ['results/fusion/test1_scores.txt', 'results/fusion/test2_scores.txt']
#score_paths = ['results_16k_coswara_18.5.22/audio_fusion/test1_scores.txt', 'results_16k_coswara_18.5.22/audio_fusion/test2_scores.txt']
#score_paths = ['/home/debarpanb/verisk_dicova2/DICOVA/icml_workshop_covid_19/srikanth_e140522_transformer/results/fusion/test1_scores.txt', '/home/debarpanb/verisk_dicova2/DICOVA/icml_workshop_covid_19/srikanth_e140522_transformer/results/fusion/test2_scores.txt']

#score_paths = ['/home/debottamd/COSWARA_SPACE/JASA_models/debarpan_fusion_result/results/fusion_combs/test1_scores.txt']
#score_paths = ['/home/debottamd/COSWARA_SPACE/JASA_models/debarpan_fusion_result/results/fusion_combs/test2_scores.txt']

df_metadata = pd.read_csv(csv_metadata_path)
df_score = pd.concat([pd.read_csv(i, header=None, delimiter=' ') for i in score_paths])
df_score.columns = ['id', 'score']
df_gnd_truth = pd.read_csv(gnd_truth_path, header=None, delimiter=' ')
df_gnd_truth.columns = ['id', 'label']
pos_ids, neg_ids = df_gnd_truth[df_gnd_truth.label=='p'].id.values, df_gnd_truth[df_gnd_truth.label=='n'].id.values

if not os.path.exists(work_dir):
        os.mkdir(work_dir)

# random sample analysis [here we can sample 100 pos and 100 neg and check the AUC. this gives us idea about bias due to random seed]
print('prevelance bias analysis using 10 seeds')
#bp()
# n_pos_sample are 100, 70, 30,15,7 for 25%, 20%, 10%, 5% and 2.5% prevelence(n_rat) respectively

n_pos_sample=7
n_rat=0.025

n_neg_sample=int((n_pos_sample/n_rat)-n_pos_sample)
seeds = [0, 42, 50, 99, 999]

save_dir = os.path.join(work_dir, 'random_sampling')
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
save_dir = os.path.join(save_dir, str(n_rat))
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

for seed in seeds:
    df_sample_pos = df_gnd_truth[df_gnd_truth.label=='p'].sample(n=n_pos_sample, random_state=seed)
    df_sample_neg = df_gnd_truth[df_gnd_truth.label=='n'].sample(n=int(n_neg_sample), random_state=seed)
    df_sample = pd.concat([df_sample_pos,df_sample_neg])
    save_bias_analysis_results(save_dir, df_sample, df_score)
    p_pos = save_MWU_test_results(df_sample_pos, df_score[df_score.id.isin(pos_ids)])
    p_neg = save_MWU_test_results(df_sample_neg, df_score[df_score.id.isin(neg_ids)])
    print(f'p-val for pos:{p_pos}')
    
    print(f'p-val for neg:{p_neg}')
    
    print(f'hmean: {scipy.stats.hmean([p_pos, p_neg])}')

    print(f'p-val for greater alternate hypothesis: {save_MWU_greater_alternate_hypothesis(df_sample_pos, df_sample_neg, df_score)}')


# age bias analysis [here we can select specific age group and check performance]
print('Age bias analysis')
age_group = [15, 30]

save_dir = os.path.join(work_dir, 'age')
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
save_dir = os.path.join(save_dir, str(age_group[0]))
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

df_age = df_metadata.loc[(df_metadata.a>age_group[0]) & (df_metadata.a<age_group[1])]
#bp()
df_sample = df_gnd_truth.loc[df_gnd_truth.id.isin(df_age.id.values)]
n_pos, n_neg = df_sample[df_sample.label=='p'].shape[0], df_sample[df_sample.label=='n'].shape[0]
print(f'no of samples: {df_sample.shape[0]}, positive: {n_pos}, negative: {n_neg}')
save_bias_analysis_results(save_dir, df_sample, df_score)
p_pos = save_MWU_test_results(df_sample[df_sample.label=='p'], df_score[df_score.id.isin(pos_ids)])
p_neg = save_MWU_test_results(df_sample[df_sample.label=='n'], df_score[df_score.id.isin(neg_ids)])
print(f'p-val for pos:{p_pos}')

print(f'p-val for neg:{p_neg}')

print(f'hmean: {scipy.stats.hmean([p_pos, p_neg])}')

df1, df2 = df_sample[df_sample.label=='p'], df_sample[df_sample.label=='n']
print(f'p-val for greater alternate hypothesis: {save_MWU_greater_alternate_hypothesis(df1, df2, df_score)}')
# for seed in seeds:
#     save_bias_analysis_results(save_dir, df_sample.sample(frac=0.6, random_state=seed), df_score)


# gender bias analysis [here we can select specific gender and check performance]
print('Gender bias analysis')
gender_group = 'male' #'male' or 'female'

save_dir = os.path.join(work_dir, 'gender')
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
save_dir = os.path.join(save_dir, gender_group)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
df_gender = df_metadata.loc[df_metadata.g==gender_group]
df_sample = df_gnd_truth.loc[df_gnd_truth.id.isin(df_gender.id.values)]
n_pos, n_neg = df_sample[df_sample.label=='p'].shape[0], df_sample[df_sample.label=='n'].shape[0]
print(f'no of samples: {df_sample.shape[0]}, positive: {n_pos}, negative: {n_neg}')
save_bias_analysis_results(save_dir, df_sample, df_score)
p_pos = save_MWU_test_results(df_sample[df_sample.label=='p'], df_score[df_score.id.isin(pos_ids)])
p_neg = save_MWU_test_results(df_sample[df_sample.label=='n'], df_score[df_score.id.isin(neg_ids)])
print(f'p-val for pos:{p_pos}')

print(f'p-val for neg:{p_neg}')

print(f'hmean: {scipy.stats.hmean([p_pos, p_neg])}')


df1, df2 = df_sample[df_sample.label=='p'], df_sample[df_sample.label=='n']
print(f'p-val for greater alternate hypothesis: {save_MWU_greater_alternate_hypothesis(df1, df2, df_score)}')

#for seed in seeds:
#    save_bias_analysis_results(save_dir, df_sample.sample(frac=0.6, random_state=seed), df_score)

# symptoms bias analysis [here we can select different severity levels of covid and absence/presence of symptoms]
print('Symptoms analysis')
positive_group = 'positive_asymp' #'positive_mild' or 'positive_moderate' or 'positive_asymp'
negative_group = 'healthy' #'healthy' or 'resp_illness_not_identified'
negative_group_symp = 'yes'

save_dir = os.path.join(work_dir, 'symptoms')
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
save_dir = os.path.join(save_dir, positive_group+'_'+negative_group+'symp_'+negative_group_symp)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
#bp()
df_symp = df_metadata.loc[(df_metadata.covid_status.isin([positive_group, negative_group]))]

symp_logic = (df_symp.cold==True) | (df_symp.cough==True) | (df_symp.fever==True) | (df_symp.diarrhoea==True) | (df_symp.st==True) | (df_symp.loss_of_smell==True) | (df_symp.mp==True) | (df_symp.ftg==True) | (df_symp.bd==True)
if negative_group=='healthy' and negative_group_symp=='yes':
    df_symp = df_symp.loc[(symp_logic) & (df_symp.covid_status=='healthy') | (df_symp.covid_status==positive_group)]
elif negative_group=='healthy' and negative_group_symp=='no':
    df_symp = df_symp.loc[~(symp_logic) & (df_symp.covid_status=='healthy') | (df_symp.covid_status==positive_group)]
    

df_sample = df_gnd_truth.loc[df_gnd_truth.id.isin(df_symp.id.values)]
n_pos, n_neg = df_sample[df_sample.label=='p'].shape[0], df_sample[df_sample.label=='n'].shape[0]
print(f'no of samples: {df_sample.shape[0]}, positive: {n_pos}, negative: {n_neg}')
save_bias_analysis_results(save_dir, df_sample, df_score)

p_pos = save_MWU_test_results(df_sample[df_sample.label=='p'], df_score[df_score.id.isin(pos_ids)])
p_neg = save_MWU_test_results(df_sample[df_sample.label=='n'], df_score[df_score.id.isin(neg_ids)])
print(f'p-val for pos:{p_pos}')

print(f'p-val for neg:{p_neg}')

print(f'hmean: {scipy.stats.hmean([p_pos, p_neg])}')

df1, df2 = df_sample[df_sample.label=='p'], df_sample[df_sample.label=='n']
print(f'p-val for greater alternate hypothesis: {save_MWU_greater_alternate_hypothesis(df1, df2, df_score)}')

#for seed in seeds:
#    save_bias_analysis_results(save_dir, df_sample.sample(frac=0.6, random_state=seed), df_score)

# specific symptoms bias analysis [here we can select absence/presence of specific symptoms]

# {
#     "id":"User ID",
#     "a":"Age (number)",
#     "covid_status":"Health status (e.g. : positive_mild, healthy,etc.)",
#     "record_date": "Date when the user recorded and submitted the samples",
#     "ep":"Proficient in English (y/n)",
#     "g":"Gender (male/female/other)" ,
#     "l_c":"Country",
#     "l_l":"Locality",
#     "l_s":"State",
#     "rU":"Returning User (y/n)",
#     "asthma":"Asthma (True/False)",
#     "cough":"Cough (True/False)",
#     "smoker":"Smoker (True/False)",
#     "test_status":"Status of COVID Test (p->Positive, n->Negative, na-> Not taken Test)",
#     "ht":"Hypertension  (True/False)",
#     "cold":"Cold  (True/False)",
#     "diabetes":"Diabetes  (True/False)",
#     "diarrhoea":"Diarrheoa  (True/False)",
#     "um":"Using Mask (y/n)",
#     "ihd":"Ischemic Heart Disease (True/False)",
#     "bd":"Breathing Difficulties (True/False)",
#     "st":"Sore Throat (True/False)",
#     "fever":"Fever (True/False)",
#     "ftg":"Fatigue (True/False)",
#     "mp":"Muscle Pain (True/False)",
#     "loss_of_smell":"Loss of Smell & Taste (True/False)",
#     "cld":"Chronic Lung Disease (True/False)",
#     "pneumonia":"Pneumonia (True/False)",
#     "ctScan":"CT-Scan (y/n if the user has taken a test)",
#     "testType":"Type of test (RAT/RT-PCR)",
#     "test_date":"Date of COVID Test (if taken)",
#     "vacc":"Vaccination status (y->both doses, p->one dose(partially vaccinated), n->no doses)",
#     "ctDate":"Date of CT-Scan",
#     "ctScore":"CT-Score",
#     "others_resp":"Respiratory illnesses other than the listed ones (True/False)",
#     "others_preexist":"Pre-existing conditions other than the listed ones (True/False)"
# }

print('Specific symptoms analysis')

#symp_name = 'asthma'
#symp_name = 'smoker'
symp_name = 'vacc'
#symp_name = 'ep'
#symp_name = 'um'
#symp_name = 'l_s'

extra_dir = 'no'

save_dir = os.path.join(work_dir, 'specific_symptom')
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
save_dir = os.path.join(save_dir, symp_name)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
save_dir = os.path.join(save_dir, extra_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
#bp()
#df_symp = df_metadata.loc[df_metadata[symp_name]==True] # for symp_name = 'asthma'
#df_symp = df_metadata.loc[df_metadata[symp_name].isin(['True', 'y'])] # for symp_name = 'smoker'
#df_symp = df_metadata.loc[df_metadata[symp_name].isin(['n'])] # for symp_name = 'smoker'
df_symp = df_metadata.loc[~(df_metadata[symp_name].isin(['y','p']))] # for symp_name = 'vacc'
#df_symp = df_metadata.loc[~(df_metadata[symp_name].isin([extra_dir]))] # for symp_name = 'l_s'
#df_symp = df_metadata.loc[df_metadata[symp_name].isin(['Karnataka'])] # for symp_name = 'l_s'
#df_symp = df_metadata.loc[df_metadata[symp_name].isin(['West Bengal'])] # for symp_name = 'l_s'
#df_symp = df_metadata.loc[~(df_metadata[symp_name].isin(['Karnataka','Tamil Nadu', 'Kerala', 'Maharashtra']))] # for symp_name = 'l_s'

df_sample = df_gnd_truth.loc[df_gnd_truth.id.isin(df_symp.id.values)]
n_pos, n_neg = df_sample[df_sample.label=='p'].shape[0], df_sample[df_sample.label=='n'].shape[0]
print(f'no of samples: {df_sample.shape[0]}, positive: {n_pos}, negative: {n_neg}')
save_bias_analysis_results(save_dir, df_sample, df_score)
p_pos = save_MWU_test_results(df_sample[df_sample.label=='p'], df_score[df_score.id.isin(pos_ids)])
p_neg = save_MWU_test_results(df_sample[df_sample.label=='n'], df_score[df_score.id.isin(neg_ids)])
print(f'p-val for pos:{p_pos}')

print(f'p-val for neg:{p_neg}')

print(f'hmean: {scipy.stats.hmean([p_pos, p_neg])}')

df1, df2 = df_sample[df_sample.label=='p'], df_sample[df_sample.label=='n']
print(f'p-val for greater alternate hypothesis: {save_MWU_greater_alternate_hypothesis(df1, df2, df_score)}')

# for seed in seeds:
#     save_bias_analysis_results(save_dir, df_sample.sample(frac=0.6, random_state=seed), df_score)

# test1 and test2 bias analysis
print('Strain bias analysis')

testset_name = 'test2'

if testset_name=='test1':
    strain_ids = test1_ids
elif testset_name=='test2':
    strain_ids = test2_ids
else:
    print('unknown strain type..exiting.')
    exit()

save_dir = os.path.join(work_dir, 'strain')
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
save_dir = os.path.join(save_dir, testset_name)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

df_strain = df_metadata.loc[df_metadata.id.isin(strain_ids)]
#bp()
df_sample = df_gnd_truth.loc[df_gnd_truth.id.isin(df_strain.id.values)]
n_pos, n_neg = df_sample[df_sample.label=='p'].shape[0], df_sample[df_sample.label=='n'].shape[0]
print(f'no of samples: {df_sample.shape[0]}, positive: {n_pos}, negative: {n_neg}')
save_bias_analysis_results(save_dir, df_sample, df_score)
#bp()
p_pos = save_MWU_test_results(df_sample[df_sample.label=='p'], df_score[df_score.id.isin(pos_ids)])
p_neg = save_MWU_test_results(df_sample[df_sample.label=='n'], df_score[df_score.id.isin(neg_ids)])
print(f'p-val for pos:{p_pos}')

print(f'p-val for neg:{p_neg}')

print(f'hmean: {scipy.stats.hmean([p_pos, p_neg])}')

df1, df2 = df_sample[df_sample.label=='p'], df_sample[df_sample.label=='n']
print(f'p-val for greater alternate hypothesis: {save_MWU_greater_alternate_hypothesis(df1, df2, df_score)}')
