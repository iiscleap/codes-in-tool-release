stage=0

results_directory=results

train_config=conf/train_config
model_config=conf/model_config
feature_config=conf/feature_config

datadir=data
featsdir=feats

. parse_options.sh

if [ $stage -le 0 ];then

	mkdir -p $results_directory
	cp $train_config $results_directory/train_config
	cp $feature_config $results_directory/feature_config
	cp $model_config $results_directory/model_config

	for audio in cough-heavy cough-shallow breathing-deep breathing-shallow vowel-a vowel-e vowel-o counting-normal counting-fast;do
		result_folder=${results_directory}/${audio}
		mkdir -p $result_folder
		echo "=================== Train on dev data============================="

		mkdir -p $result_folder

		/home/srikanthr/.conda/envs/espenv/bin/python local/train.py -c $results_directory/train_config -m $results_directory/model_config -f $featsdir/$audio/feats.scp -t $datadir/train -v $datadir/val -o $result_folder

		for item in val test1_test2;do
			/home/srikanthr/.conda/envs/espenv/bin/python local/infer.py -c $results_directory/train_config -f $results_directory/feature_config -m $result_folder/models/final.mdl -i $datadir/$audio/${item}.scp -o $result_folder/${item}_scores.txt
			/home/srikanthr/.conda/envs/espenv/bin/python local/scoring.py -r $datadir/$item -t $result_folder/${item}_scores.txt -o $result_folder/${item}_results.pkl
		done
	done
	# Score fusion when the test set is available
	mkdir -p $results_directory/audio_fusion
	for item in test1_test2;do
		/home/srikanthr/.conda/envs/espenv/bin/python local/score_fusion.py \
		${results_directory}/breathing-deep/${item}_scores.txt \
		${results_directory}/breathing-shallow/${item}_scores.txt \
		${results_directory}/cough-heavy/${item}_scores.txt \
		${results_directory}/cough-shallow/${item}_scores.txt \
		${results_directory}/vowel-a/${item}_scores.txt \
		${results_directory}/vowel-e/${item}_scores.txt \
		${results_directory}/vowel-o/${item}_scores.txt \
		${results_directory}/counting-fast/${item}_scores.txt \
		${results_directory}/counting-normal/${item}_scores.txt \
		${results_directory}/audio_fusion/${item}_scores.txt False

		/home/srikanthr/.conda/envs/espenv/bin/python local/scoring.py -r $datadir/${item} -t ${results_directory}/audio_fusion/${item}_scores.txt -o ${results_directory}/audio_fusion/${item}_results.pkl
	done
fi

if [ $stage -le 1 ];then
	/home/srikanthr/.conda/envs/espenv/bin/python local/classifier_on_symptoms.py data ${results_directory}/symptoms
	for item in val test1_test2;do
		/home/srikanthr/.conda/envs/espenv/bin/python local/infer_symptoms.py data $item ${results_directory}/symptoms
		/home/srikanthr/.conda/envs/espenv/bin/python local/scoring.py -r data/$item -t ${results_directory}/symptoms/${item}_scores.txt -o  ${results_directory}/symptoms/${item}_results.pkl
	done
fi

if [ $stage -le 2 ];then
	mkdir -p ${results_directory}/fusion
	for item in test1_test2;do
		/home/srikanthr/.conda/envs/espenv/bin/python local/score_fusion.py \
		${results_directory}/breathing-deep/${item}_scores.txt \
		${results_directory}/breathing-shallow/${item}_scores.txt \
		${results_directory}/cough-heavy/${item}_scores.txt \
		${results_directory}/cough-shallow/${item}_scores.txt \
		${results_directory}/vowel-a/${item}_scores.txt \
		${results_directory}/vowel-e/${item}_scores.txt \
		${results_directory}/vowel-o/${item}_scores.txt \
		${results_directory}/counting-fast/${item}_scores.txt \
		${results_directory}/counting-normal/${item}_scores.txt \
		${results_directory}/symptoms/${item}_scores.txt \
		${results_directory}/fusion/${item}_scores.txt False

		/home/srikanthr/.conda/envs/espenv/bin/python local/scoring.py -r $datadir/${item} -t ${results_directory}/fusion/${item}_scores.txt -o ${results_directory}/fusion/${item}_results.pkl
	done
fi
