stage=2


listsdir='/data1/srikanthr/Coswara/data_preparation/LISTS/'
audiodir='/data1/srikanthr/Coswara/data_preparation/Coswara-Data-Extracted'
audiocategory='cough-heavy'

datadir_name='data'
datadir=$datadir_name/$audiocategory
feature_dir_name='feats'
feature_dir=$feature_dir_name/$audiocategory
output_dir='results'

train_config='conf/train.conf'
feats_config='conf/feature.conf'

. parse_options.sh

# if [ $stage -le 0 ]; then
# 	echo "==== Preparing data folders ====="
# 	mkdir -p $datadir
# 	cat $listsdir/*_list | sort | uniq >$datadir/allfiles.scp
# 	awk -v audiodir=$audiodir -v audiocat=$audiocategory '{print $1" "audiodir"/"$1"/"audiocat".wav"}' < $datadir/allfiles.scp >$datadir/wav.scp
# 	for fold in $(seq 1 5);do
# 		mkdir -p $datadir/fold_$fold
# 		for item in train val;do
# 			awk -v audiodir=$audiodir -v audiocat=$audiocategory '{print $1" "audiodir"/"$1"/"audiocat".wav"}' < $listsdir/${item}_fold_${fold}_list >$datadir/fold_$fold/$item.scp	
# 			awk '{print $0}'  $listsdir/${item}_fold_${fold}_list >$datadir/fold_$fold/${item}_labels
# 		done
# 	done
# 	awk -v audiodir=$audiodir -v audiocat=$audiocategory '{print $1" "audiodir"/"$1"/"audiocat".wav"}' < $listsdir/test_list >$datadir/test.scp	
# 	awk '{print $0}' $listsdir/test_list >$datadir/test_labels
# fi

if [ $stage -le 1 ]; then
	# Creates a separate pickle file containing feature matrix for each recording in the wav.scp
	# Expects a data folder, with train_dev and eval folders inside. each folder has a wav.scp file
	# Each row in wav.scp is formatted as: <wav_id> <wav_file_path>
	# Feature matrices are written to: $feature_dir/{train_dev/eval_set}/<wav_id>_<feature_type>.pkl
	# feature.conf specifies configuration settings for feature extraction

	echo "==== Feature extraction ====="
	mkdir -p $feature_dir
	python feature_extraction.py -c $feats_config -i $datadir/all.scp -o $feature_dir
	cp $feature_dir/feats.scp $datadir/feats.scp
fi
#exit
# Logistic Regression
if [ $stage -le 2 ]; then
	output_dir_name='results_lr'
	output_dir=$output_dir_name/$audiocategory
	train_config='conf/train_lr.conf'

	mkdir -p $output_dir
	echo "========= Logistic regression classifier ======================"
	cat $train_config
	for fold in $(seq 0 0);do
		mkdir -p $output_dir
		#cp $datadir/feats.scp $datadir/fold_${fold}/
		# Train
		#python train.py	-c $train_config -d $datadir -o $output_dir
		# Validate
		python infer.py --modelfil $output_dir/model.pkl --featsfil $datadir/feats.scp --file_list $datadir/val.scp --outfil $output_dir/val_scores.txt
		# Score
		python scoring.py --ref_file $datadir_name/val --target_file $output_dir/val_scores.txt --output_file $output_dir/val_results.pkl
	
		python infer.py --modelfil $output_dir/model.pkl --featsfil $datadir/feats.scp --file_list $datadir/test1.scp --outfil $output_dir/test1_scores.txt
		# Score
		python scoring.py --ref_file $datadir_name/test1 --target_file $output_dir/test1_scores.txt --output_file $output_dir/test1_results.pkl
	
		python infer.py --modelfil $output_dir/model.pkl --featsfil $datadir/feats.scp --file_list $datadir/test2.scp --outfil $output_dir/test2_scores.txt
		# Score
		python scoring.py --ref_file $datadir_name/test2 --target_file $output_dir/test2_scores.txt --output_file $output_dir/test2_results.pkl
	
		python infer.py --modelfil $output_dir/model.pkl --featsfil $datadir/feats.scp --file_list $datadir/test1_test2.scp --outfil $output_dir/test1_test2_scores.txt
		# Score
		python scoring.py --ref_file $datadir_name/test1_test2 --target_file $output_dir/test1_test2_scores.txt --output_file $output_dir/test1_test2_results.pkl
	
	done
	# below file can be uploaded to scoring server to appear on leaderboard for development set performance 
	#cat $output_dir/fold_1/val_scores.txt $output_dir/fold_2/val_scores.txt $output_dir/fold_3/val_scores.txt $output_dir/fold_4/val_scores.txt $output_dir/fold_5/val_scores.txt > $output_dir/val_scores_allfolds.txt
	# summarize all folds performance
	#python summarize.py $output_dir

fi
#exit
# Random Forest 
if [ $stage -le 3 ]; then
	output_dir_name='results_rf'
	output_dir=$output_dir_name/$audiocategory
	train_config='conf/train_rf.conf'

	mkdir -p $output_dir
	echo "========= RF classifier ======================"
	cat $train_config
	for fold in $(seq 0 0);do
		mkdir -p $output_dir
		#cp $datadir/feats.scp $datadir/fold_${fold}/
		# Train
		#python train.py	-c $train_config -d $datadir -o $output_dir
		# Validate
		python infer.py --modelfil $output_dir/model.pkl --featsfil $datadir/feats.scp --file_list $datadir/val.scp --outfil $output_dir/val_scores.txt
		# Score
		python scoring.py --ref_file $datadir_name/val --target_file $output_dir/val_scores.txt --output_file $output_dir/val_results.pkl
	
		python infer.py --modelfil $output_dir/model.pkl --featsfil $datadir/feats.scp --file_list $datadir/test1.scp --outfil $output_dir/test1_scores.txt
		# Score
		python scoring.py --ref_file $datadir_name/test1 --target_file $output_dir/test1_scores.txt --output_file $output_dir/test1_results.pkl
	
		python infer.py --modelfil $output_dir/model.pkl --featsfil $datadir/feats.scp --file_list $datadir/test2.scp --outfil $output_dir/test2_scores.txt
		# Score
		python scoring.py --ref_file $datadir_name/test2 --target_file $output_dir/test2_scores.txt --output_file $output_dir/test2_results.pkl
	
		python infer.py --modelfil $output_dir/model.pkl --featsfil $datadir/feats.scp --file_list $datadir/test1_test2.scp --outfil $output_dir/test1_test2_scores.txt
		# Score
		python scoring.py --ref_file $datadir_name/test1_test2 --target_file $output_dir/test1_test2_scores.txt --output_file $output_dir/test1_test2_results.pkl
	
	done
	# below file can be uploaded to scoring server to appear on leaderboard for development set performance 
	#cat $output_dir/fold_1/val_scores.txt $output_dir/fold_2/val_scores.txt $output_dir/fold_3/val_scores.txt $output_dir/fold_4/val_scores.txt $output_dir/fold_5/val_scores.txt > $output_dir/val_scores_allfolds.txt
	# summarize all folds performance
	#python summarize.py $output_dir

fi

# Multi-Layer Perceptron
if [ $stage -le 4 ]; then
	output_dir_name='results_mlp'
	output_dir=$output_dir_name/$audiocategory
	train_config='conf/train_mlp.conf'

	mkdir -p $output_dir
	echo "========= MLP classifier ======================"
	cat $train_config
	for fold in $(seq 0 0);do
		mkdir -p $output_dir
		#cp $datadir/feats.scp $datadir/fold_${fold}/
		# Train
		#python train.py	-c $train_config -d $datadir -o $output_dir
		# Validate
		python infer.py --modelfil $output_dir/model.pkl --featsfil $datadir/feats.scp --file_list $datadir/val.scp --outfil $output_dir/val_scores.txt
		# Score
		python scoring.py --ref_file $datadir_name/val --target_file $output_dir/val_scores.txt --output_file $output_dir/val_results.pkl
		
		python infer.py --modelfil $output_dir/model.pkl --featsfil $datadir/feats.scp --file_list $datadir/test1.scp --outfil $output_dir/test1_scores.txt
		# Score
		python scoring.py --ref_file $datadir_name/test1 --target_file $output_dir/test1_scores.txt --output_file $output_dir/test1_results.pkl
	
		python infer.py --modelfil $output_dir/model.pkl --featsfil $datadir/feats.scp --file_list $datadir/test2.scp --outfil $output_dir/test2_scores.txt
		# Score
		python scoring.py --ref_file $datadir_name/test2 --target_file $output_dir/test2_scores.txt --output_file $output_dir/test2_results.pkl
	
		python infer.py --modelfil $output_dir/model.pkl --featsfil $datadir/feats.scp --file_list $datadir/test1_test2.scp --outfil $output_dir/test1_test2_scores.txt
		# Score
		python scoring.py --ref_file $datadir_name/test1_test2 --target_file $output_dir/test1_test2_scores.txt --output_file $output_dir/test1_test2_results.pkl
	
	done
	# below file can be uploaded to scoring server to appear on leaderboard for development set performance 
	#cat $output_dir/fold_1/val_scores.txt $output_dir/fold_2/val_scores.txt $output_dir/fold_3/val_scores.txt $output_dir/fold_4/val_scores.txt $output_dir/fold_5/val_scores.txt > $output_dir/val_scores_allfolds.txt
	# summarize all folds performance
	#python summarize.py $output_dir

fi
exit
if [ $stage -le 5 ]; then
	output_dir_name='results_linearSVM'
	output_dir=$output_dir_name/$audiocategory
	train_config='conf/train_linearSVM.conf'

	mkdir -p $output_dir
	echo "========= linearSVM classifier ======================"
	cat $train_config
	for fold in $(seq 0 0);do
		mkdir -p $output_dir
		#cp $datadir/feats.scp $datadir/fold_${fold}/
		# Train
		python train.py	-c $train_config -d $datadir -o $output_dir
		# Validate
		python infer.py --modelfil $output_dir/model.pkl --featsfil $datadir/feats.scp --file_list $datadir/val.scp --outfil $output_dir/val_scores.txt
		# Score
		python scoring.py --ref_file $datadir_name/val --target_file $output_dir/val_scores.txt --output_file $output_dir/val_results.pkl
	done
	# below file can be uploaded to scoring server to appear on leaderboard for development set performance 
	#cat $output_dir/fold_1/val_scores.txt $output_dir/fold_2/val_scores.txt $output_dir/fold_3/val_scores.txt $output_dir/fold_4/val_scores.txt $output_dir/fold_5/val_scores.txt > $output_dir/val_scores_allfolds.txt
	# summarize all folds performance
	#python summarize.py $output_dir

fi

echo "Done!!!"
