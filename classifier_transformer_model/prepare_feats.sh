feature_config=$1
scp=$2
outdir=$3

mkdir -p $outdir
python local/feature_extraction.py -c $feature_config -f $scp -o $outdir
