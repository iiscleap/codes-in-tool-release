feature_config=$1
scp=$2
outdir=$3

#feature_config should be conf/feature_config.
#
# scp shd be a data/<audio-cat>/dev.scpdata/breathing-deep.scp/dev.scp).
#
#finally,outdir shd be feats/<audio_cat> (e.g. feats/breathing-deep)
#
#so, this code shd be run seperately for 9 audio categories (breathing-deep etc.)
mkdir -p $outdir
python local/feature_extraction.py -c $feature_config -f $scp -o $outdir
