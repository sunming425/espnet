#!/bin/bash

# Update for multi-lingual TTS (Ming Sun)

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# This is a baseline for "JSALT'18 Multilingual End-to-end ASR for Incomplete Data"
# We use 5 Babel language (Assamese Tagalog Swahili Lao Zulu), Librispeech (English), and CSJ (Japanese)
# as a target language, and use 10 Babel language (Cantonese Bengali Pashto Turkish Vietnamese
# Haitian Tamil Kurmanji Tok-Pisin Georgian) as a non-target language.
# The recipe first build language-independent ASR by using non-target languages

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=0       # start from -1 if you need to start from data download
gpu=            # will be deprecated, please use ngpu
ngpu=1          # number of gpus ("0" uses cpu, otherwise use gpu)
nj=32
dumpdir=dump   # directory to dump full features
verbose=0      # verbose option
seed=1
resume=        # Resume the training from snapshot (if set empty, no effect)

debugmode=1
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.

# feature configuration
fs=16000         # sampling frequency
fmax=""          # maximum frequency
fmin=""          # minimum frequency
n_mels=80        # number of mel basis
n_fft=1024       # number of fft points
n_shift=512      # number of shift points
win_length=""  # number of samples in analysis window


# network archtecture
# encoder related
embed_dim=512
elayers=1
eunits=512
econv_layers=3 # if set 0, no conv layer is used
econv_chans=512
econv_filts=5
do_delta=false # true when using CNN (MUST BE ALLWAYS FALSE FOR TACOTRON)
#etype=blstmp # encoder architecture type
#eprojs=320
#subsample=1_2_2_1_1 # skip every n frame from input to nth layers


# decoder related
spk_embed_dim=512
dlayers=2
dunits=1024
prenet_layers=2  # if set 0, no prenet is used
prenet_units=256
postnet_layers=5 # if set 0, no postnet is used
postnet_chans=512
postnet_filts=5


# attention related
adim=128
aconv_chans=32
aconv_filts=15      # resulting in filter_size = aconv_filts * 2 + 1
cumulate_att_w=true # whether to cumulate attetion weight
use_batch_norm=true # whether to use batch normalization in conv layer
use_concate=true    # whether to concatenate encoder embedding with decoder lstm outputs
use_residual=false  # whether to concatenate encoder embedding with decoder lstm outputs
use_masking=true    # whether to mask the padded part in loss calculation
bce_pos_weight=1.0


# minibatch related
batchsize=32
batch_sort_key="output" # empty or input or output (if empty, shuffled batch will be used)
maxlen_in=150     # if input length  > maxlen_in, batchsize is reduced (if batch_sort_key="", not effect)
maxlen_out=400    # if output length > maxlen_out, batchsize is reduced (if batch_sort_key="", not effect)


# optimization related
lr=1e-3
eps=1e-6
weight_decay=0.0
dropout=0.5
zoneout=0.1
epochs=30


# decoding related
model=model.loss.best
threshold=0.5 # threshold to stop the generation
maxlenratio=10.0
minlenratio=0.0


# exp tag
tag="" # tag for managing experiments.

langs="101 102 103 104 105 106 202 203 204 205 206 207 301 302 303 304 305 306 401 402 403"
recog="107 201 307 404"

. utils/parse_options.sh || exit 1;


# check gpu option usage
if [ ! -z $gpu ]; then
    echo "WARNING: --gpu option will be deprecated."
    echo "WARNING: please use --ngpu option."
    if [ $gpu -eq -1 ]; then
	ngpu=0
    else
	ngpu=1
    fi
fi

# exp tag
tag="" # tag for managing experiments.

# data set
# non-target languages: cantonese bengali pashto turkish vietnamese haitian tamil kurmanji tokpisin georgian
train_set=tr_babel1_bengali
train_dev=dt_babel1_bengali
# non-target
recog_set="dt_babel_bengali et_babel_bengali" # dt_babel_haitian et_babel_haitian\
# dt_babel_tamil et_babel_tamil dt_babel_kurmanji et_babel_kurmanji dt_babel_tokpisin et_babel_tokpisin dt_babel_georgian et_babel_georgian"
# target
recog_set="dt_babel_bengali et_babel_bengali" #dt_babel_swahili et_babel_swahili dt_babel_lao et_babel_lao dt_babel_zulu et_babel_zulu" # dt_csj_japanese et_csj_japanese_1 et_csj_japanese_2 et_csj_japanese_3 dt_libri_english_clean dt_libri_english_other et_libri_english_clean et_libri_english_other

# whole set
recog_set="dt_babel_bengali et_babel_bengali" # dt_babel_bengali et_babel_bengali dt_babel_pashto et_babel_pashto dt_babel_turkish et_babel_turkish  dt_babel_vietnamese et_babel_vietnamese dt_babel_haitian et_babel_haitian dt_babel_swahili et_babel_swahili dt_babel_lao et_babel_lao dt_babel_tagalog et_babel_tagalog  dt_babel_tamil et_babel_tamil dt_babel_kurmanji et_babel_kurmanji dt_babel_zulu et_babel_zulu dt_babel_tokpisin et_babel_tokpisin dt_babel_georgian et_babel_georgian" #  dt_csj_japanese et_csj_japanese_1 et_csj_japanese_2 et_csj_japanese_3 dt_libri_english_clean dt_libri_english_other et_libri_english_clean et_libri_english_other

# subset options
# select the number of speakers for subset training experiments. (e.g. 1000; select 1000 speakers). Default: select the whole train set.
subset_num_spk=""

. utils/parse_options.sh || exit 1;

# data set
train_set=tr_babel1_bengali${subset_num_spk:+_${subset_num_spk}spk}

# data directories
#csjdir=../../csj
#libridir=../../librispeech
babeldir=../../babel

. ./path.sh
. ./cmd.sh

# check gpu option usage
if [ ! -z $gpu ]; then
    echo "WARNING: --gpu option will be deprecated."
    echo "WARNING: please use --ngpu option."
    if [ $gpu -eq -1 ]; then
        ngpu=0
    else
        ngpu=1
    fi
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

if [ ${stage} -le 0 ]; then
    # TODO
    # add a check whether the following data preparation is completed or not

    # # CSJ Japanese
    # if [ ! -d "$csjdir/asr1/data" ]; then
    # 	echo "run $csjdir/asr1/run.sh first"
    # 	exit 1
    # fi
    # lang_code=csj_japanese
    # utils/copy_data_dir.sh --utt-suffix -${lang_code} ../../csj/asr1/data/train_nodup data/tr_${lang_code}
    # utils/copy_data_dir.sh --utt-suffix -${lang_code} ../../csj/asr1/data/train_dev   data/dt_${lang_code}
    # utils/copy_data_dir.sh --utt-suffix -${lang_code} ../../csj/asr1/data/eval1       data/et_${lang_code}_1
    # utils/copy_data_dir.sh --utt-suffix -${lang_code} ../../csj/asr1/data/eval2       data/et_${lang_code}_2
    # utils/copy_data_dir.sh --utt-suffix -${lang_code} ../../csj/asr1/data/eval3       data/et_${lang_code}_3
    # # 1) change wide to narrow chars
    # # 2) lower to upper chars
    # for x in data/*${lang_code}*; do
    #     utils/copy_data_dir.sh ${x} ${x}_org
    #     cat ${x}_org/text | nkf -Z |\
    #         awk '{for(i=2;i<=NF;++i){$i = toupper($i)} print}' > ${x}/text
    #     rm -fr ${x}_org
    # done

    # # librispeech
    # lang_code=libri_english
    # if [ ! -d "$libridir/asr1/data" ]; then
    # 	echo "run $libridir/asr1/run.sh first"
    # 	exit 1
    # fi
    # utils/copy_data_dir.sh --utt-suffix -${lang_code} ../../librispeech/asr1/data/train_960  data/tr_${lang_code}
    # utils/copy_data_dir.sh --utt-suffix -${lang_code} ../../librispeech/asr1/data/dev_clean  data/dt_${lang_code}_clean
    # utils/copy_data_dir.sh --utt-suffix -${lang_code} ../../librispeech/asr1/data/dev_other  data/dt_${lang_code}_other
    # utils/copy_data_dir.sh --utt-suffix -${lang_code} ../../librispeech/asr1/data/test_clean data/et_${lang_code}_clean
    # utils/copy_data_dir.sh --utt-suffix -${lang_code} ../../librispeech/asr1/data/test_other data/et_${lang_code}_other

    # Babel
    for x in 103-bengali; do #101-cantonese 102-assamese 103-bengali 104-pashto 105-turkish 106-tagalog 107-vietnamese 201-haitian 202-swahili 203-lao 204-tamil 205-kurmanji 206-zulu 207-tokpisin 404-georgian; do
	langid=`echo $x | cut -f 1 -d"-"`
	lang_code=`echo $x | cut -f 2 -d"-"`
	if [ ! -d "$babeldir/tts1_${lang_code}/data" ]; then
	    echo "run $babeldir/tts1/local/run_all.sh first"
	    exit 1
	fi
        utils/copy_data_dir.sh --utt-suffix -${lang_code} ${babeldir}/tts1_${lang_code}/data/train          data/tr_babel_${lang_code}
        utils/copy_data_dir.sh --utt-suffix -${lang_code} ${babeldir}/tts1_${lang_code}/data/dev            data/dt_babel_${lang_code}
        utils/copy_data_dir.sh --utt-suffix -${lang_code} ${babeldir}/tts1_${lang_code}/data/eval_${langid} data/et_babel_${lang_code}
    done
fi


feat_tr_dir=${dumpdir}/${train_set}_${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}_${train_set}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ]; then
    utils/combine_data.sh data/tr_babel1_bengali_org data/tr_babel_bengali  #data/tr_babel_cantonese data/tr_babel_bengali data/tr_babel_pashto data/tr_babel_turkish data/tr_babel_vietnamese data/tr_babel_haitian data/tr_babel_tamil data/tr_babel_kurmanji data/tr_babel_tokpisin data/tr_babel_georgian
    utils/combine_data.sh data/dt_babel1_bengali_org data/dt_babel_bengali  #data/dt_babel_cantonese data/dt_babel_bengali data/dt_babel_pashto data/dt_babel_turkish data/dt_babel_vietnamese data/dt_babel_haitian data/dt_babel_tamil data/dt_babel_kurmanji data/dt_babel_tokpisin data/dt_babel_georgian

    if [ ! -z $subset_num_spk ]; then
        # create a trainng subset with ${subset_num_spk} speakers (in total 7470)
        head -n $subset_num_spk <(utils/shuffle_list.pl data/tr_babel1_bengali_org/spk2utt | awk '{print $1}') > data/tr_babel1_bengali_org/spk_list_${subset_num_spk}spk
        utils/subset_data_dir.sh \
        --spk-list data/tr_babel1_bengali_org/spk_list_${subset_num_spk}spk \
        data/tr_babel1_bengali_org data/${train_set}_org
    fi

    # remove utt having more than 2000 frames or less than 10 frames or
    # remove utt having more than 200 characters or no more than 0 characters
    remove_longshortdata.sh --maxframes 2000 --maxchars 200 data/${train_set}_org data/${train_set}
    remove_longshortdata.sh --maxframes 2000 --maxchars 200 data/${train_dev}_org data/${train_dev}

    # rm -rf data/${train_set}
    # cp -r data/${train_set}_org data/${train_set}
    # rm -rf data/${train_dev}
    # cp -r data/${train_dev}_org data/${train_dev}
    
    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark
    
    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{01,02,03,04}/${USER}/espnet-data/egs/jsalt18e2e/tts1/dump/${train_set}/delta${do_delta}/storage \
        ${feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{01,02,03,04}/${USER}/espnet-data/egs/jsalt18e2e/tts1/dump/${train_dev}/delta${do_delta}/storage \
        ${feat_dt_dir}/storage
    fi

    [ ! -d ${feat_tr_dir}/feats.scp ] && dump.sh --cmd "$train_cmd" --nj 40 --do_delta $do_delta \
						 data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${train_set}_${train_set} ${feat_tr_dir}

    [ ! -d ${feat_dt_dir}/feats.scp ] && dump.sh --cmd "$train_cmd" --nj 40 --do_delta $do_delta \
						 data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${train_dev}_${train_set} ${feat_dt_dir}

    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}_${train_set}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        [ ! -d ${feat_recog_dir}/feats.scp ] && dump.sh --cmd "$train_cmd" --nj 40 --do_delta $do_delta \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask}_${train_set} \
            ${feat_recog_dir}
    done
fi

dict=data/lang_1char/train_units.txt
nlsyms=data/lang_1char/non_lang_syms.txt
echo "dictionary: ${dict}"

if [ ${stage} -le 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/
    
    # echo "make a non-linguistic symbol list for all languages"
    # cut -f 2- data/tr_*/text | grep -o -P '\[.*?\]|\<.*?\>' | sort | uniq > ${nlsyms}
    # cat ${nlsyms}
    
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    cat data/tr_*/text | text2token.py -s 1 -n 1 | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict} # -l ${nlsyms}
    wc -l ${dict}
    
    # make json labels
    data2json.sh --feat ${feat_tr_dir}/feats.scp \
         data/${train_set} ${dict} > ${feat_tr_dir}/data.json # --nlsyms ${nlsyms} 
    data2json.sh --feat ${feat_dt_dir}/feats.scp \
         data/${train_dev} ${dict} > ${feat_dt_dir}/data.json # --nlsyms ${nlsyms} 
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}_${train_set}/delta${do_delta};
        data2json.sh --feat ${feat_recog_dir}/feats.scp \
		     data/${rtask} ${dict} > ${feat_recog_dir}/data.json # --nlsyms ${nlsyms} 
    done
fi


if [ ${stage} -le 3 ]; then
    echo "stage 3: x-vector extraction"
    # Babel
    for x in 103-bengali; do #101-cantonese 102-assamese 103-bengali 104-pashto 105-turkish 106-tagalog 107-vietnamese 201-haitian 202-swahili 203-lao 204-tamil 205-kurmanji 206-zulu 207-tokpisin 404-georgian; do
	langid=`echo $x | cut -f 1 -d"-"`
	lang_code=`echo $x | cut -f 2 -d"-"`
	if [ ! -d "$babeldir/tts1_${lang_code}/data" ]; then
	    echo "run $babeldir/tts1/local/run_all.sh first"
	    exit 1
	fi
	
	pathPrefix="/export/b11/msun/espnet_ming/egs/babel/tts1_${lang_code}/"

	rm -rf $babeldir/tts1_${lang_code}/data/train_xvector
	cp -r $babeldir/tts1_${lang_code}/data/train ../../babel/tts1_${lang_code}/data/train_xvector
	cat $babeldir/tts1_${lang_code}/exp/xvector_nnet_1a/xvectors_train/feats.scp | awk -v var="${pathPrefix}" '{print $1" "var$2}' > $babeldir/tts1_${lang_code}/data/train_xvector/feats.scp
	utils/fix_data_dir.sh ${babeldir}/tts1_${lang_code}/data/train_xvector # filter files by common utterances
	utils/copy_data_dir.sh --utt-suffix -${lang_code} ${babeldir}/tts1_${lang_code}/data/train_xvector data/tr_babel_${lang_code}_xvector

	rm -rf $babeldir/tts1_${lang_code}/data/dev_xvector
	cp -r $babeldir/tts1_${lang_code}/data/dev ../../babel/tts1_${lang_code}/data/dev_xvector
	cat $babeldir/tts1_${lang_code}/exp/xvector_nnet_1a/xvectors_dev/feats.scp | awk -v var="${pathPrefix}" '{print $1" "var$2}' > $babeldir/tts1_${lang_code}/data/dev_xvector/feats.scp
	utils/fix_data_dir.sh ${babeldir}/tts1_${lang_code}/data/dev_xvector # filter files by common utterances
	utils/copy_data_dir.sh --utt-suffix -${lang_code} ${babeldir}/tts1_${lang_code}/data/dev_xvector data/dt_babel_${lang_code}_xvector

	rm -rf $babeldir/tts1_${lang_code}/data/eval_${langid}_xvector
	cp -r $babeldir/tts1_${lang_code}/data/eval_${langid} $babeldir/tts1_${lang_code}/data/eval_${langid}_xvector
	cat $babeldir/tts1_${lang_code}/exp/xvector_nnet_1a/xvectors_eval_${langid}/feats.scp | awk -v var="${pathPrefix}" '{print $1" "var$2}' > $babeldir/tts1_${lang_code}/data/eval_${langid}_xvector/feats.scp
	utils/fix_data_dir.sh ${babeldir}/tts1_${lang_code}/data/eval_${langid}_xvector # filter files by common utterances
	utils/copy_data_dir.sh --utt-suffix -${lang_code} ${babeldir}/tts1_${lang_code}/data/eval_${langid}_xvector data/et_babel_${lang_code}_xvector
    done

    utils/combine_data.sh data/tr_babel1_bengali_org_xvector data/tr_babel_bengali_xvector #data/tr_babel_cantonese data/tr_babel_bengali data/tr_babel_pashto data/tr_babel_turkish data/tr_babel_vietnamese data/tr_babel_haitian data/tr_babel_tamil data/tr_babel_kurmanji data/tr_babel_tokpisin data/tr_babel_georgian
    utils/combine_data.sh data/dt_babel1_bengali_org_xvector data/dt_babel_bengali_xvector #data/dt_babel_cantonese data/dt_babel_bengali data/dt_babel_pashto data/dt_babel_turkish data/dt_babel_vietnamese data/dt_babel_haitian data/dt_babel_tamil data/dt_babel_kurmanji data/dt_babel_tokpisin data/dt_babel_georgian
    
    # Update json
    local/update_json.sh ${feat_tr_dir}/data.json data/tr_babel1_bengali_org_xvector/feats.scp
    # This should be recog set
    local/update_json.sh ${feat_dt_dir}/data.json data/dt_babel1_bengali_org_xvector/feats.scp
    for rtask in ${recog_set}; do
	if [[ $rtask = *"dt_"* ]]; then # dev set
	    local/update_json.sh ${dumpdir}/${rtask}_${train_set}/delta${do_delta}/data.json data/dt_babel_${lang_code}_xvector/feats.scp
	else # eval set
	    local/update_json.sh ${dumpdir}/${rtask}_${train_set}/delta${do_delta}/data.json data/et_babel_${lang_code}_xvector/feats.scp
	fi
    done
fi


if [ -z ${tag} ];then
    expdir=exp/${train_set}_taco2_enc${embed_dim}
    if [ ${econv_layers} -gt 0 ];then
	expdir=${expdir}-${econv_layers}x${econv_filts}x${econv_chans}
    fi
    expdir=${expdir}-${elayers}x${eunits}_dec${dlayers}x${dunits}
    if [ ${prenet_layers} -gt 0 ];then
	expdir=${expdir}_pre${prenet_layers}x${prenet_units}
    fi
    if [ ${postnet_layers} -gt 0 ];then
	expdir=${expdir}_post${postnet_layers}x${postnet_filts}x${postnet_chans}
    fi
    expdir=${expdir}_att${adim}-${aconv_filts}x${aconv_chans}
    if ${cumulate_att_w};then
	expdir=${expdir}_cm
    fi
    if ${use_batch_norm};then
	expdir=${expdir}_bn
    fi
    if ${use_residual};then
	expdir=${expdir}_rs
    fi
    if ${use_concate};then
	expdir=${expdir}_cc
    fi
    if ${use_masking};then
	expdir=${expdir}_msk_pw${bce_pos_weight}
    fi
    expdir=${expdir}_do${dropout}_zo${zoneout}_lr${lr}_ep${eps}_wd${weight_decay}_bs$((batchsize*ngpu))
    if [ ! -z ${batch_sort_key} ];then
	expdir=${expdir}_sort_by_${batch_sort_key}_mli${maxlen_in}_mlo${maxlen_out}
    fi
    expdir=${expdir}_sd${seed}
else
    expdir=exp/${train_set}_${tag}
fi


if [ ${stage} -le 4 ]; then
    echo "stage 4: Text-to-speech model training"
    tr_json=${feat_tr_dir}/data.json
    dt_json=${feat_dt_dir}/data.json
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        tts_train.py \
	--ngpu ${ngpu} \
	--outdir ${expdir}/results \
	--verbose ${verbose} \
	--seed ${seed} \
	--resume ${resume} \
	--train-json ${tr_json} \
	--valid-json ${dt_json} \
	--embed_dim ${embed_dim} \
	--elayers ${elayers} \
	--eunits ${eunits} \
	--econv_layers ${econv_layers} \
	--econv_chans ${econv_chans} \
	--econv_filts ${econv_filts} \
	--spk_embed_dim ${spk_embed_dim} \
	--dlayers ${dlayers} \
	--dunits ${dunits} \
	--prenet_layers ${prenet_layers} \
	--prenet_units ${prenet_units} \
	--postnet_layers ${postnet_layers} \
	--postnet_chans ${postnet_chans} \
	--postnet_filts ${postnet_filts} \
	--adim ${adim} \
	--aconv-chans ${aconv_chans} \
	--aconv-filts ${aconv_filts} \
	--cumulate_att_w ${cumulate_att_w} \
	--use_batch_norm ${use_batch_norm} \
	--use_concate ${use_concate} \
	--use_residual ${use_residual} \
	--use_masking ${use_masking} \
	--bce_pos_weight ${bce_pos_weight} \
	--lr ${lr} \
	--eps ${eps} \
	--dropout-rate ${dropout} \
	--zoneout-rate ${zoneout} \
	--weight-decay ${weight_decay} \
	--batch_sort_key ${batch_sort_key} \
	--batch-size ${batchsize} \
	--maxlen-in ${maxlen_in} \
	--maxlen-out ${maxlen_out} \
	--epochs ${epochs}
fi
exit


outdir=${expdir}/outputs_${model}_th${threshold}_mlr${minlenratio}-${maxlenratio}
if [ ${stage} -le 5 ];then
    echo "stage 5: Decoding"
    for sets in ${recog_set};do
	[ ! -e  ${outdir}/${sets} ] && mkdir -p ${outdir}/${sets}
	cp ${dumpdir}/${sets}/delta${do_delta}/data.json ${outdir}/${sets}
	splitjson.py --parts ${nj} ${outdir}/${sets}/data.json
	# decode in parallel
	${train_cmd} JOB=1:$nj ${outdir}/${sets}/log/decode.JOB.log \
	    tts_decode.py \
	        --backend pytorch \
		--ngpu 0 \
		--verbose ${verbose} \
		--out ${outdir}/${sets}/feats.JOB \
		--json ${outdir}/${sets}/split${nj}utt/data.JOB.json \
		--model ${expdir}/results/${model} \
		--model-conf ${expdir}/results/model.conf \
		--threshold ${threshold} \
		--maxlenratio ${maxlenratio} \
		--minlenratio ${minlenratio}
	# concatenate scp files
	for n in $(seq $nj); do
	    cat "${outdir}/${sets}/feats.$n.scp" || exit 1;
	done > ${outdir}/${sets}/feats.scp
    done
fi


if [ ${stage} -le 6 ];then
    echo "stage 6: Synthesis"
    for sets in ${recog_set};do
	[ ! -e ${outdir}_denorm/${sets} ] && mkdir -p ${outdir}_denorm/${sets}
	apply-cmvn --norm-vars=true --reverse=true data/${train_set}/cmvn.ark \
		   scp:${outdir}/${sets}/feats.scp \
		   ark,scp:${outdir}_denorm/${sets}/feats.ark,${outdir}_denorm/${sets}/feats.scp
	local/convert_fbank.sh --nj ${nj} --cmd "${train_cmd}" \
			       --fs ${fs} --fmax "${fmax}" --fmin "${fmin}" \
			       --n_mels ${n_mels} --n_fft ${n_fft} --n_shift ${n_shift} \
			       ${outdir}_denorm/${sets} \
			       ${outdir}_denorm/${sets}/log \
			       ${outdir}_denorm/${sets}/wav
    done
fi
