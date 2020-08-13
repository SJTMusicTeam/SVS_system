mkdir ../ch_asr/exp/alignment 
mkdir ../ch_asr/exp/alignment/clean_set
mkdir ../ch_asr/exp/alignment/other_set 

phone_path=../ch_asr/exp/chain/tdnn_clean_set/phones.txt
phone_post_path=../ch_asr/exp/chain
text_path=../ch_asr/data
output_path=../ch_asr/exp/alignment

python3 alignment.py $phone_path $phone_post_path/tdnn_clean_set $text_path/clean_set/text $output_path/clean_set TDNN

python3 alignment.py $phone_path $phone_post_path/tdnn_other_set $text_path/other_set/text $output_path/other_set TDNN

echo "finished alignment"
