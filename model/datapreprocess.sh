root=/data1/gs/annotation

for dir in clean other; do
    python3 datapreprocess.py $root/$dir
done
