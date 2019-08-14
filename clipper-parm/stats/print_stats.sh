#!/bin/bash

NUMARG=1
if [ "$#" -ne "$NUMARG" ]; then
  echo "Usage: $0 <results_file>"
  echo "  <results_file>: Tar file containg stats"
  exit 1
fi

results_file=$1

tmpdir=.parm_stats
mkdir $tmpdir
tar -xf $results_file -C $tmpdir
meta_file=$tmpdir/meta.txt

red_mode=$(cat $meta_file | awk -F'red_mode=' '{print $NF}' | awk -F' ' '{print $1}')
num_clients=$(cat $meta_file | awk -F'num_clients=' '{print $NF}' | awk -F' ' '{print $1}')
ec_k_val=$(cat $meta_file | awk -F'ec_k_val=' '{print $NF}' | awk -F' ' '{print $1}')
batch_size=$(cat $meta_file | awk -F'batch_size=' '{print $NF}' | awk -F' ' '{print $1}')
num_ignore=$(cat $meta_file | awk -F'num_ignore=' '{print $NF}')
num_warmup=5

full_outfile=${tmpdir}/log.txt
grepped_dir=${tmpdir}/grepped
stats_dir=${tmpdir}/stats
mkdir $grepped_dir
mkdir $stats_dir

for header in E2E_LATENCY ENCODING_LATENCY_SUM DECODING_LATENCY; do
  grep_str="${header}:"
  header_lc=$(echo "${header}" | awk '{print tolower($0)}')
  grep $grep_str $full_outfile | awk -F$grep_str '{print $2}' > /tmp/fil.txt
  python3 filter_warmup.py /tmp/fil.txt $num_clients $ec_k_val $num_warmup $num_ignore $red_mode > ${grepped_dir}/${header_lc}.txt
done

echo "E2E_LATENCY (ms)"
python3 get_latencies.py ${grepped_dir}/e2e_latency.txt \
                       ${stats_dir}/e2e \
                       $red_mode \
                       $batch_size


if [ "$red_mode" == "coded" ]; then
  echo ""
  echo "ENCODING_LATENCY (us)"
  python3 stats.py $grepped_dir/encoding_latency_sum.txt

  echo ""
  echo "DECODING_LATENCY (us)"
  python3 stats.py $grepped_dir/decoding_latency.txt
fi

rm -rf $tmpdir
