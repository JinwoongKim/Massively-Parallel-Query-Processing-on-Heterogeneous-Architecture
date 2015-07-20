#nvprof --profile-from-start off --metrics gld_transactions ./cuda128n -d 32m -m 4 -q 12288 -v 6 2>&1> /dev/null | grep -i globalMPHR2* -A 1
#nvprof --profile-from-start off --metrics gld_transactions ./cuda128n -d 32m -m 6 -q 12288 -v 6 2>&1> /dev/null | grep -i globalparent* -A 1
#nvprof --profile-from-start off --metrics gld_transactions ./cuda128n -d 32m -m 7 -q 12288 -v 6 2>&1> /dev/null | grep -i globalskip* -A 1
#
#nvprof --profile-from-start off --metrics l1_cache_global_hit_rate ./cuda128n -d 32m -m 4 -q 12288 -v 6 2>&1> /dev/null | grep -i globalMPHR2* -A 1
#nvprof --profile-from-start off --metrics l1_cache_global_hit_rate ./cuda128n -d 32m -m 6 -q 12288 -v 6 2>&1> /dev/null | grep -i globalparent* -A 1
#nvprof --profile-from-start off --metrics l1_cache_global_hit_rate ./cuda128n -d 32m -m 7 -q 12288 -v 6 2>&1> /dev/null | grep -i globalskip* -A 1

#nvprof --profile-from-start off --metrics gld_throughput ./cuda128n -d 32m -m 4 -q 12288 -v 0 2>&1> /dev/null | grep -i globalMPHR2* -A 1
#nvprof --profile-from-start off --metrics gld_throughput ./cuda128n -d 32m -m 6 -q 12288 -v 0 2>&1> /dev/null | grep -i globalparent* -A 1
#nvprof --profile-from-start off --metrics gld_throughput ./cuda128n -d 32m -m 7 -q 12288 -v 0 2>&1> /dev/null | grep -i globalskip* -A 1

#nvprof --profile-from-start off --metrics l2_l1_read_hit_rate ./cuda128n -d 32m -m 4 -q 12288 -v 0 2>&1> /dev/null | grep -i globalMPHR2* -A 1
#nvprof --profile-from-start off --metrics l2_l1_read_hit_rate ./cuda128n -d 32m -m 6 -q 12288 -v 0 2>&1> /dev/null | grep -i globalparent* -A 1
#nvprof --profile-from-start off --metrics l2_l1_read_hit_rate ./cuda128n -d 32m -m 7 -q 12288 -v 0 2>&1> /dev/null | grep -i globalskip* -A 1



nvprof --profile-from-start off --metrics gld_transactions ./cuda128n -d 32m -m 5 -q 12288 -v 0 2>&1> /dev/null | grep -i globalshort* -A 1
nvprof --profile-from-start off --metrics l1_cache_global_hit_rate ./cuda128n -d 32m -m 5 -q 12288 -v 0 2>&1> /dev/null | grep -i globalshort* -A 1
nvprof --profile-from-start off --metrics l2_l1_read_hit_rate ./cuda128n -d 32m -m 5 -q 12288 -v 0 2>&1> /dev/null | grep -i globalshort* -A 1
nvprof --profile-from-start off --metrics gld_throughput ./cuda128n -d 32m -m 5 -q 12288 -v 0 2>&1> /dev/null | grep -i globalshort* -A 1
