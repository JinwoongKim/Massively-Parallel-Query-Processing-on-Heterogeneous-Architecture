nvprof --metrics gld_transactions ./cuda2f -d 1m -m 4 -q 4096 -v 0 2>&1> /dev/null | grep -i globalMPHR2* -A 1
nvprof --metrics gld_transactions ./cuda2n -d 1m -m 4 -q 4096 -v 0 2>&1> /dev/null | grep -i globalMPHR2* -A 1
nvprof --metrics gld_transactions ./cuda2n -d 1m -m 5 -q 4096 -v 0 2>&1> /dev/null | grep -i globalshort* -A 1
nvprof --metrics gld_transactions ./cuda2f -d 1m -m 6 -q 4096 -v 0 2>&1> /dev/null | grep -i globalparent* -A 1
nvprof --metrics gld_transactions ./cuda2n -d 1m -m 6 -q 4096 -v 0 2>&1> /dev/null | grep -i globalparent* -A 1
nvprof --metrics gld_transactions ./cuda2f -d 1m -m 7 -q 4096 -v 0 2>&1> /dev/null | grep -i globalskip* -A 1
nvprof --metrics gld_transactions ./cuda2n -d 1m -m 7 -q 4096 -v 0 2>&1> /dev/null | grep -i globalskip* -A 1

