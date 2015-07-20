
for i in 2 4 8 16 32 64 128 256 512
do
nvprof --profile-from-start off --metrics warp_execution_efficiency ./cuda${i}f -d 32m -m 4 -q 12288 -v 0 2>&1> /dev/null | grep -i globalMPHR2* -A 1
nvprof --profile-from-start off --metrics warp_execution_efficiency ./cuda${i}f -d 32m -m 6 -q 12288 -v 0 2>&1> /dev/null | grep -i globalparent* -A 1
nvprof --profile-from-start off --metrics warp_execution_efficiency ./cuda${i}f -d 32m -m 7 -q 12288 -v 0 2>&1> /dev/null | grep -i globalskip* -A 1
done

for i in 2 4 8 16 32 64 128 256 512
do
nvprof --profile-from-start off --metrics warp_execution_efficiency ./cuda${i}n -d 32m -m 4 -q 12288 -v 0 2>&1> /dev/null | grep -i globalMPHR2* -A 1
nvprof --profile-from-start off --metrics warp_execution_efficiency ./cuda${i}n -d 32m -m 6 -q 12288 -v 0 2>&1> /dev/null | grep -i globalparent* -A 1
nvprof --profile-from-start off --metrics warp_execution_efficiency ./cuda${i}n -d 32m -m 7 -q 12288 -v 0 2>&1> /dev/null | grep -i globalskip* -A 1
done
