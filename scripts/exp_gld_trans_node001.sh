for i in 2 4 8 16 32 64 128 256 512
do
	nvprof --metrics gld_transactions ./cuda${i}f -d 32m -m 4 -q 81920 2>&1> /dev/null | grep -i globalMPHR2* -A 1
done

for i in 2 4 8 16 32 64 128 256 512
do
	nvprof --metrics gld_transactions ./cuda${i}n -d 32m -m 4 -q 81920 2>&1> /dev/null | grep -i globalMPHR2* -A 1
done

for i in 2 4 8 16 32 64 128 256 512
do
	nvprof --metrics gld_transactions ./cuda${i}n -d 32m -m 5 -q 81920 2>&1> /dev/null | grep -i globalshort* -A 1
done

for i in 2 4 8 16 32 64 128 256 512
do
	nvprof --metrics gld_transactions ./cuda${i}f -d 32m -m 6 -q 81920 2>&1> /dev/null | grep -i globalparent* -A 1
done
for i in 2 4 8 16 32 64 128 256 512
do
	nvprof --metrics gld_transactions ./cuda${i}n -d 32m -m 6 -q 81920 2>&1> /dev/null | grep -i globalparent* -A 1
done


for i in 2 4 8 16 32 64 128 256 512
do
	nvprof --metrics gld_transactions ./cuda${i}f -d 32m -m 7 -q 81920 2>&1> /dev/null | grep -i globalskip* -A 1
done
for i in 2 4 8 16 32 64 128 256 512
do
	nvprof --metrics gld_transactions ./cuda${i}n -d 32m -m 7 -q 81920 2>&1> /dev/null | grep -i globalskip* -A 1
done
