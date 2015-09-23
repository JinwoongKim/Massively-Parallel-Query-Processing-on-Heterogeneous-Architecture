for i in 512 256 128 64 32 16 8 4 2
do
	nvprof --events gld_inst_32bit ./cuda${i}n -d 32m -m 5 -q 81920 2>&1> /dev/null | grep -i globalshort* -A 1
	nvprof --events gld_inst_64bit ./cuda${i}n -d 32m -m 5 -q 81920 2>&1> /dev/null | grep -i globalshort* -A 1
done

for i in 512 256 128 64 32 16 8 4 2
do
	nvprof --events gld_inst_32bit ./cuda${i}f -d 32m -m 6 -q 81920 2>&1> /dev/null | grep -i globalparent* -A 1
	nvprof --events gld_inst_64bit ./cuda${i}f -d 32m -m 6 -q 81920 2>&1> /dev/null | grep -i globalparent* -A 1
	nvprof --events gld_inst_32bit ./cuda${i}n -d 32m -m 6 -q 81920 2>&1> /dev/null | grep -i globalparent* -A 1
	nvprof --events gld_inst_64bit ./cuda${i}n -d 32m -m 6 -q 81920 2>&1> /dev/null | grep -i globalparent* -A 1
done

for i in 512 256 128 64 32 16 8 4 2
do
	nvprof --events gld_inst_32bit ./cuda${i}f -d 32m -m 7 -q 81920 2>&1> /dev/null | grep -i globalskip* -A 1
	nvprof --events gld_inst_64bit ./cuda${i}f -d 32m -m 7 -q 81920 2>&1> /dev/null | grep -i globalskip* -A 1
	nvprof --events gld_inst_32bit ./cuda${i}n -d 32m -m 7 -q 81920 2>&1> /dev/null | grep -i globalskip* -A 1
	nvprof --events gld_inst_64bit ./cuda${i}n -d 32m -m 7 -q 81920 2>&1> /dev/null | grep -i globalskip* -A 1
done


