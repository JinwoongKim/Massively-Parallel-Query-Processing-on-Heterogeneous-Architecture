rm -rf elapsed_time.log
rm -rf visited_nodes.log

#TP 
if [  0 == 1 ]
then
for i in 0 
do
for f in 512 256 128 64 32 16 8 4 2
do 
./cuda${f}f -d 32m -i $i -m 4  -q 12288 -v 0
./cuda${f}f -d 32m -i $i -m 6  -q 12288 -v 0
./cuda${f}f -d 32m -i $i -m 7  -q 12288 -v 0
done
done

fi

# original 
for i in 0 
do
for f in 512 256 128 64 32 16 8 4 2
do 
./cuda${f}n -d 32m -i $i -m 4 5  -q 12288 -v 0
./cuda${f}n -d 32m -i $i -m 6  -q 12288 -v 0
./cuda${f}n -d 32m -i $i -m 7  -q 12288 -v 0
done
done



if [ 0 == 1 ]
then


for f in 512 256 128 64 32 16 8 4 2
do
./cuda${f}n -d 32m -m 7 -q 12288 
done


fi











