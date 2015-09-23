rm -rf elapsed_time.log
rm -rf visited_nodes.log

# number of dim
#braided version
for i in  4
do

./cuda2d -d 64m -i $i -m 1 4 5  -s 1
./cuda2d -d 64m -i $i -m 6 -s 1
./cuda2d -d 64m -i $i -m 7 -s 1

./cuda4d -d 32m -i $i -m 1 4 5  -s 1
./cuda4d -d 32m -i $i -m 6  -s 1
./cuda4d -d 32m -i $i -m 7  -s 1

./cuda8d -d 16m -i $i -m 1 4 5 -s 1
./cuda8d -d 16m -i $i -m 6  -s 1
./cuda8d -d 16m -i $i -m 7  -s 1

./cuda16d -d 8m -i $i -m 1 4 5 -s 1
./cuda16d -d 8m -i $i -m 6  -s 1
./cuda16d -d 8m -i $i -m 7  -s 1

./cuda32d -d 4m -i $i -m 1 4 5  -s 1
./cuda32d -d 4m -i $i -m 6  -s 1
./cuda32d -d 4m -i $i -m 7  -s 1

./cuda64d -d 2m -i $i -m 1 4 -s 1
./cuda64d -d 2m -i $i -m 6  -s 1
./cuda64d -d 2m -i $i -m 7  -s 1

done
