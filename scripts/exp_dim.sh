# number of dim
#braided version
for p in 1 128
do
for i in  0 4
do

./cuda2d -d 32m -i $i -m 1 4 5 -s 1 -p $p
./cuda2d -d 32m -i $i -m 6 -s 1 -p $p
./cuda2d -d 32m -i $i -m 7 -s 1 -p $p

./cuda4d -d 16m -i $i -m 1 4 5 -s 1 -p $p
./cuda4d -d 16m -i $i -m 6 -s 1 -p $p
./cuda4d -d 16m -i $i -m 7 -s 1 -p $p

./cuda8d -d 8m -i $i -m 1 4 5 -s 1 -p $p
./cuda8d -d 8m -i $i -m 6 -s 1 -p $p
./cuda8d -d 8m -i $i -m 7 -s 1 -p $p

./cuda16d -d 4m -i $i -m 1 4 5 -s 1 -p $p
./cuda16d -d 4m -i $i -m 6 -s 1 -p $p
./cuda16d -d 4m -i $i -m 7 -s 1 -p $p

./cuda32d -d 2m -i $i -m 1 4 5 -s 1 -p $p
./cuda32d -d 2m -i $i -m 6 -s 1 -p $p
./cuda32d -d 2m -i $i -m 7 -s 1 -p $p

./cuda64d -d 1m -i $i -m 1 4 5 -s 1 -p $p
./cuda64d -d 1m -i $i -m 6 -s 1 -p $p
./cuda64d -d 1m -i $i -m 7 -s 1 -p $p

done
done
