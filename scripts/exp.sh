rm -rf elapsed_time.log
rm -rf visited_nodes.log

# number of data
#braided version

#partitioned version
for i in 3
do
for d in 1 2 4 8 16 32
do 
./cuda3d -d ${d}m -i $i -m 1 2 3 4 5 -p 128 -o 1
./cuda3d -d ${d}m -i $i -m 6 -p 128 -o 1
./cuda3d -d ${d}m -i $i -m 7 -p 128 -o 1
done
done

# selection ratio

#partitioned version
for i in 3 
do
for s in 0.01 0.05 0.25 1.25 6.25 
do 
./cuda3d -d 32m -m 1 2 3 4 5 -s $s -i $i -p 128 -o 1
./cuda3d -d 32m -m 6 -s $s -i $i -p 128 -o 1
./cuda3d -d 32m -m 7 -s $s -i $i -p 128 -o 1
done
done


# number of dim
#braided version

# number of dim
#braided version

for p in 128
do
for i in 3
do

./cuda2d -d 64m -i $i -m 1 2 3 4 5 -s 1 -p $p -o 1
./cuda2d -d 64m -i $i -m 6 -s 1 -p $p -o 1
./cuda2d -d 64m -i $i -m 7 -s 1 -p $p -o 1

./cuda4d -d 32m -i $i -m 1 2 3 4 5 -s 1 -p $p -o 1
./cuda4d -d 32m -i $i -m 6 -s 1 -p $p -o 1
./cuda4d -d 32m -i $i -m 7 -s 1 -p $p -o 1

./cuda8d -d 16m -i $i -m 1 2 3 4 5 -s 1 -p $p -o 1
./cuda8d -d 16m -i $i -m 6 -s 1 -p $p -o 1
./cuda8d -d 16m -i $i -m 7 -s 1 -p $p -o 1

./cuda16d -d 8m -i $i -m 1 2 3 4 5 -s 1 -p $p -o 1
./cuda16d -d 8m -i $i -m 6 -s 1 -p $p -o 1
./cuda16d -d 8m -i $i -m 7 -s 1 -p $p -o 1

./cuda32d -d 4m -i $i -m 1 2 3 4 5 -s 1 -p $p -o 1
./cuda32d -d 4m -i $i -m 6 -s 1 -p $p -o 1
./cuda32d -d 4m -i $i -m 7 -s 1 -p $p -o 1

./cuda64d -d 2m -i $i -m 1 2 3 4 5 -s 1 -p $p -o 1
./cuda64d -d 2m -i $i -m 6 -s 1 -p $p -o 1
./cuda64d -d 2m -i $i -m 7 -s 1 -p $p -o 1

done
done






