rm -rf elapsed_time.log
rm -rf visited_nodes.log

# number of dim
#braided version
for i in 4
do
for n in 2 4 8 16 32 64
do
./cuda${n}d -d 2m -i $i -m 1 -s 1
done
done


# number of dim
#braided version
for i in 4 
do
for n in 2 4 8 16 32 64
do
./cuda${n}d -d 2m -i $i -m 1  -p 128 -s 1
done
done














