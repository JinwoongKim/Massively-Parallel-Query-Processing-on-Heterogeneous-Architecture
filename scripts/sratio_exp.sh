rm -rf elapsed_time.log
rm -rf visited_nodes.log

# selection ratio
#braided version
for i in 0 3
do
for s in 0.01 0.05 0.25 1.25 6.25 
do 
./cuda3d -d 32m -m 4 5 -s $s -i $i 
./cuda3d -d 32m -m 6 -s $s -i $i 
./cuda3d -d 32m -m 7 -s $s -i $i
done
done


#partitioned version
#for i in 0 2 3
#do
#for s in 0.01 0.05 0.25 1.25 6.25 
#do 
#./cuda3d -d 32m -m 1 2 3 4 5 -s $s -i $i -p 128
#./cuda3d -d 32m -m 6 -s $s -i $i -p 128
#./cuda3d -d 32m -m 7 -s $s -i $i -p 128
#done
#done



