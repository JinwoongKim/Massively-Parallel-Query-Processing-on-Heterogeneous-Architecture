rm -rf elapsed_time.log
rm -rf visited_nodes.log

#number of data
#braided version

for i in 0 
do
for d in 1 2 4 8 16 32
do 
./cuda -d ${d}m -i $i -m 4 # 5  
 #./cuda -d ${d}m -i $i -m 6 
 #./cuda -d ${d}m -i $i -m 7  
done
done

for i in 0 
do
for d in 1 2 4 8 16 32
do 
./cuda -d ${d}m -i $i -m 4  -b 1
 #./cuda -d ${d}m -i $i -m 6  -b 1
 #./cuda -d ${d}m -i $i -m 7  -b 1
done
done


for i in 0 
do
for d in 1 2 4 8 16 32
do 
./cuda -d ${d}m -i $i -m 4 -p 128 -o 1
 #./cuda -d ${d}m -i $i -m 6 -p 128 -o 1
 #./cuda -d ${d}m -i $i -m 7 -p 128 -o 1
done
done







