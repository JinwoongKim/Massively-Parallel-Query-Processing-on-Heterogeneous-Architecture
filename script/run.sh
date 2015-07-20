
#pddartitioned version
for i in 0
do
for d in 1 2 4 8 16
do 
./cuda -d $d -m 1 2 3 4 5 -i $i 
./cuda -d $d -m 6  -i $i 
./cuda -d $d -m 7  -i $i 
done
done









