cp -f main.cu main.cu.bak # backup main.cu

if [ 1 == 0 ]
then
#for simd efficiency
for i in 2 4 8 16 32 64 128
do
cp -f header/index_${i}t.h header/index.h
make
mv -f cuda cuda${i}t
done
fi


#for tp
cp -f main_tp.cu main.cu 
for i in 2 4 8 16 32 64 128 256 512
do
cp -f header/index_${i}.h header/index.h
make
mv -f cuda cuda${i}f
done

#for non ilp
cp -f main_nonilp.cu main.cu 
for i in 2 4 8 16 32 64 128 256 512
do
cp -f header/index_${i}.h header/index.h
make
mv -f cuda cuda${i}n
done


