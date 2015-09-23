nvprof --aggregate-mode off --events l1_shared_bank_conflict $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12}  ${13}  ${14}  ${15}  ${16}  ${17}  ${18}  ${19}  ${20}  2>&1> /dev/null | grep Shortstack -A 1


