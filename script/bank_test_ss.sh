nvprof --events l1_shared_bank_conflict cuda -d 1 -q 1000 -b 1 -m 5  2>&1> /dev/null | grep Shortstack -A 1
#nvprof --aggregate-mode off --events l1_shared_bank_conflict cuda -d 1 -q 1000 -b 1 -m 5  2>&1> /dev/null | grep Shortstack -A 1
