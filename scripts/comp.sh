nvprof --profile-from-start off --events l1_local_load_hit ./cuda32f -d 32m -m 4   -q 12288
nvprof --profile-from-start off --events l1_local_load_miss ./cuda32f -d 32m -m 4  -q 12288
nvprof --profile-from-start off --events l1_global_load_hit ./cuda32f -d 32m -m 4  -q 12288
nvprof --profile-from-start off --events l1_global_load_miss ./cuda32f -d 32m -m 4 -q 12288

nvprof --profile-from-start off --events l1_local_load_hit ./cuda32n -d 32m -m 4   -q 12288
nvprof --profile-from-start off --events l1_local_load_miss ./cuda32n -d 32m -m 4  -q 12288
nvprof --profile-from-start off --events l1_global_load_hit ./cuda32n -d 32m -m 4  -q 12288
nvprof --profile-from-start off --events l1_global_load_miss ./cuda32n -d 32m -m 4 -q 12288
