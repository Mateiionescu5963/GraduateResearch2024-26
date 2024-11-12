test_set=0.25
mal_ben=0.5
for window_size in $(seq 200 100 1000); do
  for stride in $(seq 200 50 $window_size); do
	for dataset in $(seq 0 1 1); do
		python3 main.py $window_size $stride $test_set $mal_ben $dataset
	done
  done
done
