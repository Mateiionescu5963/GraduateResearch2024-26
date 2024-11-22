test_set=0.25
mal_ben=0.5

for window_size in 128 256 512 1024; do
	for j in 1 2; do
		stride=$(($window_size/$j))
		for i in 8 16 32 64; do
			python3 main.py $window_size $stride $test_set $mal_ben $i
		done
	done
done
