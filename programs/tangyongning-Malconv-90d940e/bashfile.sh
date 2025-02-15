test_set=0.25
mal_ben=0.5
#window_size=256
#stride=256
embed=24
mode="maltf"

for window_size in $(seq 320 64 512); do
	for stride in $(seq 64 64 $window_size); do
		python3 main.py $window_size $stride $test_set $mal_ben $embed $mode "True"
	done
done

