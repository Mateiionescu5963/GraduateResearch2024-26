test_set=0.40
mal_ben=0.5
window_size=256
stride=256
embed=32
for i in 1 2 3; do
	for mode in "malconv" "malLSTM"; do
		python3 main.py $window_size $stride $test_set $mal_ben $embed $mode
	done
done

