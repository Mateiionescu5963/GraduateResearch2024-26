test_set=0.25
mal_ben=0.5
window_size=256
stride=256
embed=32
for mode in "STD" "LSTM"; do
	python3 main.py $window_size $stride $test_set $mal_ben $embed $mode
done

