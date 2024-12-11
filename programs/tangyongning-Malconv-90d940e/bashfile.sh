test_set=0.25

window_size=256
stride=256
embed=32
mode="mallstm"

for mal_ben in $(seq 0.59 0.01 1.00); do
	python3 main.py $window_size $stride "1" $mal_ben $embed $mode 
done

