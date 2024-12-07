test_set=0.25

window_size=256
stride=256
embed=32
mode="mallstm"

python3 main.py $window_size $stride "0.25" "0.5" $embed $mode 
for mal_ben in {0.01..0.99..0.01}; do
	python3 main.py $window_size $stride "1" $mal_ben $embed $mode 
done

