for corr in 0.01 0.05 0.1 0.2 0.3 0.4 0.5; do
	python3 corruptor.py $corr
	for run in {1..200}; do
		python3 main.py "True" "True"	
	done
done
