for fraction in $(seq 0.05 0.05 0.5); do
	python3 analysis.py $fraction "shapctrl"
done
