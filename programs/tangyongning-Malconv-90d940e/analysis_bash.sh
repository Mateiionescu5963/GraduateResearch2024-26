dir="./gridsearch_malratio_12-4-24/"
type="assemble"
for file in $dir*.txt; do
	python3 analysis.py $file $type
done
