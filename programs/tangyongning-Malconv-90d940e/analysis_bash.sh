dir="./11-11-24_GRIDSEARCH/"
type="assemble"
for file in $dir*.txt; do
	python3 analysis.py $file $type
done
