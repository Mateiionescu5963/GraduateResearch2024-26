dir="./maltf_gridsearch_2-4-25/"
type="assemble"
for file in $dir*.txt; do
	python3 analysis.py $file $type
done
