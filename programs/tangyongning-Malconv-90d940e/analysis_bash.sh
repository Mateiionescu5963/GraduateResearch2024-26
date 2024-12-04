dir="./TGB_24_grid/"
type="assemble"
for file in $dir*.txt; do
	python3 analysis.py $file $type
done
