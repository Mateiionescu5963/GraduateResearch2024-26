dir="./12-2-24_CNNLSTM/"
type="assemble"
for file in $dir*.txt; do
	python3 analysis.py $file $type
done
