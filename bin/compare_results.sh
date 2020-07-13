if [ "$2" = "" ]; then
    echo "Usage: compare_results.sh <模式:r(recognize)|c(correct)> <url> <标签文件.txt> <识别结果文件.html>"
    echo "Example: compare_results.sh rc http:0.0.0.0:8080 data/test.txt data/result.html"
    exit
fi

python -m crnn.compare_results --mode $1 --url $2 --label $3 --report $4