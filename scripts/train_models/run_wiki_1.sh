python run_scholar.py data/wiki/processed/ --topk 1 -k 50 --test-prefix test --device 0 --o ./outputs/scholar/wiki_50_1 --epochs 500 --model contrastiveScholar --batch-size 500 -l 0.001 --alpha 0.01