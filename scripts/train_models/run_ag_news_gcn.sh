python run_scholar.py data/ag_news/processed/ data/ag_news/graph -k 50 -l 0.001 --alpha 1 --seed 42 --test-prefix test --device 3 --o ./outputs/scholar/ag_news_50_gcn --epochs 100 --model GCNContrastiveScholar --batch-size 1000 --dist 0
