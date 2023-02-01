for seed in 42 #2022 2048 1024 1000 4396 #100 999
do
  for lr in 0.002
  do
    for epoch in 200 300 400 500
    do
      echo =========now:${seed}_${lr}_${epoch}========
      python run_scholar.py data/20ng/processed/ data/20ng/graph -k 50 --seed $seed -l $lr --epochs $epoch --device 2 --test-prefix test --o ./outputs/myModel_journal/20ng_without_cl/50_${seed}_${lr}_${epoch}_gcn --model gcnContrastiveScholar --ablation None --bg-freq tfidf --dist 0
      python compute_npmi.py ./outputs/myModel_journal/20ng_without_cl/50_${seed}_${lr}_${epoch}_gcn/topics.txt ./data/20ng/processed/train.npz ./data/20ng/processed/train.vocab.json
    done
  done
done