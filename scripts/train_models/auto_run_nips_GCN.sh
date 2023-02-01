for seed in 42 2022 2048 1024 1000 4396 100 999
do
  for lr in 0.005 0.006 0.007
  do
    for epoch in 500
    do
      echo now:${seed}_${lr}
      python run_scholar.py data/nips/processed/ data/nips/graph -k 50 --seed $seed -l $lr --epochs $epoch --device 8 --batch-size 500 --dev-folds 4 --dev-fold 3 --test-prefix test --o ./outputs/myModel_journal/nips_full_cl/50_${seed}_${lr}_${epoch}_GCN --model GCNContrastiveScholar --dist 0
      python compute_npmi.py ./outputs/myModel_journal/nips_full_cl/50_${seed}_${lr}_${epoch}_GCN/topics.txt ./data/nips/processed/train.npz ./data/nips/processed/train.vocab.json
    done
  done
done