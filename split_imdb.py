import file_handling as fh

train_jsonlist = fh.read_jsonlist("./data/imdb/train.jsonlist")
test_jsonlist = fh.read_jsonlist("./data/imdb/test.jsonlist")

train_dev_jsonlist = train_jsonlist + test_jsonlist[:12500]
test_only_jsonlist = test_jsonlist[12500:]

fh.write_jsonlist(train_dev_jsonlist, "./data/imdb/train_dev.jsonlist")
fh.write_jsonlist(test_only_jsonlist, "./data/imdb/test_only.jsonlist")