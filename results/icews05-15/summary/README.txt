for each pattern xxx :
xxx.txt:     the permise instances, which have a conclusion in test set with >=50% probability in dictionary.
xxx_test_train_pair.csv:      (ri -> rj) has pattern xxx, where ri in train set, rj in test set. And all relation pari have >=50% probability in dictionary.
freq_xxx.csv:      compute how many  relation (ri --> ?) has pattern xxx in train set.
pair_xxx.csv:      for each relation pair (ri --> rj), compute the percentage of ri has pattern to rj.
pair2d_xxx.csv:    same as pair_xxx.csv but in 2d table.
xxx_Distribution.png:     visualization of freq_xxx.csv
pair2d_xxx.png:     heatmap for pair2d_xxx.csv

