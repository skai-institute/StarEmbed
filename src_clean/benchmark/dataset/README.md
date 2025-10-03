# Raw light Curve and Embedding Dataset

## ZTF

###  7-class 40k ZTF data with expert-labed from CSDR1

The raw light file before splitting is located at:
```
/projects/p32795/weijian/hf_csdr1_raw4_catflags_filtered_with_labels_multiband
```

The train-val-test split is located at
```
/projects/p32795/hongyu/hf_csdr1_multiband_raw_lc_subclass_class_str_v2
```

###  Other classes out of above 7-class 
The minority classes train-val-test split used for OOD detection is located at
```
/projects/p32795/hongyu/hf_csdr1_multiband_raw_lc_minority_class_str_v2
```


## Macho

Macho is the dataset used for Astromer 1/2.

The Macho raw light curve is located at 
```
/projects/p32795/hongyu/hf_macho_light_curves
```

The train-val-test splits is at
```
/projects/p32795/hongyu/hf_macho_70-10-20
```