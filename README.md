# image-matching

To check the result with the heads:
```bash
for f in $(find dni_v3/prep/head/ -name "*.png"); do 
  python image-matching.py -t template/head_v2.png template/tail.png -q $f -v -p; 
done
```

To check the result with the tails:
```bash
for f in $(find dni_v3/prep/tails/ -name "*.png"); do 
  python image-matching.py -t template/head_v2.png template/tail.png -q $f -v -p; 
done
```

