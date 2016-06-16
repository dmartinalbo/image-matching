# image-matching

How to use this:

```bash
for f in $(find . -name "*.png"); do 
  python image-matching.py -t <TEMPLATES> -q $f -v -p; 
done
```

