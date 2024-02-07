# Latex generator

## Setup:

clone the repository:
```
git clone https://github.com/Drizzr/Img2Latex_generator
```

install the dependencies:
```
pip install -r requirements.txt
```

run the script
```
python3 run.py
```

The programm will now ask for the image path and then generates the formula.

with:
```
python3 run.py --beam --beam_width <int>
```
you can switch to beam search (greedy decoding is default).

You can also change the max token length:
```
python3 run.py --max_len <int>
```
The default is 150.