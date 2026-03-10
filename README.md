# Advanced AI

### Pong Agent
tryna get this thang to learn pong

---

Try this to run it:

- Create a Python venv:
```
python -m venv venv
```
- Activate that venv (you might have to do it differently if you're on Windows):
```
source venv/bin/activate
```
- Install the dependencies from `requirements.txt`:
```
pip install -r requirements.txt
```
- After that's done, try and run it:
```
python main.py
```

This won't stop until it reaches `1,000,000` iterations, which will take a while, but you can just cancel it with `Ctrl-C` and this will save your model `.pth` file into a `results` folder.

If you want to see how your model is performing with the human eye run the `play.py` script with the path to the `.pth` file:
```
python play.py path/to/model.pth
```
