# APEX
*AP*proximate *EX*haustive virtual screening

* [Link to preprint]()

### Building
Download CCCL
```bash
git clone \
  -b christinaz/nondeterministic_AIR_topK \
  https://github.com/ChristinaZ/cccl.git
```

Build and install the wheel,
```bash
python3 setup.py build
python3 setup.py bdist_wheel
pip install dist/*.whl
```

### Testing
Tests are organized into the [`test`](test) directory.
```bash
pytest -v test
```

## Licence
TBA
