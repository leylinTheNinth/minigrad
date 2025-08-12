from mnist_train import mnist
import gzip, os, urllib.request, urllib.error, numpy as np

FILES = {
  "train_images": "train-images-idx3-ubyte.gz",
  "train_labels": "train-labels-idx1-ubyte.gz",
  "t10k_images":  "t10k-images-idx3-ubyte.gz",
  "t10k_labels":  "t10k-labels-idx1-ubyte.gz",
}

MIRRORS = [
  "https://storage.googleapis.com/cvdf-datasets/mnist/",
  "https://ossci-datasets.s3.amazonaws.com/mnist/",
  "http://yann.lecun.com/exdb/mnist/",   # keeping as last resort, although probably won't work
]

def fetch(path="data"):
    os.makedirs(path, exist_ok=True)
    for k, fname in FILES.items():
        out = os.path.join(path, k + ".gz")
        if os.path.exists(out):
            continue
        last_err = None
        for base in MIRRORS:
            url = base + fname
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as r, open(out, "wb") as f:
                    f.write(r.read())
                print("Downloaded:", url)
                break
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
                print("Mirror failed:", url, "-", e)
                last_err = e
        else:
            raise last_err
    return path


def parse_images(fp):
    with gzip.open(fp, "rb") as f:
        _ = f.read(16)
        buf = f.read()
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(-1, 28*28).astype(np.float32)/255.0
    return arr

def parse_labels(fp):
    with gzip.open(fp, "rb") as f:
        _ = f.read(8)
        buf = f.read()
    return np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

if __name__ == "__main__":
    d = fetch()
    train_x = parse_images(os.path.join(d, "train_images.gz"))
    train_y = parse_labels(os.path.join(d, "train_labels.gz"))
    test_x  = parse_images(os.path.join(d, "t10k_images.gz"))
    test_y  = parse_labels(os.path.join(d, "t10k_labels.gz"))

    # split 5k for validation
    val_x, val_y = test_x[:5000], test_y[:5000]
    val_accs = mnist(train_x, train_y, val_x, val_y)
    print("Final validation accuracy:", val_accs[-1] if val_accs else None)