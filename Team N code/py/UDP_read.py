# receiver_text_protocol_v2.py  (lightweight + status)
import socket
import csv
import select
from pathlib import Path

TYPE_FILE = Path("../ref/bootcamp_list.dat")
OUT_ROOT = Path("../data")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

RECV_IP = "192.168.0.5"
RECV_PORT = 7001
SELECT_TIMEOUT = 0.5


def load_types(path: Path):
    m = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, t = line.split(",", 1)
            t = t.strip()
            if t not in ("0", "1"):
                raise ValueError(f"[TYPE_FILE] invalid: {line}")
            m[name.strip()] = 0 if t == "0" else 1
    return m


def parse_msg(text: str):
    # ("header"/"data"/"terminate", test_id, sec, payload)
    text = text.strip()
    if text == "TERMINATE":
        return ("terminate", None, None, None)

    bar = text.find("|")
    if bar < 0 or not text.startswith("test"):
        return None

    left = text[:bar].strip()
    payload = text[bar + 1 :].strip()

    j = 4
    while j < len(left) and left[j].isdigit():
        j += 1
    if j == 4:
        return None
    test_id = int(left[4:j])

    rest = left[j:].strip()
    if rest.startswith("header"):
        return ("header", test_id, None, payload)

    if rest.startswith("sec"):
        k = 3
        m = k
        while m < len(rest) and rest[m].isdigit():
            m += 1
        if m == k:
            return None
        sec = int(rest[k:m])
        return ("data", test_id, sec, payload)

    return None


def main():
    type_map = load_types(TYPE_FILE)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((RECV_IP, RECV_PORT))

    print(f"[RECEIVER] {RECV_IP}:{RECV_PORT} (Ctrl+C to stop)")

    # states[test_id] = (header, type_flags, test_dir)
    states = {}

    try:
        while True:
            r, _, _ = select.select([sock], [], [], SELECT_TIMEOUT)
            if not r:
                continue

            data, _ = sock.recvfrom(65535)
            msg = data.decode("utf-8", errors="replace")
            parsed = parse_msg(msg)
            if not parsed:
                continue

            kind, test_id, sec, payload = parsed

            if kind == "terminate":
                print("[TERMINATE]")
                break

            if kind == "header":
                header = [h.strip() for h in payload.split(",") if h.strip()]
                type_flags = [type_map[h] for h in header]
                test_dir = OUT_ROOT / f"test{test_id}"
                test_dir.mkdir(parents=True, exist_ok=True)
                states[test_id] = (header, type_flags, test_dir)
                print(f"[HEADER] test{test_id}  cols={len(header)}")
                continue

            st = states.get(test_id)
            if st is None:
                print(f"[WARN] test{test_id}  sec={sec}  no_header")
                continue

            header, type_flags, test_dir = st
            raw = payload.split(",")
            if len(raw) != len(header):
                print(f"[WARN] test{test_id}  sec={sec}  len_mismatch")
                continue

            out_path = test_dir / f"test{test_id}_sec{sec}.csv"

            with open(out_path, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)
                row = []
                for s, t in zip(raw, type_flags):
                    s = s.strip()
                    if s == "" or s.lower() in ("nan", "none", "null"):
                        row.append(0 if t == 0 else float("nan"))
                    else:
                        row.append(int(float(s)) if t == 0 else float(s))
                w.writerow(row)

            print(f"[SAVE] test{test_id}  sec={sec}  -> {out_path.name}")

    except KeyboardInterrupt:
        print("\n[CTRL+C]")
    finally:
        sock.close()
        print("[EXIT]")


if __name__ == "__main__":
    main()
