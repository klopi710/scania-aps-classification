from pathlib import Path
from urllib.request import Request, urlopen
import zipfile


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
ZIP_PATH = RAW / "aps_failure_at_scania_trucks.zip"
URL = "https://archive.ics.uci.edu/static/public/421/aps%2Bfailure%2Bat%2Bscania%2Btrucks.zip"


def main() -> None:
    request = Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=180) as response, ZIP_PATH.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
    if ZIP_PATH.stat().st_size < 1_000_000:
        raise RuntimeError(f"Download is unexpectedly small: {ZIP_PATH.stat().st_size} bytes")
    with zipfile.ZipFile(ZIP_PATH) as archive:
        archive.extractall(RAW)
        print("Extracted:", ", ".join(archive.namelist()))
    print(f"Archive size: {ZIP_PATH.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
