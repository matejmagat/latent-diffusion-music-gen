
import argparse
import os
import torchaudio

from pathlib import Path
from itertools import islice


def iter_chunks(output_dir: str | Path, split: str, T: float, n_songs: int | None = None):
    output_dir = Path(output_dir)
    T_str = f"{T:g}"
    chunks_folder = f"chunks_{T_str}"

    song_dirs = (
        song_dir
        for song_dir in sorted((output_dir / split).iterdir())
        if song_dir.is_dir()
    )

    for song_dir in islice(song_dirs, n_songs):
        chunks_dir = song_dir / chunks_folder
        if not chunks_dir.is_dir():
            continue
        yield from sorted(chunks_dir.glob("chunk_*.wav"))


def chunk_track(wav_path: str, T: float) -> None:
    waveform, sr = torchaudio.load(wav_path)
    chunk_samples = int(T * sr)

    if chunk_samples <= 0:
        raise ValueError(f"T={T} seconds results in 0 samples at {sr} Hz.")

    total_samples = waveform.shape[1]
    num_chunks = total_samples // chunk_samples

    if num_chunks == 0:
        print(f"    [SKIP] Track shorter than {T}s, no chunks produced.")
        return

    song_dir = os.path.dirname(wav_path)
    T_str = f"{T:g}"
    chunks_dir = os.path.join(song_dir, f"chunks_{T_str}")
    os.makedirs(chunks_dir, exist_ok=True)

    for i in range(num_chunks):
        start = i * chunk_samples
        end = start + chunk_samples
        chunk = waveform[:, start:end]
        out_path = os.path.join(chunks_dir, f"chunk_{i:04d}.wav")
        torchaudio.save(out_path, chunk, sr)

    print(f"    {num_chunks} chunks saved -> {chunks_dir}")


def process_output_dir(output_dir: str, T: float) -> None:
    for split in ("train", "test"):
        split_path = os.path.join(output_dir, split)
        if not os.path.isdir(split_path):
            print(f"[WARN] Split folder not found, skipping: {split_path}")
            continue

        song_dirs = sorted(
            d for d in os.listdir(split_path)
            if os.path.isdir(os.path.join(split_path, d))
        )

        print(f"\n[{split.upper()}] Processing {len(song_dirs)} tracks...")

        for song_name in song_dirs:
            wav_path = os.path.join(split_path, song_name, "mixture.wav")
            if not os.path.isfile(wav_path):
                print(f"  [SKIP] No mixture.wav found in: {song_name}")
                continue
            print(f"  {song_name}")
            try:
                chunk_track(wav_path, T)
            except Exception as e:
                print(f"    [ERROR] {e}")

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(
        description="Cut MUSDB18 mixture WAVs into T-second chunks (incomplete chunks discarded)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the folder produced by export_musdb18_mixtures.py.",
    )
    parser.add_argument(
        "--T",
        type=float,
        required=True,
        help="Chunk duration in seconds (e.g. 5 or 2.5).",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        raise FileNotFoundError(f"Output directory not found: {args.output_dir}")

    process_output_dir(args.output_dir, args.T)


if __name__ == "__main__":
    main()