import argparse
import os
import stempeg
import torchaudio
import torch


def export_mixtures(musdb_root: str, output_dir: str) -> None:
    for split in ("train", "test"):
        split_path = os.path.join(musdb_root, split)
        if not os.path.isdir(split_path):
            print(f"[WARN] Split folder not found, skipping: {split_path}")
            continue

        track_files = sorted(
            f for f in os.listdir(split_path)
            if f.endswith(".stem.mp4") or f.endswith(".mp4")
        )

        if not track_files:
            print(f"[WARN] No track files found in: {split_path}")
            continue

        print(f"\n[{split.upper()}] Found {len(track_files)} tracks.")

        for track_file in track_files:
            track_path = os.path.join(split_path, track_file)
            track_name = os.path.splitext(os.path.splitext(track_file)[0])[0]

            try:
                audio, rate = stempeg.read_stems(
                    track_path,
                    stem_id=0,
                )
            except Exception as e:
                print(f"  [ERROR] Could not read {track_file}: {e}")
                continue


            waveform = torch.from_numpy(audio.T).float()


            track_out_dir = os.path.join(output_dir, split, track_name)
            os.makedirs(track_out_dir, exist_ok=True)

            out_path = os.path.join(track_out_dir, "mixture.wav")
            torchaudio.save(out_path, waveform, rate)

            print(f"  Saved: {os.path.join(split, track_name, 'mixture.wav')}  "
                  f"[{waveform.shape[1]} samples @ {rate} Hz, {waveform.shape[0]}ch]")

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(
        description="Export MUSDB18 mixtures as WAV files (one folder per track)."
    )
    parser.add_argument(
        "--musdb_root",
        type=str,
        required=True,
        help="Path to the MUSDB18 root directory (contains 'train' and 'test' sub-folders).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory where WAV files will be saved.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.musdb_root):
        raise FileNotFoundError(f"MUSDB18 root not found: {args.musdb_root}")

    os.makedirs(args.output_dir, exist_ok=True)
    export_mixtures(args.musdb_root, args.output_dir)


if __name__ == "__main__":
    main()
