import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from tqdm import tqdm

import whisperx

ENV_DIR = Path(__file__).parents[2] / "env"

def make_output_dir(input_dir: Path, output_dir: Optional[Union[str, Path]] =None) -> Path:
    if output_dir is None:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = input_dir / f"output_{current_time}"

        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            raise FileNotFoundError(f"{str(output_dir)} was not found.")

    return output_dir

def set_device(device: Optional[str] =None):
    if isinstance(device, str):
        return device

    if torch.cuda.is_available():
        return "cuda"
    
    return "cpu"

def read_configs() -> Dict[str, Any]: 
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", "-i", type=str, required=True)
    parser.add_argument("--output_dir", "-o", type=str, default=None)
    parser.add_argument("--device", "-d", type=str, default=None)
    parser.add_argument("--device_asr", "-da", type=str, default="cpu")
    parser.add_argument("--batch_size", "-bs", type=int, default=16)
    parser.add_argument("--language", "-l", type=str, default="en")
    parser.add_argument("--compute_type", "-ct", type=str, default="float16")
    parser.add_argument("--min_speakers", type=int, default=None)
    parser.add_argument("--max_speakers", type=int, default=None)
    parser.add_argument("--forced", action="store_true") # default=True

    args = parser.parse_args()

    config = {
        "input_dir": Path(args.input_dir),
        "output_dir": make_output_dir(Path(args.input_dir), args.output_dir),
        "device": set_device(args.device),
        "device_asr": args.device_asr,
        "batch_size": args.batch_size,
        "language": args.language,
        "compute_type": args.compute_type,
        "min_speakers": args.min_speakers,
        "max_speakers": args.max_speakers,
        "forced": args.forced
    }

    return config

def check_files(input_dir: Path) -> int:
    if not input_dir.exists():
        raise FileNotFoundError(f"{str(input_dir)} was not found.")

    if not input_dir.is_dir():
        raise ValueError(f"argument \"input_dir\" must be directory. {str(input_dir)} is a file.")
    
    n_files = 0
    
    for filepath in input_dir.glob("*.wav"):
        if not filepath.exists():
            raise FileNotFoundError(f"{str(filepath)} was not found.")
        n_files += 1
    
    return n_files

def read_pyannote_access_token() -> str:
    access_token_path = ENV_DIR / "pyannote_access_token.json"
    with open(access_token_path, "r") as f:
        access_token_dir = json.load(f)

    return access_token_dir["pyannoteAccessToken"]

def load_pipeline(config: Dict[str, Any]):
    device = config["device"]
    device_asr = config["device_asr"]
    language = config["language"]
    compute_type = config["compute_type"]

    pyannote_access_token = read_pyannote_access_token()

    print("Loading WhisperX Pipeline...", end=" ")
    transcriber = whisperx.load_model("large-v2", device=device_asr, compute_type=compute_type)
    aligner, fa_metadata = whisperx.load_align_model(language_code=language, device=device)
    diarizer = whisperx.DiarizationPipeline(use_auth_token=pyannote_access_token, device=device)
    print("DONE!")

    return transcriber, aligner, fa_metadata, diarizer

def save_result(wav_path: Path, save_path: Path, result: Dict[str, Any]) -> None:
    segments = {
        "segments": result["segments"]
    }
    with open(save_path, "w") as f:
        json.dump(segments, f, indent=4)

def main() -> None:
    # 1. read configs
    config = read_configs()

    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    device = config["device"]
    batch_size = config["batch_size"]
    min_speakers = config["min_speakers"]
    max_speakers = config["max_speakers"]
    is_forced = config["forced"]

    # 2. check audio files
    n_files = check_files(input_dir)
    if n_files == 0:
        print(f"Wav files were not found in {str(input_dir)}.")
        return
    print(f"Identified {n_files} wav file(s).")

    # 3. load_models
    transcriber, aligner, fa_metadata, diarizer = load_pipeline(config)

    # 4. apply pipleine
    pbar = tqdm(input_dir.glob("*.wav"), total=n_files)
    for wav_path in pbar:
        save_path = output_dir / f"{wav_path.stem}.json"
        if save_path.exists() and not(is_forced):
            pbar.set_description(f"[{wav_path.stem}] Skip ...")
            continue

        audio = whisperx.load_audio(wav_path)
        
        pbar.set_description(f"[{wav_path.stem}] Transcribing ...")
        result = transcriber.transcribe(audio, batch_size=batch_size)

        pbar.set_description(f"[{wav_path.stem}] Aligning ...")
        result = whisperx.align(result["segments"], aligner, fa_metadata, audio, device, return_char_alignments=False)

        pbar.set_description(f"[{wav_path.stem}] Diarizing ...")
        dialize_segments = diarizer(audio, min_speakers=min_speakers, max_speakers=max_speakers)
        result = whisperx.assign_word_speakers(dialize_segments, result)

        save_result(wav_path, save_path, result)

if __name__ == "__main__":
    main()