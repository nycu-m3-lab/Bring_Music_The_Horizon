import argparse
import json
import os

# _DEFAULT_AUDIO = "/mnt/M3_Lab/Chiikawa/songs/devotion.mp3"
# _DEFAULT_PROMPT = (
#     "Colossal gnarled trees form a cathedral canopy over a misty forest with a glowing moss-covered shrine."
# )
# _DEFAULT_WORKSPACE = os.path.join(_COMPONENT_DIR, "workspace")

_COMPONENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_COMPONENT_DIR, ".."))
_BIG_FILES_DIR = os.path.join(_PROJECT_ROOT, "big_files")
_DEFAULT_VA_MODEL = os.path.join(_BIG_FILES_DIR, "music2VA_best_model.pth")
_DEFAULT_EIT_MODEL = os.path.join(_BIG_FILES_DIR, "emoti_best_model.pth")
_DEFAULT_SDXL_MODEL = os.path.join(_BIG_FILES_DIR, "sdxl_model")


def main():
    parser = argparse.ArgumentParser(description="Save pipeline arguments")
    # nargs="?" so omitted args use defaults (plain positional + default is still required in argparse).
    parser.add_argument(
        "audio_path",
        nargs="?",
        # default=_DEFAULT_AUDIO,
        help=f"Path to the input audio file",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        # default=_DEFAULT_PROMPT,
        help="Base text prompt",
    )
    parser.add_argument(
        "negative_prompt",
        nargs="?",
        default="",
        help="Negative text prompt (default: empty)",
    )
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--workspace",
        type=str,
        # default=_DEFAULT_WORKSPACE,
        help="Folder to pass data between components (default: Final_App_As_Components/workspace)",
    )
    parser.add_argument("--va_model", type=str, default=_DEFAULT_VA_MODEL)
    parser.add_argument("--eit_model", type=str, default=_DEFAULT_EIT_MODEL)
    parser.add_argument("--sdxl_path", type=str, default=_DEFAULT_SDXL_MODEL)
    args = parser.parse_args()

    # Resolve to absolute paths so subsequent components don't get lost
    args.audio_path = os.path.abspath(args.audio_path)
    args.va_model = os.path.abspath(args.va_model)
    args.eit_model = os.path.abspath(args.eit_model)
    args.workspace = os.path.abspath(args.workspace)

    os.makedirs(args.workspace, exist_ok=True)
    config_path = os.path.join(args.workspace, "args.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    print(f"[Component 0] Configuration saved to {config_path}")


if __name__ == "__main__":
    main()