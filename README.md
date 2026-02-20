# Speech Technology Project 7

This project uses `uv` for extremely fast and reliable Python dependency management.

## Prerequisites

1.  **Install `uv`**:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    *(See [uv documentation](https://github.com/astral/sh/uv) for other installation methods.)*

2.  **Install FFmpeg** (required by `torchcodec` for audio decoding):
    - **macOS**: `brew install ffmpeg`
    - **Linux**: `sudo apt update && sudo apt install ffmpeg`

3.  **Hugging Face Access**:
    - Accept the [Emilia Dataset terms](https://huggingface.co/datasets/amphion/Emilia-Dataset) on Hugging Face.
    - Create a `.env` file in the project root with your token:
      ```env
      HUGGINGFACE_TOKEN=your_token_here
      ```

## Setup

Run the following command in the project root to create a virtual environment and install all dependencies:

```bash
uv sync
```

## Running the Notebook

You can run the notebook directly through your IDE (VS Code/Cursor will detect the `.venv` folder) or via the command line:

```bash
uv run jupyter notebook
```

Open `scr/embedding_test.ipynb` and run the cells. The first run will download the XTTS model (approx. 2GB).

**Note on coqui-tts license**: We've included `os.environ["COQUI_TOS_AGREED"] = "1"` in the notebook to bypass the interactive license prompt.
