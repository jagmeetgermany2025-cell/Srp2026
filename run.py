#!/usr/bin/env python3
"""Cross-platform launcher for EDA.py.

Run it with whatever Python you have and it sorts the rest out:

    python run.py            (Windows: py run.py)

It looks for an interpreter that can import everything EDA.py needs. If none can,
it builds a project-local virtual environment in .venv/ and installs
requirements.txt into it, then runs the analysis there. VS Code auto-detects a
.venv/ in the workspace root, so creating one also fixes the run button without
anyone having to hardcode an interpreter path.

Uses only the standard library, so it works before anything is installed.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(HERE, ".venv")
REQUIREMENTS = os.path.join(HERE, "requirements.txt")
SCRIPT = os.path.join(HERE, "EDA.py")

# Import names, which differ from the pip names for scikit-learn.
MODULES = ["numpy", "pandas", "scipy", "sklearn", "matplotlib"]


def venv_python(root):
    """Path to the interpreter inside a virtualenv, per platform layout."""
    if os.name == "nt":
        return os.path.join(root, "Scripts", "python.exe")
    return os.path.join(root, "bin", "python")


def has_modules(python):
    """True if `python` can import every module EDA.py needs."""
    if not python or not os.path.exists(python):
        return False
    probe = "import " + ", ".join(MODULES)
    return subprocess.call(
        [python, "-c", probe],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ) == 0


def candidates():
    """Interpreters worth trying, best first, without hardcoding any one machine."""
    seen, out = set(), []
    for path in [venv_python(VENV), venv_python(os.path.join(HERE, "venv")), sys.executable]:
        if path and path not in seen:
            seen.add(path)
            out.append(path)

    # Anything on PATH, plus the common conda/homebrew locations that are often
    # installed but not first on PATH.
    from shutil import which
    for name in ("python3", "python", "py"):
        found = which(name)
        if found and found not in seen:
            seen.add(found)
            out.append(found)
    for guess in (
        "/opt/anaconda3/bin/python3", "/opt/miniconda3/bin/python3",
        os.path.expanduser("~/anaconda3/bin/python3"),
        os.path.expanduser("~/miniconda3/bin/python3"),
        "/opt/homebrew/bin/python3", "/usr/local/bin/python3",
    ):
        if os.path.exists(guess) and guess not in seen:
            seen.add(guess)
            out.append(guess)
    return out


def build_venv():
    """Create .venv and install requirements into it. Returns its interpreter."""
    print(f"No interpreter found with {', '.join(MODULES)}.")
    print(f"Creating a virtual environment in {VENV} ...")
    subprocess.check_call([sys.executable, "-m", "venv", VENV])

    python = venv_python(VENV)
    print("Installing requirements (this takes a minute the first time) ...")
    subprocess.check_call([python, "-m", "pip", "install", "--upgrade", "pip", "-q"])
    subprocess.check_call([python, "-m", "pip", "install", "-r", REQUIREMENTS])
    return python


def main():
    python = next((c for c in candidates() if has_modules(c)), None)

    if python is None:
        try:
            python = build_venv()
        except subprocess.CalledProcessError as exc:
            print(f"\nCould not build the environment automatically: {exc}", file=sys.stderr)
            print("Install the dependencies manually with:", file=sys.stderr)
            print(f"  {sys.executable} -m pip install -r requirements.txt", file=sys.stderr)
            return 1

    # flush before handing stdout to the child, or our line lands after its output
    # whenever this is piped rather than attached to a terminal.
    print(f"Using {python}\n", flush=True)
    # cwd=HERE so EDA.py finds its data and writes its outputs beside the script.
    return subprocess.call([python, SCRIPT], cwd=HERE)


if __name__ == "__main__":
    sys.exit(main())
