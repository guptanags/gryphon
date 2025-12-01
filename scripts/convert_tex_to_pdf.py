import os
import subprocess
import sys
import shutil

def check_pdflatex():
    """Checks if pdflatex is installed."""
    if not shutil.which("pdflatex"):
        print("Error: 'pdflatex' not found in PATH.")
        print("Please install a LaTeX distribution.")
        print("  - macOS: brew install --cask basictex")
        print("  - Ubuntu: sudo apt-get install texlive-latex-base")
        return False
    return True

def convert_tex_to_pdf(tex_file):
    """Converts a .tex file to .pdf using pdflatex."""
    if not os.path.exists(tex_file):
        print(f"Error: File '{tex_file}' not found.")
        return

    # Get absolute path
    tex_path = os.path.abspath(tex_file)
    work_dir = os.path.dirname(tex_path)
    file_name = os.path.basename(tex_path)

    print(f"Compiling {file_name}...")
    try:
        # Add /Library/TeX/texbin to PATH for macOS
        env = os.environ.copy()
        tex_bin_dir = "/Library/TeX/texbin"
        if os.path.exists(tex_bin_dir):
            env["PATH"] = f"{tex_bin_dir}:{env.get('PATH', '')}"

        # 1. First Pass: pdflatex
        print("Pass 1/4: pdflatex...")
        cmd_latex = ["pdflatex", "-interaction=nonstopmode", "-output-directory", work_dir, tex_path]
        subprocess.run(cmd_latex, check=True, cwd=work_dir, env=env)

        # 2. BibTeX Pass
        # Only run if .aux file exists (it should after pass 1)
        aux_file = os.path.splitext(file_name)[0]
        print(f"Pass 2/4: bibtex {aux_file}...")
        try:
            cmd_bib = ["bibtex", aux_file]
            subprocess.run(cmd_bib, check=True, cwd=work_dir, env=env)
        except subprocess.CalledProcessError:
             print("Warning: BibTeX failed (maybe no citations?). Continuing...")

        # 3. Second Pass: pdflatex (to link citations)
        print("Pass 3/4: pdflatex...")
        subprocess.run(cmd_latex, check=True, cwd=work_dir, env=env)

        # 4. Third Pass: pdflatex (to resolve cross-references)
        print("Pass 4/4: pdflatex...")
        subprocess.run(cmd_latex, check=True, cwd=work_dir, env=env)
        
        print(f"Success! PDF generated in {work_dir}")
        
    except subprocess.CalledProcessError as e:
        print("Error during compilation.")
        print("Ensure all required packages are installed.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default to the known file if no arg provided
        default_file = "docs/slm-paper.tex"
        if os.path.exists(default_file):
            convert_tex_to_pdf(default_file)
        else:
            print("Usage: python convert_tex_to_pdf.py <path_to_tex_file>")
    else:
        convert_tex_to_pdf(sys.argv[1])
