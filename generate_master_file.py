import os
import glob
import sys

def generate_master_file(root_dir, output_file):
    # Exclusion rules
    exclude_dirs = {
        '.git', 'venv', 'venv312', 'env', '.venv', 'node_modules', 
        '__pycache__', '.npm', 'dist', 'build', 'faiss_db', 'assets', 'output_visuals'
    }
    exclude_extensions = {
        '.pt', '.safetensors', '.bin', '.pdf', '.png', '.jpg', '.jpeg', '.svg', '.gif', '.ico'
    }
    exclude_files = {
        '.env', '.secret', '.DS_Store', 'package-lock.json', 'yarn.lock', 'generate_master_file.py'
    }

    def should_exclude_dir(dname):
        return dname in exclude_dirs

    def should_exclude_file(fname):
        if fname in exclude_files:
            return True
        ext = os.path.splitext(fname)[1].lower()
        if ext in exclude_extensions:
            return True
        return False

    with open(output_file, 'w', encoding='utf-8') as out:
        out.write("# Project Master File\n\n")

        # 1. Project Overview
        out.write("## 1. Project Overview\n\n")
        out.write("This project is an AI-powered mock interviewer platform. It consists of a React-based frontend and a Python backend (likely FastAPI or similar). The backend includes components for analyzing behavioral responses, evaluating answers against rubrics, NER extraction, and generating evaluation results. It leverages NLP techniques (such as sentence transformers and LLMs) to provide quantitative and qualitative assessments of user interviews.\n\n")

        # 2. Architecture & Directory Structure
        out.write("## 2. Architecture & Directory Structure\n\n")
        out.write("```text\n")
        out.write(os.path.basename(root_dir) + "/\n")
        
        # Traverse for tree
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Filter dirs
            dirnames[:] = [d for d in dirnames if not should_exclude_dir(d)]
            dirnames.sort()
            filenames.sort()

            rel_path = os.path.relpath(dirpath, root_dir)
            if rel_path == '.':
                level = 0
            else:
                level = rel_path.count(os.sep) + 1
                indent = ' ' * 4 * level
                out.write(f"{indent}{os.path.basename(dirpath)}/\n")
            
            sub_indent = ' ' * 4 * (level + 1)
            for f in filenames:
                if not should_exclude_file(f):
                    out.write(f"{sub_indent}{f}\n")
        out.write("```\n\n")

        # Gather files
        source_files = []
        config_files = []
        doc_results_files = []

        config_names = {
            'requirements.txt', 'package.json', 'Dockerfile', 'webpack.config.js',
            'vite.config.js', 'tailwind.config.js', 'postcss.config.js', 'eslint.config.js',
            '.eslintrc', '.prettierrc', 'docker-compose.yml'
        }

        for dirpath, dirnames, filenames in os.walk(root_dir):
            dirnames[:] = [d for d in dirnames if not should_exclude_dir(d)]
            for f in filenames:
                if should_exclude_file(f):
                    continue
                
                full_path = os.path.join(dirpath, f)
                rel_path = os.path.relpath(full_path, root_dir)
                
                # Determine category
                ext = os.path.splitext(f)[1].lower()
                
                if f in config_names:
                    config_files.append((rel_path, full_path))
                elif ext in {'.md'} or f in {'job_description.txt', 'ner_extracted_profile.txt'} or 'results' in f or 'metrics' in f or ext == '.csv':
                    doc_results_files.append((rel_path, full_path))
                elif ext in {'.py', '.js', '.jsx', '.ts', '.tsx', '.html', '.css', '.sh'}:
                    source_files.append((rel_path, full_path))
                else:
                    # Default to source if text
                    source_files.append((rel_path, full_path))

        def write_file_section(files_list, section_title):
            out.write(f"## {section_title}\n\n")
            for rel_path, full_path in sorted(files_list):
                out.write(f"### {rel_path}\n\n")
                
                # Try to determine markdown language block
                ext = os.path.splitext(full_path)[1].lower()
                lang_map = {
                    '.py': 'python', '.js': 'javascript', '.jsx': 'jsx', 
                    '.ts': 'typescript', '.tsx': 'tsx', '.html': 'html',
                    '.css': 'css', '.json': 'json', '.md': 'markdown', '.sh': 'bash',
                    '.txt': 'text', '.csv': 'csv'
                }
                lang = lang_map.get(ext, '')

                try:
                    size = os.path.getsize(full_path)
                    if size > 150 * 1024: # > 150KB
                        out.write(f"*(File too large to display entirely. Truncated first 150KB...)*\n\n")
                        with open(full_path, 'r', encoding='utf-8') as f_in:
                            content = f_in.read(150 * 1024)
                    else:
                        with open(full_path, 'r', encoding='utf-8') as f_in:
                            content = f_in.read()
                    
                    out.write(f"```{lang}\n{content}\n```\n\n")
                except UnicodeDecodeError:
                    out.write("*(Binary or unreadable file)*\n\n")
                except Exception as e:
                    out.write(f"*(Error reading file: {e})*\n\n")

        # 3. Core Application & Source Code
        write_file_section(source_files, "3. Core Application & Source Code")

        # 4. Configurations & Dependencies
        write_file_section(config_files, "4. Configurations & Dependencies")

        # 5. Documentation & Results
        write_file_section(doc_results_files, "5. Documentation & Results")

if __name__ == '__main__':
    root_dir = '/Users/ahmednadeem/Desktop/ai-interviewer copy 16'
    output_file = os.path.join(root_dir, 'Master_File.md')
    generate_master_file(root_dir, output_file)
    print(f"Master file generated successfully at: {output_file}")
