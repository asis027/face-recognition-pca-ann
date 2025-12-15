"""
Repository Verification Script
Checks for common errors and missing components in the face-recognition-pca-ann project
"""

import os
import sys
from pathlib import Path
import re

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.ENDC}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ {text}{Colors.ENDC}")

class RepositoryVerifier:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info = []
        self.root_dir = Path.cwd()
        
    def check_file_naming(self):
        """Check for files with invalid naming conventions"""
        print_header("Checking File Naming Conventions")
        
        # Files that should exist with clean names
        required_files = {
            'pca_ann.py': 'Core PCA module',
            'train.py': 'Training script',
            'evaluate.py': 'Evaluation script',
            'predict.py': 'Prediction script',
            'visualize.py': 'Visualization tools',
            'test_pca_ann.py': 'Unit tests',
            'config.yaml': 'Configuration file',
            'requirements.txt': 'Python dependencies',
            'README.md': 'Main documentation',
            'main.py': 'Entry point script'
        }
        
        # Check for files with incorrect names (containing descriptions)
        invalid_patterns = [
            r'.* - .*',  # Files with " - " in name
            r'.*\s+\(.*\).*',  # Files with parentheses
        ]
        
        all_files = [f for f in os.listdir(self.root_dir) if os.path.isfile(f)]
        
        for file in all_files:
            for pattern in invalid_patterns:
                if re.match(pattern, file):
                    self.errors.append(f"Invalid filename: '{file}'")
                    print_error(f"Invalid filename: '{file}'")
                    
                    # Try to suggest correct name
                    clean_name = file.split(' - ')[0].strip() if ' - ' in file else file
                    print_info(f"  → Should be: '{clean_name}'")
        
        # Check if required files exist
        for req_file, description in required_files.items():
            file_path = self.root_dir / req_file
            if file_path.exists():
                print_success(f"Found {req_file} ({description})")
            else:
                # Check if file exists with wrong name
                wrong_names = [f for f in all_files if f.startswith(req_file.replace('.', ' '))]
                if wrong_names:
                    self.errors.append(f"Missing {req_file} - found with wrong name: {wrong_names[0]}")
                    print_error(f"Missing {req_file}")
                    print_warning(f"  Found with wrong name: {wrong_names[0]}")
                else:
                    if req_file == 'main.py':
                        self.warnings.append(f"Optional file missing: {req_file}")
                        print_warning(f"Optional file missing: {req_file}")
                    else:
                        self.errors.append(f"Missing required file: {req_file}")
                        print_error(f"Missing required file: {req_file}")
    
    def check_directory_structure(self):
        """Check if required directories exist"""
        print_header("Checking Directory Structure")
        
        required_dirs = {
            'data': 'Data storage',
            'data/train': 'Training data',
            'data/test': 'Test data',
            'data/validation': 'Validation data',
            'models': 'Saved models',
            'output': 'Output files',
            'output/visualizations': 'Visualization outputs',
            'output/logs': 'Log files',
            'tests': 'Unit tests',
        }
        
        optional_dirs = {
            'src': 'Source code package',
            'docs': 'Documentation',
            'notebooks': 'Jupyter notebooks',
            'configs': 'Configuration files',
        }
        
        for dir_path, description in required_dirs.items():
            full_path = self.root_dir / dir_path
            if full_path.exists():
                print_success(f"Found {dir_path}/ ({description})")
            else:
                self.errors.append(f"Missing directory: {dir_path}/")
                print_error(f"Missing directory: {dir_path}/ ({description})")
        
        for dir_path, description in optional_dirs.items():
            full_path = self.root_dir / dir_path
            if full_path.exists():
                print_success(f"Found {dir_path}/ ({description})")
            else:
                self.warnings.append(f"Optional directory missing: {dir_path}/")
                print_warning(f"Optional directory missing: {dir_path}/ ({description})")
    
    def check_python_files(self):
        """Check Python files for basic syntax and imports"""
        print_header("Checking Python Files")
        
        python_files = list(self.root_dir.glob('*.py'))
        
        if not python_files:
            self.errors.append("No Python files found in root directory")
            print_error("No Python files found in root directory")
            return
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check if file is empty
                if not content.strip():
                    self.warnings.append(f"{py_file.name} is empty")
                    print_warning(f"{py_file.name} is empty")
                    continue
                
                # Try to compile (basic syntax check)
                try:
                    compile(content, py_file.name, 'exec')
                    print_success(f"{py_file.name} - Syntax OK")
                except SyntaxError as e:
                    self.errors.append(f"{py_file.name} has syntax error: {e}")
                    print_error(f"{py_file.name} - Syntax Error: {e}")
                
                # Check for common imports
                if 'import numpy' in content or 'from numpy' in content:
                    print_info(f"  Uses NumPy")
                if 'import cv2' in content or 'from cv2' in content:
                    print_info(f"  Uses OpenCV")
                if 'import tensorflow' in content or 'import keras' in content:
                    print_info(f"  Uses TensorFlow/Keras")
                    
            except Exception as e:
                self.errors.append(f"Cannot read {py_file.name}: {e}")
                print_error(f"Cannot read {py_file.name}: {e}")
    
    def check_requirements(self):
        """Check requirements.txt file"""
        print_header("Checking Requirements")
        
        req_file = self.root_dir / 'requirements.txt'
        
        if not req_file.exists():
            self.errors.append("requirements.txt not found")
            print_error("requirements.txt not found")
            return
        
        try:
            with open(req_file, 'r') as f:
                requirements = f.read().strip().split('\n')
            
            if not requirements or requirements == ['']:
                self.errors.append("requirements.txt is empty")
                print_error("requirements.txt is empty")
                return
            
            print_success(f"Found requirements.txt with {len(requirements)} packages")
            
            # Check for common ML packages
            common_packages = {
                'numpy': 'NumPy',
                'opencv': 'OpenCV',
                'tensorflow': 'TensorFlow',
                'keras': 'Keras',
                'scikit-learn': 'Scikit-learn',
                'matplotlib': 'Matplotlib',
                'pillow': 'Pillow',
                'pyyaml': 'PyYAML'
            }
            
            req_lower = ' '.join(requirements).lower()
            for package, name in common_packages.items():
                if package in req_lower:
                    print_info(f"  Includes {name}")
                else:
                    self.warnings.append(f"Missing common package: {name}")
                    print_warning(f"  Missing common package: {name}")
                    
        except Exception as e:
            self.errors.append(f"Error reading requirements.txt: {e}")
            print_error(f"Error reading requirements.txt: {e}")
    
    def check_config(self):
        """Check configuration file"""
        print_header("Checking Configuration")
        
        config_file = self.root_dir / 'config.yaml'
        
        if not config_file.exists():
            self.warnings.append("config.yaml not found")
            print_warning("config.yaml not found")
            return
        
        try:
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            if config:
                print_success(f"Found valid config.yaml")
                print_info(f"  Configuration keys: {list(config.keys())}")
            else:
                self.warnings.append("config.yaml is empty")
                print_warning("config.yaml is empty")
                
        except ImportError:
            print_warning("PyYAML not installed, cannot validate config.yaml")
        except Exception as e:
            self.errors.append(f"Error reading config.yaml: {e}")
            print_error(f"Error reading config.yaml: {e}")
    
    def check_gitignore(self):
        """Check .gitignore file"""
        print_header("Checking .gitignore")
        
        gitignore_file = self.root_dir / '.gitignore'
        
        if not gitignore_file.exists():
            self.warnings.append(".gitignore not found")
            print_warning(".gitignore not found")
            return
        
        try:
            with open(gitignore_file, 'r') as f:
                content = f.read()
            
            important_patterns = [
                '__pycache__',
                '*.pyc',
                'venv/',
                '.env',
                '*.log',
                'models/',
                'data/'
            ]
            
            missing_patterns = []
            for pattern in important_patterns:
                if pattern not in content:
                    missing_patterns.append(pattern)
            
            if missing_patterns:
                self.warnings.append(f".gitignore missing patterns: {', '.join(missing_patterns)}")
                print_warning(".gitignore missing important patterns:")
                for pattern in missing_patterns:
                    print(f"    - {pattern}")
            else:
                print_success(".gitignore looks comprehensive")
                
        except Exception as e:
            self.warnings.append(f"Error reading .gitignore: {e}")
            print_warning(f"Error reading .gitignore: {e}")
    
    def generate_report(self):
        """Generate final report"""
        print_header("Verification Report")
        
        total_issues = len(self.errors) + len(self.warnings)
        
        if len(self.errors) == 0 and len(self.warnings) == 0:
            print_success("✓ No issues found! Repository structure looks good.")
        else:
            if self.errors:
                print(f"\n{Colors.RED}{Colors.BOLD}Critical Errors ({len(self.errors)}):{Colors.ENDC}")
                for i, error in enumerate(self.errors, 1):
                    print(f"{Colors.RED}  {i}. {error}{Colors.ENDC}")
            
            if self.warnings:
                print(f"\n{Colors.YELLOW}{Colors.BOLD}Warnings ({len(self.warnings)}):{Colors.ENDC}")
                for i, warning in enumerate(self.warnings, 1):
                    print(f"{Colors.YELLOW}  {i}. {warning}{Colors.ENDC}")
        
        print(f"\n{Colors.BOLD}Summary:{Colors.ENDC}")
        print(f"  Critical Errors: {Colors.RED if self.errors else Colors.GREEN}{len(self.errors)}{Colors.ENDC}")
        print(f"  Warnings: {Colors.YELLOW if self.warnings else Colors.GREEN}{len(self.warnings)}{Colors.ENDC}")
        
        if self.errors:
            print(f"\n{Colors.RED}{Colors.BOLD}⚠ Action Required: Please fix critical errors before using the repository{Colors.ENDC}")
            return 1
        elif self.warnings:
            print(f"\n{Colors.YELLOW}⚠ Consider addressing warnings for better functionality{Colors.ENDC}")
            return 0
        else:
            print(f"\n{Colors.GREEN}✓ Repository is ready to use!{Colors.ENDC}")
            return 0
    
    def run(self):
        """Run all verification checks"""
        print_header("Face Recognition PCA-ANN Repository Verification")
        
        self.check_file_naming()
        self.check_directory_structure()
        self.check_python_files()
        self.check_requirements()
        self.check_config()
        self.check_gitignore()
        
        return self.generate_report()

def main():
    verifier = RepositoryVerifier()
    exit_code = verifier.run()
    sys.exit(exit_code)

if __name__ == '__main__':
    main()
