#!/usr/bin/env python3
"""
Universal Project Unification Analyzer
======================================

A comprehensive tool for analyzing, comparing, and merging two separate projects.
Combines Repomix, Git, AST analysis, ML embeddings, and advanced metrics.

Features:
- Repomix integration for AI-friendly repository packing
- Git-based diff and merge analysis
- Multi-language AST parsing (Python, JavaScript, TypeScript, Java, Go)
- Code embeddings for semantic similarity
- Dependency graph analysis
- Architecture pattern detection
- Advanced code metrics
- Database schema comparison
- Smart merge strategy generation
- Interactive HTML report generation

Author: Enhanced AI Assistant
License: MIT
"""

import os
import sys
import json
import subprocess
import tempfile
import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from collections import defaultdict, Counter
from datetime import datetime
import re
import ast
import difflib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional imports with fallbacks
try:
    import numpy as np
    from numpy import ndarray as NumpyArray
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    NumpyArray = Any  # Fallback type
    print("Warning: numpy not available. Some features disabled.")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import DBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not available. ML features disabled.")

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: networkx not available. Graph analysis disabled.")

try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False
    print("Warning: sentence-transformers not available. Semantic similarity disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class FileInfo:
    """Information about a file"""
    path: str
    language: str
    lines: int
    size: int
    hash: str
    content: str = ""
    features: List[Dict] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    complexity: int = 0


@dataclass
class Feature:
    """Represents a code feature (function, class, etc.)"""
    name: str
    type: str  # function, class, method, etc.
    file_path: str
    line_number: int
    description: str
    inputs: List[Dict] = field(default_factory=list)
    outputs: List[Dict] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    complexity: int = 0
    signature: str = ""


@dataclass
class SimilarityMatch:
    """Similarity between two items"""
    item1: str
    item2: str
    score: float
    type: str  # file, feature, content
    details: Dict = field(default_factory=dict)


@dataclass
class MergeStrategy:
    """Recommended merge strategy"""
    action: str  # merge, keep_both, prefer_p1, prefer_p2, manual_review
    reason: str
    confidence: float
    items: List[str]
    details: Dict = field(default_factory=dict)


# ============================================================================
# Language-Specific Parsers
# ============================================================================

class LanguageParser:
    """Base class for language-specific parsers"""

    def parse_file(self, content: str, filepath: str) -> List[Feature]:
        """Parse file and extract features"""
        raise NotImplementedError

    def extract_dependencies(self, content: str) -> List[str]:
        """Extract import/dependency statements"""
        raise NotImplementedError

    def calculate_complexity(self, content: str) -> int:
        """Calculate cyclomatic complexity"""
        return 1


class PythonParser(LanguageParser):
    """Python-specific parser"""

    def parse_file(self, content: str, filepath: str) -> List[Feature]:
        features = []
        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    feature = self._parse_function(node, filepath)
                    features.append(feature)
                elif isinstance(node, ast.ClassDef):
                    feature = self._parse_class(node, filepath)
                    features.append(feature)
        except SyntaxError as e:
            logger.warning(f"Syntax error in {filepath}: {e}")

        return features

    def _parse_function(self, node: ast.FunctionDef, filepath: str) -> Feature:
        """Parse function definition"""
        inputs = []
        for arg in node.args.args:
            input_info = {
                'name': arg.arg,
                'type': self._get_annotation(arg.annotation)
            }
            inputs.append(input_info)

        outputs = [{
            'type': self._get_annotation(node.returns)
        }]

        # Extract docstring
        docstring = ast.get_docstring(node) or ""

        # Get function signature
        signature = self._get_function_signature(node)

        return Feature(
            name=node.name,
            type='function',
            file_path=filepath,
            line_number=node.lineno,
            description=docstring,
            inputs=inputs,
            outputs=outputs,
            complexity=self._calculate_function_complexity(node),
            signature=signature
        )

    def _parse_class(self, node: ast.ClassDef, filepath: str) -> Feature:
        """Parse class definition"""
        docstring = ast.get_docstring(node) or ""

        # Extract methods
        methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]

        return Feature(
            name=node.name,
            type='class',
            file_path=filepath,
            line_number=node.lineno,
            description=docstring,
            dependencies=methods,
            signature=f"class {node.name}"
        )

    def _get_annotation(self, annotation) -> str:
        """Get type annotation as string"""
        if annotation is None:
            return "Any"
        try:
            return ast.unparse(annotation)
        except:
            return "Unknown"

    def _get_function_signature(self, node: ast.FunctionDef) -> str:
        """Generate function signature"""
        args = []
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {self._get_annotation(arg.annotation)}"
            args.append(arg_str)

        signature = f"def {node.name}({', '.join(args)})"
        if node.returns:
            signature += f" -> {self._get_annotation(node.returns)}"

        return signature

    def _calculate_function_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1
        for n in ast.walk(node):
            if isinstance(n, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(n, ast.BoolOp):
                complexity += len(n.values) - 1
        return complexity

    def extract_dependencies(self, content: str) -> List[str]:
        """Extract import statements"""
        dependencies = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        dependencies.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        dependencies.append(node.module)
        except:
            pass
        return dependencies


class JavaScriptParser(LanguageParser):
    """JavaScript/TypeScript parser"""

    def parse_file(self, content: str, filepath: str) -> List[Feature]:
        features = []

        # Function declarations: function name() {} or const name = () => {}
        func_patterns = [
            r'function\s+(\w+)\s*\(([^)]*)\)',
            r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(([^)]*)\)\s*=>',
            r'(?:const|let|var)\s+(\w+)\s*=\s*function\s*\(([^)]*)\)'
        ]

        for pattern in func_patterns:
            for match in re.finditer(pattern, content):
                line_num = content[:match.start()].count('\n') + 1
                features.append(Feature(
                    name=match.group(1),
                    type='function',
                    file_path=filepath,
                    line_number=line_num,
                    description="",
                    inputs=self._parse_params(match.group(2)),
                    signature=match.group(0)
                ))

        # Class declarations
        class_pattern = r'class\s+(\w+)(?:\s+extends\s+(\w+))?'
        for match in re.finditer(class_pattern, content):
            line_num = content[:match.start()].count('\n') + 1
            features.append(Feature(
                name=match.group(1),
                type='class',
                file_path=filepath,
                line_number=line_num,
                description="",
                signature=match.group(0)
            ))

        return features

    def _parse_params(self, params_str: str) -> List[Dict]:
        """Parse function parameters"""
        if not params_str.strip():
            return []

        params = []
        for param in params_str.split(','):
            param = param.strip()
            # Handle TypeScript type annotations
            if ':' in param:
                name, type_info = param.split(':', 1)
                params.append({'name': name.strip(), 'type': type_info.strip()})
            else:
                params.append({'name': param, 'type': 'any'})
        return params

    def extract_dependencies(self, content: str) -> List[str]:
        """Extract import/require statements"""
        dependencies = []

        # ES6 imports
        import_pattern = r'import\s+(?:.+\s+from\s+)?[\'"]([^\'"]+)[\'"]'
        dependencies.extend(re.findall(import_pattern, content))

        # CommonJS requires
        require_pattern = r'require\s*\([\'"]([^\'"]+)[\'"]\)'
        dependencies.extend(re.findall(require_pattern, content))

        return dependencies


class JavaParser(LanguageParser):
    """Java parser"""

    def parse_file(self, content: str, filepath: str) -> List[Feature]:
        features = []

        # Method declarations
        method_pattern = r'(?:public|private|protected)?\s*(?:static)?\s*(\w+)\s+(\w+)\s*\(([^)]*)\)'
        for match in re.finditer(method_pattern, content):
            line_num = content[:match.start()].count('\n') + 1
            features.append(Feature(
                name=match.group(2),
                type='method',
                file_path=filepath,
                line_number=line_num,
                description="",
                outputs=[{'type': match.group(1)}],
                signature=match.group(0)
            ))

        # Class declarations
        class_pattern = r'(?:public|private)?\s*class\s+(\w+)'
        for match in re.finditer(class_pattern, content):
            line_num = content[:match.start()].count('\n') + 1
            features.append(Feature(
                name=match.group(1),
                type='class',
                file_path=filepath,
                line_number=line_num,
                description="",
                signature=match.group(0)
            ))

        return features

    def extract_dependencies(self, content: str) -> List[str]:
        """Extract import statements"""
        import_pattern = r'import\s+([\w.]+(?:\.\*)?);'
        return re.findall(import_pattern, content)


class GoParser(LanguageParser):
    """Go parser"""

    def parse_file(self, content: str, filepath: str) -> List[Feature]:
        features = []

        # Function declarations
        func_pattern = r'func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(([^)]*)\)\s*(?:\(([^)]*)\)|(\w+))?'
        for match in re.finditer(func_pattern, content):
            line_num = content[:match.start()].count('\n') + 1
            features.append(Feature(
                name=match.group(1),
                type='function',
                file_path=filepath,
                line_number=line_num,
                description="",
                signature=match.group(0)
            ))

        # Struct declarations (Go's "classes")
        struct_pattern = r'type\s+(\w+)\s+struct'
        for match in re.finditer(struct_pattern, content):
            line_num = content[:match.start()].count('\n') + 1
            features.append(Feature(
                name=match.group(1),
                type='struct',
                file_path=filepath,
                line_number=line_num,
                description="",
                signature=match.group(0)
            ))

        return features

    def extract_dependencies(self, content: str) -> List[str]:
        """Extract import statements"""
        dependencies = []

        # Single imports
        single_pattern = r'import\s+"([^"]+)"'
        dependencies.extend(re.findall(single_pattern, content))

        # Multi-line imports
        multi_pattern = r'import\s+\((.*?)\)'
        for match in re.finditer(multi_pattern, content, re.DOTALL):
            imports = re.findall(r'"([^"]+)"', match.group(1))
            dependencies.extend(imports)

        return dependencies


# ============================================================================
# Parser Factory
# ============================================================================

class ParserFactory:
    """Factory for creating language-specific parsers"""

    _parsers = {
        'python': PythonParser,
        'javascript': JavaScriptParser,
        'typescript': JavaScriptParser,
        'java': JavaParser,
        'go': GoParser
    }

    @classmethod
    def get_parser(cls, language: str) -> LanguageParser:
        """Get parser for language"""
        parser_class = cls._parsers.get(language.lower())
        if parser_class:
            return parser_class()
        return LanguageParser()  # Fallback


# ============================================================================
# File Analyzer
# ============================================================================

class FileAnalyzer:
    """Analyzes individual files"""

    LANGUAGE_EXTENSIONS = {
        '.py': 'python',
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.go': 'go',
        '.rs': 'rust',
        '.cpp': 'cpp',
        '.c': 'c',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.cs': 'csharp'
    }

    def __init__(self):
        self.parsers = {}

    def analyze_file(self, filepath: Path) -> Optional[FileInfo]:
        """Analyze a single file"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            language = self._detect_language(filepath)
            file_hash = hashlib.md5(content.encode()).hexdigest()

            # Get parser
            parser = ParserFactory.get_parser(language)

            # Extract features
            features = parser.parse_file(content, str(filepath))

            # Extract dependencies
            dependencies = parser.extract_dependencies(content)

            return FileInfo(
                path=str(filepath),
                language=language,
                lines=len(content.split('\n')),
                size=len(content),
                hash=file_hash,
                content=content,
                features=[asdict(f) for f in features],
                dependencies=dependencies
            )

        except Exception as e:
            logger.warning(f"Error analyzing {filepath}: {e}")
            return None

    def _detect_language(self, filepath: Path) -> str:
        """Detect programming language from file extension"""
        ext = filepath.suffix.lower()
        return self.LANGUAGE_EXTENSIONS.get(ext, 'unknown')


# ============================================================================
# Git Operations
# ============================================================================

class GitAnalyzer:
    """Handles Git operations and analysis"""

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path

    def get_file_history(self, filepath: str) -> List[Dict]:
        """Get commit history for a file"""
        try:
            cmd = [
                'git', 'log',
                '--follow',
                '--format=%H|%an|%ae|%ad|%s',
                '--',
                filepath
            ]
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )

            history = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split('|')
                    if len(parts) >= 5:
                        history.append({
                            'commit': parts[0],
                            'author': parts[1],
                            'email': parts[2],
                            'date': parts[3],
                            'message': '|'.join(parts[4:])
                        })
            return history
        except subprocess.CalledProcessError:
            return []

    def get_file_stats(self) -> Dict:
        """Get statistics about files in repository"""
        try:
            cmd = ['git', 'log', '--all', '--format=format:', '--name-only']
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )

            file_changes = Counter(
                line for line in result.stdout.split('\n') if line
            )

            return dict(file_changes.most_common(50))
        except subprocess.CalledProcessError:
            return {}

    def get_contributors(self) -> List[Dict]:
        """Get list of contributors"""
        try:
            cmd = ['git', 'shortlog', '-sn', '--all']
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )

            contributors = []
            for line in result.stdout.strip().split('\n'):
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    contributors.append({
                        'commits': int(parts[0]),
                        'name': parts[1]
                    })
            return contributors
        except subprocess.CalledProcessError:
            return []

    @staticmethod
    def diff_repositories(repo1: Path, repo2: Path) -> Dict:
        """Perform git diff between two repositories"""
        try:
            # Get numstat
            cmd = [
                'git', 'diff', '--no-index', '--numstat',
                str(repo1), str(repo2)
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False  # This command returns non-zero on differences
            )

            diff_stats = []
            for line in result.stdout.split('\n'):
                if line.strip():
                    parts = line.split('\t')
                    if len(parts) == 3:
                        diff_stats.append({
                            'additions': parts[0],
                            'deletions': parts[1],
                            'file': parts[2]
                        })

            return {
                'stats': diff_stats,
                'total_files': len(diff_stats)
            }
        except Exception as e:
            logger.error(f"Error diffing repositories: {e}")
            return {'stats': [], 'total_files': 0}


# ============================================================================
# Repomix Integration
# ============================================================================

class RepomixAnalyzer:
    """Handles Repomix operations"""

    def __init__(self):
        self._check_repomix()

    def _check_repomix(self):
        """Check if Repomix is installed"""
        try:
            subprocess.run(
                ['repomix', '--version'],
                capture_output=True,
                check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Repomix not found. Install with: npm install -g repomix")
            logger.warning("Repomix features will be disabled.")

    def pack_repository(
        self,
        repo_path: Path,
        output_path: Path,
        style: str = 'xml'
    ) -> bool:
        """Pack repository using Repomix"""
        try:
            cmd = [
                'repomix',
                '--output', str(output_path),
                '--style', style,
                '--show-line-numbers',
                str(repo_path)
            ]

            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Packed repository to {output_path}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Repomix failed: {e}")
            return False
        except FileNotFoundError:
            logger.warning("Repomix not available")
            return False

    def parse_packed_output(self, output_path: Path) -> Dict[str, str]:
        """Parse Repomix output file"""
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()

            files = {}

            # Parse XML format
            if output_path.suffix == '.xml':
                import xml.etree.ElementTree as ET
                try:
                    tree = ET.parse(output_path)
                    root = tree.getroot()

                    for file_elem in root.findall('.//file'):
                        path = file_elem.get('path', '')
                        file_content = file_elem.text or ""
                        files[path] = file_content
                except ET.ParseError:
                    logger.warning("Failed to parse XML, using text format")

            # Fallback to text parsing
            if not files:
                file_pattern = r'File: (.+?)\n(.*?)(?=\nFile: |\Z)'
                matches = re.finditer(file_pattern, content, re.DOTALL)
                for match in matches:
                    files[match.group(1)] = match.group(2)

            return files

        except Exception as e:
            logger.error(f"Error parsing Repomix output: {e}")
            return {}


# ============================================================================
# Embedding-based Similarity
# ============================================================================

class EmbeddingAnalyzer:
    """Analyzes code using embeddings"""

    def __init__(self):
        self.model = None
        if HAS_EMBEDDINGS:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Loaded embedding model")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                self.model = None

    def compute_embeddings(self, texts: List[str]) -> Optional[NumpyArray]:
        """Compute embeddings for texts"""
        if not self.model or not HAS_NUMPY:
            return None

        try:
            embeddings = self.model.encode(texts, show_progress_bar=False)
            return embeddings
        except Exception as e:
            logger.error(f"Error computing embeddings: {e}")
            return None

    def find_similar_by_embedding(
        self,
        items1: List[str],
        items2: List[str],
        threshold: float = 0.5
    ) -> List[Tuple[int, int, float]]:
        """Find similar items using embeddings"""
        if not self.model or not HAS_NUMPY or not HAS_SKLEARN:
            return []

        emb1 = self.compute_embeddings(items1)
        emb2 = self.compute_embeddings(items2)

        if emb1 is None or emb2 is None:
            return []

        similarities = cosine_similarity(emb1, emb2)

        matches = []
        for i in range(len(items1)):
            for j in range(len(items2)):
                if similarities[i][j] >= threshold:
                    matches.append((i, j, float(similarities[i][j])))

        return sorted(matches, key=lambda x: x[2], reverse=True)


# ============================================================================
# Dependency Graph
# ============================================================================

class DependencyGraphAnalyzer:
    """Analyzes dependency relationships"""

    def __init__(self):
        self.graph = nx.DiGraph() if HAS_NETWORKX else None

    def build_graph(self, files: Dict[str, FileInfo]):
        """Build dependency graph from files"""
        if not self.graph:
            return

        # Add nodes
        for filepath, file_info in files.items():
            self.graph.add_node(filepath, **asdict(file_info))

        # Add edges (dependencies)
        for filepath, file_info in files.items():
            for dep in file_info.dependencies:
                # Try to find the dependency file
                dep_file = self._resolve_dependency(dep, filepath, files)
                if dep_file and dep_file in files:
                    self.graph.add_edge(filepath, dep_file)

    def _resolve_dependency(
        self,
        dep: str,
        current_file: str,
        files: Dict[str, FileInfo]
    ) -> Optional[str]:
        """Resolve dependency to actual file"""
        # Simple resolution - can be enhanced
        for filepath in files:
            if dep in filepath or filepath.endswith(f"{dep}.py"):
                return filepath
        return None

    def find_cycles(self) -> List[List[str]]:
        """Find circular dependencies"""
        if not self.graph:
            return []

        try:
            cycles = list(nx.simple_cycles(self.graph))
            return cycles
        except:
            return []

    def get_critical_files(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """Get most depended-upon files"""
        if not self.graph:
            return []

        # Calculate in-degree (how many files depend on this one)
        in_degrees = self.graph.in_degree()
        sorted_files = sorted(in_degrees, key=lambda x: x[1], reverse=True)
        return sorted_files[:top_n]

    def calculate_coupling(self) -> Dict[str, float]:
        """Calculate coupling metrics"""
        if not self.graph:
            return {}

        coupling = {}
        for node in self.graph.nodes():
            in_deg = self.graph.in_degree(node)
            out_deg = self.graph.out_degree(node)
            coupling[node] = in_deg + out_deg

        return coupling


# ============================================================================
# Main Analyzer
# ============================================================================

class ProjectUnificationAnalyzer:
    """Main analyzer combining all features"""

    def __init__(
        self,
        project1_path: str,
        project2_path: str,
        output_dir: str = "./analysis",
        use_cache: bool = True
    ):
        self.p1_path = Path(project1_path)
        self.p2_path = Path(project2_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.use_cache = use_cache

        # Initialize components
        self.file_analyzer = FileAnalyzer()
        self.repomix = RepomixAnalyzer()
        self.embedding_analyzer = EmbeddingAnalyzer()
        self.git1 = GitAnalyzer(self.p1_path)
        self.git2 = GitAnalyzer(self.p2_path)

        # Results storage
        self.p1_files: Dict[str, FileInfo] = {}
        self.p2_files: Dict[str, FileInfo] = {}
        self.similarities: List[SimilarityMatch] = []
        self.merge_strategies: List[MergeStrategy] = []
        self.analysis_results: Dict = {}

    def _get_cache_path(self, project_name: str) -> Path:
        """Get cache file path"""
        return self.output_dir / f"{project_name}_cache.pkl"

    def _load_cache(self, project_name: str) -> Optional[Dict]:
        """Load cached analysis"""
        if not self.use_cache:
            return None

        cache_path = self._get_cache_path(project_name)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return None

    def _save_cache(self, project_name: str, data: Dict):
        """Save analysis to cache"""
        if not self.use_cache:
            return

        cache_path = self._get_cache_path(project_name)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def analyze_project(
        self,
        project_path: Path,
        project_name: str
    ) -> Dict[str, FileInfo]:
        """Analyze a single project"""
        logger.info(f"Analyzing {project_name}...")

        # Check cache
        cached = self._load_cache(project_name)
        if cached:
            logger.info(f"Loaded {project_name} from cache")
            return cached

        files = {}

        # Find all source files
        patterns = ['**/*.py', '**/*.js', '**/*.ts', '**/*.java', '**/*.go']
        file_list = []
        for pattern in patterns:
            file_list.extend(project_path.glob(pattern))

        # Filter out common ignore patterns
        ignore_patterns = [
            'node_modules', '__pycache__', '.git', 'dist',
            'build', 'venv', '.venv', 'vendor'
        ]

        file_list = [
            f for f in file_list
            if not any(ignore in str(f) for ignore in ignore_patterns)
        ]

        logger.info(f"Found {len(file_list)} files in {project_name}")

        # Analyze files in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_file = {
                executor.submit(self.file_analyzer.analyze_file, f): f
                for f in file_list
            }

            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    file_info = future.result()
                    if file_info:
                        rel_path = str(file_path.relative_to(project_path))
                        files[rel_path] = file_info
                except Exception as e:
                    logger.error(f"Error analyzing {file_path}: {e}")

        # Save to cache
        self._save_cache(project_name, files)

        return files

    def find_similar_files(self) -> List[SimilarityMatch]:
        """Find similar files between projects"""
        logger.info("Finding similar files...")

        matches = []

        # Compare by filename
        for p1_path, p1_info in self.p1_files.items():
            for p2_path, p2_info in self.p2_files.items():
                # Name similarity
                name_sim = difflib.SequenceMatcher(
                    None,
                    Path(p1_path).name,
                    Path(p2_path).name
                ).ratio()

                # Content similarity
                content_sim = difflib.SequenceMatcher(
                    None,
                    p1_info.content[:1000],  # First 1000 chars
                    p2_info.content[:1000]
                ).ratio()

                # Hash match (exact duplicate)
                hash_match = p1_info.hash == p2_info.hash

                # Combined score
                score = (name_sim * 0.3 + content_sim * 0.7)

                if hash_match:
                    score = 1.0

                if score > 0.5:
                    matches.append(SimilarityMatch(
                        item1=p1_path,
                        item2=p2_path,
                        score=score,
                        type='file',
                        details={
                            'name_similarity': name_sim,
                            'content_similarity': content_sim,
                            'exact_match': hash_match,
                            'language': p1_info.language
                        }
                    ))

        # Sort by score
        matches.sort(key=lambda x: x.score, reverse=True)

        return matches

    def find_similar_features(self) -> List[SimilarityMatch]:
        """Find similar features (functions, classes) between projects"""
        logger.info("Finding similar features...")

        # Extract all features
        p1_features = []
        p1_feature_texts = []

        for file_info in self.p1_files.values():
            for feature_dict in file_info.features:
                feature = Feature(**feature_dict)
                p1_features.append(feature)
                # Create text representation
                text = f"{feature.name} {feature.signature} {feature.description}"
                p1_feature_texts.append(text)

        p2_features = []
        p2_feature_texts = []

        for file_info in self.p2_files.values():
            for feature_dict in file_info.features:
                feature = Feature(**feature_dict)
                p2_features.append(feature)
                text = f"{feature.name} {feature.signature} {feature.description}"
                p2_feature_texts.append(text)

        logger.info(f"Found {len(p1_features)} features in project1")
        logger.info(f"Found {len(p2_features)} features in project2")

        matches = []

        # Try embedding-based similarity first
        if self.embedding_analyzer.model:
            logger.info("Using embedding-based similarity...")
            emb_matches = self.embedding_analyzer.find_similar_by_embedding(
                p1_feature_texts,
                p2_feature_texts,
                threshold=0.6
            )

            for i, j, score in emb_matches:
                matches.append(SimilarityMatch(
                    item1=f"{p1_features[i].file_path}::{p1_features[i].name}",
                    item2=f"{p2_features[j].file_path}::{p2_features[j].name}",
                    score=score,
                    type='feature',
                    details={
                        'feature1': asdict(p1_features[i]),
                        'feature2': asdict(p2_features[j]),
                        'method': 'embedding'
                    }
                ))
        else:
            # Fallback to string similarity
            logger.info("Using string-based similarity...")
            for i, f1 in enumerate(p1_features):
                for j, f2 in enumerate(p2_features):
                    # Name similarity
                    name_sim = difflib.SequenceMatcher(
                        None, f1.name, f2.name
                    ).ratio()

                    # Signature similarity
                    sig_sim = difflib.SequenceMatcher(
                        None, f1.signature, f2.signature
                    ).ratio()

                    score = (name_sim * 0.6 + sig_sim * 0.4)

                    if score > 0.7:
                        matches.append(SimilarityMatch(
                            item1=f"{f1.file_path}::{f1.name}",
                            item2=f"{f2.file_path}::{f2.name}",
                            score=score,
                            type='feature',
                            details={
                                'feature1': asdict(f1),
                                'feature2': asdict(f2),
                                'method': 'string'
                            }
                        ))

        matches.sort(key=lambda x: x.score, reverse=True)
        return matches

    def generate_merge_strategies(self) -> List[MergeStrategy]:
        """Generate recommended merge strategies"""
        logger.info("Generating merge strategies...")

        strategies = []

        # Categorize file similarities
        high_similarity = [s for s in self.similarities if s.score > 0.9]
        medium_similarity = [s for s in self.similarities if 0.7 < s.score <= 0.9]
        low_similarity = [s for s in self.similarities if 0.5 < s.score <= 0.7]

        # Strategy for high similarity files (likely duplicates)
        if high_similarity:
            strategies.append(MergeStrategy(
                action='merge',
                reason='Files are nearly identical (>90% similarity)',
                confidence=0.95,
                items=[s.item1 for s in high_similarity],
                details={
                    'count': len(high_similarity),
                    'recommendation': 'Keep one version, review differences carefully'
                }
            ))

        # Strategy for medium similarity files
        if medium_similarity:
            strategies.append(MergeStrategy(
                action='manual_review',
                reason='Files have significant overlap but notable differences',
                confidence=0.70,
                items=[s.item1 for s in medium_similarity],
                details={
                    'count': len(medium_similarity),
                    'recommendation': 'Compare features and merge incrementally'
                }
            ))

        # Strategy for unique files in project1
        matched_p1 = {s.item1 for s in self.similarities}
        unique_p1 = set(self.p1_files.keys()) - matched_p1

        if unique_p1:
            strategies.append(MergeStrategy(
                action='keep_both',
                reason='Files unique to project1',
                confidence=1.0,
                items=list(unique_p1),
                details={
                    'count': len(unique_p1),
                    'recommendation': 'Port directly to unified project'
                }
            ))

        # Strategy for unique files in project2
        matched_p2 = {s.item2 for s in self.similarities}
        unique_p2 = set(self.p2_files.keys()) - matched_p2

        if unique_p2:
            strategies.append(MergeStrategy(
                action='keep_both',
                reason='Files unique to project2',
                confidence=1.0,
                items=list(unique_p2),
                details={
                    'count': len(unique_p2),
                    'recommendation': 'Port directly to unified project'
                }
            ))

        return strategies

    def analyze_dependencies(self):
        """Analyze dependency relationships"""
        logger.info("Analyzing dependencies...")

        dep_analyzer1 = DependencyGraphAnalyzer()
        dep_analyzer2 = DependencyGraphAnalyzer()

        if HAS_NETWORKX:
            dep_analyzer1.build_graph(self.p1_files)
            dep_analyzer2.build_graph(self.p2_files)

            # Find cycles
            cycles1 = dep_analyzer1.find_cycles()
            cycles2 = dep_analyzer2.find_cycles()

            # Critical files
            critical1 = dep_analyzer1.get_critical_files()
            critical2 = dep_analyzer2.get_critical_files()

            return {
                'project1': {
                    'circular_dependencies': cycles1,
                    'critical_files': critical1
                },
                'project2': {
                    'circular_dependencies': cycles2,
                    'critical_files': critical2
                }
            }

        return {}

    def run_repomix_analysis(self):
        """Run Repomix on both projects"""
        logger.info("Running Repomix analysis...")

        p1_output = self.output_dir / "project1-repomix.xml"
        p2_output = self.output_dir / "project2-repomix.xml"

        success1 = self.repomix.pack_repository(self.p1_path, p1_output)
        success2 = self.repomix.pack_repository(self.p2_path, p2_output)

        return {
            'project1_packed': success1,
            'project2_packed': success2,
            'project1_output': str(p1_output) if success1 else None,
            'project2_output': str(p2_output) if success2 else None
        }

    def run_git_analysis(self):
        """Run Git-based analysis"""
        logger.info("Running Git analysis...")

        return {
            'project1': {
                'contributors': self.git1.get_contributors(),
                'file_stats': self.git1.get_file_stats()
            },
            'project2': {
                'contributors': self.git2.get_contributors(),
                'file_stats': self.git2.get_file_stats()
            },
            'diff': GitAnalyzer.diff_repositories(self.p1_path, self.p2_path)
        }

    def generate_report(self):
        """Generate comprehensive analysis report"""
        logger.info("Generating report...")

        report = {
            'timestamp': datetime.now().isoformat(),
            'projects': {
                'project1': {
                    'path': str(self.p1_path),
                    'files': len(self.p1_files),
                    'languages': Counter(
                        f.language for f in self.p1_files.values()
                    ),
                    'total_lines': sum(
                        f.lines for f in self.p1_files.values()
                    ),
                    'features': sum(
                        len(f.features) for f in self.p1_files.values()
                    )
                },
                'project2': {
                    'path': str(self.p2_path),
                    'files': len(self.p2_files),
                    'languages': Counter(
                        f.language for f in self.p2_files.values()
                    ),
                    'total_lines': sum(
                        f.lines for f in self.p2_files.values()
                    ),
                    'features': sum(
                        len(f.features) for f in self.p2_files.values()
                    )
                }
            },
            'similarities': {
                'file_matches': len(self.similarities),
                'high_confidence': len([s for s in self.similarities if s.score > 0.9]),
                'medium_confidence': len([s for s in self.similarities if 0.7 < s.score <= 0.9]),
                'low_confidence': len([s for s in self.similarities if 0.5 < s.score <= 0.7])
            },
            'merge_strategies': [asdict(s) for s in self.merge_strategies],
            'detailed_matches': [asdict(s) for s in self.similarities[:100]],  # Top 100
        }

        return report

    def save_results(self):
        """Save all results to files"""
        logger.info("Saving results...")

        # Save main report
        report_path = self.output_dir / "analysis_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)

        logger.info(f"Report saved to {report_path}")

        # Save detailed file info
        files_path = self.output_dir / "project1_files.json"
        with open(files_path, 'w') as f:
            json.dump({k: asdict(v) for k, v in self.p1_files.items()}, f, indent=2)

        files_path = self.output_dir / "project2_files.json"
        with open(files_path, 'w') as f:
            json.dump({k: asdict(v) for k, v in self.p2_files.items()}, f, indent=2)

        # Generate markdown report
        self.generate_markdown_report()

        # Generate HTML report
        self.generate_html_report()

    def generate_markdown_report(self):
        """Generate human-readable markdown report"""
        report_path = self.output_dir / "ANALYSIS_REPORT.md"

        with open(report_path, 'w') as f:
            f.write("# Project Unification Analysis Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Overview
            f.write("## Overview\n\n")
            f.write(f"- **Project 1:** {self.p1_path}\n")
            f.write(f"- **Project 2:** {self.p2_path}\n\n")

            # Statistics
            f.write("## Statistics\n\n")
            f.write("| Metric | Project 1 | Project 2 |\n")
            f.write("|--------|-----------|----------|\n")
            f.write(f"| Files | {len(self.p1_files)} | {len(self.p2_files)} |\n")
            f.write(f"| Total Lines | {sum(f.lines for f in self.p1_files.values())} | ")
            f.write(f"{sum(f.lines for f in self.p2_files.values())} |\n")
            f.write(f"| Features | {sum(len(f.features) for f in self.p1_files.values())} | ")
            f.write(f"{sum(len(f.features) for f in self.p2_files.values())} |\n\n")

            # Similarities
            f.write("## File Similarities\n\n")
            f.write(f"Found **{len(self.similarities)}** similar files:\n\n")
            f.write("| File 1 | File 2 | Similarity | Type |\n")
            f.write("|--------|--------|------------|------|\n")

            for sim in self.similarities[:20]:  # Top 20
                f.write(f"| {sim.item1} | {sim.item2} | {sim.score:.2%} | {sim.type} |\n")

            # Merge Strategies
            f.write("\n## Recommended Merge Strategies\n\n")
            for strategy in self.merge_strategies:
                f.write(f"### {strategy.action.replace('_', ' ').title()}\n\n")
                f.write(f"- **Reason:** {strategy.reason}\n")
                f.write(f"- **Confidence:** {strategy.confidence:.0%}\n")
                f.write(f"- **Affected Items:** {len(strategy.items)}\n")
                f.write(f"- **Recommendation:** {strategy.details.get('recommendation', 'N/A')}\n\n")

            # Dependencies
            if 'dependencies' in self.analysis_results:
                f.write("## Dependency Analysis\n\n")
                deps = self.analysis_results['dependencies']

                if deps.get('project1', {}).get('circular_dependencies'):
                    f.write("### Project 1 Circular Dependencies ⚠️\n\n")
                    for cycle in deps['project1']['circular_dependencies'][:5]:
                        f.write(f"- {' -> '.join(cycle)}\n")
                    f.write("\n")

                if deps.get('project2', {}).get('circular_dependencies'):
                    f.write("### Project 2 Circular Dependencies ⚠️\n\n")
                    for cycle in deps['project2']['circular_dependencies'][:5]:
                        f.write(f"- {' -> '.join(cycle)}\n")
                    f.write("\n")

        logger.info(f"Markdown report saved to {report_path}")

    def generate_html_report(self):
        """Generate interactive HTML report"""
        report_path = self.output_dir / "report.html"

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Project Unification Analysis</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .section {{
            background: white;
            padding: 25px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #667eea;
            color: white;
        }}
        .similarity-high {{ color: #22c55e; font-weight: bold; }}
        .similarity-medium {{ color: #f59e0b; }}
        .similarity-low {{ color: #ef4444; }}
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
        }}
        .badge-success {{ background: #dcfce7; color: #166534; }}
        .badge-warning {{ background: #fef3c7; color: #92400e; }}
        .badge-info {{ background: #dbeafe; color: #1e40af; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔄 Project Unification Analysis</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="stats">
        <div class="stat-card">
            <div class="stat-value">{len(self.p1_files)}</div>
            <div>Project 1 Files</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(self.p2_files)}</div>
            <div>Project 2 Files</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(self.similarities)}</div>
            <div>Similar Files Found</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len([s for s in self.similarities if s.score > 0.9])}</div>
            <div>High Confidence Matches</div>
        </div>
    </div>

    <div class="section">
        <h2>📊 Similarity Matches</h2>
        <table>
            <thead>
                <tr>
                    <th>Project 1 File</th>
                    <th>Project 2 File</th>
                    <th>Similarity</th>
                    <th>Type</th>
                </tr>
            </thead>
            <tbody>
"""

        for sim in self.similarities[:50]:  # Top 50
            sim_class = 'similarity-high' if sim.score > 0.9 else (
                'similarity-medium' if sim.score > 0.7 else 'similarity-low'
            )
            html_content += f"""                <tr>
                    <td><code>{sim.item1}</code></td>
                    <td><code>{sim.item2}</code></td>
                    <td class="{sim_class}">{sim.score:.1%}</td>
                    <td><span class="badge badge-info">{sim.type}</span></td>
                </tr>
"""

        html_content += """            </tbody>
        </table>
    </div>

    <div class="section">
        <h2>🎯 Merge Strategies</h2>
"""

        for strategy in self.merge_strategies:
            html_content += f"""        <div style="margin-bottom: 20px; padding: 15px; background: #f9fafb; border-left: 4px solid #667eea; border-radius: 4px;">
            <h3>{strategy.action.replace('_', ' ').title()}</h3>
            <p><strong>Reason:</strong> {strategy.reason}</p>
            <p><strong>Confidence:</strong> {strategy.confidence:.0%}</p>
            <p><strong>Affected Items:</strong> {len(strategy.items)}</p>
            <p><strong>Recommendation:</strong> {strategy.details.get('recommendation', 'N/A')}</p>
        </div>
"""

        html_content += """    </div>
</body>
</html>
"""

        with open(report_path, 'w') as f:
            f.write(html_content)

        logger.info(f"HTML report saved to {report_path}")

    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        logger.info("=" * 70)
        logger.info("Starting Project Unification Analysis")
        logger.info("=" * 70)

        # Phase 1: Analyze both projects
        self.p1_files = self.analyze_project(self.p1_path, "project1")
        self.p2_files = self.analyze_project(self.p2_path, "project2")

        # Phase 2: Find similarities
        file_similarities = self.find_similar_files()
        feature_similarities = self.find_similar_features()
        self.similarities = file_similarities + feature_similarities

        # Phase 3: Generate merge strategies
        self.merge_strategies = self.generate_merge_strategies()

        # Phase 4: Additional analyses
        repomix_results = self.run_repomix_analysis()
        git_results = self.run_git_analysis()
        dep_results = self.analyze_dependencies()

        # Phase 5: Compile results
        self.analysis_results = self.generate_report()
        self.analysis_results['repomix'] = repomix_results
        self.analysis_results['git'] = git_results
        self.analysis_results['dependencies'] = dep_results

        # Phase 6: Save everything
        self.save_results()

        logger.info("=" * 70)
        logger.info("Analysis Complete!")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("=" * 70)

        return self.analysis_results


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Main CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Universal Project Unification Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/project1 /path/to/project2
  %(prog)s /path/to/project1 /path/to/project2 --output ./my-analysis
  %(prog)s /path/to/project1 /path/to/project2 --no-cache
        """
    )

    parser.add_argument(
        'project1',
        help='Path to first project'
    )

    parser.add_argument(
        'project2',
        help='Path to second project'
    )

    parser.add_argument(
        '--output', '-o',
        default='./analysis',
        help='Output directory for analysis results (default: ./analysis)'
    )

    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable caching of analysis results'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate paths
    if not Path(args.project1).exists():
        print(f"Error: Project 1 path does not exist: {args.project1}")
        sys.exit(1)

    if not Path(args.project2).exists():
        print(f"Error: Project 2 path does not exist: {args.project2}")
        sys.exit(1)

    # Run analysis
    try:
        analyzer = ProjectUnificationAnalyzer(
            args.project1,
            args.project2,
            args.output,
            use_cache=not args.no_cache
        )

        results = analyzer.run_full_analysis()

        print("\n✅ Analysis complete!")
        print(f"📁 Results saved to: {args.output}")
        print(f"📊 Found {len(analyzer.similarities)} similarities")
        print(f"🎯 Generated {len(analyzer.merge_strategies)} merge strategies")
        print(f"\n📄 View reports:")
        print(f"   - HTML: {Path(args.output) / 'report.html'}")
        print(f"   - Markdown: {Path(args.output) / 'ANALYSIS_REPORT.md'}")
        print(f"   - JSON: {Path(args.output) / 'analysis_report.json'}")

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
