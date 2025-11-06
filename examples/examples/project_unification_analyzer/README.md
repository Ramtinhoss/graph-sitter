# Project Unification Analyzer

A comprehensive toolkit for analyzing, comparing, and merging two separate software projects. This analyzer combines multiple advanced techniques including AST parsing, knowledge graphs, semantic similarity, dependency analysis, and automated merge strategy generation.

## Overview

The Project Unification Analyzer provides two powerful tools:

1. **Universal Analyzer** (`universal_analyzer.py`): A feature-rich analyzer that combines Repomix, Git analysis, AST parsing, ML embeddings, and dependency graphs
2. **Enhanced Analyzer** (`enhanced_analyzer.py`): An advanced version with knowledge graphs, full-text search indexing, code flow analysis, and migration planning

## Features

### Universal Analyzer Features

- **Multi-Language AST Parsing**: Supports Python, JavaScript, TypeScript, Java, and Go
- **Repomix Integration**: AI-friendly repository packing for LLM analysis
- **Git Analysis**: Commit history, contributor stats, and repository comparison
- **Semantic Similarity**: ML-based code embeddings for intelligent feature matching
- **Dependency Graphs**: Visualize and analyze code dependencies
- **Merge Strategies**: Automated recommendations for merging projects
- **Interactive Reports**: Generate HTML, Markdown, and JSON reports
- **Caching**: Speed up repeated analysis with intelligent caching

### Enhanced Analyzer Features

- **Knowledge Graphs**: Build comprehensive graphs of code relationships using NetworkX
- **Full-Text Search**: Fast code search using Whoosh indexing
- **Code Flow Analysis**: Trace call paths and data dependencies
- **Pattern Detection**: Identify design patterns (Singleton, Factory, Observer)
- **Migration Planning**: Generate step-by-step migration plans with risk assessment
- **Cross-Reference Mapping**: Map equivalent features between projects
- **Graph Export**: Export knowledge graphs in GraphML and JSON formats

## Installation

### Basic Installation

```bash
# Navigate to the graph-sitter directory
cd /path/to/graph-sitter

# Install the package (includes basic dependencies)
uv pip install -e .
```

### Optional Dependencies

For full functionality, install optional dependencies:

```bash
# For ML-based similarity (embeddings)
pip install sentence-transformers scikit-learn numpy

# For full-text search (Enhanced Analyzer)
pip install whoosh

# For knowledge graphs
pip install networkx

# Repomix (for repository packing)
npm install -g repomix
```

Or install all at once:

```bash
pip install -r examples/examples/project_unification_analyzer/requirements.txt
```

## Usage

### Universal Analyzer

Basic usage:

```bash
cd examples/examples/project_unification_analyzer

# Analyze two projects
python universal_analyzer.py /path/to/project1 /path/to/project2

# Specify output directory
python universal_analyzer.py /path/to/project1 /path/to/project2 --output ./my-analysis

# Disable caching
python universal_analyzer.py /path/to/project1 /path/to/project2 --no-cache

# Verbose logging
python universal_analyzer.py /path/to/project1 /path/to/project2 --verbose
```

### Enhanced Analyzer

Basic usage:

```bash
cd examples/examples/project_unification_analyzer

# Analyze two projects
python enhanced_analyzer.py /path/to/project1 /path/to/project2

# Specify output directory
python enhanced_analyzer.py /path/to/project1 /path/to/project2 --output ./enhanced-analysis

# Search for code
python enhanced_analyzer.py /path/to/project1 /path/to/project2 --search "authentication"
```

### Using as a Library

You can also import and use the analyzers in your Python code:

```python
from universal_analyzer import ProjectUnificationAnalyzer
from enhanced_analyzer import EnhancedUnificationAnalyzer

# Universal Analyzer
analyzer = ProjectUnificationAnalyzer(
    project1_path="/path/to/project1",
    project2_path="/path/to/project2",
    output_dir="./analysis",
    use_cache=True
)

results = analyzer.run_full_analysis()

# Access results
print(f"Found {len(analyzer.similarities)} similar files")
print(f"Merge strategies: {len(analyzer.merge_strategies)}")

# Enhanced Analyzer
enhanced = EnhancedUnificationAnalyzer(
    project1_path="/path/to/project1",
    project2_path="/path/to/project2",
    output_dir="./enhanced_analysis"
)

results = enhanced.run_full_analysis()

# Search code
search_results = enhanced.search_code("login function")
```

## Output

### Universal Analyzer Output

The analyzer generates several output files:

```
analysis/
├── analysis_report.json        # Complete analysis in JSON format
├── ANALYSIS_REPORT.md          # Human-readable markdown report
├── report.html                 # Interactive HTML report
├── project1_files.json         # Detailed file information for project 1
├── project2_files.json         # Detailed file information for project 2
├── project1-repomix.xml        # Repomix packed output (if available)
├── project2-repomix.xml        # Repomix packed output (if available)
├── project1_cache.pkl          # Cached analysis (for faster reruns)
└── project2_cache.pkl          # Cached analysis (for faster reruns)
```

### Enhanced Analyzer Output

```
enhanced_analysis/
├── knowledge_graph_project1.json    # KG for project 1
├── knowledge_graph_project2.json    # KG for project 2
├── kg_project1.graphml              # GraphML for visualization
├── kg_project2.graphml              # GraphML for visualization
├── entity_mappings.json             # Entity equivalence mappings
├── migration_plan.json              # Step-by-step migration plan
└── search_index/                    # Whoosh full-text search index
    └── ...
```

## Report Contents

### Analysis Report Includes

1. **Project Statistics**
   - Number of files per project
   - Lines of code
   - Language distribution
   - Feature counts (functions, classes, etc.)

2. **Similarity Analysis**
   - File-level similarities
   - Feature-level similarities (functions, classes)
   - Confidence scores
   - Exact duplicates

3. **Merge Strategies**
   - High similarity files (>90%): Merge recommendations
   - Medium similarity (70-90%): Manual review required
   - Unique files: Keep both
   - Confidence scores and recommendations

4. **Dependency Analysis** (if NetworkX available)
   - Circular dependencies
   - Critical files (most depended upon)
   - Coupling metrics

5. **Git Analysis**
   - Contributors
   - File change frequency
   - Repository diff statistics

6. **Knowledge Graph Insights** (Enhanced Analyzer)
   - Entity relationships
   - Design patterns detected
   - Critical entities
   - Isolated components

## Examples

### Example 1: Analyzing Two Microservices

```bash
# Compare two similar microservices
python universal_analyzer.py \
    ~/projects/user-service \
    ~/projects/auth-service \
    --output ./service-comparison
```

Output shows:
- 45 similar files found
- 12 high-confidence matches (>90% similarity)
- 8 unique authentication-related features in auth-service
- Migration strategy: Merge common utilities, keep service-specific logic

### Example 2: Refactoring Monolith to Microservices

```bash
# Analyze monolith and extract patterns
python enhanced_analyzer.py \
    ~/projects/monolith \
    ~/projects/new-architecture \
    --output ./refactoring-analysis
```

Output shows:
- Knowledge graph with 1,247 entities
- 34 detected design patterns
- Migration plan with 89 steps
- Risk assessment for each migration step

### Example 3: Code Search Across Projects

```bash
# Search for authentication code
python enhanced_analyzer.py \
    ~/projects/frontend \
    ~/projects/backend \
    --search "jwt token validation"
```

Output shows:
- 12 matches for "jwt token validation"
- Similar implementations in both projects
- Opportunities for code reuse

## Advanced Configuration

### Custom Parser Extensions

You can extend the parsers to support additional languages:

```python
from universal_analyzer import ParserFactory, LanguageParser

class RustParser(LanguageParser):
    def parse_file(self, content: str, filepath: str):
        # Custom Rust parsing logic
        pass

    def extract_dependencies(self, content: str):
        # Extract Rust dependencies
        pass

# Register the parser
ParserFactory._parsers['rust'] = RustParser
```

### Custom Similarity Metrics

```python
# Override similarity calculation
def custom_similarity(file1, file2):
    # Your custom logic
    return score

analyzer.find_similar_files = custom_similarity
```

## Visualizing Knowledge Graphs

The Enhanced Analyzer exports knowledge graphs in GraphML format, which can be visualized using tools like:

- **Gephi**: https://gephi.org/
- **Cytoscape**: https://cytoscape.org/
- **yEd**: https://www.yworks.com/products/yed

Example workflow:

1. Run enhanced analyzer
2. Open `kg_project1.graphml` in Gephi
3. Apply force-directed layout
4. Visualize dependencies and patterns

## Performance Considerations

### Caching

The Universal Analyzer caches analysis results to speed up subsequent runs:

```bash
# First run (slow)
python universal_analyzer.py project1 project2 --output ./analysis

# Subsequent runs (fast - uses cache)
python universal_analyzer.py project1 project2 --output ./analysis
```

To force re-analysis:

```bash
python universal_analyzer.py project1 project2 --output ./analysis --no-cache
```

### Large Projects

For very large projects (>10k files):

1. Use caching to avoid re-analysis
2. Limit file patterns to specific languages
3. Use parallel processing (automatically enabled)
4. Consider analyzing subdirectories separately

## Troubleshooting

### Missing Dependencies

If you see warnings about missing features:

```
Warning: numpy not available. Some features disabled.
Warning: scikit-learn not available. ML features disabled.
Warning: networkx not available. Graph analysis disabled.
Warning: whoosh not available. Install with: pip install whoosh
```

Install the missing dependencies:

```bash
pip install numpy scikit-learn networkx whoosh sentence-transformers
```

### Repomix Not Found

```
Repomix not found. Install with: npm install -g repomix
```

Install Repomix:

```bash
npm install -g repomix
```

### Memory Issues

For very large projects, you may encounter memory issues. Solutions:

1. Analyze projects separately
2. Increase Python memory limit
3. Use sampling (analyze subset of files)

## Integration with Graph-Sitter

These analyzers complement graph-sitter's capabilities:

```python
from graph_sitter import Codebase
from universal_analyzer import ProjectUnificationAnalyzer

# Use graph-sitter for detailed analysis
codebase1 = Codebase("/path/to/project1")
codebase2 = Codebase("/path/to/project2")

# Use unification analyzer for comparison
analyzer = ProjectUnificationAnalyzer(
    "/path/to/project1",
    "/path/to/project2"
)

results = analyzer.run_full_analysis()

# Combine insights from both tools
```

## API Reference

### ProjectUnificationAnalyzer

Main class for universal analysis.

**Methods:**

- `analyze_project(project_path, project_name)`: Analyze a single project
- `find_similar_files()`: Find similar files between projects
- `find_similar_features()`: Find similar functions/classes
- `generate_merge_strategies()`: Generate merge recommendations
- `run_full_analysis()`: Run complete analysis pipeline
- `save_results()`: Save all results to disk

### EnhancedUnificationAnalyzer

Main class for enhanced analysis with knowledge graphs.

**Methods:**

- `build_knowledge_graphs()`: Build KGs for both projects
- `find_equivalent_features()`: Find equivalent code entities
- `generate_migration_plan()`: Create migration roadmap
- `search_code(query)`: Search code using full-text index
- `run_full_analysis()`: Run complete enhanced analysis
- `export_results()`: Export all results

### KnowledgeGraph

Knowledge graph for code relationships.

**Methods:**

- `add_entity(entity)`: Add code entity to graph
- `add_relationship(relationship)`: Add relationship between entities
- `find_similar_entities(entity_id)`: Find similar entities
- `get_dependencies(entity_id)`: Get all dependencies
- `find_circular_dependencies()`: Detect cycles
- `get_critical_entities()`: Get most important entities
- `export_to_graphml(filepath)`: Export for visualization

## Contributing

Contributions are welcome! To add features:

1. Fork the repository
2. Add your feature to the analyzer
3. Write tests
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Related Tools

- **Repomix**: https://github.com/yamadashy/repomix
- **Graph-sitter**: The parent project this tool is part of
- **Tree-sitter**: https://tree-sitter.github.io/tree-sitter/

## Support

For issues, questions, or contributions:

- GitHub Issues: https://github.com/anthropics/graph-sitter/issues
- Documentation: https://docs.graph-sitter.dev/

## Citation

If you use this tool in research, please cite:

```bibtex
@software{project_unification_analyzer,
  title = {Project Unification Analyzer},
  author = {Graph-Sitter Contributors},
  year = {2025},
  url = {https://github.com/anthropics/graph-sitter}
}
```
