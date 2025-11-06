#!/usr/bin/env python3
"""
Example Usage of Project Unification Analyzers
==============================================

This script demonstrates how to use the analyzers as a library.
"""

from pathlib import Path
from universal_analyzer import ProjectUnificationAnalyzer
from enhanced_analyzer import EnhancedUnificationAnalyzer


def example_universal_analyzer():
    """Example using Universal Analyzer"""
    print("=" * 70)
    print("Universal Analyzer Example")
    print("=" * 70)

    # Create analyzer instance
    analyzer = ProjectUnificationAnalyzer(
        project1_path="./sample_project1",  # Replace with actual paths
        project2_path="./sample_project2",
        output_dir="./example_output",
        use_cache=True
    )

    # You can run individual analysis steps
    print("\nStep 1: Analyzing projects...")
    # p1_files = analyzer.analyze_project(analyzer.p1_path, "project1")
    # p2_files = analyzer.analyze_project(analyzer.p2_path, "project2")

    print("\nStep 2: Finding similarities...")
    # similarities = analyzer.find_similar_files()

    print("\nStep 3: Generating merge strategies...")
    # strategies = analyzer.generate_merge_strategies()

    # Or run full analysis at once
    print("\nRunning full analysis...")
    # results = analyzer.run_full_analysis()

    print("\nAnalysis complete! Check ./example_output/ for results")


def example_enhanced_analyzer():
    """Example using Enhanced Analyzer"""
    print("\n")
    print("=" * 70)
    print("Enhanced Analyzer Example")
    print("=" * 70)

    # Create enhanced analyzer instance
    analyzer = EnhancedUnificationAnalyzer(
        project1_path="./sample_project1",  # Replace with actual paths
        project2_path="./sample_project2",
        output_dir="./enhanced_example_output"
    )

    # Build knowledge graphs
    print("\nStep 1: Building knowledge graphs...")
    # analyzer.build_knowledge_graphs()

    # Find equivalent features
    print("\nStep 2: Finding equivalent features...")
    # equivalents = analyzer.find_equivalent_features()

    # Generate migration plan
    print("\nStep 3: Generating migration plan...")
    # migration_plan = analyzer.generate_migration_plan()

    # Search code
    print("\nStep 4: Searching code...")
    # results = analyzer.search_code("authentication")

    # Or run full analysis
    print("\nRunning full enhanced analysis...")
    # results = analyzer.run_full_analysis()

    print("\nAnalysis complete! Check ./enhanced_example_output/ for results")


def example_specific_operations():
    """Example of specific operations"""
    print("\n")
    print("=" * 70)
    print("Specific Operations Examples")
    print("=" * 70)

    # Example: Just find similar files
    print("\nExample: Find similar files only")
    # analyzer = ProjectUnificationAnalyzer("./p1", "./p2")
    # analyzer.p1_files = analyzer.analyze_project(analyzer.p1_path, "project1")
    # analyzer.p2_files = analyzer.analyze_project(analyzer.p2_path, "project2")
    # similarities = analyzer.find_similar_files()
    # print(f"Found {len(similarities)} similar files")

    # Example: Search for specific code
    print("\nExample: Search for specific code")
    # enhanced = EnhancedUnificationAnalyzer("./p1", "./p2")
    # enhanced.build_knowledge_graphs()
    # results = enhanced.search_code("def login")
    # for result in results:
    #     print(f"  - {result['name']} in {result['file_path']}")


def main():
    """Main function"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║         Project Unification Analyzer - Example Usage            ║
╚══════════════════════════════════════════════════════════════════╝

This script demonstrates how to use the analyzers.
To run with real projects, uncomment the code and provide actual paths.

Usage:
  1. Create sample projects or use existing ones
  2. Uncomment the code in the example functions
  3. Run: python example_usage.py

Features demonstrated:
  - Universal Analyzer for comprehensive analysis
  - Enhanced Analyzer for knowledge graphs
  - Code search capabilities
  - Migration planning

For more information, see README.md
""")

    # Uncomment to run examples:
    # example_universal_analyzer()
    # example_enhanced_analyzer()
    # example_specific_operations()

    print("\n✅ Examples defined. Uncomment code to run with your projects.\n")


if __name__ == "__main__":
    main()
