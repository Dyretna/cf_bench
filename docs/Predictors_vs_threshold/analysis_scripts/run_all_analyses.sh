#!/bin/bash
# Run all analysis scripts in sequence
# Usage: bash analysis_scripts/run_all_analyses.sh

echo "================================================================================"
echo "Running Generation 2 Experimental Analysis"
echo "================================================================================"
echo ""

# Create output directory
mkdir -p docs/Predictors_vs_threshold/analysis_scripts/output

echo "Step 1: Analyzing baseline models vs thresholds..."
echo "--------------------------------------------------------------------------------"
python3 docs/Predictors_vs_threshold/analysis_scripts/01_analyze_base_vs_thresholds.py
if [ $? -ne 0 ]; then
    echo "ERROR: Script 01 failed!"
    exit 1
fi
echo ""

echo "Step 2: Analyzing SMOTE and optimized models..."
echo "--------------------------------------------------------------------------------"
python3 docs/Predictors_vs_threshold/analysis_scripts/02_analyze_smote_and_optimized.py
if [ $? -ne 0 ]; then
    echo "ERROR: Script 02 failed!"
    exit 1
fi
echo ""

echo "Step 3: Generating comprehensive comparison..."
echo "--------------------------------------------------------------------------------"
python3 docs/Predictors_vs_threshold/analysis_scripts/03_comprehensive_comparison.py
if [ $? -ne 0 ]; then
    echo "ERROR: Script 03 failed!"
    exit 1
fi
echo ""

echo "================================================================================"
echo "Analysis complete!"
echo "================================================================================"
echo ""
echo "Output files generated:"
echo "  - docs/Predictors_vs_threshold/analysis_scripts/output/base_vs_thresholds_summary.csv"
echo "  - docs/Predictors_vs_threshold/analysis_scripts/output/smote_xgb_optimized_summary.csv"
echo "  - docs/Predictors_vs_threshold/analysis_scripts/output/all_experiments_combined.csv"
echo ""
echo "See docs/Generation_2_Experimental_Results_Analysis.md for detailed findings."
echo ""
